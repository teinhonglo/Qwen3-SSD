#!/usr/bin/env python3
"""Evaluate Qwen3-SSD predictions with the WhiStress corpus protocol."""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import wave
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import pyphen
except ImportError:
    pyphen = None

HYPHENATOR = pyphen.Pyphen(lang="en") if pyphen is not None else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number} in {path}: {error}") from error
    return rows


def extract_stress_binary(stress_pattern: str) -> tuple[list[int], list[str]]:
    tokens = (
        (stress_pattern or "")
        .replace("<stress>", " <stress> ")
        .replace("</stress>", " </stress> ")
        .split()
    )
    predictions = []
    words = []
    stressed = False
    for token in tokens:
        if token == "<stress>":
            stressed = True
        elif token == "</stress>":
            stressed = False
        else:
            predictions.append(int(stressed))
            words.append(token)
    return predictions, words


def normalize_words(text: str) -> list[str]:
    text = (text or "").upper()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row_index, reference_token in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + int(reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def binary_metrics(predictions: list[int], references: list[int]) -> dict[str, Any]:
    tp = sum(prediction == 1 and reference == 1 for prediction, reference in zip(predictions, references))
    fp = sum(prediction == 1 and reference == 0 for prediction, reference in zip(predictions, references))
    fn = sum(prediction == 0 and reference == 1 for prediction, reference in zip(predictions, references))
    tn = sum(prediction == 0 and reference == 0 for prediction, reference in zip(predictions, references))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "num_words": len(references),
    }


def count_syllables(word: str) -> int:
    if HYPHENATOR is None:
        return 1
    hyphenated = HYPHENATOR.inserted(word)
    return len(hyphenated.split("-")) if hyphenated else 1


def audio_duration(path: str) -> float:
    try:
        if sf is not None:
            info = sf.info(path)
            return info.frames / info.samplerate
        with wave.open(path, "rb") as audio_file:
            return audio_file.getnframes() / audio_file.getframerate()
    except Exception:
        return 0.0


def reference_labels(row: dict[str, Any]) -> list[int]:
    if isinstance(row.get("stress_pattern_binary"), list):
        return [int(value) for value in row["stress_pattern_binary"]]
    words = row["transcription"].strip().split()
    emphasis = set(row.get("emphasis_indices", []))
    return [int(index in emphasis) for index in range(len(words))]


def find_manifest_stats(
    manifest_path: Path | None,
    corpus: str,
    split: str,
    num_samples: int,
) -> dict[str, Any]:
    stats = {
        "num_original_samples": num_samples,
        "num_filtered_invalid_emphasis": 0,
        "num_filtered_by_protocol": 0,
        "num_retained_samples": num_samples,
    }
    if manifest_path is None or not manifest_path.is_file():
        return stats
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("splits", []):
        if item.get("corpus") == corpus and item.get("split") == split:
            stats.update(item)
            break
    return stats


def gender_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any] | None:
    pairs = [
        (prediction.lower(), reference.lower())
        for prediction, reference in pairs
        if reference.lower() in {"female", "male"}
    ]
    if not pairs:
        return None
    tp = sum(prediction == "female" and reference == "female" for prediction, reference in pairs)
    tn = sum(prediction == "male" and reference == "male" for prediction, reference in pairs)
    fp = sum(prediction == "female" and reference == "male" for prediction, reference in pairs)
    fn = sum(prediction == "male" and reference == "female" for prediction, reference in pairs)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": (tp + tn) / len(pairs),
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "num_samples": len(pairs),
        "reference_female_rate": sum(reference == "female" for _, reference in pairs) / len(pairs),
    }


def evaluate_rows(
    prediction_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    corpus: str,
    split: str,
    manifest_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions_by_id = {
        str(row.get("id", row.get("text_id"))): row for row in prediction_rows
    }
    duplicate_count = len(prediction_rows) - len(predictions_by_id)

    all_predictions = []
    all_references = []
    gender_pairs = []
    error_cases = []
    skip_reasons = Counter()
    asr_errors = 0
    tagged_errors = 0
    reference_word_count = 0

    for reference_row in reference_rows:
        sample_id = str(reference_row.get("id", reference_row.get("text_id")))
        prediction_row = predictions_by_id.get(sample_id)
        if prediction_row is None:
            skip_reasons["missing_prediction"] += 1
            continue
        if prediction_row.get("parse_error"):
            skip_reasons["parse_error"] += 1
            continue

        predicted_labels, predicted_words = extract_stress_binary(
            prediction_row.get("pred_stress", "")
        )
        labels = reference_labels(reference_row)
        words = reference_row["transcription"].strip().split()
        if len(predicted_labels) != len(labels):
            skip_reasons["word_length_mismatch"] += 1
            continue

        all_predictions.extend(predicted_labels)
        all_references.extend(labels)
        gender_pairs.append(
            (
                str(prediction_row.get("pred_gender", "") or ""),
                str(reference_row.get("gender", "") or ""),
            )
        )

        reference_tokens = normalize_words(reference_row["transcription"])
        asr_tokens = normalize_words(prediction_row.get("pred_transcription", ""))
        tagged_tokens = normalize_words(" ".join(predicted_words))
        asr_errors += edit_distance(reference_tokens, asr_tokens)
        tagged_errors += edit_distance(reference_tokens, tagged_tokens)
        reference_word_count += len(reference_tokens)

        duration = audio_duration(reference_row.get("audio", ""))
        word_results = []
        for index, (word, reference, prediction) in enumerate(
            zip(words, labels, predicted_labels)
        ):
            result_type = {
                (1, 1): "TP",
                (0, 0): "TN",
                (1, 0): "FN",
                (0, 1): "FP",
            }[(reference, prediction)]
            word_results.append(
                {
                    "index": index,
                    "word": word,
                    "gt": reference,
                    "pred": prediction,
                    "type": result_type,
                    "word_len": len(word),
                    "syllable_count": count_syllables(word),
                }
            )
        error_cases.append(
            {
                "id": sample_id,
                "source_dataset": reference_row.get("source_dataset", corpus),
                "transcription": reference_row["transcription"],
                "pred_transcription": prediction_row.get("pred_transcription", ""),
                "pred_stress": prediction_row.get("pred_stress", ""),
                "utt_len": len(words),
                "utt_duration": duration,
                "speaking_rate": len(words) / duration if duration else 0.0,
                "gt_stresses": labels,
                "pred_stresses": predicted_labels,
                "words": word_results,
            }
        )

    num_samples = len(reference_rows)
    num_skipped = sum(skip_reasons.values())
    num_decode_attempted = num_samples - skip_reasons["missing_prediction"]
    num_decode_failures = skip_reasons["parse_error"]
    metrics = binary_metrics(all_predictions, all_references) if all_references else None
    stats = find_manifest_stats(manifest_path, corpus, split, num_samples)
    results = {
        "dataset": corpus,
        "split": split,
        "num_original_samples": stats["num_original_samples"],
        "num_filtered_invalid_emphasis": stats.get("num_filtered_invalid_emphasis", 0),
        "num_filtered_by_protocol": stats.get("num_filtered_by_protocol", 0),
        "num_samples": num_samples,
        "metrics": metrics,
        "transcription_metrics": {
            "wer": asr_errors / reference_word_count if reference_word_count else None,
            "tagged_transcript_wer": (
                tagged_errors / reference_word_count if reference_word_count else None
            ),
            "reference_words": reference_word_count,
            "asr_errors": asr_errors,
            "tagged_transcript_errors": tagged_errors,
        },
        "gender_metrics": gender_metrics(gender_pairs),
        "coverage": {
            "num_samples": num_samples,
            "num_evaluated": num_samples - num_skipped,
            "num_skipped": num_skipped,
            "coverage_rate": (num_samples - num_skipped) / num_samples if num_samples else 0.0,
            "num_decode_attempted": num_decode_attempted,
            "num_decode_failures": num_decode_failures,
            "decode_failure_rate": (
                num_decode_failures / num_decode_attempted
                if num_decode_attempted
                else 0.0
            ),
            "skip_reasons": dict(skip_reasons),
            "duplicate_prediction_ids": duplicate_count,
        },
    }
    return results, error_cases


def evaluate(
    predictions_path: Path,
    references_path: Path,
    corpus: str,
    split: str,
    manifest_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return evaluate_rows(
        load_jsonl(predictions_path),
        load_jsonl(references_path),
        corpus,
        split,
        manifest_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    results, error_cases = evaluate(
        args.predictions,
        args.references,
        args.corpus,
        args.split,
        args.manifest,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = args.results_dir / "qwen3_ssd_evaluation.json"
    error_path = args.results_dir / "qwen3_ssd_error_analysis.json"
    evaluation_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    error_path.write_text(
        json.dumps(error_cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved evaluation: {evaluation_path}")
    print(f"Saved error analysis: {error_path}")


if __name__ == "__main__":
    main()
