#!/usr/bin/env python3
"""Prepare Qwen3-SSD JSONL from the canonical WhiStress corpus adapters."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpora import SUPPORTED_CORPORA, compute_stress_binary, load_corpus


TARGET_FORMATS = ("ts", "gts", "s", "gs", "tgs")


def safe_file_stem(value: Any) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return stem[:80] or "sample"


def tagged_stress_pattern(transcription: str, emphasis_indices: list[int]) -> str:
    emphasis = set(emphasis_indices)
    return " ".join(
        f"<stress> {word} </stress>" if index in emphasis else word
        for index, word in enumerate(transcription.strip().split())
    )


def build_target(
    transcription: str,
    stress_pattern: str,
    gender: str,
    target_format: str,
) -> str:
    if target_format == "ts":
        payload = json.dumps({"stress_pattern": stress_pattern}, ensure_ascii=False)
        return f"language English<asr_text>{transcription}<ssd>{payload}"
    if target_format == "gts":
        payload = json.dumps(
            {"gender": gender, "stress_pattern": stress_pattern},
            ensure_ascii=False,
        )
        return f"language English<asr_text>{transcription}<ssd>{payload}"
    if target_format == "s":
        return f"language English<asr_text><ssd>{stress_pattern}"
    if target_format == "gs":
        return f"language English<asr_text><gender>{gender}<ssd>{stress_pattern}"
    if target_format == "tgs":
        return (
            f"language English<asr_text>{transcription}"
            f"<gender>{gender}<ssd>{stress_pattern}"
        )
    raise ValueError(f"Unsupported target format: {target_format}")


def write_audio(audio: dict[str, Any], path: Path) -> None:
    samples = np.asarray(audio["array"], dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=-1)
    if samples.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {samples.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        samples,
        int(audio["sampling_rate"]),
        subtype="PCM_16",
    )


def dataset_statistics(dataset) -> dict[str, Any]:
    defaults = {
        "num_original_samples": len(dataset),
        "num_filtered_invalid_emphasis": 0,
        "num_filtered_by_protocol": 0,
        "num_retained_samples": len(dataset),
    }
    description = getattr(dataset.info, "description", "")
    if description:
        try:
            defaults.update(json.loads(description))
        except (TypeError, json.JSONDecodeError):
            pass
    return defaults


def write_split(
    dataset,
    corpus: str,
    split: str,
    data_root: Path,
    jsonl_root: Path,
    prompt: str,
    target_format: str,
) -> dict[str, Any]:
    output_path = jsonl_root / corpus / f"{split}.jsonl"
    audio_root = data_root / "audio" / corpus / split
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, sample in enumerate(dataset):
            sample_id = str(sample["id"])
            transcription = sample["transcription"].strip()
            emphasis_indices = list(sample["emphasis_indices"])
            stress_pattern = tagged_stress_pattern(
                transcription,
                emphasis_indices,
            )
            gender = str(sample.get("gender", "") or "")
            audio_path = audio_root / f"{index:08d}_{safe_file_stem(sample_id)}.wav"
            write_audio(sample["audio"], audio_path)
            row = {
                "id": sample_id,
                "text_id": sample_id,
                "audio": str(audio_path.resolve()),
                "prompt": prompt,
                "text": build_target(
                    transcription,
                    stress_pattern,
                    gender,
                    target_format,
                ),
                "transcription": transcription,
                "stress": stress_pattern,
                "stress_pattern_binary": compute_stress_binary(
                    transcription,
                    emphasis_indices,
                ),
                "emphasis_indices": emphasis_indices,
                "source_dataset": sample.get("source_dataset", corpus),
                "target_format": target_format,
                "gender": gender,
            }
            for optional_key in ("speaker_id", "style", "voice"):
                if sample.get(optional_key) is not None:
                    row[optional_key] = sample[optional_key]
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = dataset_statistics(dataset)
    stats.update(
        corpus=corpus,
        split=split,
        num_samples=len(dataset),
        jsonl=str(output_path),
        audio_root=str(audio_root),
    )
    print(f"{corpus}/{split}: wrote {len(dataset)} samples to {output_path}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--jsonl-root", type=Path, default=Path("data-json"))
    parser.add_argument("--prompt-file", type=Path, default=Path("prompt/prompt_ts.txt"))
    parser.add_argument("--target", choices=TARGET_FORMATS, default="ts")
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=SUPPORTED_CORPORA,
        default=list(SUPPORTED_CORPORA),
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=66)
    args = parser.parse_args()

    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1")
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt is empty: {args.prompt_file}")

    manifests = []
    training = load_corpus("tinystress", "train", args.data_root / "raw")
    split_dataset = training.train_test_split(
        test_size=args.validation_ratio,
        seed=args.seed,
    )
    manifests.append(
        write_split(
            split_dataset["train"],
            "tinystress",
            "train",
            args.data_root,
            args.jsonl_root,
            prompt,
            args.target,
        )
    )
    manifests.append(
        write_split(
            split_dataset["test"],
            "tinystress",
            "dev",
            args.data_root,
            args.jsonl_root,
            prompt,
            args.target,
        )
    )

    for corpus in args.corpora:
        dataset = load_corpus(corpus, "test", args.data_root / "raw")
        manifests.append(
            write_split(
                dataset,
                corpus,
                "test",
                args.data_root,
                args.jsonl_root,
                prompt,
                args.target,
            )
        )

    manifest = {
        "target_format": args.target,
        "prompt_file": str(args.prompt_file),
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "splits": manifests,
    }
    manifest_path = args.jsonl_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
