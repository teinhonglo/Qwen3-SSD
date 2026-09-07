#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Union

import librosa
import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")
_TARGET_FORMATS = ("ts", "gts", "s", "gs", "tgs")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        match = _CKPT_RE.match(name)
        path = os.path.join(output_dir, name)
        if match and os.path.isdir(path):
            candidates.append((int(match.group(1)), path))
    return max(candidates)[1] if candidates else None


def resolve_checkpoint(
    exp_dir: str,
    use_best: bool,
    use_latest: bool,
    use_last: bool = False,
) -> str:
    if sum((use_best, use_latest, use_last)) > 1:
        raise ValueError(
            "--auto_best_checkpoint, --auto_latest_checkpoint, and "
            "--auto_last_checkpoint are mutually exclusive"
        )
    if use_best:
        checkpoint = os.path.join(exp_dir, "checkpoint-best")
        if not os.path.isdir(checkpoint):
            raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
        return checkpoint
    if use_last:
        checkpoint = os.path.join(exp_dir, "checkpoint-last")
        if not os.path.isdir(checkpoint):
            raise FileNotFoundError(f"Last checkpoint not found: {checkpoint}")
        return checkpoint
    if use_latest:
        checkpoint = find_latest_checkpoint(exp_dir)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint-* directory found under: {exp_dir}")
        return checkpoint
    return exp_dir


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array=None):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def build_prefix_text(processor, prompt: str) -> str:
    prefix_text = processor.apply_chat_template(
        [build_prefix_messages(prompt, None)],
        add_generation_prompt=True,
        tokenize=False,
    )
    return prefix_text[0] if isinstance(prefix_text, list) else prefix_text


def move_inputs_to_device(inputs: Dict[str, Any], device, model_dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(model_dtype)
        moved[key] = value
    return moved


def batch_decode_text(processor, token_ids):
    decoder = processor.batch_decode if hasattr(processor, "batch_decode") else processor.tokenizer.batch_decode
    return decoder(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def unwrap_generate_output(generate_output):
    if hasattr(generate_output, "sequences"):
        return generate_output.sequences
    if isinstance(generate_output, dict) and "sequences" in generate_output:
        return generate_output["sequences"]
    if isinstance(generate_output, (tuple, list)):
        return generate_output[0]
    return generate_output


def infer_one(
    asr_wrapper,
    audio_path: str,
    prompt: str,
    sr: int,
    generation: Dict[str, Any],
    decoding_mode: str,
    dola_conf: Optional[Dict[str, Any]] = None,
) -> Union[str, List[str]]:
    processor = asr_wrapper.processor
    model = asr_wrapper.model
    device = next(model.parameters()).device
    model_dtype = getattr(model, "dtype", torch.float16)

    inputs = processor(
        text=[build_prefix_text(processor, prompt)],
        audio=[load_audio(audio_path, sr=sr)],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    prefix_len = int(inputs["attention_mask"][0].sum().item())
    inputs = move_inputs_to_device(inputs, device, model_dtype)

    num_return_sequences = int(generation["num_return_sequences"])
    beam_size = max(int(generation["beam_size"]), num_return_sequences)
    generate_kwargs = {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "do_sample": bool(generation["do_sample"]),
        "repetition_penalty": float(generation["repetition_penalty"]),
        "num_return_sequences": num_return_sequences,
    }
    if generate_kwargs["do_sample"]:
        generate_kwargs.update(
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
        )
        if int(generation["top_k"]) > 0:
            generate_kwargs["top_k"] = int(generation["top_k"])
    if beam_size > 1:
        generate_kwargs["num_beams"] = beam_size

    if decoding_mode == "dola":
        dola_conf = dola_conf or {}
        generate_kwargs["dola_layers"] = dola_conf.get("layers", "high")
        generate_kwargs["repetition_penalty"] = float(
            dola_conf.get("repetition_penalty", generate_kwargs["repetition_penalty"])
        )
    elif decoding_mode not in {"basic", "layer_lmhead"}:
        raise ValueError(f"Unsupported decoding mode: {decoding_mode}")

    model.eval()
    with torch.inference_mode():
        output_ids = unwrap_generate_output(model.generate(**inputs, **generate_kwargs))
    if not torch.is_tensor(output_ids):
        raise TypeError(f"generate() returned unsupported type: {type(output_ids)}")
    if output_ids.ndim == 1:
        output_ids = output_ids.unsqueeze(0)
    generated_ids = output_ids[:, prefix_len:] if output_ids.shape[1] > prefix_len else output_ids
    decoded = [text.strip() for text in batch_decode_text(processor, generated_ids)]
    return decoded if num_return_sequences > 1 else (decoded[0] if decoded else "")


def extract_first_json_dict(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start:index + 1])
                    return value if isinstance(value, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def remove_stress_tags(text: str) -> str:
    return " ".join(
        (text or "").replace("<stress>", " ").replace("</stress>", " ").split()
    )


def strip_asr_prefix(text: str) -> str:
    text = re.sub(r"^language\s+[^<\s]+", "", (text or "").strip(), flags=re.IGNORECASE)
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    return text.strip()


def parse_ssd_output(raw_text: str, target_format: str) -> Dict[str, Any]:
    result = {
        "pred_transcription": "",
        "pred_stress": "",
        "pred_gender": "",
        "parse_error": "",
    }
    if target_format not in _TARGET_FORMATS:
        result["parse_error"] = f"unsupported target format: {target_format}"
        return result
    if "<ssd>" not in raw_text:
        result["parse_error"] = "missing <ssd> separator"
        return result

    prefix, payload = raw_text.split("<ssd>", 1)
    prefix = strip_asr_prefix(prefix)
    payload = payload.strip()

    if target_format in {"ts", "gts"}:
        parsed = extract_first_json_dict(payload)
        result["pred_stress"] = str(parsed.get("stress_pattern", "") or "").strip()
        result["pred_gender"] = str(parsed.get("gender", "") or "").strip()
        result["pred_transcription"] = prefix
        if not parsed:
            result["parse_error"] = "invalid SSD JSON payload"
    elif target_format == "s":
        result["pred_stress"] = payload
        result["pred_transcription"] = remove_stress_tags(payload)
    elif target_format == "gs":
        if "<gender>" in prefix:
            _, result["pred_gender"] = prefix.split("<gender>", 1)
        result["pred_gender"] = result["pred_gender"].strip()
        result["pred_stress"] = payload
        result["pred_transcription"] = remove_stress_tags(payload)
    elif target_format == "tgs":
        if "<gender>" in prefix:
            transcription, gender = prefix.split("<gender>", 1)
            result["pred_transcription"] = transcription.strip()
            result["pred_gender"] = gender.strip()
        else:
            result["pred_transcription"] = prefix
            result["parse_error"] = "missing <gender> separator"
        result["pred_stress"] = payload

    if not result["pred_stress"] and not result["parse_error"]:
        result["parse_error"] = "empty stress prediction"
    if not result["pred_transcription"] and result["pred_stress"]:
        result["pred_transcription"] = remove_stress_tags(result["pred_stress"])
    return result


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number} in {path}: {error}") from error
    return rows


def load_train_conf(exp_dir: str):
    path = os.path.join(exp_dir, "train_conf.json")
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, list) or len(config) != 2 or not all(isinstance(x, dict) for x in config):
        raise ValueError("train_conf.json must contain [training_args, model_args]")
    return config


def load_decoding_conf(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("decoding config must be a JSON object")
    return config


def resolve_decoding_conf(model_args: Dict[str, Any], decoding_conf: Dict[str, Any]):
    decoding = decoding_conf.get("decoding", {})
    generation = decoding.get("generation", {})
    num_return_sequences = int(generation.get("num_return_sequences", model_args.get("num_return_sequences", 1)))
    beam_size = int(generation.get("beam_size", generation.get("num_beams", model_args.get("beam_size", 1))))
    return {
        "mode": str(decoding.get("mode", "basic")),
        "strict": bool(decoding.get("strict", False)),
        "generation": {
            "max_new_tokens": int(generation.get("max_new_tokens", model_args.get("max_new_tokens", 256))),
            "do_sample": bool(generation.get("do_sample", model_args.get("do_sample", False))),
            "temperature": float(generation.get("temperature", model_args.get("temperature", 0.0))),
            "top_p": float(generation.get("top_p", model_args.get("top_p", 1.0))),
            "top_k": int(generation.get("top_k", 0)),
            "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
            "num_return_sequences": max(1, num_return_sequences),
            "beam_size": max(1, beam_size, num_return_sequences),
        },
        "dola": decoding.get("dola", {}),
        "layer_lmhead": decoding.get("layer_lmhead", {}),
    }


def resolve_dtype(dtype_string: str, device: str):
    explicit = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_string in explicit:
        return explicit[dtype_string]
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


def build_model(model_args, checkpoint_path, device, dtype, resolved_decoding):
    lora_config = model_args.get("lora_config")
    if lora_config:
        wrapper = Qwen3ASRModel.from_pretrained(
            model_args["model_path"],
            dtype=dtype,
            device_map=device,
        )
        wrapper.model = PeftModel.from_pretrained(
            wrapper.model,
            checkpoint_path,
            torch_dtype=dtype,
        )
    else:
        wrapper = Qwen3ASRModel.from_pretrained(
            checkpoint_path,
            dtype=dtype,
            device_map=device,
        )
    if resolved_decoding["mode"] == "layer_lmhead":
        setter = getattr(wrapper.model, "set_layer_lmhead_index", None)
        if setter is None:
            raise AttributeError("The model does not support layer_lmhead decoding")
        setter(int(resolved_decoding["layer_lmhead"].get("layer_index", -1)))
    return wrapper


def output_subdir(input_jsonl: str, decoding_conf_path: str) -> str:
    data_name = os.path.splitext(os.path.basename(input_jsonl))[0]
    decoding_name = os.path.splitext(os.path.basename(decoding_conf_path))[0]
    return f"{data_name}_{decoding_name}"


def infer_rows(
    asr_wrapper,
    rows: List[Dict[str, Any]],
    sr: int,
    resolved_decoding: Dict[str, Any],
    target: Optional[str] = None,
    log_every: int = 1,
    progress_label: str = "",
) -> List[Dict[str, Any]]:
    output_rows = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        text_id = str(row.get("text_id", row.get("id", f"line{index}")))
        target_format = target or row.get("target_format", "ts")
        prediction = {
            "id": text_id,
            "text_id": text_id,
            "source_dataset": row.get("source_dataset", ""),
            "transcription": row.get("transcription", ""),
            "emphasis_indices": row.get("emphasis_indices", []),
            "pred_raw": "",
            "pred_transcription": "",
            "pred_stress": "",
            "pred_gender": "",
            "parse_error": "",
        }
        audio_path = row.get("audio", "")
        if not audio_path or not os.path.isfile(audio_path):
            prediction["parse_error"] = f"missing audio: {audio_path}"
            output_rows.append(prediction)
            continue
        try:
            generated = infer_one(
                asr_wrapper,
                audio_path=audio_path,
                prompt=str(row.get("prompt", "") or ""),
                sr=sr,
                generation=resolved_decoding["generation"],
                decoding_mode=resolved_decoding["mode"],
                dola_conf=resolved_decoding["dola"],
            )
            nbest = generated if isinstance(generated, list) else [generated]
            prediction["pred_raw"] = nbest[0] if nbest else ""
            prediction["nbest"] = nbest
            prediction.update(parse_ssd_output(prediction["pred_raw"], target_format))
        except Exception as error:
            prediction["parse_error"] = f"{type(error).__name__}: {error}"
        output_rows.append(prediction)
        if log_every > 0 and (index % log_every == 0 or index == total):
            label = f"{progress_label} " if progress_label else ""
            print(f"[{label}{index}/{total}] {text_id}: {prediction['parse_error'] or 'ok'}")
    return output_rows


def parse_args():
    parser = argparse.ArgumentParser("Qwen3-ASR SSD inference")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--auto_latest_checkpoint", action="store_true")
    parser.add_argument("--auto_best_checkpoint", action="store_true")
    parser.add_argument("--auto_last_checkpoint", action="store_true")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_root", default="checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decoding_conf", default="conf/decoding/basic_decoding.json")
    parser.add_argument("--target", choices=_TARGET_FORMATS, default=None)
    parser.add_argument("--num_return_sequences", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    _, model_args = load_train_conf(args.exp_dir)
    resolved_decoding = resolve_decoding_conf(model_args, load_decoding_conf(args.decoding_conf))
    if args.num_return_sequences is not None:
        resolved_decoding["generation"]["num_return_sequences"] = max(1, args.num_return_sequences)
    if args.beam_size is not None:
        resolved_decoding["generation"]["beam_size"] = max(1, args.beam_size)
    resolved_decoding["generation"]["beam_size"] = max(
        resolved_decoding["generation"]["beam_size"],
        resolved_decoding["generation"]["num_return_sequences"],
    )

    checkpoint_path = resolve_checkpoint(
        args.exp_dir,
        args.auto_best_checkpoint,
        args.auto_latest_checkpoint,
        args.auto_last_checkpoint,
    )
    print(f"[info] use checkpoint: {checkpoint_path}")
    dtype = resolve_dtype(str(model_args.get("dtype", "auto")), args.device)
    wrapper = build_model(model_args, checkpoint_path, args.device, dtype, resolved_decoding)

    save_dir = os.path.join(args.output_root, output_subdir(args.input_jsonl, args.decoding_conf))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "resolved_decoding_config.json"), "w", encoding="utf-8") as file:
        json.dump(resolved_decoding, file, ensure_ascii=False, indent=2)

    rows = load_jsonl(args.input_jsonl)
    output_rows = infer_rows(
        wrapper,
        rows=rows,
        sr=int(model_args.get("sr", 16000)),
        resolved_decoding=resolved_decoding,
        target=args.target,
    )

    output_path = os.path.join(save_dir, "predictions.jsonl")
    with open(output_path, "w", encoding="utf-8") as file:
        for row in output_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[info] saved: {output_path}")


if __name__ == "__main__":
    main()
