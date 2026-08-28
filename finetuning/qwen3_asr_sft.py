# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import random

import librosa
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments, BitsAndBytesConfig)
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel

def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


def extract_default_prompt(dataset) -> str:
    prompts = []
    for ex in dataset:
        p = str(ex.get("prompt", "") or "").strip()
        if p:
            prompts.append(p)

    if not prompts:
        return ""

    first = prompts[0]
    if any(p != first for p in prompts[1:]):
        print("[warn] Multiple prompt values found in train set; using the first non-empty prompt for prompt.txt")
    return first


def save_prompt_txt(save_dir: str, prompt: str):
    os.makedirs(save_dir, exist_ok=True)
    prompt_path = os.path.join(save_dir, "prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt or "")

class CastFloatInputsTrainer(Trainer):
    def __init__(self, *args, spec_aug_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec_aug_config = spec_aug_config or {}

    @staticmethod
    def _mask_axis(features, axis, max_width, valid_width=None):
        if max_width <= 0:
            return features
        axis_size = features.shape[axis]
        usable = min(axis_size, valid_width) if valid_width is not None else axis_size
        if usable <= 1:
            return features
        width = int(torch.randint(0, min(max_width, usable) + 1, (1,)).item())
        if width == 0:
            return features
        start = int(torch.randint(0, usable - width + 1, (1,)).item())
        index = [slice(None)] * features.ndim
        index[axis] = slice(start, start + width)
        features[tuple(index)] = 0
        return features

    def _apply_spec_augment(self, inputs):
        config = self.spec_aug_config
        if not self.model.training or not config.get("apply", False):
            return inputs
        features = inputs.get("input_features")
        if not torch.is_tensor(features) or features.ndim != 3:
            return inputs

        features = features.clone()
        feature_mask = inputs.get("feature_attention_mask")
        mask_length = feature_mask.shape[-1] if torch.is_tensor(feature_mask) else None
        if mask_length == features.shape[1]:
            time_axis, freq_axis = 1, 2
        elif mask_length == features.shape[2]:
            time_axis, freq_axis = 2, 1
        else:
            # Qwen processors may expose either [B, T, F] or [B, F, T].
            # The longer non-batch dimension is treated as time when no mask is available.
            time_axis, freq_axis = (1, 2) if features.shape[1] >= features.shape[2] else (2, 1)

        for batch_index in range(features.shape[0]):
            sample = features[batch_index]
            sample_time_axis = time_axis - 1
            sample_freq_axis = freq_axis - 1
            valid_frames = None
            if torch.is_tensor(feature_mask):
                valid_frames = int(feature_mask[batch_index].sum().item())
            self._mask_axis(
                sample,
                sample_freq_axis,
                int(config.get("freq_mask_param", 27)),
            )
            self._mask_axis(
                sample,
                sample_time_axis,
                int(config.get("time_mask_param", 50)),
                valid_width=valid_frames,
            )
        inputs["input_features"] = features
        return inputs

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return self._apply_spec_augment(inputs)


def apply_freeze_components(model, freeze_components):
    if isinstance(freeze_components, str):
        freeze_components = [freeze_components] if freeze_components.strip() else []
    if not isinstance(freeze_components, list):
        raise ValueError("model_args.freeze_components must be a string or list of strings")

    named_modules = dict(model.named_modules())
    named_parameters = dict(model.named_parameters())
    frozen = []
    for requested_name in freeze_components:
        requested_name = str(requested_name).strip()
        if not requested_name:
            continue
        if requested_name in named_parameters:
            named_parameters[requested_name].requires_grad = False
            frozen.append(f"parameter:{requested_name}")
            continue

        matches = [
            name for name in named_modules
            if name == requested_name or name.endswith(f".{requested_name}")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"freeze component {requested_name!r} matched {matches or 'nothing'}; "
                "use an unambiguous name from model.named_modules()"
            )
        for parameter in named_modules[matches[0]].parameters():
            parameter.requires_grad = False
        frozen.append(f"module:{matches[0]}")

    if frozen:
        print(f"[freeze] {', '.join(frozen)}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable={trainable:,} / total={total:,} ({100.0 * trainable / total:.2f}%)")
    return model

class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, processor, model=None, default_prompt: str = ""):
        self.processor = processor
        self.model = model
        self.default_prompt = default_prompt

    def _save_infer_files(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        self.processor.save_pretrained(save_dir)

        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.save_pretrained(save_dir)

        if self.model is not None and getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.save_pretrained(save_dir)

        save_prompt_txt(save_dir, self.default_prompt)

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        self._save_infer_files(ckpt_dir)
        return control

def save_best_checkpoint(
    best_src: str,
    output_dir: str,
    processor=None,
    model=None,
    default_prompt: str = "",
    best_ckpt_name: str = "checkpoint-best",
):
    if not best_src or not os.path.isdir(best_src):
        print(
            "[best] checkpoint-best not created: no best_model_checkpoint was selected. "
            "Please make sure evaluation runs and load_best_model_at_end=true."
        )
        return

    best_ckpt_dir = os.path.join(output_dir, best_ckpt_name)
    if os.path.exists(best_ckpt_dir):
        shutil.rmtree(best_ckpt_dir)
    shutil.copytree(best_src, best_ckpt_dir)

    if processor is not None:
        processor.save_pretrained(best_ckpt_dir)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.save_pretrained(best_ckpt_dir)

    if model is not None and getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(best_ckpt_dir)

    save_prompt_txt(best_ckpt_dir, default_prompt)
    print(f"[best] Saved best checkpoint from {best_src} to {best_ckpt_dir}")


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--train_conf", type=str, required=True,
                   help="JSON config path with format: [training_args, model_args]")
    p.add_argument('--seed', type=int, default=66)
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="dev.jsonl")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Resume / warm start
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    p.add_argument(
        "--init_from_checkpoint",
        type=str,
        default="",
        help="Warm-start model/adapter weights from a checkpoint without resuming optimizer/scheduler state",
    )

    return p.parse_args()

def load_train_conf(train_conf_path: str) -> Optional[List[Dict[str, Any]]]:
    if not train_conf_path:
        return None

    with open(train_conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, list) or len(cfg) != 2:
        raise ValueError("train_conf must be a list in format: [training_args, model_args]")

    training_args, model_args = cfg
    if not isinstance(training_args, dict) or not isinstance(model_args, dict):
        raise ValueError("train_conf entries must both be dictionaries")
    return [training_args, model_args]

def main():
    args_cli = parse_args()

    seed = args_cli.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    train_conf = load_train_conf(args_cli.train_conf)
    if train_conf is None:
        raise ValueError("--train_conf is required")

    training_args_conf, model_args_conf = train_conf
    training_args_conf = dict(training_args_conf)

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    model_path = model_args_conf.get("model_path")
    if not model_path:
        raise KeyError("model_args.model_path is required in train_conf")

    sr = int(model_args_conf.get("sr", 16000))
    eval_max_new_tokens = int(model_args_conf.get("eval_max_new_tokens", 256))

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    # LoRA
    lora_config = model_args_conf.get("lora_config", None)
    lora_type = model_args_conf.get("lora_type", "default")
    
    if lora_type == "qlora":
        # load pretrained model (reload)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            quantization_config=bnb_config,
            device_map=None,
        )
    else:
        # load pretrained model
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            device_map=None,
        )
        
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    
    init_from_checkpoint = (args_cli.init_from_checkpoint or "").strip()
    if init_from_checkpoint and not os.path.isdir(init_from_checkpoint):
        raise FileNotFoundError(f"init_from_checkpoint not found: {init_from_checkpoint}")

    if lora_config:
        if lora_type not in ["default", "qlora"]:
            raise ValueError(f"lora_type: {lora_type} is NOT implemented yet.")

        print(f"LoRA Finetuning {lora_type}")
        if init_from_checkpoint:
            print(f"[init] warm-start LoRA adapter from checkpoint = {init_from_checkpoint}")
            model = PeftModel.from_pretrained(
                model,
                init_from_checkpoint,
                is_trainable=True,
            )
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                **lora_config
            )
            model = get_peft_model(model, peft_config)
        print("="*100)
        model.print_trainable_parameters()
        print("="*100)
    else:
        if init_from_checkpoint:
            raise ValueError("--init_from_checkpoint currently supports LoRA/QLoRA checkpoints only")
        print("Full Finetuning")

    model = apply_freeze_components(
        model,
        model_args_conf.get("freeze_components", []),
    )
    
    if training_args_conf["gradient_checkpointing"]:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            "validation": args_cli.eval_file,
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    default_prompt = extract_default_prompt(ds["train"])

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=sr)

    training_args_conf["run_name"] = os.path.basename(args_cli.output_dir)
    if model_args_conf.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = model_args_conf["wandb_project"]
    os.environ["WANDB_LOG_MODEL"] = str(model_args_conf.get("wandb_log_model", "false")).lower()

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        do_eval=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        **training_args_conf
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[
            MakeEveryCheckpointInferableCallback(
                processor=processor,
                model=model,
                default_prompt=default_prompt,
            ),
        ],
        spec_aug_config=model_args_conf.get("spec_aug", {}),
    )

    os.makedirs(training_args.output_dir, exist_ok=True)

    if train_conf is not None and trainer.args.process_index == 0:
        saved_train_conf = os.path.join(training_args.output_dir, "train_conf.json")
        with open(saved_train_conf, "w", encoding="utf-8") as f:
            json.dump(train_conf, f, ensure_ascii=False, indent=4)

    processor.save_pretrained(training_args.output_dir)

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.save_pretrained(training_args.output_dir)

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(training_args.output_dir)

    resume_from = (args_cli.resume_from or "").strip()
    if init_from_checkpoint and (resume_from or args_cli.resume == 1):
        raise ValueError("--init_from_checkpoint warm-starts weights and cannot be combined with --resume/--resume_from")
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    if trainer.args.process_index == 0:
        save_best_checkpoint(
            best_src=getattr(trainer.state, "best_model_checkpoint", None),
            output_dir=training_args.output_dir,
            processor=processor,
            model=model,
            default_prompt=default_prompt,
        )
        save_prompt_txt(training_args.output_dir, default_prompt)


if __name__ == "__main__":
    main()
