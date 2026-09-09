# Qwen3-SSD finetuning

The repository-level `run.sh` is the recommended entry point for the complete
preparation, training, inference, evaluation, and plotting pipeline. This
directory contains the two core commands used by that pipeline.

## JSONL contract

Training, development, and test files use one JSON object per line:

```json
{"audio": "/absolute/or/relative/path.wav", "prompt": "task instruction", "text": "target response", "target_format": "ts"}
```

Use `local/prepare_ssd_jsonl.py` to generate this format consistently for all
supported corpora and target formats.

For the sentence `I bought a new record`, with `record` stressed at zero-based
index 4, Stage 0 writes the following `text` values:

| `target_format` | JSONL `text` value |
| --- | --- |
| `ts` | `language English<asr_text>I bought a new record<ssd>{"stress_pattern": "I bought a new <stress> record </stress>"}` |
| `gts` | `language English<asr_text>I bought a new record<ssd>{"gender": "female", "stress_pattern": "I bought a new <stress> record </stress>"}` |
| `s` | `language English<asr_text><ssd>I bought a new <stress> record </stress>` |
| `gs` | `language English<asr_text><gender>female<ssd>I bought a new <stress> record </stress>` |
| `tgs` | `language English<asr_text>I bought a new record<gender>female<ssd>I bought a new <stress> record </stress>` |
| `asr_ssd_json` | `{"asr_text": "I bought a new record", "ssd": [{"record": 4}]}` |

The other JSONL fields stay the same. `target_format` is stored in every row
and is the parser contract used by both training-time evaluation and test
inference.

## Training

```bash
python finetuning/qwen3_asr_sft.py \
  --train_conf conf/tinystress_qwen3_asr_06b.json \
  --train_file data-json/tinystress/train.jsonl \
  --eval_file data-json/tinystress/dev.jsonl \
  --decoding_conf conf/decoding/basic_decoding.json \
  --manifest data-json/manifest.json \
  --output_dir exp/tinystress/ts
```

A training config is a two-object JSON array: `[training_args, model_args]`.
The first object is forwarded to Hugging Face `TrainingArguments`. The second
selects the base model and model-specific settings such as sample rate,
generation defaults, component freezing, SpecAugment, LoRA/QLoRA, and W&B
metadata. Use `--resume`, `--resume_from`, or `--init_from_checkpoint` when
continuing an experiment.

The provided adaptation configs follow the Qwen3-SLU LoRA layout:

| Config | Trainable scope |
| --- | --- |
| `tinystress_qwen3_asr_06b.json` | Full model |
| `tinystress_qwen3_asr_06b_freeze.json` | Full fine-tuning except `thinker.audio_tower` |
| `tinystress_qwen3_asr_06b_lora.json` | LoRA plus trainable token embeddings and LM head |
| `tinystress_qwen3_asr_06b_lora_woemblmhead_frozenaudio.json` | Text-backbone LoRA only; audio tower, token embeddings, and LM head remain frozen |

The frozen-audio LoRA config uses both the PEFT `exclude_modules` regex and
`freeze_components` as an explicit safeguard. It requires PEFT 0.14.0 or
newer. Select a variant through the existing `--train_conf` option.

Set `model_args.eval_generation_metrics` to `true` to run the same greedy
generation and SSD evaluation used by the inference/evaluation stages after
each validation pass. The Trainer logs `eval_ssd_precision`,
`eval_ssd_recall`, `eval_ssd_f1`, `eval_coverage_rate`,
`eval_decode_failure_rate`, `eval_wer`, and `eval_tagged_transcript_wer` in
addition to `eval_loss`. This evaluates the full development set by default.
Set `model_args.eval_generation_max_samples` to a positive integer only when a
faster, approximate validation metric is preferred. The TinyStress configs
evaluate and save at the end of every epoch, let Trainer reload the checkpoint
selected by `eval_ssd_f1`, and preserve the evaluated final epoch as
`checkpoint-last`. The selected checkpoint is also copied to
`checkpoint-best`, and the final epoch is not decoded a second time.

## Inference

```bash
python finetuning/qwen3_asr_test.py \
  --exp_dir exp/tinystress/ts \
  --auto_best_checkpoint \
  --input_jsonl data-json/tinystress/test.jsonl \
  --output_root exp/tinystress/ts/tinystress \
  --decoding_conf conf/decoding/basic_decoding.json \
  --device cuda:0
```

`--auto_latest_checkpoint` selects the most recent numbered periodic
checkpoint. `--auto_last_checkpoint` selects the evaluated final model in
`checkpoint-last`. The supported SSD targets are `ts`, `gts`, `s`, `gs`,
`tgs`, and `asr_ssd_json`. The latter generates a pure JSON object such as
`{"asr_text": "I bought a new record", "ssd": [{"record": 4}]}`, where word
indices are zero-based. Both training-time generation evaluation and test
inference select their parser from each JSONL row's `target_format`. The
optional `--target` argument is only a compatibility fallback for legacy JSONL
files without that field. Inference prints the decode failure count and rate to `stage2.log`.
A decode failure is an output row with a non-empty `parse_error`, including a
missing audio file, a generation exception, or an invalid SSD output.

For the normal end-to-end workflow, run
`./run.sh --stage 0 --stop_stage 5` from the repository root. This uses
`prompts/prompt_ts.txt` by default. Pass `--prompt_file` to select another
relative or absolute prompt path. The prompt filename without its extension is
included in the experiment directory name.
