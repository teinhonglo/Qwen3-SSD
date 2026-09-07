# Qwen3-SSD finetuning

The repository-level `run.sh` is the recommended entry point for the complete
preparation, training, inference, evaluation, and plotting pipeline. This
directory contains the two core commands used by that pipeline.

## JSONL contract

Training, development, and test files use one JSON object per line:

```json
{"audio": "/absolute/or/relative/path.wav", "prompt": "task instruction", "text": "target response"}
```

Use `local/prepare_ssd_jsonl.py` to generate this format consistently for all
supported corpora and target formats.

## Training

```bash
python finetuning/qwen3_asr_sft.py \
  --train_conf conf/TinyStress_qwen3_asr_06b.json \
  --train_file data-json/tinystress/train.jsonl \
  --eval_file data-json/tinystress/dev.jsonl \
  --decoding_conf conf/decoding/basic_decoding.json \
  --target ts \
  --manifest data-json/manifest.json \
  --output_dir exp/tinystress/ts
```

A training config is a two-object JSON array: `[training_args, model_args]`.
The first object is forwarded to Hugging Face `TrainingArguments`. The second
selects the base model and model-specific settings such as sample rate,
generation defaults, component freezing, SpecAugment, LoRA/QLoRA, and W&B
metadata. Use `--resume`, `--resume_from`, or `--init_from_checkpoint` when
continuing an experiment.

Set `model_args.eval_generation_metrics` to `true` to run the same greedy
generation and SSD evaluation used by the inference/evaluation stages after
each validation pass. The Trainer logs `eval_ssd_precision`,
`eval_ssd_recall`, `eval_ssd_f1`, `eval_coverage_rate`, `eval_wer`, and
`eval_tagged_transcript_wer` in addition to `eval_loss`. This evaluates the
full development set by default. Set `model_args.eval_generation_max_samples`
to a positive integer only when a faster, approximate validation metric is
preferred. The TinyStress configs select periodic checkpoints by
`eval_ssd_f1`. After training, the untouched final model is evaluated once
more and saved as `checkpoint-last`; its `eval_last_ssd_f1` is compared with
the best periodic score before `checkpoint-best` is created.

## Inference

```bash
python finetuning/qwen3_asr_test.py \
  --exp_dir exp/tinystress/ts \
  --auto_best_checkpoint \
  --input_jsonl data-json/tinystress/test.jsonl \
  --output_root exp/tinystress/ts/tinystress \
  --decoding_conf conf/decoding/basic_decoding.json \
  --target ts \
  --device cuda:0
```

`--auto_latest_checkpoint` selects the most recent numbered periodic
checkpoint. `--auto_last_checkpoint` selects the evaluated final model in
`checkpoint-last`. The supported SSD targets are `ts`, `gts`, `s`, `gs`, and
`tgs`.

For the normal end-to-end workflow, run `./run.sh --stage 0 --stop_stage 5`
from the repository root.
