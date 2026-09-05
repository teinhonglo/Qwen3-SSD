#!/bin/bash
# Unified Qwen3-SSD pipeline aligned with Qwen3-SLU and WhiStress.

set -euo pipefail

stage=0
stop_stage=1000

# Data and task configuration
data_root=data
json_root=data-json
test_corpora="tinystress stresstest stresspresso expresso emphassess"
target=ts
prompt_file=
validation_ratio=0.1
force_download=false

# Training and inference configuration
train_conf=conf/TinyStress_qwen3_asr_06b.json
decoding_conf=conf/decoding/basic_decoding.json
gpuid=0
seed=66
checkpoint=
inference_mode=best
force_train=false

. ./local/parse_options.sh
. ./path.sh

if [ -z "$prompt_file" ]; then
    prompt_file="prompt/prompt_${target}.txt"
fi

for required_file in "$train_conf" "$decoding_conf" "$prompt_file"; do
    if [ ! -f "$required_file" ]; then
        echo "[ERROR] required file not found: $required_file" >&2
        exit 1
    fi
done

conf_tag=$(basename -s .json "$train_conf")
decoding_tag=$(basename -s .json "$decoding_conf")
exp_dir="exp/TinyStress/${conf_tag}_${target}"

training_opts=()
if [ -n "$checkpoint" ]; then
    training_opts+=(--resume_from "$checkpoint" --resume 1)
fi

inference_opts=()
case "$inference_mode" in
    best)
        inference_opts+=(--auto_best_checkpoint)
        ;;
    latest)
        inference_opts+=(--auto_latest_checkpoint)
        ;;
    root)
        ;;
    *)
        echo "[ERROR] inference_mode must be best, latest, or root" >&2
        exit 1
        ;;
esac

if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
    echo "Stage 0: Download canonical corpora and prepare SSD JSONL"
    download_args=(
        --data_root "$data_root/raw"
        --corpora tinystress $test_corpora
    )
    if [ "$force_download" = true ]; then
        download_args+=(--force)
    fi
    python local/download_corpora.py "${download_args[@]}"

    python local/prepare_ssd_jsonl.py +        --data-root "$data_root" +        --jsonl-root "$json_root" +        --prompt-file "$prompt_file" +        --target "$target" +        --corpora $test_corpora +        --validation-ratio "$validation_ratio" +        --seed "$seed"
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
    echo "Stage 1: Finetune Qwen3-ASR on TinyStress"
    if [ "$force_train" = true ] || [ ! -f "$exp_dir/.done" ]; then
        CUDA_VISIBLE_DEVICES="$gpuid" +            python finetuning/qwen3_asr_sft.py +                --seed "$seed" +                "${training_opts[@]}" +                --train_conf "$train_conf" +                --train_file "$json_root/tinystress/train.jsonl" +                --eval_file "$json_root/tinystress/dev.jsonl" +                --output_dir "$exp_dir"
        touch "$exp_dir/.done"
    else
        echo "[info] training already completed: $exp_dir/.done"
    fi
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
    echo "Stage 2: Run SSD inference"
    for corpus in $test_corpora; do
        input_jsonl="$json_root/$corpus/test.jsonl"
        results_root="$exp_dir/test/$corpus"
        mkdir -p "$results_root"
        CUDA_VISIBLE_DEVICES="$gpuid" +            python finetuning/qwen3_asr_test.py +                "${inference_opts[@]}" +                --exp_dir "$exp_dir" +                --input_jsonl "$input_jsonl" +                --output_root "$results_root" +                --device cuda:0 +                --decoding_conf "$decoding_conf" +                --target "$target" +                > "$results_root/stage2.log"
    done
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
    echo "Stage 3: Evaluate predictions with WhiStress-aligned protocol"
    for corpus in $test_corpora; do
        results_dir="$exp_dir/test/$corpus/test_${decoding_tag}"
        prediction_file="$results_dir/predictions.jsonl"
        reference_file="$json_root/$corpus/test.jsonl"
        if [ ! -f "$prediction_file" ]; then
            echo "[WARNING] prediction file not found: $prediction_file" >&2
            continue
        fi
        python local/evaluate_ssd.py +            --predictions "$prediction_file" +            --references "$reference_file" +            --results-dir "$results_dir" +            --corpus "$corpus" +            --split test +            --manifest "$json_root/manifest.json" +            | tee "$results_dir/metrics.txt"
    done
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
    echo "Stage 4: Plot error analysis"
    for corpus in $test_corpora; do
        results_dir="$exp_dir/test/$corpus/test_${decoding_tag}"
        error_file="$results_dir/qwen3_ssd_error_analysis.json"
        if [ ! -f "$error_file" ]; then
            echo "[WARNING] error analysis not found: $error_file" >&2
            continue
        fi
        python local/plot_evaluation_results.py +            --error_case_path "$error_file" +            --save_fig_dir "$results_dir/imgs"
    done
fi

if [ "$stage" -le 5 ] && [ "$stop_stage" -ge 5 ]; then
    echo "Stage 5: Summary"
    for corpus in $test_corpora; do
        evaluation_file="$exp_dir/test/$corpus/test_${decoding_tag}/qwen3_ssd_evaluation.json"
        if [ -f "$evaluation_file" ]; then
            echo "========== $corpus =========="
            cat "$evaluation_file"
        fi
    done
fi
