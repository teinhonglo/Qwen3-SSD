#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa, huggingface_hub

set -euo pipefail

# data config
repo_id=/datas/store162/annhung/WhiStress_ws0202/EmphAssess_Dataset
data_root="data/Emphassess"
download_dir=/datas/store162/annhung/WhiStress_ws0202/EmphAssess_Dataset/audio
extract_root=${data_root}/audio
audio_dir=${data_root}/audio
json_root=data-json/Emphassess
prompt_file=/datas/store162/annhung/Qwen3-SLU/prompt/prompt_s.txt   # 可指定外部 prompt 檔案，空字串則使用 prepare_macslu_jsonl.py 內建 prompt
target="text_s"

# training config
nj=4
gpuid=0
suffix=_s
train_conf=conf/TinyStress_qwen3_asr_06b_freeze.json
seed=66

# stage
stage=0
stop_stage=1000
test_set_name="Emphassess"
test_sets="test"

. ./local/parse_options.sh
. ./path.sh

if [ ! -f "$train_conf" ]; then
    echo "[ERROR] train_conf not found: $train_conf"
    exit 1
fi

conf_tag=$(basename -s .json $train_conf)
exp_root=exp/TinyStress/${conf_tag}${suffix}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Download Emphassess and prepare jsonl"

    prep_cmd=(
        python local/prepare_Emphassess_jsonl.py
        --raw-dir "$repo_id"
        --audio-dir "$download_dir"
        # --extract-root "$extract_root"
        --output-dir "$json_root"
        --splits test
    )

    if [ -n "$prompt_file" ]; then
        prep_cmd+=(--prompt_file "$prompt_file")
    fi

    "${prep_cmd[@]}"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Finetuning on TinyStress"

    data_dir=$json_root
    exp_dir=$exp_root

    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_sft.py --seed $seed \
            --train_conf $train_conf \
            --train_file $data_dir/train.jsonl \
            --eval_file $data_dir/test.jsonl \
            --output_dir $exp_dir \
            --target "$target" \
            --prompt_file $prompt_file
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Inference on Emphassess test"

    data_dir=$json_root
    exp_dir=$exp_root

    for test_set in $test_sets; do
        test_jsonl=${data_dir}/${test_set}.jsonl

        mkdir -p ${exp_dir}/${test_set_name}

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test.py \
                --exp_dir $exp_dir \
                --auto_latest_checkpoint \
                --input_jsonl $test_jsonl \
                --output_root $exp_dir/${test_set_name} \
                --device cuda:0 \
                --prompt_file $prompt_file \
                --target "$target"
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Evaluate Emphassess predictions"

    for test_set in $test_sets; do
        pred_file=${exp_root}/${test_set_name}/${test_set}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl

        if [ ! -f "$pred_file" ]; then
            echo "[WARNING] prediction file not found: $pred_file"
            continue
        fi

        python local/metrics_TinyStress.py "$pred_file" "$gt_file" "$target" "Emphassess" | tee ${exp_root}/${test_set_name}/${test_set}/metrics.txt
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Summary (Emphassess)"

    for test_set in $test_sets; do
        metrics_file=${exp_root}/${test_set_name}/${test_set}/metrics.txt
        if [ ! -f "$metrics_file" ]; then
            echo "[WARNING] metrics file not found: $metrics_file"
            continue
        fi

        echo "========== ${test_set} =========="
        cat "$metrics_file"
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    # metadata_fn=$exp_dir/best/metadata.json
    results_dir=${exp_root}/${test_set_name}/test
    
    python local/plot_evaluation_results_pos.py \
        --pred_file ${exp_root}/${test_set_name}/${test_sets}/predictions.jsonl \
        --gt_file ${json_root}/${test_sets}.jsonl \
        --save_fig_dir $results_dir/imgs
fi