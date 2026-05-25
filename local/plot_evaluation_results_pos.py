import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import re

from e2e_stt.nlp_models import NlpModel

def extract_stress_binary(stress_pattern: str):
    """
    將包含 <stress> 標籤的字串，轉換為 0/1 陣列與乾淨的單字列表。
    """
    # 1. 預處理防呆：確保標籤前後一定有空白，避免 LLM 生成時標籤與單字黏連
    safe_pattern = stress_pattern.replace("<stress>", " <stress> ").replace("</stress>", " </stress> ")
    tokens = safe_pattern.split()
    
    binary_array = []
    clean_words = []
    is_stressed = False  # 狀態開關
    punctuation = {",", ".", "?", "!", ";", ":"}
    
    # 2. 逐一掃描 Token
    for token in tokens:
        if token == "<stress>":
            is_stressed = True   # 開啟重音狀態
        elif token == "</stress>":
            is_stressed = False  # 關閉重音狀態
        elif token in punctuation:
            continue
        else:
            # 遇到一般單字，根據目前的狀態紀錄 0 或 1
            binary_array.append(1 if is_stressed else 0)
            clean_words.append(token)
            
    return binary_array, clean_words

def plot_fnr_fpr_by(df, feature, save_dir, bins=None, labels=None):
    df_copy = df.copy()

    if bins:
        df_copy["bin"] = pd.cut(df_copy[feature], bins=bins, labels=labels, include_lowest=True)
    else:
        df_copy["bin"] = df_copy[feature]

    # 計算 FNR/FPR
    grouped = df_copy.groupby(["bin", "type"], observed=False).size().unstack(fill_value=0)
    grouped["FNR"] = grouped["FN"] / (grouped["FN"] + grouped["TP"]).replace(0, np.nan)
    grouped["FPR"] = grouped["FP"] / (grouped["FP"] + grouped["TN"]).replace(0, np.nan)

    # 計算 count 並依 count 排序
    count_by_bin = df_copy.groupby("bin", observed=False).size()
    sorted_index = count_by_bin.sort_values(ascending=False).index

    # 根據排序重新排列
    grouped = grouped.loc[sorted_index]
    count_by_bin = count_by_bin.loc[sorted_index]

    # 建立圖
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 左軸：FNR / FPR 柱狀圖
    grouped[["FNR", "FPR"]].plot(kind="bar", ax=ax1, stacked=False, colormap="Set2", width=0.8)
    ax1.set_ylabel("FNR / FPR Rate")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(feature)
    ax1.set_xticks(range(len(grouped.index)))
    ax1.set_xticklabels(grouped.index, rotation=45)
    ax1.grid(True, axis="y")

    # 右軸：Count 折線圖
    ax2 = ax1.twinx()
    ax2.plot(range(len(count_by_bin)), count_by_bin.values, color='black', marker='o', linestyle='-', label='Count')
    ax2.set_ylabel("Sample Count")
    ax2.set_ylim(0, max(count_by_bin.values) * 1.1)

    # 合併 Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    #plt.title(f"FNR / FPR by {feature}")
    plt.tight_layout()
    output_path = save_dir / f"fnr_fpr_by_{feature}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved plot: {output_path}")

def flatten_words_with_pos(predict_lines, ground_truth_lines, nlp_model):
    rows = []
    for i, (pred_line, gt_line) in enumerate(zip(predict_lines, ground_truth_lines), start=1):
        pred_data = json.loads(pred_line.strip())
        gt_data = json.loads(gt_line.strip())
        pred_transcription = pred_data.get("pred_transcription", "")
        gt_transcription = gt_data.get("transcription", "")

        pred_transcription = pred_transcription.lower()
        pred_transcription = re.sub(r'[^\w\s]', '', pred_transcription)
        pred_transcription.strip()

        gt_transcription = gt_transcription.lower()
        gt_transcription = re.sub(r'[^\w\s]', '', gt_transcription)
        gt_transcription.strip()
        
        pred_stress = pred_data.get("pred_stress", "")
        gt_stress = gt_data.get("stress", "")
        pred_stress_pattern,_ = extract_stress_binary(pred_stress)
        gt_stress_pattern,_ = extract_stress_binary(gt_stress)

        words = pred_transcription
        vp_feats = nlp_model.vocab_profile_feats(words.split())
        pos_feats = vp_feats["pos_list"]
        assert len(words.split()) == len(pos_feats)
        if len(pred_stress_pattern) != len(gt_stress_pattern):
            continue
        utt_pred = words.split(" ")
        utt_gt = words.split(" ")
        if len(pred_stress_pattern) != len(utt_pred):
            continue
        for i, (word_pred, word_gt, pred, gt) in enumerate(zip(utt_pred, utt_gt, pred_stress_pattern, gt_stress_pattern)):
            rows.append({
                "word": word_pred,
                "gt": word_gt,
                "pred": word_pred,
                "type": "TP" if gt == 1 and pred == 1 else "FN" if gt == 1 and pred == 0 else "FP" if gt == 0 and pred == 1 else "TN",
                # "word_len": word["word_len"],
                # "syllable_count": word.get("syllable_count", None),
                # "utt_len": utt["utt_len"],
                # "utt_duration": utt["utt_duration"],
                # "speaking_rate": utt["speaking_rate"],
                "pos": pos_feats[i]
            })
    return pd.DataFrame(rows)

    # for utt in error_cases:
    #     words = " ".join([w["word"] for w in utt["words"]])
    #     vp_feats = nlp_model.vocab_profile_feats(words.split())
    #     pos_feats = vp_feats["pos_list"]
    #     assert len(words.split()) == len(pos_feats)

    #     for i, word in enumerate(utt["words"]):
    #         rows.append({
    #             "word": word["word"],
    #             "gt": word["gt"],
    #             "pred": word["pred"],
    #             "type": word["type"],
    #             # "word_len": word["word_len"],
    #             # "syllable_count": word.get("syllable_count", None),
    #             # "utt_len": utt["utt_len"],
    #             # "utt_duration": utt["utt_duration"],
    #             # "speaking_rate": utt["speaking_rate"],
    #             "pos": pos_feats[i]
    #         })
    # return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default="evaluation_results/whistress_error_analysis.json", help="Path to whistress_error_analysis.json")
    parser.add_argument("--gt_file", type=str, default="evaluation_results/whistress_error_analysis.json", help="Path to whistress_error_analysis.json")
    parser.add_argument("--save_fig_dir", type=str, default="evaluation_results/figs", help="Directory to save figures")
    args = parser.parse_args()
    

    # error_case_path = Path(args.error_case_path)
    save_fig_dir = Path(args.save_fig_dir)
    save_fig_dir.mkdir(parents=True, exist_ok=True)

    # with open(error_case_path, "r") as f:
    #     error_cases = json.load(f)
    with open(args.pred_file, 'r', encoding='utf-8') as f_pred:
        predict_lines = f_pred.readlines()
    with open(args.gt_file, 'r', encoding='utf-8') as f_gt:
        ground_truth_lines = f_gt.readlines()

    if len(predict_lines) != len(ground_truth_lines):
        print(f"Error: line count mismatch. pred={len(predict_lines)} gt={len(ground_truth_lines)}", file=sys.stderr)
        sys.exit(1)

    nlp_model = NlpModel(tokenize_pretokenized=True)
    # df = flatten_words_with_pos(error_cases, nlp_model)
    df = flatten_words_with_pos(predict_lines, ground_truth_lines, nlp_model)

    # plot_fnr_fpr_by(df, "syllable_count", save_fig_dir)
    # plot_fnr_fpr_by(df, "word_len", save_fig_dir, bins=[0, 3, 6, 10, 20], labels=["1-3", "4-6", "7-10", "11+"])
    # plot_fnr_fpr_by(df, "speaking_rate", save_fig_dir, bins=[0, 2, 4, 6, 10], labels=["0-2", "2-4", "4-6", "6+"])

    plot_fnr_fpr_by(df, "pos", save_fig_dir)
