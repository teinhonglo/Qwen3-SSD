#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional

import librosa
import torch
from qwen_asr import Qwen3ASRModel
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

def plot_split_attention_heatmap(gen_out, inputs, asr_wrapper, audio_path, target_layer=-1):
    """
    gen_out: return from model.generate
    inputs: inputs of model.generate (找音訊位置)
    asr_wrapper: model
    audio_path: 圖片存檔名稱
    target_layer: 預設最後一層
    """
    if gen_out.attentions is None:
        print("錯誤: gen_out.attentions 為 None")
        return

    tokenizer = asr_wrapper.processor.tokenizer
    # print(gen_out) # shape: sequences[[...]], sttentions, past_key_values
    # print(inputs) # shape: input_ids[[...]], attention_mas, feature_attenion_mask, input_features
    # print(gen_out.attentions)
    full_ids = gen_out.sequences[0].cpu().tolist()
    all_tokens_text = [tokenizer.decode([tid]) for tid in full_ids]
    
    prefix_len = inputs["input_ids"].shape[1]
    total_seq_len = len(full_ids)
    num_gen_steps = len(gen_out.attentions)

    # 1. get indices of Audio, Prompt, Output token
    # Audio
    audio_token_id = asr_wrapper.model.config.thinker_config.audio_token_id
    input_ids = inputs["input_ids"][0]
    audio_indices = (input_ids == audio_token_id).nonzero(as_tuple=True)[0].cpu().tolist()
    
    # Prompt (Prefix 中除去 Audio 的部分)
    prompt_indices = [i for i in range(prefix_len) if i not in audio_indices]
    
    # Output
    output_indices = list(range(prefix_len, total_seq_len))
    #2. whole heat map
    heatmap_matrix = np.zeros((num_gen_steps, total_seq_len))
    for i in range(num_gen_steps):
        step_attn = gen_out.attentions[i][target_layer][0]
        avg_attn = step_attn.mean(dim=0)[0].cpu().numpy()
        heatmap_matrix[i, :len(avg_attn)] = avg_attn

    #3. split heat map
    audio_attn = heatmap_matrix[:, audio_indices].copy()
    prompt_attn = heatmap_matrix[:, prompt_indices].copy()
    output_attn = heatmap_matrix[:, output_indices].copy()

    # set axis labels
    y_labels = all_tokens_text[prefix_len:]
    audio_x_labels = [all_tokens_text[i] for i in audio_indices]
    prompt_x_labels = [all_tokens_text[i] for i in prompt_indices]
    output_x_labels = [all_tokens_text[i] for i in output_indices]

    #4. draw heat map (1 Row, 3 Columns)
    widths = [max(len(prompt_indices), 10), max(len(audio_indices), 10), max(len(output_indices), 10)]
    fig, axes = plt.subplots(1, 3, figsize=(26, 11), gridspec_kw={'width_ratios': widths}, sharey=True)

    # Heatmap 參數
    kwargs_base = dict(cmap="viridis", cbar=True, yticklabels=y_labels)

    # 圖 1: Prompt Zone
    if prompt_attn.size > 0:
        vmax_prompt = np.percentile(prompt_attn[:, 1:], 99.9) if np.max(prompt_attn) > 0 else 0.1
        sns.heatmap(prompt_attn, ax=axes[0], xticklabels=prompt_x_labels, vmax=vmax_prompt, **kwargs_base)
        axes[0].set_xlabel("Instruction / Context", fontsize=11)
        plt.setp(axes[0].get_xticklabels(), rotation=90, fontsize=5)
    axes[0].set_title(f"1. PROMPT ZONE\n(Locally Scaled vmax={vmax_prompt:.4f})", fontsize=14, color='blue', weight='bold')

    # 圖 2: Audio Zone
    # vmax: values to anchor the colormap
    if audio_attn.size > 0:
        vmax_audio = np.percentile(audio_attn, 99.9) if np.max(audio_attn) > 0 else 0.1
        # audio_indices 幀數表示時間
        sns.heatmap(audio_attn, ax=axes[1], xticklabels=False, vmax=vmax_audio, **kwargs_base)
        axes[1].set_xlabel(f"Audio Time Frames (0 to {len(audio_indices)})", fontsize=11)
    axes[1].set_title(f"2. AUDIO ZONE\n(Locally Scaled vmax={vmax_audio:.4f})", fontsize=14, color='red', weight='bold')

    # 圖 3: Output Zone
    if output_attn.size > 0:
        vmax_output = np.percentile(output_attn, 99.9) if np.max(output_attn) > 0 else 0.1
        sns.heatmap(output_attn, ax=axes[2], xticklabels=output_x_labels, vmax=vmax_output, **kwargs_base)
        axes[2].set_xlabel("Previous Generated Tokens", fontsize=11)
        plt.setp(axes[2].get_xticklabels(), rotation=90, fontsize=5)
    axes[2].set_title(f"3. OUTPUT\n(Locally Scaled vmax={vmax_output:.4f})", fontsize=14, color='green', weight='bold')


    # <stress> 標籤畫虛線
    for ax in axes:
        for i, label in enumerate(y_labels):
            if "stress" in label or "gender" in label:
                ax.axhline(y=i+0.5, color='white', linestyle=':', alpha=0.8, linewidth=1.5)

    plt.suptitle(f"Attention Analysis: {os.path.basename(audio_path)}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    filename = os.path.basename(audio_path).replace(".wav", "")
    save_path = f"split_attn_independent_{filename}_ts.png"
    plt.savefig(save_path, dpi=300)
    print(f"[success] split heat map saved: {save_path}")
    plt.show()
    plt.close()

# 4/19 combined pic
def plot_full_attention_heatmap(gen_out, inputs, asr_wrapper, audio_path, target_layer=-1):
    """
    繪製完整的注意力矩陣熱圖
    gen_out: model.generate 的回傳物件
    inputs: 傳給 model.generate 的原始 inputs (用於找音訊位置)
    asr_wrapper: 包含 processor 和 model 的物件
    audio_path: 用於存檔名稱
    target_layer: 觀察哪一層的注意力 (預設最後一層)
    """
    if gen_out.attentions is None:
        print("錯誤: gen_out.attentions 為 None，請確認已開啟 eager 模式並修改 forward 回傳值。")
        return

    # --- 1. 準備 Token 標籤 ---
    tokenizer = asr_wrapper.processor.tokenizer
    full_ids = gen_out.sequences[0].cpu().tolist()
    # 將所有 ID 轉為文字標籤 (用於 X 軸和 Y 軸)
    all_tokens_text = [tokenizer.decode([tid]) for tid in full_ids]
    
    # 找出 Prefix (輸入部分) 的長度
    prefix_len = inputs["input_ids"].shape[1]
    # Y 軸標籤：只包含生成的 Token (從 prefix 之後開始)
    y_labels = all_tokens_text[prefix_len:]
    # X 軸標籤：包含整個序列 (Input + Generated)
    x_labels = all_tokens_text

    # --- 2. 提取並對齊注意力矩陣 ---
    # gen_out.attentions 是一個 tuple (step)，每步包含 (layer, batch, head, 1, current_len)
    num_gen_steps = len(gen_out.attentions)
    total_seq_len = len(full_ids)
    
    # 建立一個空矩陣 (Rows: 生成步數, Cols: 總序列長度)
    heatmap_matrix = np.zeros((num_gen_steps, total_seq_len))

    for i in range(num_gen_steps):
        # 提取該步生成的注意力 (最後一層, Batch 0, 平均所有 Head)
        # step_attn shape: (num_heads, 1, current_len)
        step_attn = gen_out.attentions[i][target_layer][0]
        avg_attn = step_attn.mean(dim=0)[0].cpu().numpy()
        
        # 將數據填入矩陣 (因為是自回歸，後半部未來的部分會是 0)
        heatmap_matrix[i, :len(avg_attn)] = avg_attn

    # --- 3. 獲取音訊 Token 的範圍 (用於在圖上畫框標註) ---
    audio_token_id = getattr(asr_wrapper.model.config.thinker_config, "audio_token_id", None)
    if audio_token_id is None:
        audio_token_id = asr_wrapper.model.thinker.config.audio_token_id
        
    input_ids = inputs["input_ids"][0]
    audio_indices = (input_ids == audio_token_id).nonzero(as_tuple=True)[0]
    audio_start, audio_end = (audio_indices[0].item(), audio_indices[-1].item() + 1) if len(audio_indices) > 0 else (0, 0)

    # --- 4. 繪製熱圖 ---
    plt.figure(figsize=(20, 10))
    # 使用 log scale 或稍微增加亮度 (vmax) 讓分佈更明顯
    ax = sns.heatmap(heatmap_matrix, 
                     xticklabels=x_labels, 
                     yticklabels=y_labels, 
                     cmap="viridis", 
                     cbar_kws={'label': 'Attention Weight'},
                     vmax=np.percentile(heatmap_matrix, 99)) # 排除極端值讓顏色更均勻

    # --- 5. 增加視覺輔助標記 ---
    # 在 X 軸上標註音訊區間 (畫一條紅色的粗線在底部)
    if audio_end > 0:
        plt.axvspan(audio_start, audio_end, color='red', alpha=0.1, label="Audio Tokens")
        plt.text((audio_start + audio_end) / 2, -1, "AUDIO ZONE", 
                 color='red', ha='center', weight='bold')

    # 標註 <stress> 標籤所在的行，方便一眼定位
    for i, label in enumerate(y_labels):
        if "<stress>" in label or "stress" in label:
            plt.axhline(y=i+0.5, color='white', linestyle='--', alpha=0.5)

    plt.title(f"Full Sequence Self-Attention Heatmap\nAudio: {os.path.basename(audio_path)}", fontsize=15)
    plt.xlabel("Key Tokens (Input Context + History)", fontsize=12)
    plt.ylabel("Generated Tokens (Query)", fontsize=12)
    
    # 優化標籤顯示
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(fontsize=8)
    
    plt.tight_layout()
    filename = os.path.basename(audio_path).replace(".wav", "")
    plt.savefig(f"heatmap_{filename}.png", dpi=300)
    print(f"[success] 全序列熱圖已保存至: heatmap_{filename}.png")
    plt.show()

def get_audio_token_range(asr_wrapper, inputs):
    """
    找出音訊 Token 在輸入序列中的起始與結束索引
    """
    # audio_token_id = asr_wrapper.model.config.audio_token_id
    if hasattr(asr_wrapper.model.config, "thinker_config"):
        audio_token_id = asr_wrapper.model.config.thinker_config.audio_token_id
    else:
        # 備用方案：直接從 thinker 物件拿
        audio_token_id = asr_wrapper.model.thinker.config.audio_token_id
    print(f"audio_token_id: {audio_token_id}")
    input_ids = inputs["input_ids"][0]
    print(f"input_ids: {input_ids}")
    # 找出所有等於 audio_token_id 的位置
    audio_indices = (input_ids == audio_token_id).nonzero(as_tuple=True)[0]
    
    if len(audio_indices) == 0:
        return None, None
    
    return audio_indices[0].item(), audio_indices[-1].item() + 1

def visualize_multi_stress(gen_out, stress_indices, audio_wav, audio_path, inputs, asr_wrapper, sr=16000):
    if not stress_indices or gen_out.attentions is None:
        print("沒有數據可供繪圖 (標籤缺失或 Attention 為 None)")
        return

    # 1. 確定音訊 Token 在序列中的範圍 (Key 的範圍)
    audio_start, audio_end = get_audio_token_range(asr_wrapper, inputs)
    if audio_start is None:
        print("無法在序列中定位音訊 Token")
        return
    
    num_stress = len(stress_indices)
    fig, axes = plt.subplots(num_stress + 1, 1, figsize=(15, 3 * (num_stress + 1)), sharex=True)
    
    duration = len(audio_wav) / sr
    time_axis = np.linspace(0, duration, len(audio_wav))

    # --- 繪製原始波形 ---
    axes[0].plot(time_axis, audio_wav, color='gray', alpha=0.4)
    axes[0].set_title(f"Audio: {os.path.basename(audio_path)}")
    axes[0].set_ylabel("Amplitude")

    # --- 遍歷每個 <stress> 標籤 ---
    for i, token_idx in enumerate(stress_indices):
        ax = axes[i + 1]
        
        # 提取該步生成的 Self-Attention (最後一層)
        # shape: (num_heads, 1, current_seq_len)
        step_attn = gen_out.attentions[token_idx][-1][0]
        
        # 平均所有 Head 並轉為 numpy
        # shape: (current_seq_len,)
        full_attn_weights = step_attn.mean(dim=0)[0].cpu().numpy()
        
        # 【核心修正】只切出音訊 Token 對應的那一段 Key
        stress_to_audio_attn = full_attn_weights[audio_start:audio_end]
        
        # 將這段權重映射到時間軸
        attn_time = np.linspace(0, duration, len(stress_to_audio_attn))
        
        # 繪製曲線
        ax.fill_between(attn_time, stress_to_audio_attn, color='orange', alpha=0.6, 
                        label=f'Stress @ Gen Step {token_idx}')
        ax.set_ylabel("Attn Weight")
        ax.legend(loc='upper right')
        
        # 尋找最強音訊關聯點
        max_idx = np.argmax(stress_to_audio_attn)
        peak_time = attn_time[max_idx]
        
        # 在波形圖畫紅線標記
        axes[0].axvline(x=peak_time, color='red', linestyle='--', alpha=0.5)
        axes[0].text(peak_time, axes[0].get_ylim()[1], f" S{i+1}", color='red', weight='bold')

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    
    filename = os.path.basename(audio_path).replace(".wav", "")
    plt.savefig(f"plot_{filename}.png")
    print(f"[success] 視覺化已保存至: plot_{filename}.png")
    plt.close()

# def visualize_multi_stress(gen_out, stress_indices, audio_wav, audio_path, sr=16000, title="Multi-Stress Alignment Analysis"):
#     """
#     gen_out: model.generate 的回傳物件 (需含 cross_attentions)
#     stress_indices: 包含所有 <stress> 位置的 list
#     audio_wav: 原始音訊數值 (numpy array or tensor)
#     """
#     if not stress_indices:
#         print("沒有找到 <stress> 標籤，跳過繪圖。")
#         return

#     num_stress = len(stress_indices)
#     # 建立畫布：1 個波形圖 + num_stress 個注意力圖
#     fig, axes = plt.subplots(num_stress + 1, 1, figsize=(15, 3 * (num_stress + 1)), sharex=True)
    
#     # 取得音訊時間軸
#     duration = len(audio_wav) / sr
#     time_axis = np.linspace(0, duration, len(audio_wav))

#     # --- 1. 繪製原始波形 ---
#     axes[0].plot(time_axis, audio_wav, color='gray', alpha=0.4)
#     axes[0].set_title("Original Audio Waveform")
#     axes[0].set_ylabel("Amplitude")

#     # --- 2. 遍歷每個 <stress> 標籤並繪圖 ---
#     for i, token_idx in enumerate(stress_indices):
#         ax = axes[i + 1]
        
#         # 提取該 Token 的 Cross-Attention (取最後一層，平均所有 Head)
#         # 結構: [step][layer][batch, head, query_len, key_len]
#         # 我們取最後一層 [-1]，第 0 個 batch [0]，平均所有 head [mean(0)]
#         # query_len 通常為 1 (因為是逐個生成的)，所以拿 [0, :]
#         attn_weights = gen_out.attentions[token_idx][-1][0].mean(dim=0)[0].cpu().numpy()
#         audio_token_len = 32
#         attn_weights = attn_weights[:audio_token_len]
#         # 將 Attention 幀數映射到時間軸
#         attn_time = np.linspace(0, duration, len(attn_weights))
        
#         # 繪製注意力曲線
#         ax.fill_between(attn_time, attn_weights, color='orange', alpha=0.6, label=f'Stress Token @ Index {token_idx}')
#         ax.set_ylabel("Attn Weight")
#         ax.legend(loc='upper right')
        
#         # 在波形圖上標註對應的高亮區 (找出注意力最高的地方)
#         max_idx = np.argmax(attn_weights)
#         peak_time = attn_time[max_idx]
#         axes[0].axvline(x=peak_time, color='red', linestyle='--', alpha=0.5)
#         axes[0].text(peak_time, ax.get_ylim()[1], f" S{i+1}", color='red', verticalalignment='bottom')

#     axes[-1].set_xlabel("Time (seconds)")
#     plt.suptitle(title, fontsize=16)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     audio_path = audio_path.strip(".wav")
#     filename = os.path.basename(audio_path)
#     plt.savefig(f"plot_{filename}.png")
#     plt.show()

# def plot_stress_attention(gen_out, token_index, audio_wav, sr=16000, id):
#     # 1. 取得特定 Token 的 Cross-Attention (假設取最後一層，並對 Head 取平均)
#     # gen_out.cross_attentions 結構: [step][layer][batch, head, query_len, key_len]
#     last_layer_attn = gen_out.cross_attentions[token_index][-1] 
#     attn_weights = last_layer_attn[0].mean(dim=0).cpu().numpy() # (1, audio_frames)
    
#     # 2. 建立畫布：上方波形，下方熱圖
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
#     # 上方：原始波形
#     time_axis = torch.linspace(0, len(audio_wav)/sr, len(audio_wav))
#     ax1.plot(time_axis, audio_wav, color='gray', alpha=0.5)
#     ax1.set_title("Audio Waveform")
    
#     # 下方：Attention 熱圖
#     sns.heatmap(attn_weights, ax=ax2, cmap="YlGnBu", cbar=False)
#     ax2.set_title(f"Cross-Attention for '<stress>' (Token Index: {token_index})")
    
#     plt.tight_layout()
#     plt.savefig(f"plot_{id}.png")
#     plt.show()

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


def build_prefix_messages(prompt: str, audio_array=None):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def build_prefix_text(processor, prompt: str) -> str:
    prefix_msgs = build_prefix_messages(prompt, None)
    prefix_text = processor.apply_chat_template(
        [prefix_msgs],
        add_generation_prompt=True,
        tokenize=False,
    )
    if isinstance(prefix_text, list):
        prefix_text = prefix_text[0]
    return prefix_text


def move_inputs_to_device(inputs: Dict[str, Any], device: str, model_dtype: torch.dtype):
    new_inputs = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(model_dtype)
        new_inputs[k] = v
    return new_inputs


def batch_decode_text(processor, token_ids):
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return processor.tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def unwrap_generate_output(gen_out):
    if hasattr(gen_out, "sequences"):
        return gen_out.sequences
    if isinstance(gen_out, dict) and "sequences" in gen_out:
        return gen_out["sequences"]
    if isinstance(gen_out, (tuple, list)):
        return gen_out[0]
    return gen_out


def _extract_first_json_dict(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text


# def try_parse_tasks_list(text: str) -> List[Dict[str, Any]]:
#     payload = _extract_first_json_array(text)
#     try:
#         obj = json.loads(payload)
#         if isinstance(obj, list):
#             return obj
#     except Exception:
#         pass
#     return []

def try_parse_tasks_dict(text: str) -> Dict[str, Any]:
    payload = _extract_first_json_dict(text)
    try:
        obj = json.loads(payload)
        # 🌟 修正：確保解析出來的是 dict 而不是 list
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def infer_one(
    asr_wrapper,
    audio_path: str,
    prompt: str = "",
    sr: int = 16000,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    ################ set eager mode
    asr_wrapper.model.thinker.config._attn_implementation = "eager"
    # asr_wrapper.model.config._attn_implementation = "eager"
    # 1. 檢查類型 (看它屬於哪個 Class)
    # print(f"model 的類型: {type(asr_wrapper.model)}")
    # print(f"Thinker 的類型: {type(asr_wrapper.model.thinker)}")

    # 2. 檢查屬性 (看它下面還有哪些按鈕可以按)
    # print(f"Config 裡面的所有變數: {dir(asr_wrapper.model.thinker.config)}")

    # print(f"Check mode: {asr_wrapper.model.thinker.config._attn_implementation}")

    processor = asr_wrapper.processor
    model = asr_wrapper.model
    device = next(model.parameters()).device
    model_dtype = getattr(model, "dtype", torch.float16)

    wav = load_audio(audio_path, sr=sr)
    prefix_text = build_prefix_text(processor, prompt)

    inputs = processor(
        text=[prefix_text],
        audio=[wav],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    prefix_len = int(inputs["attention_mask"][0].sum().item())
    inputs = move_inputs_to_device(inputs, device=device, model_dtype=model_dtype)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        # gen_out = model.generate(**inputs, **gen_kwargs)
        ############ allow attention
        gen_out = model.generate(**inputs, **gen_kwargs, output_attentions=True)

    output_ids = unwrap_generate_output(gen_out)
    if not torch.is_tensor(output_ids):
        raise TypeError(f"generate() returned unsupported type: {type(output_ids)}")

    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)

    if output_ids.size(1) > prefix_len:
        gen_only_ids = output_ids[:, prefix_len:]
    else:
        gen_only_ids = output_ids

    decoded = batch_decode_text(processor, gen_only_ids)[0].strip()

    ########### Attention Heat map ############
    # print(gen_out.keys())
    # print(gen_out)
    # find token id
    test_text = "<stress>"
    target_ids = processor.tokenizer.encode(test_text, add_special_tokens=False)
    print(f"{test_text} token id: {target_ids}")
    plot_split_attention_heatmap(gen_out, inputs, asr_wrapper, audio_path, target_layer=-1)
    #######################

    return decoded


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_id} in {path}: {e}")
    return data


def resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32

    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            major = torch.cuda.get_device_capability(device=device)[0]
        except Exception:
            major = torch.cuda.get_device_capability()[0]
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_jsonl_name(input_jsonl: str) -> str:
    base = os.path.basename(input_jsonl)
    name, _ = os.path.splitext(base)
    return name


def write_ssd_prediction_jsonl(rows_out: List[Dict[str, Any]], output_root: str, jsonl_name: str):
    save_dir = os.path.join(output_root, jsonl_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "predictions.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            item = {
                "id": row["text_id"],
                "pred_gender": row.get("pred_gender", ""),
                "pred_transcription": row.get("pred_transcription", ""),
                "pred_transcription_0": row.get("pred_transcription_0", ""),
                "pred_stress": row.get("pred_stress", ""),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[info] saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR SLU test script")

    p.add_argument("--exp_dir", type=str, required=True,
                   help="Experiment directory. Will load train_conf.json from this directory")
    p.add_argument("--auto_latest_checkpoint", action="store_true",
                   help="If exp_dir contains checkpoints, automatically use latest checkpoint")

    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL with fields like text_id, query, audio, prompt")

    p.add_argument("--output_root", type=str, default="checkpoints",
                   help='Root output dir. Default: "checkpoints"')

    p.add_argument("--device", type=str, default="cuda:0",
                   help='e.g. "cuda:0", "cuda:1", "cpu"')
    p.add_argument("--prompt_file", default="/datas/store162/annhung/Qwen3-SLU/prompt/prompt_ts.txt")
    p.add_argument("--target", default="text_ts")
    return p.parse_args()


def load_train_conf_from_exp_dir(exp_dir: str) -> Optional[List[Dict[str, Any]]]:
    if not exp_dir:
        return None

    train_conf_path = os.path.join(exp_dir, "train_conf.json")
    if not os.path.isfile(train_conf_path):
        raise FileNotFoundError(f"train_conf.json not found under exp_dir: {train_conf_path}")

    with open(train_conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, list) or len(cfg) != 2:
        raise ValueError("train_conf.json must be [training_args, model_args]")
    if not isinstance(cfg[0], dict) or not isinstance(cfg[1], dict):
        raise ValueError("Both train_conf entries must be dictionaries")
    return cfg


def main():
    args = parse_args()

    prompt = "" #DEFAULT_PROMPT
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    else:
        raise ValueError("No prompt")

    if args.target:
        target = args.target
    else:
        raise ValueError("No prompt")

    train_conf = load_train_conf_from_exp_dir(args.exp_dir)
    if train_conf is None:
        raise ValueError("Unable to load train_conf from exp_dir")

    training_args_conf, model_args_conf = train_conf
    sr = int(training_args_conf.get("sr", 16000))
    max_new_tokens = int(training_args_conf.get("max_new_tokens", 256))
    do_sample = bool(training_args_conf.get("do_sample", False))
    temperature = float(training_args_conf.get("temperature", 1.0))
    top_p = float(training_args_conf.get("top_p", 1.0))
    dtype_str = str(model_args_conf.get("dtype", "auto"))

    model_path = args.exp_dir
    if args.auto_latest_checkpoint:
        latest_ckpt = find_latest_checkpoint(model_path)
        if latest_ckpt is None:
            raise ValueError(f"No checkpoint-* found under: {model_path}")
        model_path = latest_ckpt
        print(f"[info] use latest checkpoint: {model_path}")

    dtype = resolve_dtype(dtype_str, args.device)
    jsonl_name = get_jsonl_name(args.input_jsonl)

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        # "Qwen/Qwen3-ASR-0.6B",
        dtype=dtype,
        device_map=args.device,
    )

    rows = load_jsonl(args.input_jsonl)
    rows_out = []

    for i, row in enumerate(rows, start=1):
        text_id = str(row.get("text_id", f"line{i}")).strip()
        audio_path = row.get("audio", "")
        # prompt = row.get("prompt", "")
        # print(prompt)
        transcription = row.get("transcription", "")

        if not audio_path:
            print(f"[skip] line {i}: no audio field")
            continue

        pred_raw = infer_one(
            asr_wrapper=asr_wrapper,
            audio_path=audio_path,
            prompt=prompt,
            sr=sr,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        if target == "text_ts":
            if "<ssd>" in pred_raw:
                pred_transcription_raw = pred_raw.split("<ssd>")[0]
                pred_raw_text = pred_raw.split("<ssd>")[1]
                pred_transcription = pred_transcription_raw.replace("language English", "").replace("<asr_text>", "").strip()
                pred_raw_dict = try_parse_tasks_dict(pred_raw_text)
                pred_stress = pred_raw_dict.get("stress_pattern", "")

                pred_transcription_0 = pred_stress.replace("<stress>", "").replace("</stress>", "").strip()
                pred_transcription_0 = " ".join(pred_transcription_0.split())

                
            else:
                pred_transcription = ""

            print(f"pred_<ssd>{pred_raw_text}")
            rows_out.append({
                "text_id": text_id,
                "pred_transcription_0": pred_transcription_0,
                "pred_transcription": pred_transcription,
                "pred_stress": pred_stress
            })

        if target == "text_gts":
            if "<ssd>" in pred_raw:
                pred_transcription_raw = pred_raw.split("<ssd>")[0]
                pred_raw_text = pred_raw.split("<ssd>")[1]
                pred_transcription = pred_transcription_raw.replace("language English", "").replace("<asr_text>", "").strip()
                pred_raw_dict = try_parse_tasks_dict(pred_raw_text)
                pred_stress = pred_raw_dict.get("stress_pattern", "")
                pred_gender = pred_raw_dict.get("gender", "")

                pred_transcription_0 = pred_stress.replace("<stress>", "").replace("</stress>", "").strip()
                pred_transcription_0 = " ".join(pred_transcription_0.split())

            else:
                pred_transcription = ""

            print(f"pred_<ssd>{pred_raw_text}")
            rows_out.append({
                "text_id": text_id,
                "pred_transcription_0": pred_transcription_0,
                "pred_transcription": pred_transcription,
                "pred_stress": pred_stress,
                "pred_gender": pred_gender
            })
        
        if target == "text_s":
            if "<ssd>" in pred_raw:
                pred_stress = pred_raw.split("<ssd>")[1]
                pred_transcription = pred_stress.replace("<stress>", "").replace("</stress>", "").strip()
                pred_transcription = " ".join(pred_transcription.split())
            else:
                pred_stress = ""

            print(f"transcript:{pred_transcription}\nstress:{pred_stress}")
            rows_out.append({
                "text_id": text_id,
                "pred_transcription": pred_transcription,
                "pred_stress": pred_stress,
            })

        if target == "text_gs":
            if "<ssd>" in pred_raw:
                print(pred_raw)
                pred_gender_raw = pred_raw.split("<ssd>")[0]
                pred_stress = pred_raw.split("<ssd>")[1]
                pred_transcription = pred_stress.replace("<stress>", "").replace("</stress>", "").strip()
                pred_transcription = " ".join(pred_transcription.split())
                if "<gender>" in pred_gender_raw:
                    pred_gender = pred_gender_raw.split("<gender>")[1]
                    print(pred_gender)
                else:
                    pred_gender = ""
            else:
                pred_stress = ""

            print(f"pred: <gender>{pred_gender}<ssd>{pred_stress}")
            rows_out.append({
                "text_id": text_id,
                "pred_gender": pred_gender,
                "pred_transcription": pred_transcription,
                "pred_stress": pred_stress,
            })
        
        if target == "text_tgs":
            if "<ssd>" in pred_raw:
                print(pred_raw)
                pred_gender_raw = pred_raw.split("<ssd>")[0]
                pred_stress = pred_raw.split("<ssd>")[1]
                pred_transcription_0 = pred_stress.replace("<stress>", "").replace("</stress>", "").strip()
                pred_transcription_0 = " ".join(pred_transcription_0.split())
                if "<gender>" in pred_gender_raw:
                    pred_transcription_raw = pred_gender_raw.split("<gender>")[0]
                    pred_gender = pred_gender_raw.split("<gender>")[1]
                    print(pred_gender)
                    if "<asr_text>" in pred_transcription_raw:
                        pred_transcription = pred_transcription_raw.split("<asr_text>")[1]
                        print(pred_gender)
                    else:
                        pred_transcription = ""
                else:
                    pred_gender = ""
                    if "<asr_text>" in pred_gender_raw:
                        pred_transcription = pred_gender_raw.split("<asr_text>")[1]
                        print(pred_gender)
                    else:
                        pred_transcription = ""
            else:
                pred_stress = ""

            print(f"pred: <transcription>{pred_transcription}<gender>{pred_gender}<ssd>{pred_stress}")
            rows_out.append({
                "text_id": text_id,
                "pred_gender": pred_gender,
                "pred_transcription_0": pred_transcription_0,
                "pred_transcription": pred_transcription,
                "pred_stress": pred_stress,
            })

        print(f"[{i}/{len(rows)}] done: {text_id}")

    write_ssd_prediction_jsonl(rows_out=rows_out, output_root=args.output_root, jsonl_name=jsonl_name)


if __name__ == "__main__":
    main()
