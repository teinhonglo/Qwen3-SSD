# qwen3_asr_test.py
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
    # find token id
    test_text = "<stress>"
    target_ids = processor.tokenizer.encode(test_text, add_special_tokens=False)
    print(f"{test_text} token id: {target_ids}")
    plot_split_attention_heatmap(gen_out, inputs, asr_wrapper, audio_path, target_layer=-1)
    #######################

    return decoded

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
            if "stress" in label:
                ax.axhline(y=i+0.5, color='white', linestyle=':', alpha=0.8, linewidth=1.5)

    plt.suptitle(f"Attention Analysis: {os.path.basename(audio_path)}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    filename = os.path.basename(audio_path).replace(".wav", "")
    save_path = f"split_attn_independent_{filename}.png"
    plt.savefig(save_path, dpi=300)
    print(f"[success] split heat map saved: {save_path}")
    plt.show()
    plt.close()