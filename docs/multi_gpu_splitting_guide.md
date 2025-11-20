# FlashVSR 多 GPU 长视频切片指南

为了在多 GPU 上并行处理长视频并确保拼接无缝，必须遵循特定的切片逻辑。这主要是由 `FlashVSRTinyLongPipeline` 中的 **VAE 4x 下采样约束** 和 **流式推理的上下文依赖（KV Cache）** 决定的。

## 核心约束

1.  **单片帧数约束 (`4k + 1`)**:
    每个切片的长度必须满足 `num_frames % 4 == 1`。代码中会强制执行此操作，如果不满足会被向上取整，导致帧数对不上。

2.  **起始位置对齐 (`Multiple of 4`)**:
    每个切片在原始视频中的**起始帧索引**必须是 **4 的倍数**（0, 4, 8, ...）。这是为了保证 VAE 的 Latent Grid 在不同切片间是对齐的（Latent 0 对应 Frame 0, Latent 1 对应 Frame 4...）。

3.  **重叠预热 (`Overlap Warmup`)**:
    由于每个 GPU 实例是独立启动的，第二个切片在开始时没有前一个切片的 KV Cache（记忆）。为了避免拼接处的“断层”或颜色突变，后一个切片必须**向前回溯**一段距离开始处理，利用这段重叠区域作为“预热（Warmup）”，让模型状态稳定下来。拼接时，丢弃这段预热生成的帧。

---

## 切片设计方案

假设原始视频总帧数为 `Total_Frames`，我们要切成多个 `Chunk`。

### 1. 参数建议
*   **重叠帧数 (`Overlap`)**: 建议 **40 帧** (或者至少 32 帧)。
    *   *原因*: 代码中 `First Block` 处理前 6 个 Latent（约 24 帧）。为了确保拼接点处于稳定的“流式（Streaming）”阶段，预热长度应大于 24 帧。
*   **切片长度 (`Length`)**: 根据显存决定，但必须满足 `Length % 4 == 1`。

### 2. 切割逻辑 (以 2 个 GPU 为例)
假设我们要在一个点 `Split_Point` 进行拼接：

*   **GPU 1 (前段)**:
    *   **输入范围**: `[0, End1]`
    *   **保留输出**: `[0, Split_Point]`
    *   **要求**: `End1` 必须足够长，覆盖 `Split_Point`。通常让 `End1 = Split_Point` 或者稍微多几帧（多出的丢弃）。

*   **GPU 2 (后段)**:
    *   **输入范围**: `[Start2, End2]`
    *   **保留输出**: `[Split_Point + 1, End2]`
    *   **Start2 计算**: `Start2 = Split_Point - Warmup_Frames`
        *   `Start2` 必须是 **4 的倍数**。
        *   `Warmup_Frames` 建议 $\ge 40$。
    *   **End2 计算**: 视频结尾或下一段的结束点。
    *   **长度检查**: 确保 `(End2 - Start2 + 1) % 4 == 1`。如果不满足，调整 `End2` 或 `Start2`。

### 3. 拼接操作
*   **最终视频** = `GPU1_Output[0 : Split_Point+1]` + `GPU2_Output[Warmup_Frames+1 : ]`
    *   注意：Python 切片是左闭右开，所以是 `[:Split_Point+1]` (包含第 Split_Point 帧)。
    *   GPU 2 生成了 `Warmup + (End2 - Split_Point)` 帧，我们丢弃前 `Warmup` 帧（即丢弃 `Split_Point` 及其之前的帧，从 `Split_Point + 1` 开始保留）。

---

## Python 计算代码示例

你可以使用这段代码来计算每个 GPU 应该处理的 `start` 和 `end` 帧索引：

```python
def calculate_video_splits(total_frames, num_gpus, overlap=40):
    """
    计算多GPU处理的切片范围。
    
    Args:
        total_frames: 视频总帧数
        num_gpus: GPU 数量
        overlap: 重叠预热帧数 (建议 >= 32, 且为 4 的倍数)
    
    Returns:
        splits: List of dict, 每个 dict 包含:
            - 'input_range': (start, end)  -> 送入 GPU 的帧范围 (闭区间)
            - 'keep_range': (start, end)   -> 拼接时保留的相对范围 (闭区间)
            - 'global_range': (start, end) -> 保留的帧在原视频中的绝对范围
    """
    # 确保 overlap 是 4 的倍数
    overlap = (overlap + 3) // 4 * 4
    
    # 理想的每段长度
    base_len = total_frames // num_gpus
    
    splits = []
    current_global_start = 0
    
    for i in range(num_gpus):
        is_last = (i == num_gpus - 1)
        
        # 1. 确定拼接点 (Split Point)
        # 这是这一段有效数据的结束点（绝对索引）
        if is_last:
            global_end = total_frames - 1
        else:
            # 粗略计算结束点，必须调整为 4k (方便下一段从 4k+1 开始? 不，下一段start必须是4k)
            # 我们让这一段的有效结束点 global_end 满足: 下一段的 start (global_end + 1) 减去 overlap 后是 4 的倍数
            target_end = current_global_start + base_len
            # 对齐到 4 的倍数
            global_end = (target_end // 4) * 4
            # 此时 global_end 是 4的倍数 (例如 100)。
            # GPU1 保留 0~100。GPU2 需要从 101 开始保留。
            # GPU2 的输入 start = 101 - 1 - overlap? 
            # 简化逻辑：拼接点设在 global_end。
            
        # 2. 确定该 GPU 的输入 Start (Input Start)
        if i == 0:
            input_start = 0
            warmup = 0
        else:
            # 这一段的有效数据从 current_global_start 开始
            # 需要向前回溯 overlap 帧
            input_start = current_global_start - overlap
            # 修正 input_start 为 4 的倍数
            input_start = (input_start // 4) * 4
            warmup = current_global_start - input_start
        
        # 3. 确定该 GPU 的输入 End (Input End)
        # 必须满足 (input_end - input_start + 1) % 4 == 1
        # 先设定目标 input_end
        input_end = global_end
        current_len = input_end - input_start + 1
        
        # 调整长度以满足 % 4 == 1
        remainder = current_len % 4
        if remainder != 1:
            # 需要增加帧数来补齐 (因为不能减少有效帧)
            needed = (1 - remainder + 4) % 4
            input_end += needed
            
            # 如果超出视频总长度，不仅要截断，还要反向调整 start 以保持 % 4 == 1
            if input_end >= total_frames:
                input_end = total_frames - 1
                # 重新计算长度并调整 start
                final_len = input_end - input_start + 1
                remainder = final_len % 4
                if remainder != 1:
                    # 向前扩展 start (start 减小)
                    needed_back = (1 - remainder + 4) % 4
                    input_start -= needed_back
                    # 重新计算 warmup
                    warmup = current_global_start - input_start

        # 记录分片信息
        # keep_range 是相对于该切片的局部索引
        # 第一段保留: 0 ~ (global_end - input_start)
        # 后续段保留: warmup ~ (global_end - input_start)
        
        # 这一段负责产出的绝对范围
        valid_global_end = min(global_end, total_frames - 1)
        
        splits.append({
            "gpu_index": i,
            "input_range": (input_start, input_end), # 传给 FlashVSR 的帧范围
            "input_length": input_end - input_start + 1,
            "keep_local_range": (warmup, warmup + (valid_global_end - current_global_start)), # 局部切片 [start:end+1]
            "global_output_range": (current_global_start, valid_global_end)
        })
        
        current_global_start = valid_global_end + 1
        
    return splits

# --- 测试示例 ---
# 假设 200 帧视频，2 个 GPU，重叠 40 帧
if __name__ == "__main__":
    res = calculate_video_splits(200, 2, overlap=40)
    for item in res:
        print(f"GPU {item['gpu_index']}:")
        print(f"  Input Frames: {item['input_range']} (Len: {item['input_length']})")
        print(f"  Keep Local:   {item['keep_local_range']} (Python Slice: [{item['keep_local_range'][0]}:{item['keep_local_range'][1]+1}])")
        print(f"  Output:       {item['global_output_range']}")
```
