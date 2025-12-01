# Transformer Fill Mode 使用指南

## 概述

Transformer 的 `forward` 方法现在支持三种补全模式，用于处理不同的序列生成场景。

## 三种模式对比

| 模式 | 速度 | 质量 | 适用场景 | 计算成本 |
|------|------|------|----------|----------|
| `one_shot` | ⚡⚡⚡ 最快 | ⭐⭐ 中等 | 实时推理、初步预测 | 1次前向传播 |
| `autoregressive` | ⚡ 慢 | ⭐⭐⭐ 最好 | 高质量补全、训练 | N次前向传播 |
| `sliding_window` | ⚡⚡ 中等 | ⭐⭐⭐ 好 | 滚动预测未来 | N次前向传播 |

## 模式详解

### 1. one_shot 模式（一次性预测）

**原理**：

```python
输入: [真实点, pad, pad, ..., pad]
输出: [p1, p2, p3, ..., p10]  # 一次性预测所有位置
```

**特点**：

- ✅ 速度最快（单次前向传播）
- ✅ 适合推理阶段
- ❌ 后面的点无法利用前面预测的信息
- ❌ 预测质量略低

**使用场景**：

- 实时推理需要快速响应
- 对精度要求不高的初步预测
- Scheduled sampling 的预测阶段（配合真值混合）

**代码示例**：

```python
# 在 agent.py 中
decoder_output = self.nets['backbone'].forward(
    enc_inputs, 
    dec_inputs, 
    return_attention_weights=False,
    autoregressive=True,
    fill_mode='one_shot'  # 使用一次性预测
)
```

### 2. autoregressive 模式（自回归补全）

**原理**：

```python
# 逐步填充空缺位置
Step 1: [真实点, pad, pad, ...] → 预测p1 → [真实点, p1, pad, ...]
Step 2: [真实点, p1, pad, ...] → 预测p2 → [真实点, p1, p2, ...]
...
Step 10: [真实点, p1, ..., p9] → 预测p10 → [真实点, p1, ..., p10]
```

**特点**：

- ✅ 预测质量最高
- ✅ 每个点都能利用之前预测的信息
- ✅ 更符合时序建模的直觉
- ❌ 速度慢（需要N次前向传播）

**使用场景**：

- 训练阶段的高质量数据生成
- 对精度要求高的场合
- 需要逐步补全缺失数据

**代码示例**：

```python
# 输入需要包含 attention_mask
dec_inputs['attention_mask'] = attention_mask  # [B, seq_len], 1表示有效

decoder_output = self.nets['backbone'].forward(
    enc_inputs, 
    dec_inputs, 
    return_attention_weights=False,
    autoregressive=True,
    fill_mode='autoregressive'  # 使用自回归补全
)
```

### 3. sliding_window 模式（滑动窗口）

**原理**：

```python
# 滚动预测未来序列
Step 1: [p0, p1, ..., p9] → 预测p10 → 窗口变为[p1, p2, ..., p10]
Step 2: [p1, p2, ..., p10] → 预测p11 → 窗口变为[p2, p3, ..., p11]
```

**特点**：

- ✅ 适合连续预测未来
- ✅ 保持固定窗口大小
- ❌ 不是补全空缺，而是预测新值

**使用场景**：

- 已有完整历史数据
- 需要预测未来N步
- 轨迹滚动预测

**代码示例**：

```python
decoder_output = self.nets['backbone'].forward(
    enc_inputs, 
    dec_inputs, 
    return_attention_weights=False,
    autoregressive=True,
    fill_mode='sliding_window'  # 使用滑动窗口
)
```

## 在 Scheduled Sampling 中的应用

### 推荐组合策略

```python
def _forward_training_with_scheduled_sampling(self, batch, epoch):
    if epoch > 100:
        k = 0.995
        teacher_forcing_ratio = k ** (epoch - 100)
        
        # 阶段1: 快速预测（使用one_shot）
        start_point = batch['obs']['robot0_eef_pos_step_traj_past'][:, 0:1]
        
        # Padding
        padded_input = torch.zeros_like(batch['obs']['robot0_eef_pos_step_traj_past'])
        padded_input[:, 0] = start_point.squeeze(1)
        
        with torch.no_grad():
            temp_obs = batch['obs'].copy()
            temp_obs['robot0_eef_pos_step_traj_past'] = padded_input
            
            # 使用 one_shot 快速预测
            temp_dists, _ = self.nets["policy"].forward_train(
                obs_dict=temp_obs,
                goal_dict=batch['goal_obs'],
                return_attention_weights=True,
                fill_mode='one_shot'  # 快速预测
            )
            
            predicted_10steps = temp_dists.mean.view(-1, 10, 3)
        
        # 阶段2: 随机混合真值（Scheduled Sampling）
        ground_truth = batch['obs']['robot0_eef_pos_step_traj_past']
        use_gt_mask = torch.bernoulli(
            torch.full((batch_size, 10), teacher_forcing_ratio, device=ground_truth.device)
        ).bool()
        
        mixed_input = torch.where(
            use_gt_mask.unsqueeze(-1),
            ground_truth,
            predicted_10steps
        )
        
        batch['obs']['robot0_eef_pos_step_traj_past'] = mixed_input
    
    # 正常训练（可以使用任何模式）
    dists, entropy_loss = self.nets["policy"].forward_train(
        obs_dict=batch["obs"],
        goal_dict=batch["goal_obs"],
        return_attention_weights=True,
        fill_mode='one_shot'  # 或 'autoregressive'
    )
```

## 性能对比

假设序列长度为10：

| 模式 | 前向传播次数 | 相对速度 | 推荐使用率 |
|------|-------------|---------|-----------|
| one_shot | 1 | 1x (基准) | 80% (推理) |
| autoregressive | 10 | 0.1x | 20% (训练/高精度) |
| sliding_window | 10 | 0.1x | 特殊场景 |

## 注意事项

1. **autoregressive 模式需要 attention_mask**：

   ```python
   dec_inputs['attention_mask'] = mask  # [B, seq_len]
   ```

2. **默认模式是 'autoregressive'**：
   - 如果不指定 `fill_mode`，默认使用 `autoregressive`
   - 保持向后兼容

3. **训练 vs 推理**：
   - 训练：推荐 `one_shot`（速度快，配合scheduled sampling效果好）
   - 推理：根据需求选择（快速用`one_shot`，高质量用`autoregressive`）

4. **位置编码**：
   - 所有模式都会对所有位置添加位置编码
   - 即使是padding位置也有位置编码（但会被mask忽略）

## 总结建议

### 你的场景（Scheduled Sampling）

✅ **推荐使用 `one_shot` 模式**

**理由**：

1. **速度快**：训练时每个batch只需2次前向传播（预测+正常训练）
2. **效果好**：配合随机混合真值，已经能有效缓解exposure bias
3. **简单**：不需要额外的mask管理
4. **实用**：在精度和速度之间取得最佳平衡

**实现**：

```python
# 在 agent.py 的 _forward_training 中
fill_mode='one_shot'  # 明确指定使用一次性预测
```

如果未来需要更高质量的补全，可以切换到 `autoregressive` 模式，只需修改一个参数即可。
