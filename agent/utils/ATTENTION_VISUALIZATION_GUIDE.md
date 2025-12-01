# Transformer 注意力可视化指南

本指南介绍如何在基于Transformer的编解码器中生成和可视化注意力图（热力图）。

## 功能特性

### 1. 注意力权重收集

- 支持收集所有层的注意力权重
- 包括编码器自注意力、解码器自注意力和交叉注意力
- 可以选择性地启用/禁用注意力存储

### 2. 多种可视化方式

- **基础热力图**：单层单头或多头平均的注意力可视化
- **多头对比**：并排展示所有注意力头的注意力模式
- **层级对比**：对比不同层的注意力分布
- **注意力流**：完整展示编码器→解码器的注意力传递
- **统计信息**：保存注意力权重的统计数据

## 快速开始

### 安装依赖

```bash
pip install matplotlib seaborn
```

### 基础使用

```python
import torch
from agent.models.transformer import Transformer, AttentionVisualizer

# 1. 创建模型
model = Transformer(
    embed_dim=512,
    context_length=100,
    num_heads=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
)

# 2. 准备输入数据
enc = {
    'text': torch.randn(2, 10, 512),
    'robot0_eef_pos': torch.randn(2, 1, 512),
    'agentview_image': torch.randn(2, 9, 512),
    'agentview_depth': torch.randn(2, 9, 512),
}
dec = {
    'text': torch.randn(2, 10, 512),
    'robot0_eef_pos_past_traj': torch.randn(2, 10, 512),
    'robot0_eef_pos_past_traj_delta': torch.randn(2, 9, 512),
}

# 3. 启用注意力存储
model.enable_attention_storage()

# 4. 前向传播并获取注意力权重
output, attention_weights = model(enc, dec, return_attention_weights=True)

# 5. 创建可视化器
visualizer = AttentionVisualizer()

# 6. 可视化注意力
visualizer.plot_attention_heatmap(
    attention_weights['encoder'][0]['self_attention'],
    title='Encoder Layer 0 Self-Attention',
    save_path='attention.png'
)
```

## 详细使用说明

### 1. 启用/禁用注意力存储

```python
# 启用注意力存储（在推理时使用）
model.enable_attention_storage()

# 前向传播
output, attn_weights = model(enc, dec, return_attention_weights=True)

# 禁用注意力存储（训练时节省内存）
model.disable_attention_storage()
```

### 2. 注意力权重结构

```python
attention_weights = {
    'encoder': [
        {
            'self_attention': Tensor(B, NH, T, T)
        },  # Layer 0
        {
            'self_attention': Tensor(B, NH, T, T)
        },  # Layer 1
        # ...
    ],
    'decoder': [
        {
            'self_attention': Tensor(B, NH, T, T),
            'cross_attention': Tensor(B, NH, T, S)
        },  # Layer 0
        # ...
    ]
}
```

其中：

- `B`: 批次大小
- `NH`: 注意力头数量
- `T`: 解码器序列长度（查询长度）
- `S`: 编码器序列长度（键/值长度）

### 3. 基础热力图可视化

```python
# 可视化特定头
visualizer.plot_attention_heatmap(
    attention_weights['encoder'][0]['self_attention'],
    head_idx=3,        # 第3个注意力头
    batch_idx=0,       # 第0个样本
    title='Encoder Layer 0, Head 3',
    figsize=(10, 8),
    cmap='viridis',
    save_path='encoder_attn.png'
)

# 可视化所有头的平均
visualizer.plot_attention_heatmap(
    attention_weights['encoder'][0]['self_attention'],
    head_idx=None,     # None表示平均所有头
    title='Encoder Layer 0, Average',
    save_path='encoder_attn_avg.png'
)
```

### 4. 多头注意力对比

```python
# 并排显示所有注意力头
visualizer.plot_multi_head_attention(
    attention_weights['encoder'][0]['self_attention'],
    title='Encoder Layer 0 - All Heads',
    figsize=(20, 12),
    save_path='multi_head.png'
)
```

### 5. 层级对比

```python
# 对比编码器所有层的自注意力
visualizer.plot_layer_comparison(
    attention_weights['encoder'],
    attention_type='self_attention',
    head_idx=None,  # 平均所有头
    title='Encoder Self-Attention Across Layers',
    save_path='encoder_layers.png'
)

# 对比解码器所有层的交叉注意力
visualizer.plot_layer_comparison(
    attention_weights['decoder'],
    attention_type='cross_attention',
    title='Decoder Cross-Attention Across Layers',
    save_path='decoder_cross_layers.png'
)
```

### 6. 完整注意力流

```python
# 可视化完整的注意力流程
visualizer.plot_attention_flow(
    encoder_attention=attention_weights['encoder'][0]['self_attention'],
    decoder_self_attention=attention_weights['decoder'][0]['self_attention'],
    decoder_cross_attention=attention_weights['decoder'][0]['cross_attention'],
    layer_idx=0,
    head_idx=None,  # 平均所有头
    save_path='attention_flow.png'
)
```

### 7. 保存统计信息

```python
# 保存注意力权重的统计信息到JSON文件
visualizer.save_attention_statistics(
    attention_weights,
    save_path='attention_stats.json'
)
```

统计信息示例：

```json
{
  "encoder": [
    {
      "layer": 0,
      "type": "self_attention",
      "mean": 0.034,
      "std": 0.028,
      "min": 0.001,
      "max": 0.215,
      "shape": [8, 29, 29]
    }
  ],
  "decoder": [
    {
      "layer": 0,
      "type": "self_attention",
      "mean": 0.052,
      "std": 0.035,
      "min": 0.002,
      "max": 0.187,
      "shape": [8, 29, 29]
    },
    {
      "layer": 0,
      "type": "cross_attention",
      "mean": 0.034,
      "std": 0.021,
      "min": 0.001,
      "max": 0.142,
      "shape": [8, 29, 29]
    }
  ]
}
```

## 高级用法

### 1. 自定义颜色映射

```python
# 使用不同的颜色映射
visualizer.plot_attention_heatmap(
    attention_weights['encoder'][0]['self_attention'],
    cmap='hot',      # 'viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool'
    vmin=0.0,        # 固定颜色范围最小值
    vmax=0.1,        # 固定颜色范围最大值
    save_path='custom_colormap.png'
)
```

### 2. 批量生成可视化

```python
# 为所有层生成可视化
for layer_idx in range(len(attention_weights['encoder'])):
    visualizer.plot_attention_heatmap(
        attention_weights['encoder'][layer_idx]['self_attention'],
        title=f'Encoder Layer {layer_idx}',
        save_path=f'encoder_layer_{layer_idx}.png',
        show=False  # 不显示，只保存
    )
```

### 3. 在训练循环中使用

```python
# 训练时定期可视化
for epoch in range(num_epochs):
    for batch in dataloader:
        # 训练步骤
        output = model(enc, dec)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # 每N个epoch可视化一次
    if epoch % 10 == 0:
        model.enable_attention_storage()
        with torch.no_grad():
            _, attn = model(enc, dec, return_attention_weights=True)
        
        visualizer.plot_attention_heatmap(
            attn['encoder'][0]['self_attention'],
            save_path=f'attention_epoch_{epoch}.png',
            show=False
        )
        
        model.disable_attention_storage()
```

### 4. 注意力权重分析

```python
import numpy as np

# 分析注意力集中度
encoder_attn = attention_weights['encoder'][0]['self_attention'][0].cpu().numpy()

# 计算每个头的熵（注意力分散程度）
def attention_entropy(attn_matrix):
    """计算注意力分布的熵"""
    # attn_matrix: (T, T)
    entropy = -np.sum(attn_matrix * np.log(attn_matrix + 1e-10), axis=-1)
    return entropy.mean()

for head_idx in range(encoder_attn.shape[0]):
    head_attn = encoder_attn[head_idx]
    entropy = attention_entropy(head_attn)
    print(f"Head {head_idx}: Entropy = {entropy:.4f}")

# 找出最重要的注意力连接
def top_k_attention_pairs(attn_matrix, k=10):
    """找出权重最大的k个注意力对"""
    attn_flat = attn_matrix.flatten()
    top_k_indices = np.argsort(attn_flat)[-k:][::-1]
    
    pairs = []
    for idx in top_k_indices:
        query_idx = idx // attn_matrix.shape[1]
        key_idx = idx % attn_matrix.shape[1]
        weight = attn_flat[idx]
        pairs.append((query_idx, key_idx, weight))
    
    return pairs

# 对平均注意力进行分析
avg_attn = encoder_attn.mean(axis=0)
top_pairs = top_k_attention_pairs(avg_attn)
print("\nTop 10 attention pairs:")
for query, key, weight in top_pairs:
    print(f"  Query {query} -> Key {key}: {weight:.4f}")
```

## 使用示例

运行完整示例：

```bash
python agent/utils/attention_visualization_example.py
```

这将生成以下可视化文件：

- `encoder_layer0_attention.png`: 编码器第0层自注意力
- `decoder_layer0_cross_attention.png`: 解码器第0层交叉注意力
- `encoder_multi_head.png`: 编码器多头注意力对比
- `encoder_layer_comparison.png`: 编码器层级对比
- `decoder_cross_attention_comparison.png`: 解码器交叉注意力层级对比
- `attention_flow_layer0.png`: 完整注意力流
- `attention_statistics.json`: 注意力统计信息

## 性能考虑

1. **内存使用**：注意力权重会占用额外内存，建议只在推理或验证时启用
2. **训练时**：使用 `model.disable_attention_storage()` 禁用存储
3. **大批次**：可视化时选择 `batch_idx=0` 只看第一个样本
4. **多层模型**：如果层数很多，可以选择性地只可视化关键层

## 故障排除

### 问题1：没有matplotlib或seaborn

```bash
pip install matplotlib seaborn
```

### 问题2：注意力权重为None

确保在前向传播前启用了注意力存储：

```python
model.enable_attention_storage()
```

### 问题3：内存不足

- 减小批次大小
- 只可视化部分层
- 使用 `show=False` 避免显示窗口

### 问题4：图片无法保存

检查保存路径的权限，或使用绝对路径：

```python
save_path='/absolute/path/to/attention.png'
```

## 代码修改说明

本功能通过以下修改实现：

1. **注意力模块修改**：
   - `NormalSelfAttention`
   - `CausalSelfAttention`
   - `CrossAttention`

   添加了：
   - `store_attention` 参数
   - `attention_weights` 属性
   - `return_attention` 参数

2. **Transformer类修改**：
   - 添加 `enable_attention_storage()` 方法
   - 添加 `disable_attention_storage()` 方法
   - 添加 `get_attention_weights()` 方法
   - `forward()` 方法支持 `return_attention_weights` 参数

3. **新增工具类**：
   - `AttentionVisualizer` 类提供多种可视化方法

## 参考资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Visualizing Attention in Transformer Models](https://jalammar.github.io/illustrated-transformer/)
- [BertViz: Attention Visualization](https://github.com/jessevig/bertviz)

## 联系方式

如有问题或建议，请提交Issue或Pull Request。
