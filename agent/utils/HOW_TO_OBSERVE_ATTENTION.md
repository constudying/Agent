# 如何观察编码器和解码器的注意力图 - 快速指南

## 📊 一分钟快速上手

```python
from agent.models.transformer import Transformer, AttentionVisualizer
import torch

# 1. 获取注意力权重
model = Transformer(...)
model.enable_attention_storage()

with torch.no_grad():
    output, attn = model(enc, dec, return_attention_weights=True)

model.disable_attention_storage()

# 2. 创建可视化器
visualizer = AttentionVisualizer()

# 3. 查看编码器注意力
visualizer.plot_attention_heatmap(
    attn['encoder'][0]['self_attention'],
    title='编码器第0层自注意力',
    save_path='encoder_attn.png'
)

# 4. 查看解码器交叉注意力
visualizer.plot_attention_heatmap(
    attn['decoder'][0]['cross_attention'],
    title='解码器第0层交叉注意力',
    save_path='decoder_cross_attn.png'
)
```

## 🎯 注意力类型说明

### 1. 编码器自注意力 (Encoder Self-Attention)

**作用**: 编码器内部，每个位置关注其他位置

```python
# 查看编码器第0层
attn['encoder'][0]['self_attention']
# 形状: (batch_size, num_heads, seq_len, seq_len)

# 可视化
visualizer.plot_attention_heatmap(
    attn['encoder'][0]['self_attention'],
    title='编码器自注意力'
)
```

**如何解读**:

- **行**: Query位置（问"我应该关注谁？"）
- **列**: Key位置（被关注的位置）
- **亮点**: 表示强注意力
- **暗点**: 表示弱注意力

**例子**:

- 如果位置5对位置3的值很亮 → 位置5在关注位置3
- 对角线亮 → 位置关注自己
- 某行全亮 → 该位置关注所有位置

### 2. 解码器交叉注意力 (Decoder Cross-Attention)

**作用**: 解码器关注编码器，这是信息从输入流向输出的关键

```python
# 查看解码器第0层的交叉注意力
attn['decoder'][0]['cross_attention']
# 形状: (batch_size, num_heads, decoder_len, encoder_len)

# 可视化
visualizer.plot_attention_heatmap(
    attn['decoder'][0]['cross_attention'],
    title='解码器交叉注意力'
)
```

**如何解读**:

- **行**: 解码器位置
- **列**: 编码器位置
- **含义**: 解码器每个位置在关注编码器的哪些部分

**例子**:

- 如果解码器位置2在编码器位置7很亮 → 生成第2个输出时主要参考输入的第7个位置
- 某行有多个亮点 → 该输出综合参考了多个输入位置

### 3. 解码器自注意力 (Decoder Self-Attention)

**作用**: 解码器内部，每个位置关注之前的位置（因果mask）

```python
# 查看解码器第0层的自注意力
attn['decoder'][0]['self_attention']
# 形状: (batch_size, num_heads, seq_len, seq_len)

# 可视化
visualizer.plot_attention_heatmap(
    attn['decoder'][0]['self_attention'],
    title='解码器自注意力（因果）'
)
```

**如何解读**:

- **下三角矩阵**: 只能看到当前和之前的位置
- **上三角全黑**: 因果mask，不能看到未来

## 🔍 六种观察方法

### 方法1: 基础热力图（最常用）

```python
# 查看某一层某一头的注意力
visualizer.plot_attention_heatmap(
    attn['encoder'][0]['self_attention'],
    head_idx=3,      # 第3个头，None=平均所有头
    batch_idx=0,     # 第0个样本
    save_path='attn.png'
)
```

**用途**: 快速查看单层注意力分布

### 方法2: 层级对比

```python
# 对比所有编码器层
visualizer.plot_layer_comparison(
    attn['encoder'],
    attention_type='self_attention',
    title='编码器各层对比'
)

# 对比所有解码器交叉注意力
visualizer.plot_layer_comparison(
    attn['decoder'],
    attention_type='cross_attention',
    title='解码器交叉注意力各层对比'
)
```

**用途**: 观察注意力随深度的变化

### 方法3: 多头分析

```python
# 并排显示所有注意力头
visualizer.plot_multi_head_attention(
    attn['encoder'][0]['self_attention'],
    title='编码器第0层所有头'
)
```

**用途**: 理解不同头学到了什么

### 方法4: 完整注意力流

```python
# 从编码器到解码器的完整流程
visualizer.plot_attention_flow(
    encoder_attention=attn['encoder'][0]['self_attention'],
    decoder_self_attention=attn['decoder'][0]['self_attention'],
    decoder_cross_attention=attn['decoder'][0]['cross_attention'],
    layer_idx=0
)
```

**用途**: 理解信息流动路径

### 方法5: 数值分析

```python
# 查看特定位置的注意力分布
decoder_pos = 5
cross_attn = attn['decoder'][0]['cross_attention']
attn_at_pos = cross_attn[0, :, decoder_pos, :].mean(dim=0)

# 找出最关注的位置
top_values, top_indices = torch.topk(attn_at_pos, 5)
print("解码器位置5最关注编码器的位置:", top_indices)
```

**用途**: 精确分析某个位置的注意力

### 方法6: 统计信息

```python
# 保存所有层的统计数据
visualizer.save_attention_statistics(
    attn,
    save_path='attention_stats.json'
)
```

**用途**: 定量分析和记录

## 📋 注意力数据结构

```python
attention_weights = {
    'encoder': [
        {'self_attention': Tensor(B, NH, T, T)},    # Layer 0
        {'self_attention': Tensor(B, NH, T, T)},    # Layer 1
        # ...
    ],
    'decoder': [
        {
            'self_attention': Tensor(B, NH, T, T),
            'cross_attention': Tensor(B, NH, T_dec, T_enc)
        },  # Layer 0
        # ...
    ]
}
```

**维度说明**:

- `B`: Batch Size（批次大小）
- `NH`: Number of Heads（注意力头数）
- `T`: Sequence Length（序列长度）
- `T_dec`: Decoder序列长度
- `T_enc`: Encoder序列长度

## 🎨 实际应用场景

### 场景1: 调试模型行为

```python
# 检查模型是否学到了合理的注意力模式
model.enable_attention_storage()
with torch.no_grad():
    output, attn = model(enc, dec, return_attention_weights=True)

# 查看解码器是否正确关注编码器
visualizer.plot_attention_heatmap(
    attn['decoder'][0]['cross_attention'],
    title='解码器关注输入的哪些位置？'
)
```

### 场景2: 分析模型学习过程

```python
# 训练不同阶段的注意力对比
for epoch in [0, 10, 50, 100]:
    load_checkpoint(f'epoch_{epoch}.pth')
    _, attn = model(enc, dec, return_attention_weights=True)
    
    visualizer.plot_attention_heatmap(
        attn['decoder'][0]['cross_attention'],
        save_path=f'attention_epoch_{epoch}.png'
    )
```

### 场景3: 可解释性分析

```python
# 解释模型的预测
# 例如：为什么模型生成了这个输出？
output, attn = model(enc, dec, return_attention_weights=True)

# 查看生成某个token时关注了输入的哪些部分
output_pos = 10  # 第10个输出token
cross_attn = attn['decoder'][-1]['cross_attention'][0, :, output_pos, :]

# 可视化：生成第10个token时的注意力分布
plt.bar(range(len(cross_attn.mean(0))), cross_attn.mean(0).cpu())
plt.title(f'生成第{output_pos}个token时的输入注意力')
plt.xlabel('输入位置')
plt.ylabel('注意力权重')
```

## 💡 常见模式解读

### 模式1: 对角线明显

```
■ □ □ □ □
□ ■ □ □ □
□ □ ■ □ □
□ □ □ ■ □
□ □ □ □ ■
```

**含义**: 位置主要关注自己，可能是模型依赖局部信息

### 模式2: 全局注意

```
■ ■ ■ ■ ■
■ ■ ■ ■ ■
■ ■ ■ ■ ■
■ ■ ■ ■ ■
■ ■ ■ ■ ■
```

**含义**: 每个位置关注所有位置，捕捉全局依赖

### 模式3: 稀疏注意

```
■ □ □ ■ □
□ □ ■ □ □
■ □ □ □ ■
□ ■ □ □ □
□ □ ■ □ ■
```

**含义**: 选择性关注，学到了特定的依赖关系

### 模式4: 因果下三角（解码器）

```
■ □ □ □ □
■ ■ □ □ □
■ ■ ■ □ □
■ ■ ■ ■ □
■ ■ ■ ■ ■
```

**含义**: 正常的因果注意力，只看当前和之前

## 🚀 快速运行完整示例

```bash
# 运行包含所有可视化方法的示例
python agent/utils/how_to_visualize_attention.py
```

这将生成：

1. `encoder_layer0_self_attention.png` - 编码器自注意力
2. `decoder_layer0_cross_attention.png` - 解码器交叉注意力
3. `decoder_layer0_self_attention.png` - 解码器自注意力
4. `encoder_all_layers_comparison.png` - 编码器层级对比
5. `decoder_all_layers_cross_comparison.png` - 解码器交叉注意力对比
6. `encoder_layer0_all_heads.png` - 多头对比
7. `attention_flow_layer0.png` - 完整注意力流
8. `attention_statistics.json` - 统计信息

## ❓ 常见问题

### Q1: 注意力权重为None？

```python
# 确保启用了注意力存储
model.enable_attention_storage()
_, attn = model(enc, dec, return_attention_weights=True)
```

### Q2: 如何只看特定的头？

```python
# head_idx=3 只看第3个头
visualizer.plot_attention_heatmap(
    attn['encoder'][0]['self_attention'],
    head_idx=3
)
```

### Q3: 如何比较不同样本？

```python
# batch_idx指定要看第几个样本
for i in range(batch_size):
    visualizer.plot_attention_heatmap(
        attn['encoder'][0]['self_attention'],
        batch_idx=i,
        save_path=f'sample_{i}.png'
    )
```

### Q4: 内存不足？

```python
# 只可视化关键层
model.enable_attention_storage()
_, attn = model(enc, dec, return_attention_weights=True)
model.disable_attention_storage()  # 立即禁用

# 只保存需要的层
visualizer.plot_attention_heatmap(
    attn['decoder'][0]['cross_attention'],  # 只看第0层
    show=False  # 不显示，只保存
)
```

## 📚 更多资源

- 详细指南: `agent/utils/ATTENTION_VISUALIZATION_GUIDE.md`
- 完整示例: `agent/utils/attention_visualization_example.py`
- 训练监控: `agent/utils/TRAINING_ATTENTION_GUIDE.md`
