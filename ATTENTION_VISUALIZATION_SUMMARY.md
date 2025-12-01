# Transformer注意力可视化功能总结

## 实现的功能

### 1. 核心修改

#### 注意力模块增强

- **NormalSelfAttention**: 添加注意力权重存储和返回功能
- **CausalSelfAttention**: 添加注意力权重存储和返回功能  
- **CrossAttention**: 添加注意力权重存储和返回功能

每个模块新增：

- `store_attention` 参数：控制是否自动存储注意力权重
- `attention_weights` 属性：存储注意力权重
- `return_attention` 参数：控制forward是否返回注意力权重

#### Transformer类增强

新增方法：

```python
# 启用所有层的注意力存储
model.enable_attention_storage()

# 禁用注意力存储（训练时节省内存）
model.disable_attention_storage()

# 收集所有层的注意力权重
attention_weights = model.get_attention_weights()

# 前向传播时返回注意力权重
output, attention_weights = model(enc, dec, return_attention_weights=True)
```

### 2. AttentionVisualizer 可视化工具类

提供5种主要可视化方法：

#### 2.1 基础热力图

```python
visualizer.plot_attention_heatmap(
    attention_weights['encoder'][0]['self_attention'],
    head_idx=3,  # 特定头，None表示平均所有头
    title='Encoder Layer 0 Self-Attention'
)
```

#### 2.2 多头注意力对比

```python
visualizer.plot_multi_head_attention(
    attention_weights['encoder'][0]['self_attention'],
    title='All Attention Heads'
)
```

#### 2.3 层级对比

```python
visualizer.plot_layer_comparison(
    attention_weights['encoder'],
    attention_type='self_attention',
    title='Encoder Layers Comparison'
)
```

#### 2.4 完整注意力流

```python
visualizer.plot_attention_flow(
    encoder_attention=...,
    decoder_self_attention=...,
    decoder_cross_attention=...,
    layer_idx=0
)
```

#### 2.5 统计信息保存

```python
visualizer.save_attention_statistics(
    attention_weights,
    save_path='stats.json'
)
```

## 使用流程

```python
# 1. 创建模型
model = Transformer(...)

# 2. 启用注意力存储
model.enable_attention_storage()

# 3. 前向传播并获取注意力
output, attn = model(enc, dec, return_attention_weights=True)

# 4. 可视化
visualizer = AttentionVisualizer()
visualizer.plot_attention_heatmap(
    attn['encoder'][0]['self_attention'],
    save_path='attention.png'
)

# 5. 训练时记得禁用
model.disable_attention_storage()
```

## 注意力权重数据结构

```python
attention_weights = {
    'encoder': [
        {'self_attention': Tensor(B, NH, T, T)},  # Layer 0
        {'self_attention': Tensor(B, NH, T, T)},  # Layer 1
        ...
    ],
    'decoder': [
        {
            'self_attention': Tensor(B, NH, T, T),
            'cross_attention': Tensor(B, NH, T, S)
        },  # Layer 0
        ...
    ]
}
```

其中：

- B: Batch Size
- NH: Number of Heads
- T: Decoder Sequence Length
- S: Encoder Sequence Length

## 文件清单

1. **主要代码**: `/home/lsy/cjh/project1/Agent/agent/models/transformer.py`
   - 修改的注意力模块
   - 修改的Transformer类
   - 新增的AttentionVisualizer类

2. **使用示例**: `/home/lsy/cjh/project1/Agent/agent/utils/attention_visualization_example.py`
   - 5个完整的使用示例
   - 涵盖所有可视化功能

3. **详细文档**: `/home/lsy/cjh/project1/Agent/agent/utils/ATTENTION_VISUALIZATION_GUIDE.md`
   - 完整的使用指南
   - API参考
   - 高级用法和故障排除

## 应用场景

1. **模型分析**：理解模型关注哪些输入部分
2. **调试**：诊断模型行为异常
3. **研究**：分析不同层和头的注意力模式
4. **可解释性**：向用户展示模型决策依据
5. **论文可视化**：生成高质量的注意力图表

## 性能建议

- ✅ **推理/验证时**: 启用注意力存储
- ❌ **训练时**: 禁用注意力存储以节省内存
- 💡 **大批次**: 只可视化batch_idx=0
- 💡 **多层模型**: 选择性可视化关键层

## 依赖要求

```bash
pip install matplotlib seaborn
```

## 快速测试

```bash
python agent/utils/attention_visualization_example.py
```

这将生成多个示例可视化图片和统计文件。
