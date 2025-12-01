# 训练时注意力监控 - 快速参考

## 一行代码集成

```python
from agent.utils.training_attention_monitor import TrainingAttentionMonitor

# 创建监控器
monitor = TrainingAttentionMonitor(save_dir='./attn_logs', save_frequency=100)

# 在训练循环中添加一行
for step, batch in enumerate(dataloader):
    output = model(enc, dec)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    monitor.log_attention(model, enc, dec, step, loss.item())  # 添加这行

# 训练结束
monitor.close()
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 🚀 **不影响训练** | 异步后台处理，主线程不阻塞 |
| 💾 **自动管理内存** | 立即释放GPU内存，防止OOM |
| 📊 **TensorBoard集成** | 实时查看注意力统计 |
| ⚙️ **高度可配置** | 控制频率、层、头、可视化类型 |
| 🎯 **两种模式** | 完整可视化 或 轻量级统计 |

## 两种监控器对比

| 特性 | TrainingAttentionMonitor | LightweightAttentionMonitor |
|------|-------------------------|---------------------------|
| 生成图像 | ✅ 是 | ❌ 否 |
| 保存统计 | ✅ 是 | ✅ 是 |
| 速度影响 | ~2% | ~0.5% |
| 磁盘占用 | 中等 | 很小 |
| 适用场景 | 验证、检查点 | 长时间训练 |

## 常用配置

### 🎯 推荐配置（平衡）

```python
monitor = TrainingAttentionMonitor(
    save_dir='./attention_logs',
    save_frequency=100,              # 每100步
    visualization_types=['heatmap', 'statistics'],
    layers_to_visualize=[0, -1],    # 首尾层
    use_tensorboard=True
)
```

### 🔍 调试配置（详细）

```python
monitor = TrainingAttentionMonitor(
    save_frequency=10,               # 更频繁
    visualization_types=['heatmap', 'multi_head', 'layer_comparison', 'statistics'],
    layers_to_visualize=None,       # 所有层
)
```

### ⚡ 高性能配置（最快）

```python
monitor = LightweightAttentionMonitor(
    save_frequency=50,
    use_tensorboard=True
)
```

## 性能影响

```
基准训练速度: 100%

+ LightweightAttentionMonitor:  100.5%  (+0.5%)
+ TrainingAttentionMonitor:     102%    (+2%)
```

## 查看结果

### 文件系统

```bash
attention_logs/
├── step_100/
│   ├── encoder_layer0_self_attn.png      # 注意力热力图
│   ├── decoder_layer0_cross_attn.png     # 交叉注意力
│   └── statistics.json                    # 统计数据
├── step_200/
└── tensorboard/                           # TensorBoard日志
```

### TensorBoard

```bash
tensorboard --logdir=./attention_logs/tensorboard
# 访问 http://localhost:6006
```

## 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| 训练变慢 | 提高`save_frequency`或使用`LightweightAttentionMonitor` |
| 内存占用高 | 设置`save_raw_weights=False`，`max_workers=1` |
| 磁盘占满 | 使用`LightweightAttentionMonitor`或只保存`statistics` |
| GPU内存不足 | 已自动处理，注意力权重会立即移到CPU |

## 完整示例

```python
from agent.utils.training_attention_monitor import TrainingAttentionMonitor
import torch.nn as nn
import torch.optim as optim

# 模型和优化器
model = Transformer(...)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# 创建监控器
with TrainingAttentionMonitor(
    save_dir='./attention_logs',
    save_frequency=100,
    use_tensorboard=True
) as monitor:
    
    # 训练循环
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            # 准备数据
            enc, dec, target = prepare_batch(batch)
            
            # 前向传播
            output = model(enc, dec)
            loss = criterion(output, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录注意力（不影响训练）
            monitor.log_attention(
                model, enc, dec, 
                global_step, loss.item(), epoch
            )

# 自动关闭和清理
```

## 更多信息

- 详细文档: `agent/utils/TRAINING_ATTENTION_GUIDE.md`
- 完整示例: `agent/utils/training_with_attention_example.py`
- 运行示例: `python agent/utils/training_with_attention_example.py`

## 关键要点

1. ✅ 使用异步处理，训练不会被阻塞
2. ✅ 自动释放GPU内存，防止OOM
3. ✅ 根据需求选择完整或轻量级模式
4. ✅ 使用TensorBoard实时监控
5. ✅ 记得在训练结束时调用`close()`
