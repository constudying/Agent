# 图像序列位置编码使用指南

## 概述

对于 `[batch, seq, c, h, w]` 格式的图像序列数据，位置编码需要同时考虑**时间维度**和**空间维度**的位置信息。

我们实现了 `ImageSequencePositionalEncoding` 类来处理这种特殊情况。

## 核心区别

### 传统1D序列位置编码

- 输入: `[batch, seq, dim]`
- 只需要编码时间位置: `t ∈ [0, seq_len)`

### 图像序列位置编码

- 输入: `[batch, seq, c, h, w]`
- 需要编码三个维度:
  - **时间位置**: `t ∈ [0, seq_len)` - 哪一帧
  - **空间位置**: `(i, j) ∈ [0, H) × [0, W)` - 帧内哪个像素

## 支持的编码类型

### 1. `temporal_only` - 仅时间编码

```python
# 适用场景: 关注帧之间的时序关系
pos_encoder = ImageSequencePositionalEncoding(
    temporal_dim=16,      # 序列长度
    spatial_height=64,    # 图像高度
    spatial_width=64,     # 图像宽度
    embed_dim=256,        # 嵌入维度
    encoding_type='temporal_only'
)
```

- 只对时间维度 `t` 编码
- 空间位置 `(i, j)` 上的编码相同
- **适用**: 动作识别、时序预测

### 2. `spatial_only` - 仅空间编码

```python
# 适用场景: 关注帧内的空间结构
pos_encoder = ImageSequencePositionalEncoding(
    temporal_dim=16,
    spatial_height=64,
    spatial_width=64,
    embed_dim=256,
    encoding_type='spatial_only'
)
```

- 只对空间位置 `(i, j)` 编码
- 不同帧的同一位置编码相同
- **适用**: 图像分割、目标检测

### 3. `spatiotemporal` - 联合时空编码

```python
# 适用场景: 完整的视频理解
pos_encoder = ImageSequencePositionalEncoding(
    temporal_dim=16,
    spatial_height=64,
    spatial_width=64,
    embed_dim=384,  # 必须能被3整除
    encoding_type='spatiotemporal'
)
```

- 时间、高度、宽度各分配 `embed_dim/3` 维度
- 编码: `[PE_t(t), PE_h(i), PE_w(j)]`
- **适用**: 视频分类、时空建模

### 4. `factorized` - 分解编码

```python
# 适用场景: 计算效率优先
pos_encoder = ImageSequencePositionalEncoding(
    temporal_dim=16,
    spatial_height=64,
    spatial_width=64,
    embed_dim=256,
    encoding_type='factorized'
)
```

- 时间和空间编码分别计算后相加
- 编码: `PE(t, i, j) = PE_temporal(t) + PE_spatial(i, j)`
- **适用**: 长序列视频、实时应用

### 5. `learned_2d` - 可学习编码

```python
# 适用场景: 数据充足，需要任务特化
pos_encoder = ImageSequencePositionalEncoding(
    temporal_dim=16,
    spatial_height=64,
    spatial_width=64,
    embed_dim=256,
    encoding_type='learned_2d'
)
```

- 时间和空间编码都是可训练参数
- 在训练中优化，适应特定任务
- **适用**: 有大量训练数据的任务

## 完整使用示例

### 场景: 视频动作识别

```python
import torch
import torch.nn as nn
from agent.models.transformer import ImageSequencePositionalEncoding

# 配置
batch_size = 4
seq_len = 16        # 16帧视频
channels = 3
height, width = 112, 112
embed_dim = 512

# 1. 输入视频数据
video = torch.randn(batch_size, seq_len, channels, height, width)
print(f"输入视频: {video.shape}")  # [4, 16, 3, 112, 112]

# 2. 图像特征提取 (例如使用3D CNN)
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 简化: 使用全局平均池化 + 线性层
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(64 * 7 * 7, embed_dim)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        # 逐帧处理
        x = x.view(B * T, C, H, W)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        x = x.view(B, T, -1)
        return x

encoder = VideoEncoder()
features = encoder(video)
print(f"特征提取: {features.shape}")  # [4, 16, 512]

# 3. 添加位置编码
pos_encoder = ImageSequencePositionalEncoding(
    temporal_dim=seq_len,
    spatial_height=height,
    spatial_width=width,
    embed_dim=embed_dim,
    encoding_type='temporal_only',  # 只关注时序
    dropout=0.1
)

features_with_pos = pos_encoder(features)
print(f"添加位置编码: {features_with_pos.shape}")  # [4, 16, 512]

# 4. 输入到Transformer进行分类
# transformer_output = transformer(features_with_pos)
# prediction = classifier(transformer_output)
```

### 场景: 视频分割 (需要空间信息)

```python
# 使用分解编码保留时空信息
pos_encoder = ImageSequencePositionalEncoding(
    temporal_dim=seq_len,
    spatial_height=height,
    spatial_width=width,
    embed_dim=embed_dim,
    encoding_type='factorized',  # 时空分解编码
    dropout=0.1
)

# 对于需要保留空间结构的任务，获取位置编码
pos_encoding_2d = pos_encoder(video)  # [B, T, H, W, D]

# 可以在特征提取的不同阶段添加
```

## 编码方式对比

| 编码类型 | 参数量 | 计算复杂度 | 适用场景 |
|---------|--------|-----------|---------|
| `temporal_only` | 0 (固定) | 最低 | 动作识别、时序建模 |
| `spatial_only` | 0 (固定) | 低 | 单帧图像任务 |
| `spatiotemporal` | 0 (固定) | 中 | 完整视频理解 |
| `factorized` | 0 (固定) | 低 | **推荐**: 平衡性能和效率 |
| `learned_2d` | T×H×W×D | 中 | 数据充足的特定任务 |

## 实现原理

### 时间编码 (Temporal Encoding)

```python
# 对序列位置 t 进行编码
PE_t(t, 2i) = sin(t / 10000^(2i/d))
PE_t(t, 2i+1) = cos(t / 10000^(2i/d))
```

### 空间编码 (Spatial Encoding)

```python
# 对空间位置 (h, w) 分别编码
PE_h(h, 2i) = sin(h / 10000^(2i/d_h))
PE_h(h, 2i+1) = cos(h / 10000^(2i/d_h))

PE_w(w, 2j) = sin(w / 10000^(2j/d_w))
PE_w(w, 2j+1) = cos(w / 10000^(2j/d_w))
```

### 组合方式

#### 联合编码 (spatiotemporal)

```python
PE(t, h, w) = [PE_t(t), PE_h(h), PE_w(w)]
# 拼接，总维度 = d_t + d_h + d_w = embed_dim
```

#### 分解编码 (factorized)

```python
PE(t, h, w) = PE_t(t) + PE_spatial(h, w)
# 相加，都是 embed_dim 维度
```

## 常见问题

### Q1: 什么时候添加位置编码？

**A:** 通常在特征提取之后、Transformer之前添加:

```
图像序列 -> CNN/ViT特征提取 -> 位置编码 -> Transformer
```

### Q2: 如何选择编码类型？

**A:**

- 不确定? → 选择 `factorized` (最平衡)
- 只关注时序? → 选择 `temporal_only` (最快)
- 需要完整时空信息? → 选择 `spatiotemporal`
- 数据充足? → 可尝试 `learned_2d`

### Q3: embed_dim 如何选择？

**A:**

- `spatiotemporal`: 必须能被3整除 (时间、高、宽各分配1/3)
- 其他类型: 任意值，建议 128/256/512/768

### Q4: 位置编码会增加多少计算量？

**A:**

- 固定编码 (sinusoidal/factorized): 几乎无额外计算，只是查表
- 可学习编码: 增加少量参数和计算

### Q5: 可以在图像展平前添加位置编码吗？

**A:** 可以，但需要确保编码的维度匹配:

```python
# 方式1: 先展平，再编码 (推荐)
features = cnn(images)  # [B, T, D]
features = pos_encoder(features)

# 方式2: 先编码，再展平
pos_encoding = pos_encoder(images)  # [B, T, H, W, D]
# 需要在CNN中融合位置信息
```

## 性能建议

1. **长序列视频 (T > 32)**: 使用 `factorized` 或 `temporal_only`
2. **高分辨率图像 (H,W > 224)**: 在降采样后添加位置编码
3. **实时应用**: 优先使用 `temporal_only`
4. **离线训练，高精度**: 可尝试 `learned_2d` 或 `spatiotemporal`

## 参考文献

- Vaswani et al., "Attention is All You Need", 2017
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", 2020
- Bertasius et al., "Is Space-Time Attention All You Need for Video Understanding?", 2021
