# FiLM 网络设备不匹配修复说明

## 错误描述

```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```

## 根本原因

这个错误有**两个问题**需要同时修复：

### 问题 1: 设备不匹配

在 `ResNetFiLM.forward()` 中创建的临时张量默认在 CPU 上：

```python
film_out = self.resnet["film"](torch.zeros(512))  # ❌ 在 CPU 上
```

### 问题 2: 形状和逻辑错误

1. FiLM 网络需要接收 **batch** 维度的输入
2. `film_out[i,:]` 的索引方式假设输出是 2D，但实际是 1D
3. `ResidualBlock.forward()` 期望接收形状为 `[batch, 2]` 的 FiLM 参数

## 完整修复方案

### 修复 1: ResNetFiLM.forward()

**之前的错误代码：**

```python
def forward(self, x):
    device = x.device
    film_out = self.resnet["film"](torch.zeros(512, device=device))
    
    for i in range(self.block_nums):
        if self.film_dict[i] is True:
            x = self.resnet[f"block{i+1}"](x, film_out[i,:])  # ❌ 错误的索引
```

**修复后的代码：**

```python
def forward(self, x):
    # 从输入 x 获取设备和 batch 大小
    device = x.device
    batch_size = x.shape[0]
    
    # 为每个 batch 中的样本计算 FiLM 参数
    # 输入形状: [batch, 512]，输出形状: [batch, 2]
    film_input = torch.zeros(batch_size, 512, device=device)
    film_out = self.resnet["film"](film_input)
    
    # 通过各个残差块
    for i in range(self.block_nums):
        if self.film_dict[i] is True:
            x = self.resnet[f"block{i+1}"](x, film_out)  # 传递 [batch, 2] 张量
        elif self.film_dict[i] is False:
            x = self.resnet[f"block{i+1}"](x)
        else:
            raise ValueError("film_dict values must be True or False")
    return x
```

### 修复 2: ResidualBlock.forward()

**之前的错误代码：**

```python
def forward(self, x, *args):
    film_params = args[0] if len(args) > 0 else None
    out = self.convnet[0](x)
    out = self.activation(out)
    out = self.convnet[1](out)
    if film_params is not None:
        gamma, beta = film_params[0], film_params[1]  # ❌ 假设是 [gamma, beta]
        out = gamma * out + beta  # ❌ 形状不匹配
```

**修复后的代码：**

```python
def forward(self, x, *args):
    film_params = args[0] if len(args) > 0 else None
    out = self.convnet[0](x)
    out = self.activation(out)
    out = self.convnet[1](out)
    
    if film_params is not None:
        # film_params 形状: [batch, 2]，包含 gamma 和 beta
        # out 形状: [batch, channels, height, width]
        gamma = film_params[:, 0]  # [batch]
        beta = film_params[:, 1]   # [batch]
        
        # 将 gamma 和 beta 扩展为 [batch, 1, 1, 1] 以便广播
        gamma = gamma.view(-1, 1, 1, 1)
        beta = beta.view(-1, 1, 1, 1)
        
        # FiLM 操作: out = gamma * out + beta
        out = gamma * out + beta
    
    identity = self.convnet[2](x)
    out += identity
    out = self.activation(out)
    return out
```

## 关键修复点

### 1. 设备一致性

```python
device = x.device  # 从输入获取设备
film_input = torch.zeros(batch_size, 512, device=device)  # 创建时指定设备
```

### 2. Batch 维度处理

```python
batch_size = x.shape[0]
film_input = torch.zeros(batch_size, 512, device=device)  # 包含 batch 维度
```

### 3. FiLM 参数的正确使用

```python
# 从 [batch, 2] 中提取 gamma 和 beta
gamma = film_params[:, 0].view(-1, 1, 1, 1)  # [batch, 1, 1, 1]
beta = film_params[:, 1].view(-1, 1, 1, 1)   # [batch, 1, 1, 1]

# 广播到 [batch, channels, height, width]
out = gamma * out + beta
```

## FiLM 工作原理

FiLM (Feature-wise Linear Modulation) 通过学习到的 γ (gamma) 和 β (beta) 参数来调制特征：

```
out = γ ⊙ out + β
```

其中：

- `γ` (gamma): 缩放因子
- `β` (beta): 偏移因子
- `⊙`: 逐元素乘法

在我们的实现中：

- FiLM 网络输入: `[batch, 512]` 的零张量
- FiLM 网络输出: `[batch, 2]`，其中第一列是 γ，第二列是 β
- 应用到特征图: `[batch, channels, H, W]`

## 测试验证

运行修复后的代码：

```bash
cd /home/lsy/cjh/project1/Agent
python agent/scripts/train.py \
    --config ./agent/configs/stage1_imagespec.json \
    --dataset 'agent/datasets/playdata/image_demo_local.hdf5'
```

## 相关文件

- `agent/models/base_nets.py` - 包含 `ResNetFiLM` 和 `ResidualBlock` 类
- `agent/configs/stage1_imagespec.json` - 配置文件，定义 FiLM 网络结构
