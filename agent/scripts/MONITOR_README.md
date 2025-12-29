# 模型中间输出监控脚本使用文档

## 概述

该脚本实现了一个`Monitor`类，用于在PyTorch模型训练过程中捕获和记录指定层的中间计算输出，而无需修改模型代码或训练脚本的核心逻辑。通过PyTorch的forward hooks机制，脚本能够在训练的高层实现模型脚本与训练脚本间的通信，特别适用于记录指定epoch（或batch位置）的中间变量，如Transformer中的注意力图。脚本使用优化后的数据结构存储输出，支持别名索引和tuple输出处理，提供灵活的数据访问接口，允许在训练循环中进行自定义处理，而非固定的通用处理逻辑。

## 功能特性

- **无侵入式监控**：不修改模型定义或训练循环的核心代码，只需在训练脚本中插入Monitor实例。
- **灵活配置**：支持指定监控的层、记录的epoch和可选的batch索引。
- **自动保存**：在满足条件时自动保存中间输出到磁盘，支持后续分析。
- **内存友好**：使用detach和CPU转移，避免GPU内存泄漏；优化数据存储结构，减少冗余。
- **可扩展**：易于扩展支持更多监控条件，如batch位置或自定义过滤。
- **Tuple输出支持**：自动处理如MultiheadAttention等返回tuple输出的层。
- **别名系统**：支持为层指定唯一别名，提高数据访问的可读性。

## 安装和依赖

### 依赖项

- Python 3.7+
- PyTorch (与您的模型兼容的版本)
- 其他依赖：`os`, `json`, `collections` (标准库)

### 安装步骤

1. 确保脚本文件位于项目目录中：
   - `agent/scripts/monitor.py`：Monitor类实现
   - `agent/scripts/train.py`：训练脚本（已集成Monitor）

2. 设置Python路径：

   ```bash
   export PYTHONPATH=/path/to/project1
   ```

3. 验证导入：

   ```bash
   python -c "from agent.scripts.monitor import Monitor; print('导入成功')"
   ```

## 配置

### 获取model的模块名称

通过model（应该是nn.Module类）的named_modules()方法获取。

### Monitor类参数

- `model`：要监控的PyTorch模型实例。
- `layers_to_monitor`：要监控的层配置。
  - **列表格式**：`['layer1', 'layer2']`（默认alias=None, record_type='single'）。
  - **字典格式**：`{'layer1': {'alias': 'attention_output', 'record_type': 'single'}, 'layer2': {'alias': None, 'record_type': 'epoch_full'}}`。
    - `alias`：可选的含义名称，用于提高可读性（默认None，使用层名）。**别名必须在所有层中唯一**。
    - `record_type`：记录类型。
      - `'single'`：记录指定epoch/batch的单次输出（默认）。**必须指定epochs_to_record和batch_indices**。
      - `'epoch_full'`：累积整个epoch的所有batch输出，拼接后计算统计，并在epoch结束时保存。**必须指定epochs_to_record**，batch_indices可选。
- `epochs_to_record`：要记录输出的epoch编号列表（整数列表）。**必须非空**。
- `output_dir`：输出保存目录（字符串）。
- `batch_indices`：可选，要记录的batch索引列表（整数列表），如果batch是顺序采样的。**当有'single'类型层时必须指定**。

### 配置验证

脚本会自动验证配置：

- `epochs_to_record`不能为空。
- 如果任何层为`'single'`，则`batch_indices`必须指定且非空。
- 如果验证失败，会抛出`ValueError`并提示具体错误。

### 在训练脚本中的配置

在`train.py`中，通过config添加自定义参数（推荐）：

```python
# 在config.json中添加
{
  "monitor_layers": {
    "transformer.encoder.layers.0.self_attn": {
      "alias": "attention_weights",
      "record_type": "single"
    },
    "transformer.encoder.layers.0.cross_attn": {
      "alias": "cross_attention",
      "record_type": "epoch_full"
    }
  },
  "monitor_epochs": [1, 5, 10, 20]
}
```

或直接在代码中硬编码：

```python
monitor_layers = {
    'transformer.layers.0.attention': {'alias': 'attn_out', 'record_type': 'single'},
    'transformer.layers.1.ffn': {'alias': 'ffn_stats', 'record_type': 'epoch_full'}
}
monitor_epochs = [1, 5, 10]
```

## 基本概念和工作原理

### Monitor的工作机制

Monitor通过PyTorch的forward hooks机制工作：

1. **注册阶段**：在模型指定层上注册forward hooks
2. **捕获阶段**：每次模型forward pass时，hooks自动触发，捕获中间输出
3. **存储阶段**：根据配置将数据存储在内存中（single_data或epoch_data）
4. **处理阶段**：在适当时候获取数据进行自定义处理
5. **保存阶段**：将处理后的数据保存到磁盘
6. **清理阶段**：释放内存和移除hooks

### 数据类型说明

- **single类型**：记录指定batch的输出，用于捕获特定时刻的状态
- **epoch_full类型**：累积整个epoch的所有batch输出，用于统计分析

### 关键方法说明

- `set_epoch(epoch)`: 必须调用，设置当前训练epoch
- `set_batch_idx(batch_idx)`: 必须调用，设置当前batch索引
- `get_single_data()` / `get_epoch_data()`: 获取数据副本进行自定义处理
- `save_single_outputs()` / `save_epoch_outputs()`: 保存原始数据
- `clear_current_epoch_data()`: 清理内存
- `remove_hooks()`: 训练结束时清理hooks

## 使用场景和方法

如果您的训练函数被封装，无法直接在训练循环中调用`set_epoch()`和`set_batch_idx()`，可以使用以下方法：

### 方法1：回调函数（推荐）

修改训练函数接受回调参数：

```python
def encapsulated_train(config, device, **kwargs):
    # 获取回调函数
    epoch_callback = kwargs.get('epoch_callback')
    batch_callback = kwargs.get('batch_callback')
    
    for epoch in range(1, config.train.num_epochs + 1):
        if epoch_callback:
            epoch_callback(epoch)  # 设置当前epoch
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_callback:
                batch_callback(batch_idx)  # 设置当前batch
            
            # 训练逻辑...
        
        # epoch结束处理
        if hasattr(config, 'monitor') and config.monitor:
            single_data = config.monitor.get_single_data()
            epoch_data = config.monitor.get_epoch_data()
            # 自定义处理...
            config.monitor.save_single_outputs()
            config.monitor.save_epoch_outputs()
            config.monitor.clear_current_epoch_data()
    
    # 训练结束
    if hasattr(config, 'monitor') and config.monitor:
        config.monitor.remove_hooks()
```

在训练脚本中使用：

```python
monitor = Monitor(...)
epoch_callback = monitor.create_epoch_callback()
batch_callback = monitor.create_batch_callback()

# 调用封装的训练函数
encapsulated_train(config, device, 
                  epoch_callback=epoch_callback, 
                  batch_callback=batch_callback)
```

### 方法2：包装器模式

使用`MonitoredTrainer`包装器：

```python
from agent.scripts.training_wrapper import MonitoredTrainer

monitor = Monitor(...)
trainer = MonitoredTrainer(monitor)
monitored_train = trainer.wrap_training_function(encapsulated_train)
monitored_train(config, device)
```

### 方法3：直接在调用点控制

如果您能控制训练循环的调用点：

```python
monitor = Monitor(...)

for epoch in range(1, num_epochs + 1):
    monitor.set_epoch(epoch)
    
    for batch_idx in range(num_batches):
        monitor.set_batch_idx(batch_idx)
        # 调用封装的单batch训练函数
        encapsulated_train_one_batch(batch_idx)
    
    # epoch结束处理
    single_data = monitor.get_single_data()
    epoch_data = monitor.get_epoch_data()
    # 自定义处理...
    monitor.save_single_outputs()
    monitor.save_epoch_outputs()
    monitor.clear_current_epoch_data()

monitor.remove_hooks()  # 训练结束清理
```

---

如果训练过程没有封装成函数，可以直接在训练循环中调用Monitor方法：

### 未封装训练过程的处理

```python
monitor = Monitor(...)

try:
    for epoch in range(1, num_epochs + 1):
        monitor.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(dataloader):
            monitor.set_batch_idx(batch_idx)
            
            # 正常的训练代码
            outputs = model(batch)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # epoch结束：处理Monitor数据
        single_data = monitor.get_single_data()
        epoch_data = monitor.get_epoch_data()
        
        # 自定义处理逻辑
        for alias, layer_info in single_data.items():
            outputs = list(layer_info['data'].values())
            # 计算平均值或其他统计
            avg_output = torch.mean(torch.stack(outputs), dim=0)
            torch.save(avg_output, f'custom_avg_{alias}_epoch_{epoch}.pt')
        
        for alias, layer_info in epoch_data.items():
            outputs = list(layer_info['data'].values())
            # 对整个epoch的数据进行处理
            all_outputs = torch.cat(outputs, dim=0)
            epoch_stats = compute_statistics(all_outputs)
            torch.save(epoch_stats, f'epoch_stats_{alias}_epoch_{epoch}.pt')
        
        # 保存原始数据
        monitor.save_single_outputs()
        monitor.save_epoch_outputs()
        
        # 清理内存
        monitor.clear_current_epoch_data()

finally:
    # 确保清理hooks
    monitor.remove_hooks()
```

## 高级用法和示例

### 混合记录类型

Monitor支持同时监控不同类型的层：

```python
layers_to_monitor = {
    'transformer.encoder.layers.0.self_attn': {
        'alias': 'attention_weights',
        'record_type': 'single'  # 记录特定batch
    },
    'transformer.encoder.layers.0.ffn': {
        'alias': 'ffn_epoch_stats',
        'record_type': 'epoch_full'  # 记录整个epoch
    }
}
```

### Tuple输出处理

Monitor自动处理返回tuple输出的层，如PyTorch的`nn.MultiheadAttention`（返回`(attn_output, attn_output_weights)`）：

```python
# 对于tuple输出，output_shape为列表，output为tuple
single_data = monitor.get_single_data()
for alias, layer_info in single_data.items():
    outputs = list(layer_info['data'].values())
    for output in outputs:
        if isinstance(output, tuple):
            attn_output, attn_weights = output
            # 处理注意力输出和权重
            print(f"Attention output shape: {attn_output.shape}")
            print(f"Attention weights shape: {attn_weights.shape}")
```

### 数据处理示例

#### 示例1：注意力权重分析

```python
# 获取single数据
single_data = monitor.get_single_data()
for alias, layer_info in single_data.items():
    if 'attention' in alias:
        # 获取所有输出
        outputs = list(layer_info['data'].values())  # list of outputs
        # 计算注意力权重的平均值
        attention_weights = outputs  # 已经是list
        avg_attention = torch.mean(torch.stack(attention_weights), dim=0)
        
        # 可视化注意力模式
        plot_attention_patterns(avg_attention)
        torch.save(avg_attention, f'attention_analysis_epoch_{epoch}.pt')
```

#### 示例2：FFN输出统计

```python
# 获取epoch数据
epoch_data = monitor.get_epoch_data()
for alias, layer_info in epoch_data.items():
    if 'ffn' in alias:
        # 获取所有输出
        outputs = list(layer_info['data'].values())
        # 拼接所有batch的输出
        all_outputs = torch.cat(outputs, dim=0)
        
        # 计算统计信息
        stats = {
            'mean': torch.mean(all_outputs, dim=0),
            'std': torch.std(all_outputs, dim=0),
            'min': torch.min(all_outputs, dim=0)[0],
            'max': torch.max(all_outputs, dim=0)[0]
        }
        
        torch.save(stats, f'ffn_stats_epoch_{epoch}.pt')
```

#### 示例3：自定义处理函数

```python
def process_attention_outputs(outputs):
    """自定义注意力输出处理"""
    weights = torch.stack(outputs)
    
    # 计算头部重要性
    head_importance = torch.mean(weights, dim=[0, 2])  # [num_heads]
    
    # 计算层间注意力模式
    layer_patterns = torch.mean(weights, dim=1)  # [batch_size, seq_len, seq_len]
    
    return {
        'head_importance': head_importance,
        'layer_patterns': layer_patterns
    }

# 在训练循环中使用
single_data = monitor.get_single_data()
for alias, layer_info in single_data.items():
    if 'attention' in alias:
        outputs = list(layer_info['data'].values())
        analysis = process_attention_outputs(outputs)
        torch.save(analysis, f'attention_analysis_{alias}_epoch_{epoch}.pt')
```

### 内存管理最佳实践

```python
# 定期清理内存
monitor.clear_current_epoch_data()

# 只在需要时保存数据
if epoch % save_interval == 0:
    monitor.save_single_outputs()
    monitor.save_epoch_outputs()

# 使用try-finally确保清理
try:
    # 训练代码...
finally:
    monitor.remove_hooks()
```

### 性能优化建议

1. **选择性监控**：只监控需要的层和epoch
2. **及时清理**：每个epoch结束及时清理数据
3. **异步保存**：对于大数据考虑异步保存
4. **内存监控**：监控内存使用情况，避免OOM

### 常见问题

1. **层名不正确**：
   - 使用`print(dict(model.named_modules()).keys())`检查可用层名。
   - 错误：`ValueError: The following layers specified in 'layers_to_monitor' do not exist in the model: [...]` 并列出所有可用层。

2. **别名重复**：
   - 别名必须在所有层中唯一。
   - 错误：`ValueError: Duplicate alias 'xxx' in layers_to_monitor.`

3. **输出目录不存在**：
   - Monitor会自动创建目录，确保有写入权限。

4. **内存溢出**：
   - 对于大型输出，考虑只保存必要的统计信息而不是完整tensor。
   - 使用`monitor.clear_current_epoch_data()`定期清理内存。

5. **batch索引不准确**：
   - 确保batch是顺序采样的；如果shuffle，索引可能不对应。

6. **钩子冲突**：
   - 避免在同一层注册多个钩子；Monitor会管理钩子。

7. **Tuple输出处理**：
   - Monitor自动处理tuple输出，无需额外配置。检查`output_shape`是否为列表来判断是否为tuple。

### 性能影响

- 钩子在forward pass中添加少量开销，主要在条件检查和保存时。
- 对于频繁记录，考虑异步保存或采样。

### 调试

- 添加日志：修改`_hook_fn`中的`print`语句查看触发情况。
- 检查输出：加载`.pt`文件验证数据。

## 扩展功能

### 添加自定义过滤

修改`_hook_fn`添加更多条件：

```python
def _hook_fn(self, layer_name):
    def hook(module, input, output):
        if self.current_epoch in self.epochs_to_record and self._custom_condition(output):
            # 保存...
    return hook
```

### 自定义数据处理

使用`get_epoch_buffers()`和`get_single_outputs()`获取数据，进行任意处理：

```python
buffers = monitor.get_epoch_buffers()
for layer, outputs in buffers.items():
    # 自定义拼接、统计、可视化等
    processed = my_custom_function(outputs)
    monitor.save_custom_output(processed, f'custom_{layer}.pt')
monitor.clear_epoch_data()
```

### 支持更多输出类型

扩展保存逻辑支持numpy数组或其他格式。

## API参考

### Monitor类方法

#### 初始化

- `__init__(model, layers_to_monitor, epochs_to_record, output_dir, batch_indices=None)`

#### 配置和设置

- `set_epoch(epoch)`: 设置当前epoch，清空之前的数据。
- `set_batch_idx(batch_idx)`: 设置当前batch索引。

#### 数据访问

- `get_single_data(layers=None)`: 获取single类型数据的副本，按别名索引。layers为None时返回所有层。返回格式：`{alias: {'layer': str, 'alias': str, 'output_shape': list, 'data': {(epoch, batch_idx): output, ...}}}`
- `get_epoch_data(layers=None)`: 获取epoch_full类型数据的副本，按别名索引。layers为None时返回所有层。返回格式同上。

#### 数据保存

- `save_single_outputs(layers=None)`: 保存single数据到文件。layers为None时保存所有。
- `save_epoch_outputs(layers=None)`: 保存epoch数据到文件。layers为None时保存所有。

#### 回调函数创建

- `create_epoch_callback()`: 创建epoch设置回调函数。
- `create_batch_callback()`: 创建batch索引设置回调函数。

#### 资源管理

- `clear_current_epoch_data()`: 清空当前epoch的数据，释放内存。
- `remove_hooks()`: 移除所有注册的hooks。

### 数据格式

Monitor内部使用优化后的数据结构存储输出：

- `single_data` / `epoch_data`：`{alias: {'layer': str, 'alias': str, 'output_shape': list, 'data': {(epoch, batch_idx): output, ...}}}`
  - `alias`：层的别名（键）
  - `layer`：原始层名
  - `output_shape`：输出形状（对于tuple输出，为列表的形状列表）
  - `data`：以(epoch, batch_idx)为键的输出数据字典

`get_single_data()` 和 `get_epoch_data()` 返回此结构的深拷贝副本。

保存到磁盘时，每个输出项包含：

- `epoch`: epoch编号
- `batch_idx`: batch索引
- `layer`: 层名
- `alias`: 别名（如果指定）
- `output_shape`: 输出tensor形状
- `output`: tensor数据（已detach到CPU，支持tuple输出）

## 结论

Monitor脚本提供了一个灵活、高效的解决方案，用于在PyTorch模型训练过程中捕获和分析中间输出。通过无侵入式的设计，它可以在不修改模型和训练核心逻辑的情况下，实现精确的监控控制。

### 主要优势

1. **无侵入性**：不修改模型定义或训练逻辑
2. **灵活性**：支持多种记录类型和自定义处理
3. **可扩展性**：易于集成到现有训练流程
4. **内存友好**：自动内存管理和清理
5. **易用性**：清晰的API和完善的文档

### 适用场景

- 模型调试和分析
- 中间表示学习
- 训练过程监控
- 研究和实验记录

根据您的训练代码封装情况选择合适的使用方法。如有问题或需要进一步定制，请参考API文档或代码注释。
