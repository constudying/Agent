```markdown
# PlotTool 使用说明文档

## 概述

`PlotTool` 是一个用于分析训练循环过程中产生数据的绘图类。该类支持两种绘制模式：在线模式（实时更新）和离线模式（一次性绘制），主要用于绘制Transformer的注意力热力图和训练指标的折线图。支持离屏渲染，避免在无GUI环境中意外弹出图像窗口。

## 功能特性

- **双模式支持**：在线实时绘制和离线批量绘制
- **多种图表类型**：折线图和注意力热力图
- **子图组合**：支持将多个图像组合成一个大图
- **在线子图面板**：支持多个子图同时在线更新
- **离屏渲染**：默认启用离屏渲染，避免图像窗口意外弹出
- **子图单独保存**：支持从子图网格中提取和保存单个子图
- **灵活的数据输入**：自动适应数据维度，支持自定义标签
- **图像管理**：支持显示、保存和关闭图像

## 安装依赖

确保安装了以下Python包：

```bash
pip install matplotlib numpy
```

## 基本使用

### 1. 初始化

```python
from agent.scripts.plot_tool import PlotTool

# 在线模式（实时更新，默认离屏渲染）
plotter_online = PlotTool(mode='online', figsize=(12, 8), offscreen_rendering=True)

# 离线模式（批量绘制）
plotter_offline = PlotTool(mode='offline', figsize=(12, 8))
```

### 2. 离屏渲染控制

```python
# 默认启用离屏渲染（推荐，用于服务器环境）
plotter = PlotTool(mode='online', offscreen_rendering=True)

# 临时启用窗口显示（用于本地开发）
plotter.set_offscreen_rendering(False)

# 切换回离屏渲染
plotter.set_offscreen_rendering(True)
```

### 3. 折线图绘制

#### 在线模式

在训练循环中使用，实时更新图表：

```python
# 在训练循环中
for epoch in range(num_epochs):
    # 训练代码...
    loss = calculate_loss()
    accuracy = calculate_accuracy()
    reward = calculate_reward()

    # 绘制训练指标
    fig, ax = plotter_online.plot_line_online(
        epoch=epoch,
        y_data=[loss, accuracy, reward],
        labels=['Loss', 'Accuracy', 'Reward'],
        title='Training Metrics',
        xlabel='Epoch',
        ylabel='Value'
    )
```

#### 离线模式

训练结束后一次性绘制所有数据：

```python
# 收集所有训练数据
epochs = list(range(num_epochs))
losses = [loss_history[i] for i in range(num_epochs)]
accuracies = [acc_history[i] for i in range(num_epochs)]
rewards = [reward_history[i] for i in range(num_epochs)]

# 绘制完整训练曲线
fig, ax = plotter_offline.plot_line_offline(
    epochs=epochs,
    y_data_list=[losses, accuracies, rewards],
    labels=['Loss', 'Accuracy', 'Reward'],
    title='Complete Training History',
    xlabel='Epoch',
    ylabel='Value'
)
```

### 4. 注意力热力图绘制

```python
import numpy as np

# 生成或获取注意力权重矩阵
attention_weights = model.get_attention_weights()  # shape: (seq_len, seq_len)

# 绘制热力图
fig, ax = plotter_offline.plot_heatmap(
    data=attention_weights,
    title='Transformer Attention Heatmap',
    xlabel='Key Positions',
    ylabel='Query Positions',
    cmap='viridis'
)
```

### 5. 在线子图面板

创建多个子图同时在线更新的面板：

```python
# 定义子图配置
subplot_configs = [
    {
        'plot_type': 'line',
        'title': 'Training Loss',
        'ylabel': 'Loss',
        'labels': ['Train Loss', 'Val Loss']
    },
    {
        'plot_type': 'line',
        'title': 'Accuracy',
        'ylabel': 'Accuracy (%)',
        'labels': ['Train Acc', 'Val Acc']
    },
    {
        'plot_type': 'heatmap',
        'title': 'Attention Weights',
        'xlabel': 'Key Position',
        'ylabel': 'Query Position',
        'cmap': 'plasma'
    }
]

# 创建在线子图面板
fig, axes = plotter_online.create_online_subplot_grid(
    subplot_configs=subplot_configs,
    grid_shape=(2, 2),  # 2x2网格
    figsize=(15, 10)
)

# 在训练循环中同时更新所有子图
for epoch in range(num_epochs):
    # 生成数据
    train_loss = calculate_train_loss()
    val_loss = calculate_val_loss()
    train_acc = calculate_train_acc()
    val_acc = calculate_val_acc()
    attention_weights = model.get_attention_weights()

    # 同时更新所有子图
    plotter_online.update_online_subplot_grid([
        [train_loss, val_loss],  # 第一个子图：两条loss曲线
        [train_acc, val_acc],    # 第二个子图：两条accuracy曲线
        attention_weights        # 第三个子图：注意力热力图
    ], epoch=epoch)
```

### 6. 子图组合

将多个图像组合成一个大图：

```python
# 创建多个图表
fig1, ax1 = plotter_offline.plot_line_offline(epochs, [losses], labels=['Loss'])
fig2, ax2 = plotter_offline.plot_heatmap(attention_weights, title='Attention')

# 组合成2x1的子图布局
plots = [(fig1, ax1), (fig2, ax2)]
combined_fig, combined_axes = plotter_offline.create_subplot_grid(
    plots=plots,
    grid_shape=(2, 1),
    titles=['Training Loss', 'Attention Heatmap']
)
```

### 7. 图像显示和保存

```python
# 显示图像（离屏渲染时会跳过显示）
plotter_offline.show_figure(combined_fig)

# 保存图像
plotter_offline.save_figure('training_analysis.png', combined_fig, dpi=300)

# 保存到指定目录
plotter_offline.save_figure('results/plots/training_analysis.png', combined_fig)
```

### 8. 子图单独保存

```python
# 从子图网格中单独保存子图
plotter.save_subplot(0, 'loss_curve.png', combined_fig, combined_axes)  # 保存第1个子图
plotter.save_subplot(1, 'accuracy_curve.png', combined_fig, combined_axes)  # 保存第2个子图

# 提取子图到新的figure（不保存）
extracted_fig, extracted_ax = plotter.extract_subplot(0, combined_fig, combined_axes)
```

### 9. 资源清理

```python
# 关闭所有图像，释放内存
plotter_online.close_all()
plotter_offline.close_all()
```

## API 参考

### PlotTool 类

#### `__init__(mode='offline', figsize=(10, 6), offscreen_rendering=True)`

初始化绘图工具。

**参数：**

- `mode`: 绘制模式，'online' 或 'offline'
- `figsize`: 默认图表尺寸 (width, height)
- `offscreen_rendering`: 是否启用离屏渲染（默认True，避免弹出窗口）

#### `set_offscreen_rendering(offscreen: bool)`

动态设置离屏渲染模式。

**参数：**

- `offscreen`: True启用离屏渲染，False启用交互式显示

#### `plot_line_online(epoch, y_data, labels=None, title=None, xlabel='Epoch', ylabel='Value')`

在线模式绘制折线图。

**参数：**

- `epoch`: 当前epoch编号
- `y_data`: y轴数据，单个值或数值列表
- `labels`: 数据系列标签列表
- `title`: 图表标题
- `xlabel`: x轴标签
- `ylabel`: y轴标签

**返回：** (figure, axes) 元组

#### `plot_line_offline(epochs, y_data_list, labels=None, title=None, xlabel='Epoch', ylabel='Value')`

离线模式绘制折线图。

**参数：**

- `epochs`: epoch编号列表
- `y_data_list`: y数据列表的列表，每个子列表对应一条线
- `labels`: 数据系列标签列表
- `title`: 图表标题
- `xlabel`: x轴标签
- `ylabel`: y轴标签

**返回：** (figure, axes) 元组

#### `plot_heatmap(data, title=None, xlabel=None, ylabel=None, cmap='viridis')`

绘制注意力热力图。

**参数：**

- `data`: 二维numpy数组
- `title`: 图表标题
- `xlabel`: x轴标签
- `ylabel`: y轴标签
- `cmap`: 颜色映射

**返回：** (figure, axes) 元组

#### `create_online_subplot_grid(subplot_configs, grid_shape=None, figsize=None)`

创建可同时在线更新的子图网格。

**参数：**

- `subplot_configs`: 子图配置列表，每个配置包含plot_type, title, labels等
- `grid_shape`: 子图网格形状 (rows, cols)，None时自动计算
- `figsize`: 整体图表尺寸，None时自动计算

**返回：** (figure, axes_array) 元组

#### `update_online_subplot_grid(subplot_data_list, epoch=None)`

同时更新在线子图网格中的所有子图。

**参数：**

- `subplot_data_list`: 每个子图的数据列表
- `epoch`: 当前epoch编号，None时使用内部计数器

**返回：** (figure, axes_array) 元组

#### `create_subplot_grid(plots, grid_shape=None, titles=None)`

将多个图表组合成子图网格。

**参数：**

- `plots`: (figure, axes) 元组列表
- `grid_shape`: 子图网格形状 (rows, cols)，None时自动计算
- `titles`: 子图标题列表

**返回：** (combined_figure, axes_array) 元组

#### `show_figure(figure=None)`

显示图表（离屏渲染时跳过显示）。

**参数：**

- `figure`: 要显示的图表，None时显示当前图表

#### `save_figure(filename, figure=None, dpi=300)`

保存图表到文件。

**参数：**

- `filename`: 保存路径
- `figure`: 要保存的图表，None时保存当前图表
- `dpi`: 分辨率

#### `save_subplot(subplot_index, filename, figure=None, axes=None, dpi=300, figsize=None)`

保存子图网格中的特定子图到单独文件。

**参数：**

- `subplot_index`: 要保存的子图索引（从0开始）
- `filename`: 保存路径
- `figure`: 包含子图的图表，None时使用当前图表
- `axes`: 子图网格的坐标轴数组，None时使用当前坐标轴
- `dpi`: 分辨率
- `figsize`: 新图表尺寸，None时使用默认尺寸

**返回：** (new_figure, new_axes) 元组

#### `extract_subplot(subplot_index, figure=None, axes=None, figsize=None)`

提取子图网格中的特定子图到新的figure。

**参数：**

- `subplot_index`: 要提取的子图索引（从0开始）
- `figure`: 包含子图的图表，None时使用当前图表
- `axes`: 子图网格的坐标轴数组，None时使用当前坐标轴
- `figsize`: 新图表尺寸，None时使用默认尺寸

**返回：** (new_figure, new_axes) 元组

#### `close_all()`

关闭所有图表并清理资源。

## 使用示例

### 离屏渲染示例

```python
from agent.scripts.plot_tool import PlotTool

# 服务器环境：启用离屏渲染，避免弹出窗口
plotter = PlotTool(mode='online', offscreen_rendering=True)

for epoch in range(100):
    loss = calculate_loss()
    plotter.plot_line_online(epoch, loss, labels=['Loss'])

# 保存结果，无需显示
plotter.save_figure('training_loss.png')
plotter.close_all()

# 本地开发：临时启用窗口显示
plotter.set_offscreen_rendering(False)
plotter.show_figure()  # 现在会弹出窗口
```

### 训练监控示例

```python
from agent.scripts.plot_tool import PlotTool
import numpy as np

# 初始化在线绘图器
plotter = PlotTool(mode='online')

# 模拟训练过程
for epoch in range(100):
    # 模拟训练指标
    loss = 1.0 * np.exp(-epoch * 0.05) + 0.1 * np.random.randn()
    acc = 1 - np.exp(-epoch * 0.03) + 0.05 * np.random.randn()

    # 实时绘制
    plotter.plot_line_online(
        epoch=epoch,
        y_data=[loss, acc],
        labels=['Training Loss', 'Validation Accuracy'],
        title='Real-time Training Monitor'
    )

# 保存最终结果
plotter.save_figure('training_progress.png')
plotter.close_all()
```

### 在线子图面板示例

```python
from agent.scripts.plot_tool import PlotTool
import numpy as np

# 初始化在线绘图器
plotter = PlotTool(mode='online')

# 定义子图配置
subplot_configs = [
    {
        'plot_type': 'line',
        'title': 'Loss Curves',
        'ylabel': 'Loss',
        'labels': ['Train', 'Validation']
    },
    {
        'plot_type': 'line',
        'title': 'Accuracy',
        'ylabel': 'Accuracy (%)',
        'labels': ['Train', 'Val']
    },
    {
        'plot_type': 'heatmap',
        'title': 'Attention Map',
        'cmap': 'viridis'
    }
]

# 创建在线子图面板
fig, axes = plotter.create_online_subplot_grid(
    subplot_configs=subplot_configs,
    grid_shape=(2, 2)
)

# 训练循环
for epoch in range(50):
    # 生成模拟数据
    train_loss = 1.0 * np.exp(-epoch * 0.08) + 0.05 * np.random.randn()
    val_loss = 1.0 * np.exp(-epoch * 0.06) + 0.08 * np.random.randn()
    train_acc = 100 * (1 - np.exp(-epoch * 0.08)) + np.random.randn()
    val_acc = 100 * (1 - np.exp(-epoch * 0.06)) + 1.5 * np.random.randn()
    attention = np.random.rand(10, 10) + np.eye(10)

    # 同时更新所有子图
    plotter.update_online_subplot_grid([
        [train_loss, val_loss],  # Loss subplot
        [train_acc, val_acc],    # Accuracy subplot
        attention                # Attention heatmap
    ], epoch=epoch)

# 保存最终面板
plotter.save_figure('training_dashboard.png', fig)
plotter.close_all()
```

### 注意力分析示例

```python
from agent.scripts.plot_tool import PlotTool
import numpy as np

# 初始化离线绘图器
plotter = PlotTool(mode='offline')

# 模拟注意力权重
seq_len = 50
attention_matrix = np.random.rand(seq_len, seq_len)
# 添加一些结构（对角线更亮）
attention_matrix += np.eye(seq_len) * 2
attention_matrix = attention_matrix / attention_matrix.max()

# 绘制注意力热力图
plotter.plot_heatmap(
    data=attention_matrix,
    title='Transformer Self-Attention Weights',
    xlabel='Key Position',
    ylabel='Query Position'
)

# 保存结果
plotter.save_figure('attention_heatmap.png')
plotter.close_all()
```

### 子图单独保存示例

```python
from agent.scripts.plot_tool import PlotTool
import numpy as np

# 创建离线绘图器
plotter = PlotTool(mode='offline')

# 创建多个图表
epochs = list(range(20))
loss_data = [1.0 * np.exp(-i * 0.08) for i in epochs]
acc_data = [100 * (1 - np.exp(-epoch * 0.06)) for i in epochs]

fig1, ax1 = plotter.plot_line_offline(epochs, [loss_data], labels=['Loss'])
fig2, ax2 = plotter.plot_line_offline(epochs, [acc_data], labels=['Accuracy'])
fig3, ax3 = plotter.plot_heatmap(np.random.rand(10, 10), title='Attention')

# 组合成子图网格
plots = [(fig1, ax1), (fig2, ax2), (fig3, ax3)]
combined_fig, combined_axes = plotter.create_subplot_grid(plots, grid_shape=(2, 2))

# 单独保存每个子图
plotter.save_subplot(0, 'individual_loss.png', combined_fig, combined_axes)
plotter.save_subplot(1, 'individual_accuracy.png', combined_fig, combined_axes)
plotter.save_subplot(2, 'individual_attention.png', combined_fig, combined_axes)

# 或者提取子图进行进一步处理
loss_fig, loss_ax = plotter.extract_subplot(0, combined_fig, combined_axes)
# 可以对提取的子图进行额外修改...

plotter.close_all()
```

## 注意事项

1. **离屏渲染**：默认启用，避免在服务器环境中弹出窗口。如需查看图像，使用 `set_offscreen_rendering(False)` 并调用 `show_figure()`。

2. **在线模式性能**：在线模式会在每次更新时重新绘制图表，适用于epoch间隔较大的情况。

3. **内存管理**：长时间运行时注意调用 `close_all()` 释放内存。

4. **数据格式**：确保输入的数据格式正确，numpy数组用于热力图，列表用于折线图。

5. **标签数量**：`labels` 列表的长度应与数据系列数量一致。

6. **文件路径**：保存图像时会自动创建不存在的目录。

7. **在线子图面板**：所有子图共享同一个epoch计数器，确保数据对齐。

## 故障排除

- **导入错误**：确保matplotlib和numpy已正确安装
- **显示问题**：在线模式下确保有图形界面支持，或使用离线模式
- **保存失败**：检查文件路径权限和磁盘空间
- **子图更新问题**：确保 `update_online_subplot_grid` 的数据列表与子图数量匹配
- **离屏渲染问题**：如果需要查看图像，临时设置 `offscreen_rendering=False`

## 版本信息

- 当前版本：v1.3.0
- 作者：AI Assistant
- 创建日期：2025-12-29
- 新增功能：离屏渲染支持、子图单独保存和提取功能、动态渲染模式切换
