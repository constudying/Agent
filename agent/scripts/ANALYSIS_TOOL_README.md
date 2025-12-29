# Analysis 使用文档

该文档说明 `analysis_tool.py` 中 `Analysis` 类的配置、训练调用与可视化流程，包含多种常见应用方式。

## 1. 概念与数据流

- **alias（别名）**：训练循环中要处理的数据名称。
- **source（真实变量名）**：训练循环传入的 payload 字典中的 key。
- **destination（输出导向位置）**：
  - `train-probe`：训练集探针数据
  - `valid-probe`：验证集探针数据
  - `epoch-probe`：按 epoch 粒度的数据
- **PCA 模式（可共存）**：
  - `token-pca`：将 `(n_samples, tokens, dim)` 展平后做 PCA
  - `sample-pca`：将 tokens 维度平均后做 PCA
  - `per-sample token-pca`：对每个 sample 的 token 做 PCA，再聚合统计

## 2. 指标说明

每次 PCA 会计算：

- `eigenvalues`：特征值（方差谱）
- `explained_variance_ratio`：解释方差比
- `cumulative_variance_ratio`：累计解释方差
- `k95`：累计解释方差达到 0.95 的最小维度
- `effective_dimension`：有效维度
- `spectral_sharpness`：谱尖锐度（最大特征值 / 平均特征值）
- `topk_cumvar`：前 k 维累计解释方差
- `components`：主成分方向（前 k 个）

`per-sample token-pca` 会输出 `*_mean` 和 `*_std` 的聚合指标。

## 3. 初始化

```python
from analysis_tool import Analysis

analysis = Analysis(
    pca_modes=["sample-pca", "token-pca"],
    visualize_mode="offline",   # "online" 或 "offline"
    plot_view="combined",       # "train-valid" / "epoch" / "combined"
    top_k=2,
    spectrum_interval=5,
    component_interval=5,
)
```

## 4. 配置 alias（必须）

```python
analysis.configure_targets({
    "tokens_train": {
        "source": "token_states",
        "destination": "train-probe",
        "pca_modes": ["token-pca"],
        "label": "train",
    },
    "tokens_valid": {
        "source": "valid_token_states",
        "destination": "valid-probe",
        "pca_modes": ["token-pca"],
        "label": "valid",
    },
    "epoch_repr": {
        "source": "epoch_repr",
        "destination": "epoch-probe",
        "pca_modes": ["sample-pca"],
    },
})
```

## 5. 可视化配置（必须）

可视化支持三种视图：`train-valid`、`epoch`、`combined`。其中 `train-valid` 需要明确指定哪些 alias 属于 train / valid。

```python
analysis.configure_plots({
    "train-valid": {
        "groups": {
            "tokens": {
                "train": ["tokens_train"],
                "valid": ["tokens_valid"],
                "metrics": ["k95", "effective_dimension"],
                "modes": ["token-pca"],
                "plot_spectrum": True,
                "plot_components": False,
            }
        }
    },
    "epoch": {
        "groups": {
            "epoch_repr": {
                "aliases": ["epoch_repr"],
                "metrics": ["spectral_sharpness", "topk_cumvar"],
                "modes": ["sample-pca"],
                "plot_spectrum": False,
                "plot_components": False,
            }
        }
    },
})
```

说明：

- `metrics` 可不写，默认使用：`effective_dimension`、`k95`、`spectral_sharpness`、`topk_cumvar`
- `modes` 可不写，自动根据数据中已有的 PCA 模式推断
- `plot_spectrum` / `plot_components` 可选，用于绘制方差谱和主成分

## 6. 训练循环中调用（核心流程）

### 6.1 每个 epoch 自动处理所有 alias

```python
for epoch in range(num_epochs):
    payload = {
        "token_states": token_states_np,          # (n_samples, tokens, dim)
        "valid_token_states": valid_states_np,    # (n_samples, tokens, dim)
        "epoch_repr": epoch_repr_np,              # (n_samples, dim)
    }
    analysis.process_epoch(payload, epoch)
```

严格模式（缺失 source 时抛错）：

```python
analysis.process_epoch(payload, epoch, strict=True)
```

### 6.2 单条数据分析

```python
analysis.process("tokens_train", token_states_np, epoch)
```

## 7. 可视化模式（在线 / 离线）

### 7.1 在线更新（训练中实时显示）

```python
analysis.configure_visualization(mode="online")
```

在线模式下，`process` 会自动触发绘图。

### 7.2 离线统一绘图（训练结束）

```python
analysis.configure_visualization(mode="offline")
analysis.plot_all(save_dir="path/to/plots")
```

### 7.3 在线保存特定 epoch 的图像

```python
for epoch in range(num_epochs):
    analysis.process_epoch(payload, epoch)
    if epoch % 10 == 0:
        analysis.save_plots(epoch, save_dir="plots")
```

### 7.4 视图选择

```python
analysis.configure_visualization(plot_view="combined")
```

### 7.5 设置绘图间隔

```python
analysis.configure_visualization(
    spectrum_interval=10,
    component_interval=10,
)
```

## 8. 常见应用示例

### 8.1 只做 epoch 曲线（没有 train/valid 对照）

```python
analysis.configure_targets({
    "epoch_repr": {
        "source": "epoch_repr",
        "destination": "epoch-probe",
        "pca_modes": ["sample-pca"],
    }
})

analysis.configure_plots({
    "epoch": {
        "groups": {
            "epoch_repr": {
                "aliases": ["epoch_repr"],
                "metrics": ["k95", "spectral_sharpness"],
            }
        }
    }
})
```

### 8.2 只做 train-valid 对照

```python
analysis.configure_targets({
    "tokens_train": {
        "source": "token_states",
        "destination": "train-probe",
        "pca_modes": ["token-pca"],
        "label": "train",
    },
    "tokens_valid": {
        "source": "valid_token_states",
        "destination": "valid-probe",
        "pca_modes": ["token-pca"],
        "label": "valid",
    },
})

analysis.configure_plots({
    "train-valid": {
        "groups": {
            "tokens": {
                "train": ["tokens_train"],
                "valid": ["tokens_valid"],
                "metrics": ["effective_dimension", "k95"],
                "plot_spectrum": True,
            }
        }
    }
})
```

### 8.3 同时画 train-valid + epoch（组合视图）

```python
analysis.configure_visualization(plot_view="combined")
```

### 8.4 启用 per-sample token-pca

```python
analysis.configure_targets({
    "tokens_train": {
        "source": "token_states",
        "destination": "train-probe",
        "pca_modes": ["per-sample token-pca"],
    },
})
```

可视化时使用 `*_mean` 或 `*_std` 指标，例如：

```python
analysis.configure_plots({
    "train-valid": {
        "groups": {
            "tokens": {
                "train": ["tokens_train"],
                "valid": [],
                "metrics": ["effective_dimension_mean", "k95_mean"],
            }
        }
    }
})
```

### 8.5 同时启用 token-pca + sample-pca

```python
analysis.configure_targets({
    "tokens_train": {
        "source": "token_states",
        "destination": "train-probe",
        "pca_modes": ["token-pca", "sample-pca"],
    },
})

analysis.configure_plots({
    "train-valid": {
        "groups": {
            "tokens": {
                "train": ["tokens_train"],
                "valid": [],
                "modes": ["token-pca"],  # 指定只画 token-pca
            }
        }
    }
})
```

## 9. 指标存储结构

所有指标保存在 `analysis.records` 中：

```python
records = analysis.records
# records["train-probe"]["tokens_train"]["token-pca"] -> List[Dict]
```

每条记录示例：

```python
{
    "epoch": 5,
    "eigenvalues": ...,
    "explained_variance_ratio": ...,
    "cumulative_variance_ratio": ...,
    "k95": 12,
    "effective_dimension": 7.3,
    "spectral_sharpness": 4.1,
    "topk_cumvar": 0.62,
    "components": ...
}
```

## 10. 注意事项

- PCA 至少需要 2 个样本。
- `token-pca` / `per-sample token-pca` 需要 3D 数据 `(n_samples, tokens, dim)`。
- `sample-pca` 可接受 `(n_samples, dim)` 或 `(n_samples, tokens, dim)`（会先平均 tokens）。
