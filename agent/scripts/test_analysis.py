import numpy as np
from analysis_tool import Analysis

# 1) 初始化：模式+可视化策略
ana = Analysis(
    pca_modes=["sample-pca", "token-pca"],
    visualize_mode="offline",        # 改成 "online" 会每次 process 后更新曲线
    metrics_to_plot=["effective_dimension", "k95", "spectral_sharpness", "topk_cumvar"],
    top_k=2,
    spectrum_interval=3,             # 每 3 个 epoch 画一次完整谱
    component_interval=5,            # 每 5 个 epoch 画一次主成分方向
)

# 2) 训练前配置数据别名 → 源变量名/输出目的地
ana.configure_targets({
    "tokens": {"source": "token_states", "destination": "train-probe"},
    "valid_tokens": {"source": "valid_token_states", "destination": "valid-probe"},
    "epoch_repr": {"source": "epoch_repr", "destination": "epoch-probe"},
})

# 3) 在训练循环里调用 process
for epoch in range(1, 11):
    # 假设拿到三份数据
    train_token_states = np.random.randn(32, 16, 64)   # (batch, tokens, dim)
    valid_token_states = np.random.randn(16, 12, 64)
    epoch_repr = np.random.randn(64, 128)              # (samples, dim)

    train_token_states = train_token_states.reshape(-1, train_token_states.shape[-1])
    valid_token_states = valid_token_states.reshape(-1, valid_token_states.shape[-1])
    ana.process("tokens", train_token_states, epoch)
    ana.process("valid_tokens", valid_token_states, epoch)
    ana.process("epoch_repr", epoch_repr, epoch)

# 4) 如果 visualize_mode="offline"，训练结束后统一绘图
ana.plot_all(show=True)  # save_dir 可选；show=True 弹窗