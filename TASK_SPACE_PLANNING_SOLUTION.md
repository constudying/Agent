# 任务空间规划的正确实现方案

## 你的设计意图（我现在理解了）

你想要构建一个分层系统：

```
High-level Policy (Highlevel_GMM_pretrain):
  输入: 当前观测 + 任务token + 历史轨迹
  输出: 未来轨迹规划 (期望的运动轨迹)
  作用: 告诉低层"应该往哪里移动"，并且用GMM建模多种可能的路径
  
Low-level Policy (Lowlevel_RNN_agent):
  输入: 当前状态 + 高层规划的轨迹
  输出: 具体的关节动作
  作用: 跟踪高层规划的轨迹
```

这个设计**理论上是正确的**，类似于：

- MPC (Model Predictive Control)
- Hierarchical Reinforcement Learning
- Goal-conditioned Policies

## 当前实现的问题

### 问题1: 预测 vs 规划

当前代码：

```python
log_probs = dists.log_prob(batch["obs"]["robot0_eef_pos_future_traj"])
```

这里`future_traj`是演示数据中**实际执行的轨迹**，但：

- 它可能不是最优轨迹（专家可能绕了弯路）
- 它只是一种可能的轨迹（可能有更好的路径）
- 它依赖于执行时的动力学（控制频率、物理参数）

### 问题2: 30维的GMM难以训练

即使任务定义合理，30维空间中：

- 5个高斯模态覆盖率极低
- 时间依赖被忽略（GMM假设独立）
- 需要大量数据才能拟合

### 问题3: 缺少"操作余量"的显式建模

你提到想让模型学习"还有多少空间余量可供操作"，但当前：

- 模型只看到轨迹，没有看到环境约束
- 没有显式建模"危险区域"或"可行空间"

## 解决方案

### 方案A: 保持轨迹预测，但降低维度（推荐）

只预测关键路径点，而不是完整轨迹：

```python
# 修改 agent/algo/agent.py line 669
def _forward_training(self, batch):
    dists, entropy_loss = self.nets["policy"].forward_train(
        obs_dict=batch["obs"],
        goal_dict=batch["goal_obs"],
        return_attention_weights=True
    )
    
    assert len(dists.batch_shape) == 1
    
    # 方案A1: 只预测终点（最简单）
    future_traj = batch["obs"]["robot0_eef_pos_future_traj"]
    target_endpoint = future_traj[:, -3:]  # 最后一个点的xyz
    log_probs = dists.log_prob(target_endpoint)
    
    # 或者 方案A2: 预测几个关键点
    # key_indices = [0, 4, 9]  # 第1, 5, 10个点
    # keypoints = future_traj.view(-1, 10, 3)[:, key_indices].reshape(-1, 9)
    # log_probs = dists.log_prob(keypoints)
    
    # ... rest of code
```

修改配置：

```json
{
  "algo": {
    "highlevel": {
      "ac_dim": 3  // 改为3（只预测终点）或9（3个关键点）
    }
  }
}
```

**优点**：

- 维度大幅降低（3维或9维）
- GMM能够建模"多条路径到达目标"的多模态性
- 训练更容易收敛

**缺点**：

- 丢失了完整轨迹的时序信息
- Low-level需要自己插值/规划中间路径

### 方案B: 使用Diffusion Policy（最强但复杂）

如果你确实想预测完整的30维轨迹，考虑用Diffusion而不是GMM：

```python
# 需要大幅修改网络架构
# Diffusion能够建模复杂的多模态分布，且能处理高维输出
```

参考论文：

- Diffusion Policy (Chi et al., 2023)
- 专门为机器人轨迹生成设计

**优点**：

- 能处理30维高维输出
- 多模态建模能力强
- 考虑时间依赖

**缺点**：

- 实现复杂，需要重写训练流程
- 推理速度慢（需要多步去噪）

### 方案C: 显式建模"操作余量"（创新）

既然你关心"空间余量"，不如直接建模它：

```python
# 输出1: 期望的终点位置 (3维)
target_pos = ...

# 输出2: 危险区域的位置和范围 (多个3D球体)
# 例如：预测K个危险点及其半径
dangerous_zones = ... # [K, 4] (xyz + radius)

# 输出3: 可行空间的编码
feasibility_map = ... # [H, W, D] 体素网格
```

这样Low-level可以：

- 避开危险区域
- 在可行空间内规划路径
- 动态调整轨迹

**优点**：

- 直接建模你真正关心的东西
- 更容易可解释
- Low-level有更大的灵活性

**缺点**：

- 需要设计新的监督信号
- 可能需要额外的标注

### 方案D: 当前方案的渐进式改进（最实用）

如果不想大改，逐步优化当前方案：

#### 步骤1: 添加诊断，确认问题

```python
# 在 _forward_training 中添加
if not hasattr(self, '_step'):
    self._step = 0
self._step += 1

if self._step % 100 == 0:
    print(f"\n[Step {self._step}] Diagnostics:")
    print(f"  Log prob: {log_probs.mean().item():.2f}")
    print(f"  Entropy: {entropy_loss.item():.4f}")
    
    # 检查预测的轨迹是否合理
    pred_traj = dists.mean[0, 0].view(10, 3)  # 第一个样本，第一个模态
    true_traj = batch["obs"]["robot0_eef_pos_future_traj"][0].view(10, 3)
    error = (pred_traj - true_traj).norm(dim=1).mean()
    print(f"  Mean trajectory error: {error.item():.4f}")
```

#### 步骤2: 调整loss权重

```python
# 在 _train_step 中
# 如果entropy太小，说明模态崩溃了
if entropy_loss.item() < 0.1:
    adaptive_weight *= 2  # 增加entropy权重
elif entropy_loss.item() > 1.0:
    adaptive_weight *= 0.5  # 减少entropy权重
```

#### 步骤3: 增加轨迹级别的约束

```python
# 在 _compute_losses 中添加
def _compute_losses(self, predictions, batch):
    action_loss = -predictions["log_probs"].mean()
    
    # 添加物理约束损失
    pred_means = dists.mean  # [batch, num_modes, 30]
    pred_traj = pred_means.view(-1, 5, 10, 3)  # [batch, modes, timesteps, xyz]
    
    # 约束1: 轨迹应该是平滑的
    traj_velocity = pred_traj[:, :, 1:] - pred_traj[:, :, :-1]
    traj_accel = traj_velocity[:, :, 1:] - traj_velocity[:, :, :-1]
    smoothness_loss = traj_accel.pow(2).mean()
    
    # 约束2: 轨迹不应该有太大的跳跃
    max_step = 0.1  # 最大单步移动距离
    step_size = traj_velocity.norm(dim=-1)
    jump_penalty = F.relu(step_size - max_step).mean()
    
    # 约束3: 轨迹应该接近当前位置
    current_pos = batch["obs"]["robot0_eef_pos"]  # [batch, 3]
    start_deviation = (pred_traj[:, :, 0] - current_pos.unsqueeze(1)).pow(2).mean()
    
    total_loss = action_loss + 0.01 * smoothness_loss + 0.1 * jump_penalty + 0.1 * start_deviation
    
    return OrderedDict(
        log_probs=-action_loss,
        entropy=predictions["entropy"],
        smoothness=smoothness_loss,
        jump_penalty=jump_penalty,
        start_deviation=start_deviation,
        action_loss=total_loss,
    )
```

#### 步骤4: 可视化预测结果

```python
# 训练时定期保存预测轨迹的可视化
if self._step % 1000 == 0:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 5))
    for mode in range(5):
        ax = fig.add_subplot(1, 5, mode+1, projection='3d')
        
        # 真实轨迹
        true_traj = batch["obs"]["robot0_eef_pos_future_traj"][0].view(10, 3).cpu()
        ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], 'g-', label='True', linewidth=2)
        
        # 预测轨迹（该模态）
        pred_traj = dists.mean[0, mode].view(10, 3).detach().cpu()
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 'r--', label=f'Pred Mode {mode}')
        
        # 当前位置
        current = batch["obs"]["robot0_eef_pos"][0].cpu()
        ax.scatter(*current, c='blue', s=100, label='Current')
        
        ax.legend()
        ax.set_title(f'Mode {mode} (p={dists.mixture_distribution.probs[0, mode].item():.2f})')
    
    plt.savefig(f'trajectory_prediction_step{self._step}.png')
    plt.close()
```

## 我的最终建议

基于你的设计意图，我建议：

### 短期方案（立即可行）

**改为只预测终点位置**：

1. 修改 `ac_dim = 3`
2. 修改目标为 `future_traj[:, -3:]`
3. 重新训练

理由：

- 保留了你的分层设计
- GMM仍然能建模"多条路径"的多模态性
- 训练会容易得多
- Low-level可以用当前位置→目标位置做局部规划

### 中期方案（更好的设计）

**预测3-5个关键路径点**：

1. `ac_dim = 9` 或 `15` (3-5个点)
2. 选择轨迹中的关键时刻（如25%, 50%, 75%, 100%）
3. 添加物理约束（平滑性、连续性）

理由：

- 给Low-level更多的引导信息
- 仍然可训练（9-15维GMM还OK）
- 能够表达"绕过障碍物"的路径

### 长期方案（如果你有时间）

**结合Diffusion Policy或VAE**：

- 用Diffusion建模完整的30维轨迹分布
- 或用VAE学习轨迹的低维embedding
- 显式建模环境约束和可行空间

## 立即可以尝试的代码修改

我可以帮你实现"方案A1: 只预测终点"，这是最快能让训练收敛的方法。需要我直接修改代码吗？

修改点：

1. `agent/algo/agent.py` line 669
2. `agent/configs/stage2_actionpre.json` 中的 `ac_dim`
3. 添加诊断和可视化代码

这样修改后，你的High-level仍然学习"任务空间规划"，只是从"预测完整轨迹"简化为"预测目标位置"，但保留了GMM的多模态建模能力。

你觉得这个方案如何？需要我实施吗？
