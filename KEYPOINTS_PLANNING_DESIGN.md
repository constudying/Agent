# 任务空间规划 - 关键路径点方案

## 修改说明

### 原始设计（30维完整轨迹）

```
输入: 当前观测 + 历史轨迹
输出: 未来10帧的完整轨迹 (10 × 3 = 30维)
问题: GMM难以拟合30维分布，训练困难
```

### 新设计（9维关键路径点）

```
输入: 当前观测 + 历史轨迹
输出: 未来轨迹的3个关键点 (3 × 3 = 9维)
  - 点1: 25%位置（第3帧，索引2）
  - 点2: 50%位置（第6帧，索引5）
  - 点3: 100%位置（第10帧，索引9）
```

## 为什么这样改？

### ✅ 保留了你的设计理念

1. **仍然是任务空间规划**
   - High-level预测"往哪里移动"
   - Low-level执行"如何移动"

2. **仍然能表达路径信息**
   - 3个点足以表达路径的方向和弯曲
   - 例如：绕过左边 vs 绕过右边 vs 直线到达

3. **GMM仍然建模多模态**
   - 不同的模态 = 不同的路径选择
   - 5个高斯模态在9维空间中覆盖率更合理

4. **仍然能传递"操作余量"信息**
   - 如果关键点接近障碍物，Low-level能感知到
   - 可以后续扩展：在关键点附近添加"危险度"信息

### ✅ 解决了训练困难问题

#### 维度对比

| 方案 | 输出维度 | GMM参数量 | 数据密度 | 训练难度 |
|------|----------|-----------|----------|----------|
| 完整轨迹 | 30维 | 305 | 极稀疏 | ⭐⭐⭐⭐⭐ |
| 关键点 | 9维 | 95 | 较密集 | ⭐⭐ |
| 仅终点 | 3维 | 35 | 密集 | ⭐ |

#### 为什么9维而不是3维？

**9维（3个关键点）的优势：**

- ✓ 能表达路径的弯曲（如绕过障碍物）
- ✓ 给Low-level更多的引导信息
- ✓ 仍然可训练（9维GMM是可行的）

**3维（仅终点）的限制：**

- ✗ 只知道目标在哪，不知道如何到达
- ✗ Low-level需要自己规划完整路径
- ✗ 无法表达"绕左边还是绕右边"

## 代码修改清单

### 1. agent/algo/agent.py (已修改)

```python
# 在 _forward_training 方法中:
# - 将30维 future_traj reshape为 [batch, 10, 3]
# - 选择索引 [2, 5, 9] 的3个关键点
# - 展平为 [batch, 9] 作为训练目标
# - 添加了诊断输出（每100次打印）
```

### 2. agent/configs/stage2_actionpre.json (已修改)

```json
{
  "algo": {
    "highlevel": {
      "ac_dim": 9  // 从30改为9
    }
  }
}
```

### 3. 需要重新训练

```bash
# 删除旧的checkpoint（因为ac_dim变了）
rm -rf trained_models_highlevel/test/*

# 重新训练
python agent/scripts/train.py --config agent/configs/stage2_actionpre.json --dataset <你的数据路径>
```

## 预期效果

### 训练阶段

1. **Loss应该能正常下降**
   - Log prob 从 -50 逐渐提升到 -10 左右
   - 9维比30维容易拟合得多

2. **诊断输出示例**

   ```
   [Iter 101] Training Diagnostics:
     Log prob: -15.23 (higher is better)
     Entropy: 0.45
     Keypoint errors: [0.023, 0.031, 0.045]
   ```

   - 关键点误差应该逐渐减小
   - 从初始的0.5+ 降到 0.05以下

3. **GMM模态分布**
   - 5个模态的权重应该比较均匀（如 [0.18, 0.22, 0.20, 0.19, 0.21]）
   - 如果某个模态>0.5，说明发生了模态崩溃

### 推理阶段（Low-level使用）

High-level输出3个关键点后，Low-level可以：

```python
# 在 Lowlevel_RNN_agent 中
with torch.no_grad():
    keypoints, latent = self.human_nets.policy._get_latent_plan(obs, goal)
    # keypoints: [batch, 9] -> reshape -> [batch, 3, 3]
    
    # 方案A: 直接使用最近的关键点作为短期目标
    current_pos = obs['robot0_eef_pos']
    distances = torch.norm(keypoints.view(-1, 3, 3) - current_pos.unsqueeze(1), dim=-1)
    nearest_idx = distances.argmin(dim=1)
    subgoal = keypoints.view(-1, 3, 3)[torch.arange(len(keypoints)), nearest_idx]
    
    # 方案B: 在关键点之间插值，生成平滑轨迹
    trajectory = interpolate_keypoints(keypoints.view(-1, 3, 3), num_steps=10)
    
    obs['subgoal'] = subgoal  # 或 trajectory
    obs['latent_plan'] = latent
```

## 进一步改进方向

### 如果训练仍然困难

1. **减少到2个关键点（6维）**

   ```python
   key_indices = [4, 9]  # 中点和终点
   ```

2. **只预测终点（3维）**

   ```python
   keypoints = future_traj_seq[:, -1, :]  # 只要最后一个点
   ac_dim = 3
   ```

### 如果训练成功，想要更多信息

1. **增加关键点数量（12维或15维）**

   ```python
   key_indices = [2, 4, 6, 9]  # 4个点，12维
   ```

2. **为每个关键点添加置信度/危险度**

   ```python
   # 输出: [batch, 12]
   # - keypoints: [9] (3个点 × 3维)
   # - confidence: [3] (每个点的置信度，0-1)
   ```

3. **添加速度信息**

   ```python
   # 输出: [batch, 15]
   # - keypoints: [9]
   # - velocities: [6] (3个点中前2个的速度向量)
   ```

## 监控指标

训练时应该关注：

1. **Log Likelihood (log_probs)**
   - 目标: > -20
   - 如果 < -50: 模型没学到东西，检查数据
   - 如果 > -5: 可能过拟合

2. **Keypoint Error**
   - 目标: < 0.05（5cm误差）
   - 按关键点分解，看是否均匀

3. **Entropy**
   - 目标: 0.3 - 0.8
   - 如果 < 0.1: 模态崩溃，增加entropy权重
   - 如果 > 1.5: 模型不确定性太大

4. **Mode Weights**
   - 目标: 各模态权重在0.1-0.3之间
   - 如果某个模态 > 0.6: 模态崩溃

## 可视化脚本

```python
# 训练后可以运行这个脚本来可视化预测结果
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载模型和数据
# ...

# 获取一个batch
with torch.no_grad():
    dists, _ = model._forward_training(batch)
    
# 可视化第一个样本的所有模态
fig = plt.figure(figsize=(20, 4))
for mode in range(5):
    ax = fig.add_subplot(1, 5, mode+1, projection='3d')
    
    # 真实的完整轨迹
    true_traj = batch["obs"]["robot0_eef_pos_future_traj"][0].view(10, 3).cpu()
    ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], 
            'g-', alpha=0.3, label='True trajectory')
    
    # 真实的关键点
    true_keypoints = true_traj[[2, 5, 9]]
    ax.scatter(true_keypoints[:, 0], true_keypoints[:, 1], true_keypoints[:, 2],
              c='green', s=100, marker='o', label='True keypoints')
    
    # 预测的关键点（该模态）
    pred_keypoints = dists.mean[0, mode].view(3, 3).cpu()
    ax.scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], pred_keypoints[:, 2],
              c='red', s=100, marker='x', label='Pred keypoints')
    
    # 连接预测的关键点
    ax.plot(pred_keypoints[:, 0], pred_keypoints[:, 1], pred_keypoints[:, 2],
           'r--', alpha=0.5)
    
    # 当前位置
    current = batch["obs"]["robot0_eef_pos"][0].cpu()
    ax.scatter(*current, c='blue', s=200, marker='*', label='Current')
    
    prob = dists.mixture_distribution.probs[0, mode].item()
    ax.set_title(f'Mode {mode} (p={prob:.2%})')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.savefig('keypoints_prediction.png', dpi=150)
plt.show()
```

## 总结

这个修改：
✅ **保留了你的分层设计理念**（任务空间规划）
✅ **大幅降低了训练难度**（30维→9维）
✅ **仍然能表达路径信息**（3个关键点足够）
✅ **GMM能够有效建模**（9维分布可行）
✅ **为Low-level提供清晰引导**（关键点作为subgoal）

现在可以重新训练，应该能看到loss正常下降了！
