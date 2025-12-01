"""
Highlevel_GMM_pretrain 训练困难问题的完整分析和解决方案

===== 问题诊断 =====

你的配置显示:

- ac_dim = 30
- 训练目标 = batch["obs"]["robot0_eef_pos_future_traj"] (很可能是 10个未来位置点 × 3维 = 30)

这意味着你在做: 图像 -> 未来轨迹位置

===== 为什么这很困难 =====

1. **任务本身就很难**
   - 从单张图像预测未来轨迹需要预知:
     - 当前机器人的速度和加速度
     - 控制器的行为
     - 环境的动力学
   - 这些信息在静态图像中完全不存在！

2. **多模态的来源不对**
   - 行为克隆的多模态: 同一个状态下，专家可能采取多种不同的动作
   - 轨迹的多模态: 同一个动作可能导致多种轨迹(噪声、动力学不确定性)
   - 你在建模后者，但BC需要的是前者！

3. **GMM无法处理时序依赖**
   - future_traj[0:3] 是t+1时刻的位置
   - future_traj[3:6] 是t+2时刻的位置
   - 它们不是独立的！t+2依赖于t+1
   - 但GMM假设所有30个维度独立同分布

===== 正确的做法 =====

方案1: 标准行为克隆 (强烈推荐)
----------------------------

预测动作，而不是轨迹

修改文件: agent/algo/agent.py
位置: Highlevel_GMM_pretrain._forward_training 方法 (约line 658-670)

原代码:

```python
def _forward_training(self, batch):
    dists, entropy_loss = self.nets["policy"].forward_train(
        obs_dict=batch["obs"],
        goal_dict=batch["goal_obs"],
        return_attention_weights=True
    )
    
    assert len(dists.batch_shape) == 1
    
    log_probs = dists.log_prob(batch["obs"]["robot0_eef_pos_future_traj"])  # ❌ 错误
    
    target_ratio = 0.02
    adaptive_weight = (log_probs.mean().item() * target_ratio) / entropy_loss.item()
    adaptive_weight = np.clip(adaptive_weight, 0.001, 150)
    
    predictions = OrderedDict(
        log_probs=log_probs,
        entropy=entropy_loss,
    )
    return predictions, adaptive_weight
```

修改为:

```python
def _forward_training(self, batch):
    dists, entropy_loss = self.nets["policy"].forward_train(
        obs_dict=batch["obs"],
        goal_dict=batch["goal_obs"],
        return_attention_weights=True
    )
    
    assert len(dists.batch_shape) == 1
    
    # ✅ 改为预测动作
    log_probs = dists.log_prob(batch["actions"])
    
    target_ratio = 0.02
    adaptive_weight = (log_probs.mean().item() * target_ratio) / entropy_loss.item()
    adaptive_weight = np.clip(adaptive_weight, 0.001, 150)
    
    predictions = OrderedDict(
        log_probs=log_probs,
        entropy=entropy_loss,
    )
    return predictions, adaptive_weight
```

BUT! 还需要检查:

1. batch["actions"].shape[1] 是否等于 30？
   - 如果是，直接改就行
   - 如果不是，需要调整 ac_dim

2. 如果actions维度不是30，比如是7 (机器人关节+夹爪):
   - 修改 stage2_actionpre.json 中的 "ac_dim": 7 (或实际维度)
   - 重新训练

方案2: 预测目标位置 (如果你坚持预测轨迹)
--------------------------------------

如果你确实想预测空间位置而不是动作:

1. 不要预测整条轨迹，只预测**终点**:
   - 提取 future_traj 的最后一个点: future_traj[:, -1, :] (维度变为3)
   - 修改 ac_dim = 3
   - 这样GMM建模的是"目标位置的多模态性"

2. 增加当前状态信息:
   - 在 obs_shapes 中确保包含:
     - robot0_eef_pos (当前位置)
     - robot0_eef_vel (如果有的话)
     - robot0_joint_pos (关节状态)

3. 修改网络架构:
   - 考虑使用更深的网络
   - 增加 context_length 以利用历史信息

===== 快速验证步骤 =====

1. 添加调试打印 (在 _forward_training 中):

```python
print(f"Actions shape: {batch['actions'].shape}")
print(f"Future traj shape: {batch['obs']['robot0_eef_pos_future_traj'].shape}")
print(f"GMM output dim: {dists.mean.shape[-1]}")
```

2. 运行一个epoch，查看输出:
   - 如果 actions.shape[1] == 30: 维度碰巧相同，但语义不对
   - 如果 actions.shape[1] == 7: 维度不匹配，需要修改ac_dim

3. 检查log_prob的值:

```python
print(f"Log prob: {log_probs.mean().item()}")
```

- 如果是 < -100: 模型完全没学到
- 如果是 -10 到 -50: 还有希望
- 如果是 > -10: 已经在学习

===== 我的建议 =====

直接使用方案1，将训练目标改为 batch["actions"]。

原因:
✓ 行为克隆的标准做法
✓ 任务定义清晰: 输入状态 -> 输出动作
✓ 多模态建模的是专家的决策多样性
✓ 不需要预测环境动力学

如果改了之后loss还是不降，那才考虑:

- 增加输入特征
- 调整网络架构  
- 检查数据质量(运行 estimate_bayes_error.py)

但现在的问题是: **任务定义就错了**！
"""

print(**doc**)

print("\n" + "="*80)
print("推荐的修改方案")
print("="*80)

print("""
立即执行:

1. 修改 agent/algo/agent.py 第662行:
   log_probs = dists.log_prob(batch["actions"])

2. 确认 ac_dim 配置:
   - 如果 actions 是 7 维 -> ac_dim 改为 7
   - 如果 actions 是 30 维 -> 保持 30 (但检查是否合理)

3. 重新训练，观察:
   - Loss 应该能正常下降
   - Log likelihood 应该从负值逐渐增大(接近0)

4. 如果还有问题，运行:
   python agent/scripts/estimate_bayes_error.py --dataset <你的数据.hdf5>

需要帮助修改代码吗？我可以直接帮你改！
""")
