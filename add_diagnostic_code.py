"""
添加到 Highlevel_GMM_pretrain._forward_training 的诊断代码

将这段代码插入到 agent/algo/agent.py 的 _forward_training 方法中
用于诊断是预测轨迹难还是预测动作难
"""

diagnostic_code = '''
def _forward_training(self, batch):
    """
    模型训练时使用，以字典形式返回网络输出
    """
    dists, entropy_loss = self.nets["policy"].forward_train(
        obs_dict=batch["obs"],
        goal_dict=batch["goal_obs"],
        return_attention_weights=True
    )

    assert len(dists.batch_shape) == 1, "@Highlevel_GMM_pretrain: action distribution must be 1D batch shape during training."

    # ========== 添加诊断代码 START ==========
    # 每100个batch打印一次
    if not hasattr(self, '_diagnostic_counter'):
        self._diagnostic_counter = 0
    
    self._diagnostic_counter += 1
    
    if self._diagnostic_counter % 100 == 1:  # 第1, 101, 201...次打印
        print("\\n" + "="*80)
        print("🔍 训练诊断信息")
        print("="*80)
        
        # 1. 维度信息
        print("\\n【维度检查】")
        print(f"GMM输出维度:")
        print(f"  - means.shape: {dists.mean.shape}")  # [batch, num_modes, ac_dim]
        print(f"  - scales.shape: {dists.scale.shape}")
        print(f"  - ac_dim配置: {self.ac_dim}")
        
        print(f"\\n目标数据维度:")
        future_traj = batch["obs"]["robot0_eef_pos_future_traj"]
        print(f"  - future_traj.shape: {future_traj.shape}")
        
        if "actions" in batch:
            actions = batch["actions"]
            print(f"  - actions.shape: {actions.shape}")
        else:
            print(f"  - actions: 未找到")
        
        # 2. 数据值范围
        print(f"\\n【数据值范围】")
        print(f"future_traj:")
        print(f"  - 均值: {future_traj.mean().item():.4f}")
        print(f"  - 标准差: {future_traj.std().item():.4f}")
        print(f"  - 最小值: {future_traj.min().item():.4f}")
        print(f"  - 最大值: {future_traj.max().item():.4f}")
        
        if "actions" in batch:
            print(f"\\nactions:")
            print(f"  - 均值: {actions.mean().item():.4f}")
            print(f"  - 标准差: {actions.std().item():.4f}")
            print(f"  - 最小值: {actions.min().item():.4f}")
            print(f"  - 最大值: {actions.max().item():.4f}")
        
        # 3. 对比预测难度
        print(f"\\n【预测难度对比】")
        
        # 计算future_traj的log_prob
        log_probs_traj = dists.log_prob(future_traj)
        print(f"\\n目标: future_traj (30维)")
        print(f"  - log_prob 均值: {log_probs_traj.mean().item():.4f}")
        print(f"  - log_prob 标准差: {log_probs_traj.std().item():.4f}")
        print(f"  - log_prob 最小值: {log_probs_traj.min().item():.4f}")
        print(f"  - log_prob 最大值: {log_probs_traj.max().item():.4f}")
        
        # 判断log_prob的健康程度
        mean_log_prob = log_probs_traj.mean().item()
        if mean_log_prob < -100:
            print(f"  ⚠️ 警告: log_prob < -100，模型基本没学到任何东西！")
        elif mean_log_prob < -50:
            print(f"  ⚠️ 注意: log_prob在[-100, -50]，学习困难")
        elif mean_log_prob < -10:
            print(f"  ✓ log_prob在[-50, -10]，模型在学习")
        else:
            print(f"  ✓✓ log_prob > -10，学习效果较好")
        
        # 如果有actions，对比一下
        if "actions" in batch:
            # 检查维度是否匹配
            if actions.shape[-1] == self.ac_dim:
                log_probs_action = dists.log_prob(actions)
                print(f"\\n对比: actions ({actions.shape[-1]}维)")
                print(f"  - log_prob 均值: {log_probs_action.mean().item():.4f}")
                print(f"  - log_prob 标准差: {log_probs_action.std().item():.4f}")
                
                diff = log_probs_action.mean().item() - log_probs_traj.mean().item()
                print(f"\\n📊 预测难度差异:")
                print(f"  - actions的log_prob - future_traj的log_prob = {diff:.4f}")
                
                if diff > 10:
                    print(f"  ✓✓ actions **明显更容易预测** (+{diff:.1f})")
                    print(f"      建议: 改为预测actions!")
                elif diff > 5:
                    print(f"  ✓ actions 更容易预测 (+{diff:.1f})")
                    print(f"      建议: 考虑改为预测actions")
                elif diff > -5:
                    print(f"  - 两者难度相当 ({diff:.1f})")
                else:
                    print(f"  ? future_traj更容易？不太可能，请检查数据")
            else:
                print(f"\\n⚠️ actions维度({actions.shape[-1]}) != ac_dim({self.ac_dim})")
                print(f"  无法直接对比，但这说明配置可能有问题")
        
        # 4. GMM组件分析
        print(f"\\n【GMM组件分析】")
        print(f"模态数量: {dists.mixture_distribution.probs.shape[-1]}")
        mode_probs = dists.mixture_distribution.probs.mean(0)  # 平均每个模态的概率
        print(f"各模态平均权重: {mode_probs.cpu().numpy()}")
        
        # 检查是否有模态崩溃
        max_prob = mode_probs.max().item()
        if max_prob > 0.8:
            print(f"  ⚠️ 警告: 模态{mode_probs.argmax().item()}占主导({max_prob:.2%})，可能发生模态崩溃")
        elif max_prob > 0.5:
            print(f"  ⚠️ 注意: 模态{mode_probs.argmax().item()}权重较高({max_prob:.2%})")
        else:
            print(f"  ✓ 模态分布较均匀")
        
        print("="*80 + "\\n")
    # ========== 添加诊断代码 END ==========

    # 原来的代码继续
    log_probs = dists.log_prob(batch["obs"]["robot0_eef_pos_future_traj"])

    target_ratio = 0.02
    adaptive_weight = (log_probs.mean().item() * target_ratio) / entropy_loss.item()
    adaptive_weight = np.clip(adaptive_weight, 0.001, 150)
    predictions = OrderedDict(
        log_probs=log_probs,
        entropy=entropy_loss,
    )
    return predictions, adaptive_weight
'''

print("="*80)
print("诊断代码使用说明")
print("="*80)

print("""
将上面的诊断代码替换 agent/algo/agent.py 中 Highlevel_GMM_pretrain 类的
_forward_training 方法（约 line 658-678）

这段代码会在训练时每100个batch打印一次诊断信息，包括：

1. 维度检查
   - GMM输出维度和配置是否匹配
   - future_traj和actions的维度

2. 数据值范围
   - 检查数据是否被正确归一化
   - 是否有异常值

3. 预测难度对比 ⭐最重要⭐
   - future_traj的log_prob有多低
   - 如果有actions，对比哪个更容易预测
   - 给出明确的建议

4. GMM组件分析
   - 检查是否发生模态崩溃
   - 各个模态的权重分布

运行训练后，你会看到类似这样的输出：

🔍 训练诊断信息
================================================================================

【维度检查】
GMM输出维度:
  - means.shape: torch.Size([32, 5, 30])
  - ac_dim配置: 30

目标数据维度:
  - future_traj.shape: torch.Size([32, 30])
  - actions.shape: torch.Size([32, 7])

【预测难度对比】
目标: future_traj (30维)
  - log_prob 均值: -245.6789
  ⚠️ 警告: log_prob < -100，模型基本没学到任何东西！

对比: actions (7维)
  - log_prob 均值: -15.2345

📊 预测难度差异:
  - actions的log_prob - future_traj的log_prob = 230.44
  ✓✓ actions **明显更容易预测** (+230.4)
      建议: 改为预测actions!

================================================================================

看到这样的输出后，你就知道应该怎么改了！
""")

print("\n" + "="*80)
print("快速修改方案")
print("="*80)

print("""
如果诊断显示 actions 明显更容易预测，做以下修改：

步骤1: 修改 agent/algo/agent.py line 669
---------------------------------------
将:
    log_probs = dists.log_prob(batch["obs"]["robot0_eef_pos_future_traj"])

改为:
    log_probs = dists.log_prob(batch["actions"])


步骤2: 修改 agent/configs/stage2_actionpre.json
------------------------------------------------
将:
    "ac_dim": 30

改为:
    "ac_dim": 7  (或你的实际action维度)


步骤3: 重新训练
--------------
删除旧的checkpoint，从头开始训练：
    rm -rf trained_models_highlevel/test/*
    python agent/scripts/train.py --config agent/configs/stage2_actionpre.json

观察loss是否能正常下降。


如果你想保留轨迹预测：
----------------------
考虑只预测终点而不是整条轨迹：

1. 修改 agent/algo/agent.py line 669:
    # 提取future_traj的最后一个点
    future_endpoint = batch["obs"]["robot0_eef_pos_future_traj"][:, -3:]
    log_probs = dists.log_prob(future_endpoint)

2. 修改 agent/configs/stage2_actionpre.json:
    "ac_dim": 3

这样GMM建模的是"最终目标位置的多模态性"，维度从30降到3，更容易训练。
""")

print("\n需要我帮你直接修改代码吗？")
