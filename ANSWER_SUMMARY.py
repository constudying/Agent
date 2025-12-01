#!/usr/bin/env python3
"""
回答你的两个问题 + 实用建议

问题1: GMM的模态数量和预测输出什么关系？
问题2: 当前输入输出设计有问题吗？
"""

print("="*70)
print("问题1: GMM模态数量与输出的关系")
print("="*70)

print("""
核心回答：模态数量和输出维度是独立的！

GMM参数结构：
  means:  [batch, num_modes, ac_dim]  # 每个模态的均值
  scales: [batch, num_modes, ac_dim]  # 每个模态的标准差  
  logits: [batch, num_modes]          # 每个模态的权重

你的配置：
  num_modes = 5   # 5个模态
  ac_dim = 9      # 9维输出（3个关键点×3维）
  
参数量: 5×9 + 5×9 + 5 = 95 (每个样本)

模态数量的含义：
  - 能同时建模多少种"不同的路径选择"
  - 例如：模态1=从左边绕, 模态2=从右边绕, 模态3=直线, ...
  
输出维度的含义：
  - 每条路径的精细程度
  - 9维 = 3个关键路径点
  - 30维 = 10个完整轨迹点

模态数量选择：
  ✓ num_modes=5: 大多数机器人任务的经验值，你的选择合理
  ✗ num_modes=1: 退化为确定性输出
  ✗ num_modes>10: 容易模态崩溃

注意模态崩溃：
  理想: [0.25, 0.22, 0.20, 0.18, 0.15] - 各模态均衡
  崩溃: [0.85, 0.10, 0.03, 0.02, 0.00] - 只用1个模态
  
  检测方法（训练时打印）：
    mode_probs = dists.mixture_distribution.probs.mean(0)
    if mode_probs.max() > 0.6:
        print("警告: 模态崩溃")
""")

print("\n" + "="*70)
print("问题2: 你的输入输出设计分析")
print("="*70)

print("""
你的设计：
  编码器输入: robot0_eef_pos + agentview_image + agentview_depth
  解码器输入: 任务token + 过去轨迹 + 轨迹增量
  输出: 未来轨迹的3个关键点 (9维)
  
设计目标：让模型学习"动作空间的执行余量"

评价：

✅ 好的方面：
  1. 任务token - 很好！不同任务有不同的"余量"概念
  2. 历史轨迹 - 合理！包含速度和加速度信息
  3. 输出关键点 - 可行！符合任务空间规划思路
  4. 分层设计 - 正确！High-level规划，Low-level执行

⚠️ 潜在问题：
  1. "执行余量"没有被显式建模
     - 当前学到的是："会往哪里移动"
     - 但你想要的是："还能怎么移动，有多安全"
     
  2. 缺少环境约束信息
     - 哪些方向安全？哪些区域危险？
     - 模型需要从图像隐式推理，很难
     
  3. 监督信号的局限
     - 演示数据只有一条路径
     - 不代表所有可能的路径和余量

❌ 不是说设计错了！
  - 作为基础的轨迹规划，设计是合理的
  - 但要真正表达"执行余量"，需要额外信息
""")

print("\n" + "="*70)
print("实用建议（分阶段）")
print("="*70)

print("""
【立即行动】先让当前设计训练起来
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 保持当前设计不变（9维关键点）
2. 开始训练，监控以下指标：

   关键指标1: Log probability
     目标: 从 -50 提升到 -10 左右
     
   关键指标2: 模态使用情况
     mode_probs = dists.mixture_distribution.probs.mean(0)
     检查: 是否各模态权重都 > 0.1
     
   关键指标3: 关键点误差  
     pred vs true 的欧氏距离
     目标: < 0.05m (5cm)

3. 如果出现模态崩溃（某个模态 > 0.6）：
   - 增加entropy权重
   - 或减少num_modes到3
   
4. 训练命令：
   $ rm -rf trained_models_highlevel/test/*
   $ python agent/scripts/train.py --config agent/configs/stage2_actionpre.json

【短期改进】添加"执行余量"的简单表达
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

等训练稳定后（loss能正常下降），添加置信度输出：

修改输出：
  output = {
      'keypoints': [batch, 9],     # 原有的关键点
      'confidence': [batch, 3],    # 新增：每个关键点的"余量分数"
  }
  
confidence的含义：
  - 高值(0.8-1.0): 这个方向很安全，操作余量大
  - 中值(0.4-0.7): 可以走，但要小心
  - 低值(0.0-0.3): 危险，余量很小

监督信号（自动生成）：
  从演示数据中，计算每个关键点到障碍物的距离
  距离越远，confidence越高

实现代码：
""")

code_snippet = '''
# 在 _forward_training 中添加
def compute_confidence(keypoints, workspace_info):
    """
    根据关键点位置计算confidence
    keypoints: [batch, 3, 3]  # 3个点，每个3维
    """
    confidence = []
    for point in keypoints:
        # 距离到最近障碍物/边界的距离
        dist = compute_clearance(point, workspace_info)
        # 转换为0-1分数，5cm为分界点
        conf = torch.sigmoid((dist - 0.05) / 0.05)
        confidence.append(conf)
    return torch.stack(confidence, dim=-1)  # [batch, 3]

# 修改loss
keypoints_loss = -dists.log_prob(keypoints_flat)
confidence_loss = F.mse_loss(pred_confidence, true_confidence)
total_loss = keypoints_loss + 0.1 * confidence_loss
'''

print(code_snippet)

print("""
【长期改进】显式建模环境约束
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果上面的都work了，可以进一步：

输出结构：
  output = {
      'keypoints': [batch, 9],                # 路径关键点
      'obstacles': [batch, M, 3],             # M个障碍物位置
      'obstacle_radius': [batch, M],          # 每个障碍物的半径
      'workspace_bounds': [batch, 6],         # [xmin,xmax,ymin,ymax,zmin,zmax]
  }

优势：
  - Low-level可以用这些约束做轨迹优化
  - 能真正表达"执行余量"
  - 可解释性强

但需要：
  - 从演示数据自动提取障碍物标签
  - 或使用图像分割得到障碍物
  - 更复杂的loss设计
""")

print("\n" + "="*70)
print("总结：现在应该做什么？")
print("="*70)

print("""
1. ✅ 你的num_modes=5是合理的，不需要改

2. ✅ 你的输入输出设计作为"轨迹规划"是合理的
   
3. ⚠️ 但要表达"执行余量"，当前设计不够显式

4. 🎯 建议的行动路线：
   
   第1步（现在）: 
     先让训练work起来
     观察9维关键点能否收敛
     
   第2步（训练稳定后）:
     添加confidence输出
     表达"每个方向的余量"
     
   第3步（如果需要）:
     显式预测环境约束
     完整表达"可行空间"

5. 📊 训练时必须监控的指标：
   - Log probability (越高越好)
   - 模态使用率 (各模态 > 0.1)
   - 关键点预测误差 (< 0.05m)
   - 模态间差异性 (不同模态应该预测不同路径)

现在可以开始训练了！有问题随时问。
""")

print("\n需要帮你实现confidence输出吗？(y/n): ", end="")
