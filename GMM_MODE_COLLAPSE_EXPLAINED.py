#!/usr/bin/env python3
"""
GMM模态崩溃详解与可视化
演示什么是模态崩溃、为什么不好、如何避免
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, MixtureSameFamily


def print_section(title, char="="):
    print(f"\n{char*70}")
    print(f" {title}")
    print(f"{char*70}\n")


def visualize_mode_usage(mode_probs, title):
    """可视化模态使用情况"""
    print(f"{title}")
    print(f"模态概率: {mode_probs}")
    
    # ASCII条形图
    max_width = 50
    for i, prob in enumerate(mode_probs):
        bar_width = int(prob * max_width)
        bar = "█" * bar_width
        print(f"  模态{i+1}: {bar} {prob:.3f}")
    
    # 判断是否崩溃
    max_prob = np.max(mode_probs)
    if max_prob > 0.6:
        print(f"\n  ⚠️  警告：模态崩溃！主导模态占比 {max_prob:.1%}")
    else:
        print(f"\n  ✓  健康：模态分布均衡")
    
    # 计算熵（衡量分布的均匀程度）
    entropy = -np.sum(mode_probs * np.log(mode_probs + 1e-10))
    max_entropy = np.log(len(mode_probs))
    normalized_entropy = entropy / max_entropy
    print(f"  分布熵: {entropy:.3f} / {max_entropy:.3f} = {normalized_entropy:.1%}")
    print()


def demonstrate_mode_collapse():
    """演示模态崩溃过程"""
    
    print_section("什么是模态崩溃？")
    
    print("""
模态崩溃(Mode Collapse)是指GMM训练时，多个模态退化为少数几个
（甚至一个）模态，失去了多模态建模能力。

GMM的核心思想：
  用多个高斯分布的混合来建模复杂的多峰分布
  每个模态(mode)代表数据的一种"模式"或"聚类"

在机器人任务中：
  模态1: 从左边绕过障碍物
  模态2: 从右边绕过障碍物  
  模态3: 从上方越过障碍物
  模态4: 直接推开障碍物
  模态5: 后退重新规划
  
如果崩溃成1个模态：
  只学会了一种策略（例如"总是从左边绕"）
  失去了处理不同情况的灵活性
    """)
    
    print_section("训练过程中的模态演化", "-")
    
    # 模拟训练过程
    epochs = [0, 10, 50, 100, 200, 500]
    
    # 场景1：健康训练
    print("【场景1：健康训练 - 模态保持多样性】\n")
    healthy_probs = [
        np.array([0.20, 0.20, 0.20, 0.20, 0.20]),  # epoch 0: 均匀初始化
        np.array([0.24, 0.22, 0.19, 0.18, 0.17]),  # epoch 10: 开始分化
        np.array([0.26, 0.23, 0.20, 0.18, 0.13]),  # epoch 50: 持续调整
        np.array([0.28, 0.24, 0.21, 0.16, 0.11]),  # epoch 100: 趋于稳定
        np.array([0.27, 0.25, 0.22, 0.15, 0.11]),  # epoch 200: 稳定
        np.array([0.26, 0.25, 0.23, 0.16, 0.10]),  # epoch 500: 最终状态
    ]
    
    for epoch, probs in zip(epochs, healthy_probs):
        visualize_mode_usage(probs, f"Epoch {epoch:3d}:")
    
    print("分析：")
    print("  ✓ 各模态权重相对均衡（0.10-0.28）")
    print("  ✓ 没有单一模态占主导")
    print("  ✓ 保留了多样性和灵活性")
    print("  → 模型可以根据不同情况选择不同策略\n")
    
    
    # 场景2：模态崩溃
    print("\n" + "="*70)
    print("【场景2：模态崩溃 - 退化为单模态】\n")
    collapsed_probs = [
        np.array([0.20, 0.20, 0.20, 0.20, 0.20]),  # epoch 0: 均匀初始化
        np.array([0.32, 0.22, 0.18, 0.16, 0.12]),  # epoch 10: 某模态开始主导
        np.array([0.51, 0.20, 0.13, 0.10, 0.06]),  # epoch 50: 加速崩溃
        np.array([0.72, 0.14, 0.08, 0.04, 0.02]),  # epoch 100: 严重崩溃
        np.array([0.85, 0.10, 0.03, 0.01, 0.01]),  # epoch 200: 接近完全崩溃
        np.array([0.94, 0.04, 0.01, 0.005, 0.005]), # epoch 500: 完全崩溃
    ]
    
    for epoch, probs in zip(epochs, collapsed_probs):
        visualize_mode_usage(probs, f"Epoch {epoch:3d}:")
    
    print("分析：")
    print("  ✗ 模态1占主导地位（94%）")
    print("  ✗ 其他模态几乎不被使用")
    print("  ✗ 退化为单一确定性策略")
    print("  → 模型失去了多样性，无法应对不同情况\n")


def why_mode_collapse_is_bad():
    """为什么模态崩溃不好"""
    
    print_section("为什么模态崩溃是问题？")
    
    print("""
1. 【失去多样性】
   
   健康GMM：
     情况A(障碍物在左) → 选择模态2(从右绕) prob=0.8
     情况B(障碍物在右) → 选择模态1(从左绕) prob=0.8
     情况C(障碍物正前方) → 选择模态3(从上越过) prob=0.7
   
   崩溃GMM：
     情况A → 模态1(从左绕) prob=0.95
     情况B → 模态1(从左绕) prob=0.95  ← 错误！应该从右绕
     情况C → 模态1(从左绕) prob=0.95  ← 错误！会撞上
   
   → 只学会了一招，无法应对变化

2. 【泛化能力差】
   
   训练数据可能偏向某种情况（例如大部分演示都从左绕）
   健康GMM：仍然学会了右绕、上越等其他策略
   崩溃GMM：只记住了"从左绕"，测试时遇到新情况就失败
   
   → 过拟合到训练分布的主要模式

3. 【无法表达不确定性】
   
   在歧义情况下（多种策略都可行）：
   健康GMM：输出 [0.3, 0.35, 0.35] 表示"这几种都可以"
   崩溃GMM：输出 [0.9, 0.05, 0.05] 表示"只能这样"
   
   → 过度自信，无法表达"我不确定"

4. 【浪费模型容量】
   
   你设置了 num_modes=5，期望学习5种策略
   但实际只用了1个模态，其他4个参数被浪费
   
   → 模型容量利用率低

5. 【优化问题】
   
   GMM的loss = -log(Σ w_i × N(x | μ_i, σ_i))
   
   如果某个模态w_1很大：
     loss ≈ -log(w_1 × N(x | μ_1, σ_1))
     其他模态的梯度 ≈ 0 (因为w_i→0)
   
   → 其他模态"死亡"，无法继续学习
    """)


def demonstrate_with_pytorch():
    """用PyTorch演示模态崩溃的数值行为"""
    
    print_section("PyTorch演示：崩溃如何影响预测")
    
    # 简化的演示：手动采样
    num_modes = 5
    ac_dim = 9
    
    # 健康的GMM
    print("【健康GMM】\n")
    torch.manual_seed(42)
    healthy_probs = torch.tensor([0.26, 0.25, 0.23, 0.16, 0.10])
    healthy_means = torch.randn(num_modes, ac_dim) * 0.5
    healthy_stds = torch.ones(num_modes, ac_dim) * 0.1
    
    print(f"模态权重: {healthy_probs.numpy()}")
    print(f"模态1预测: {healthy_means[0, :3].numpy()}")
    print(f"模态2预测: {healthy_means[1, :3].numpy()}")
    print(f"模态3预测: {healthy_means[2, :3].numpy()}")
    print(f"模态4预测: {healthy_means[3, :3].numpy()}")
    print(f"模态5预测: {healthy_means[4, :3].numpy()}")
    
    # 手动采样：根据权重选择模态，然后从该模态的高斯分布采样
    print(f"\n5次采样结果（前3维）:")
    for i in range(5):
        # 根据权重随机选择模态
        mode_idx = torch.multinomial(healthy_probs, 1).item()
        # 从该模态的高斯分布采样
        sample = torch.normal(healthy_means[mode_idx], healthy_stds[mode_idx])
        print(f"  采样{i+1}(模态{mode_idx+1}): {sample[:3].numpy()}")
    print("→ 注意预测有多样性！不同采样可能来自不同模态\n")
    
    
    # 崩溃的GMM
    print("\n【崩溃GMM】\n")
    torch.manual_seed(42)
    collapsed_probs = torch.tensor([0.94, 0.04, 0.01, 0.005, 0.005])
    collapsed_means = torch.randn(num_modes, ac_dim) * 0.5
    collapsed_stds = torch.ones(num_modes, ac_dim) * 0.1
    
    print(f"模态权重: {collapsed_probs.numpy()}")
    print(f"模态1预测: {collapsed_means[0, :3].numpy()}")
    print(f"模态2预测: {collapsed_means[1, :3].numpy()} (几乎不会被选中)")
    print(f"模态3预测: {collapsed_means[2, :3].numpy()} (几乎不会被选中)")
    print(f"模态4预测: {collapsed_means[3, :3].numpy()} (几乎不会被选中)")
    print(f"模态5预测: {collapsed_means[4, :3].numpy()} (几乎不会被选中)")
    
    # 手动采样
    print(f"\n5次采样结果（前3维）:")
    for i in range(5):
        mode_idx = torch.multinomial(collapsed_probs, 1).item()
        sample = torch.normal(collapsed_means[mode_idx], collapsed_stds[mode_idx])
        print(f"  采样{i+1}(模态{mode_idx+1}): {sample[:3].numpy()}")
    print("→ 注意几乎所有采样都来自模态1！失去了多样性\n")
    
    # 统计100次采样的模态使用情况
    print("【统计100次采样的模态分布】\n")
    mode_counts_healthy = torch.zeros(num_modes)
    mode_counts_collapsed = torch.zeros(num_modes)
    
    for _ in range(100):
        mode_counts_healthy[torch.multinomial(healthy_probs, 1)] += 1
        mode_counts_collapsed[torch.multinomial(collapsed_probs, 1)] += 1
    
    print("健康GMM (100次采样):")
    for i, count in enumerate(mode_counts_healthy):
        bar = "█" * int(count / 2)
        print(f"  模态{i+1}: {bar} {count:.0f}次")
    
    print("\n崩溃GMM (100次采样):")
    for i, count in enumerate(mode_counts_collapsed):
        bar = "█" * int(count / 2)
        print(f"  模态{i+1}: {bar} {count:.0f}次")
    print()


def how_modes_should_evolve():
    """模态应该如何演化"""
    
    print_section("模态应该如何演化？")
    
    print("""
理想的训练过程：

阶段1：探索期 (Epoch 0-50)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━���━━━━━━
  模态权重: 相对均匀 [0.20, 0.20, 0.20, 0.20, 0.20]
  模态分化: 开始形成不同的"专长"
  
  例如：
    模态1：学习处理"左侧有障碍"的情况
    模态2：学习处理"右侧有障碍"的情况
    模态3：学习处理"前方有障碍"的情况
    ...
  
  期望：各模态都在积极学习，梯度不为0


阶段2：分化期 (Epoch 50-200)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  模态权重: 开始有差异 [0.28, 0.24, 0.21, 0.16, 0.11]
  模态专业化: 每个模态负责特定的数据子集
  
  这是自然的！因为：
    - 训练数据本身有分布偏好
    - 某些情况更常见
    - 模态根据数据分布自适应
  
  期望：
    ✓ 最大权重 < 0.4 (没有单一主导)
    ✓ 最小权重 > 0.05 (没有完全死亡)
    ✓ 权重变化平缓（不是突变）


阶段3：稳定期 (Epoch 200+)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  模态权重: 趋于稳定 [0.26, 0.25, 0.23, 0.16, 0.10]
  Loss: 不再显著下降
  
  期望：
    ✓ 权重分布稳定（不再大幅变化）
    ✓ 各模态都有明确的"负责区域"
    ✓ 验证集上的表现最优
  
  测试标准：
    在验证集上，不同类型的样本应该激活不同的模态


健康指标：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 模态使用率
   ✓ 任何模态权重不超过 0.6
   ✓ 任何模态权重不低于 0.05
   
2. 模态多样性
   ✓ 不同模态的均值应该显著不同
   ✓ 计算: |μ_i - μ_j| > 0.1 (对于任意i≠j)
   
3. 条件多样性
   ✓ 对于不同的输入，应该激活不同的模态
   ✓ 例如：obs_A → 模态1最高，obs_B → 模态2最高
   
4. 熵指标
   ✓ 平均熵 > 0.8 × log(num_modes)
   ✓ 表示分布相对均匀


警告信号：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  某模态权重 > 0.6 → 开始崩溃
⚠️  某模态权重 > 0.8 → 严重崩溃  
⚠️  权重在训练中突然跳变 → 优化不稳定
⚠️  所有输入都激活同一模态 → 失去条件依赖性
    """)


def detection_and_prevention():
    """检测和预防模态崩溃"""
    
    print_section("如何检测和预防？")
    
    print("""
【检测方法】

1. 训练时实时监控
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    print("""
# 在 _forward_training 中添加
def _forward_training(self, batch):
    dists = self.nets["policy"].forward_train(...)
    
    # 计算模态使用率
    mode_probs = dists.mixture_distribution.probs  # [batch, num_modes]
    avg_mode_probs = mode_probs.mean(0)  # [num_modes]
    
    # 每100个batch打印一次
    if self.global_step % 100 == 0:
        print(f"\\n[Step {self.global_step}]")
        print(f"  模态权重: {avg_mode_probs.detach().cpu().numpy()}")
        
        # 检测崩溃
        max_prob = avg_mode_probs.max().item()
        if max_prob > 0.6:
            print(f"  ⚠️  警告：模态崩溃倾向 (max={max_prob:.2f})")
        
        # 计算熵
        entropy = -(avg_mode_probs * torch.log(avg_mode_probs + 1e-10)).sum()
        max_entropy = np.log(len(avg_mode_probs))
        print(f"  模态熵: {entropy:.3f} / {max_entropy:.3f}")
    """)
    
    print("""

2. 可视化模态分布
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
在TensorBoard中记录：
    - 每个模态的平均权重（折线图）
    - 模态熵的变化（折线图）
    - 不同模态的预测可视化（散点图）


【预防方法】

1. 增加熵正则化 ⭐ (最有效)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    print("""
# 鼓励模态分布均匀
entropy_loss = -(mode_probs * torch.log(mode_probs + 1e-10)).sum(-1).mean()
entropy_target = np.log(num_modes)  # 最大熵

# 熵越高越好，所以loss = -entropy
entropy_regularization = -entropy_loss

# 总loss
total_loss = gmm_loss + 0.1 * entropy_regularization

提示：
  - 权重0.1是起点，可以调到0.2-0.5
  - 如果崩溃严重，增大权重
  - 如果模态过于均匀导致性能下降，减小权重
    """)
    
    print("""

2. 降低模态数量
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
如果num_modes=5总是崩溃，尝试：
  - num_modes=3 (更容易保持均衡)
  - num_modes=2 (最简单的多模态)

原因：模态数量越多，越难保持均衡


3. 调整学习率
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 学习率太高 → 优化不稳定 → 容易崩溃
  - 建议：0.0001 或更小
  - 使用学习率预热（前1000步线性增加）


4. 初始化技巧
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GMM的logits初始化为0（使得初始权重均匀）：
    """)
    
    print("""
class GMMLossHead(nn.Module):
    def __init__(self, ...):
        self.logits_layer = nn.Linear(input_dim, num_modes)
        # 初始化为0
        nn.init.zeros_(self.logits_layer.weight)
        nn.init.zeros_(self.logits_layer.bias)
    """)
    
    print("""

5. 数据增强
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
如果训练数据本身就不平衡（例如90%从左绕，10%从右绕）：
  - 过采样少数类
  - 给不同类型样本不同的loss权重
  - 这样每个模态都有足够的数据学习


6. 渐进式训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
先从少数模态开始，逐渐增加：
  - Epoch 0-100: num_modes=2
  - Epoch 100-200: num_modes=3
  - Epoch 200+: num_modes=5

让模型逐步学会分化
    """)


def main():
    print("="*70)
    print(" GMM模态崩溃完全解析")
    print("="*70)
    
    demonstrate_mode_collapse()
    why_mode_collapse_is_bad()
    demonstrate_with_pytorch()
    how_modes_should_evolve()
    detection_and_prevention()
    
    print_section("总结", "=")
    print("""
模态崩溃 = 多个模态退化为一个，失去多样性

为什么不好：
  1. 失去处理不同情况的能力
  2. 泛化能力差
  3. 无法表达不确定性
  4. 浪费模型容量
  5. 优化死锁（其他模态梯度为0）

理想的演化：
  初期：均匀探索 [0.20, 0.20, 0.20, 0.20, 0.20]
  中期：适度分化 [0.28, 0.24, 0.21, 0.16, 0.11]
  后期：稳定专业化，但各模态都活跃

检测指标：
  ✓ 任何模态 < 0.6
  ✓ 任何模态 > 0.05
  ✓ 熵 > 0.8 × log(num_modes)

最佳预防：
  ⭐ 增加熵正则化 (loss += 0.1 * entropy_regularization)
  - 降低学习率
  - 减少模态数量
  - 均匀初始化

关键：在训练时实时监控模态权重！
    """)
    
    print("\n" + "="*70)
    print(" 详细文档: EXECUTION_MARGIN_DESIGN_ANALYSIS.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
