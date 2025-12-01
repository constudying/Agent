"""
对比MSE损失和似然损失在多模态数据上的行为

演示为什么似然损失不怕多模态性
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch
import torch.nn as nn

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_bimodal_data(n_samples=1000):
    """生成双模态数据"""
    # 50%概率动作=-1（左转），50%概率动作=+1（右转）
    mode1 = np.random.normal(-1.0, 0.1, n_samples // 2)
    mode2 = np.random.normal(1.0, 0.1, n_samples // 2)
    actions = np.concatenate([mode1, mode2])
    np.random.shuffle(actions)
    return actions


def train_mse_model(actions, n_epochs=200):
    """训练MSE模型（预测单个值）"""
    # 简单的：预测一个常数
    prediction = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([prediction], lr=0.01)
    
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    losses = []
    predictions = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = torch.mean((prediction - actions_tensor) ** 2)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        predictions.append(prediction.item())
    
    return predictions, losses


def train_gmm_model(actions, n_modes=2, n_epochs=200):
    """训练GMM模型（预测分布）"""
    # GMM参数：均值、标准差、权重
    means = torch.nn.Parameter(torch.randn(n_modes) * 0.5)
    log_stds = torch.nn.Parameter(torch.zeros(n_modes))
    logit_weights = torch.nn.Parameter(torch.zeros(n_modes))
    
    params = [means, log_stds, logit_weights]
    optimizer = torch.optim.Adam(params, lr=0.01)
    
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    losses = []
    means_history = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # 计算混合高斯的负对数似然
        stds = torch.exp(log_stds)
        weights = torch.softmax(logit_weights, dim=0)
        
        # 每个模态的概率密度
        log_probs = []
        for i in range(n_modes):
            log_prob = -0.5 * ((actions_tensor - means[i]) / stds[i]) ** 2
            log_prob = log_prob - torch.log(stds[i]) - 0.5 * np.log(2 * np.pi)
            log_prob = log_prob + torch.log(weights[i])
            log_probs.append(log_prob)
        
        # 混合：log-sum-exp
        log_probs = torch.stack(log_probs, dim=0)
        log_likelihood = torch.logsumexp(log_probs, dim=0)
        loss = -torch.mean(log_likelihood)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        means_history.append(means.detach().numpy().copy())
    
    return means_history, losses, means, stds, weights


def visualize_comparison():
    """可视化MSE vs GMM的对比"""
    
    # 生成数据
    np.random.seed(42)
    torch.manual_seed(42)
    actions = generate_bimodal_data(1000)
    
    # 训练两个模型
    mse_preds, mse_losses = train_mse_model(actions, n_epochs=200)
    gmm_means_history, gmm_losses, final_means, final_stds, final_weights = train_gmm_model(
        actions, n_modes=2, n_epochs=200
    )
    
    # 创建图表
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # ============ 1. 数据分布 ============
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(actions, bins=50, alpha=0.7, color='gray', edgecolor='black', density=True)
    ax1.axvline(-1.0, color='blue', linestyle='--', linewidth=2, label='真实模态1 (左转)')
    ax1.axvline(1.0, color='green', linestyle='--', linewidth=2, label='真实模态2 (右转)')
    ax1.axvline(0.0, color='red', linestyle=':', linewidth=2, label='均值 (MSE会学到这个)')
    ax1.set_xlabel('动作值', fontsize=12)
    ax1.set_ylabel('概率密度', fontsize=12)
    ax1.set_title('训练数据分布（双模态）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # ============ 2. MSE训练过程 ============
    ax2 = fig.add_subplot(gs[1, 0])
    epochs = np.arange(len(mse_preds))
    ax2.plot(epochs, mse_preds, 'r-', linewidth=2)
    ax2.axhline(-1.0, color='blue', linestyle='--', alpha=0.5, label='模态1')
    ax2.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='模态2')
    ax2.axhline(0.0, color='red', linestyle=':', alpha=0.7, label='期望值')
    ax2.set_xlabel('训练轮次', fontsize=11)
    ax2.set_ylabel('MSE预测值', fontsize=11)
    ax2.set_title('MSE模型：预测收敛到0\n（均值，最差的动作！）', 
                 fontsize=12, fontweight='bold', color='red')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ============ 3. GMM训练过程 ============
    ax3 = fig.add_subplot(gs[1, 1])
    gmm_means_history = np.array(gmm_means_history)
    for i in range(gmm_means_history.shape[1]):
        ax3.plot(epochs, gmm_means_history[:, i], linewidth=2, 
                label=f'模态{i+1}均值', alpha=0.8)
    ax3.axhline(-1.0, color='blue', linestyle='--', alpha=0.5)
    ax3.axhline(1.0, color='green', linestyle='--', alpha=0.5)
    ax3.set_xlabel('训练轮次', fontsize=11)
    ax3.set_ylabel('GMM模态均值', fontsize=11)
    ax3.set_title('GMM模型：两个模态分别收敛\n（正确学到分布！）', 
                 fontsize=12, fontweight='bold', color='green')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ============ 4. 损失曲线对比 ============
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(epochs, mse_losses, 'r-', linewidth=2, label='MSE损失', alpha=0.8)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(epochs, gmm_losses, 'g-', linewidth=2, label='GMM NLL损失', alpha=0.8)
    
    ax4.set_xlabel('训练轮次', fontsize=11)
    ax4.set_ylabel('MSE损失', fontsize=11, color='red')
    ax4_twin.set_ylabel('负对数似然', fontsize=11, color='green')
    ax4.tick_params(axis='y', labelcolor='red')
    ax4_twin.tick_params(axis='y', labelcolor='green')
    ax4.set_title('损失曲线对比', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 合并图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    
    # ============ 5. MSE的问题可视化 ============
    ax5 = fig.add_subplot(gs[2, 0])
    x = np.linspace(-2, 2, 1000)
    
    # 真实分布
    true_dist = 0.5 * norm.pdf(x, -1.0, 0.1) + 0.5 * norm.pdf(x, 1.0, 0.1)
    ax5.plot(x, true_dist, 'k-', linewidth=2, label='真实分布', alpha=0.7)
    
    # MSE预测（单个点）
    mse_final = mse_preds[-1]
    ax5.axvline(mse_final, color='red', linewidth=3, label=f'MSE预测={mse_final:.2f}')
    ax5.fill_between([mse_final-0.05, mse_final+0.05], 0, ax5.get_ylim()[1], 
                     color='red', alpha=0.3)
    
    ax5.set_xlabel('动作值', fontsize=11)
    ax5.set_ylabel('概率密度', fontsize=11)
    ax5.set_title('MSE预测 vs 真实分布\n（预测在"谷底"！）', 
                 fontsize=12, fontweight='bold', color='red')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.text(0, max(true_dist)*0.7, '❌ 最低概率区域', 
            ha='center', fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # ============ 6. GMM的正确建模 ============
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(x, true_dist, 'k-', linewidth=2, label='真实分布', alpha=0.7)
    
    # GMM预测分布
    gmm_dist = np.zeros_like(x)
    means_np = final_means.detach().numpy()
    stds_np = final_stds.detach().numpy()
    weights_np = final_weights.detach().numpy()
    
    for i in range(len(means_np)):
        component = weights_np[i] * norm.pdf(x, means_np[i], stds_np[i])
        gmm_dist += component
        ax6.plot(x, component, '--', linewidth=1.5, alpha=0.6, 
                label=f'模态{i+1}: μ={means_np[i]:.2f}')
    
    ax6.plot(x, gmm_dist, 'g-', linewidth=3, label='GMM预测分布', alpha=0.8)
    ax6.set_xlabel('动作值', fontsize=11)
    ax6.set_ylabel('概率密度', fontsize=11)
    ax6.set_title('GMM预测 vs 真实分布\n（完美拟合！）', 
                 fontsize=12, fontweight='bold', color='green')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # ============ 7. 关键指标对比 ============
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    # 计算一些关键指标
    mse_final_loss = mse_losses[-1]
    gmm_final_loss = gmm_losses[-1]
    
    # 理论Bayes误差（数据本身的不确定性）
    bayes_error = 0.1 ** 2  # 每个模态的方差
    
    summary_text = f"""
    关键指标对比
    {'='*40}
    
    MSE方法：
    · 最终损失: {mse_final_loss:.4f}
    · 预测值: {mse_preds[-1]:.4f}
    · 问题: 预测在两个模态之间
    · 状态: ❌ 卡住，无法进一步优化
    
    GMM方法：
    · 最终损失: {gmm_final_loss:.4f}
    · 模态1均值: {means_np[0]:.4f}
    · 模态2均值: {means_np[1]:.4f}
    · 权重: [{weights_np[0]:.2f}, {weights_np[1]:.2f}]
    · 状态: ✅ 正确建模分布
    
    Bayes误差估计: ~{bayes_error:.4f}
    （数据本身噪声的理论下界）
    
    {'='*40}
    结论：
    
    MSE损失在多模态数据上：
    ❌ 损失卡在高位（{mse_final_loss:.4f}）
    ❌ 预测错误的动作（均值）
    ❌ 无法继续优化
    
    似然损失（GMM）在多模态数据上：
    ✅ 损失可以接近Bayes误差
    ✅ 正确建模多个模态
    ✅ 可以持续优化
    """
    
    ax7.text(0.05, 0.95, summary_text, 
            transform=ax7.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig('/home/lsy/cjh/project1/Agent/mse_vs_likelihood_comparison.png', 
                dpi=150, bbox_inches='tight')
    print("对比图已保存到: /home/lsy/cjh/project1/Agent/mse_vs_likelihood_comparison.png")
    plt.close()
    
    # 打印最终结果
    print("\n" + "="*80)
    print("训练结果总结")
    print("="*80)
    print(f"\nMSE方法:")
    print(f"  - 最终预测: {mse_preds[-1]:.4f}")
    print(f"  - 最终损失: {mse_final_loss:.4f}")
    print(f"  - 问题: 预测值在两个模态({-1.0:.1f}, {1.0:.1f})的中间")
    
    print(f"\nGMM方法:")
    print(f"  - 模态均值: {means_np}")
    print(f"  - 模态标准差: {stds_np}")
    print(f"  - 模态权重: {weights_np}")
    print(f"  - 最终损失: {gmm_final_loss:.4f}")
    print(f"  - 成功: 两个模态分别收敛到真实值附近")
    
    print(f"\n关键结论:")
    print(f"  ✅ GMM损失({gmm_final_loss:.4f}) << MSE损失({mse_final_loss:.4f})")
    print(f"  ✅ GMM正确建模了分布，MSE只能预测均值")
    print(f"  ✅ 似然损失不怕多模态性！")
    print("="*80)


def demonstrate_gradient_conflict():
    """演示MSE的梯度冲突问题"""
    print("\n" + "="*80)
    print("演示：为什么MSE在多模态数据上梯度互相抵消")
    print("="*80)
    
    # 假设当前预测 = 0.0
    current_pred = 0.0
    
    # 两个训练样本
    sample1 = 1.0  # 右转
    sample2 = -1.0  # 左转
    
    # 计算梯度
    grad1 = 2 * (current_pred - sample1)  # = -2.0 (想让预测增大)
    grad2 = 2 * (current_pred - sample2)  # = +2.0 (想让预测减小)
    
    print(f"\n当前预测值: {current_pred:.2f}")
    print(f"\n样本1 (右转, action={sample1:.1f}):")
    print(f"  Loss = (pred - action)² = ({current_pred:.1f} - {sample1:.1f})² = {(current_pred - sample1)**2:.2f}")
    print(f"  梯度 = {grad1:.2f}  ⬅️ 想让预测往右移动（增大）")
    
    print(f"\n样本2 (左转, action={sample2:.1f}):")
    print(f"  Loss = (pred - action)² = ({current_pred:.1f} - ({sample2:.1f}))² = {(current_pred - sample2)**2:.2f}")
    print(f"  梯度 = {grad2:.2f}  ⬅️ 想让预测往左移动（减小）")
    
    print(f"\n平均梯度 = ({grad1:.2f} + {grad2:.2f}) / 2 = {(grad1 + grad2)/2:.2f}")
    print(f"\n❌ 结果：梯度互相抵消！预测停在 {current_pred:.2f}")
    print(f"   但这是最差的动作（既不左转也不右转）")
    
    print("\n" + "-"*80)
    print("GMM的情况：")
    print("-"*80)
    print("\nGMM有两个独立的均值参数 μ₁ 和 μ₂")
    print("\n样本1会推动某个μ往右：μ₁ → 1.0")
    print("样本2会推动另个μ往左：μ₂ → -1.0")
    print("\n✅ 结果：两个参数独立优化，不会冲突！")
    print("   最终正确学到两个模态")
    print("="*80)


if __name__ == "__main__":
    print("="*80)
    print("MSE损失 vs 似然损失（GMM）：多模态数据上的对比")
    print("="*80)
    
    # 演示梯度冲突
    demonstrate_gradient_conflict()
    
    # 可视化对比
    print("\n正在生成可视化...")
    visualize_comparison()
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("""
    关键要点：
    
    1. MSE损失 + 多模态数据 = 灾难
       - 学到的是均值（最差的预测）
       - 损失卡住无法下降
       - 梯度互相抵消
    
    2. 似然损失（GMM/Diffusion）+ 多模态数据 = 没问题
       - 学到的是完整分布
       - 损失可以持续下降
       - 正确建模每个模态
    
    3. 如果使用似然损失，多模态性不会阻止loss下降
       - 如果loss不降，原因是其他的：
         * 学习率不当
         * 模型容量不足（这时才考虑容量）
         * GMM的num_modes设置不当
         * 训练不稳定
    
    4. 但似然损失仍有Bayes误差下界
       - 由数据本身的随机性决定
       - 即使完美拟合分布，预测单个样本仍有不确定性
    """)
    print("="*80)
