"""
快速判断：训练是否接近Bayes误差

这个脚本提供快速的命令行工具来判断训练状态
"""

import argparse
import numpy as np


def quick_diagnosis(training_loss, bayes_error=None, 
                   train_loss=None, val_loss=None,
                   action_variance=None):
    """
    快速诊断训练状态
    
    Args:
        training_loss: 主要的训练损失（通常用验证集损失）
        bayes_error: 估算的Bayes误差（如果已知）
        train_loss: 训练集损失（用于检查过拟合）
        val_loss: 验证集损失（用于检查过拟合）
        action_variance: 数据中动作的方差（用于粗略估计）
    """
    
    print("\n" + "="*80)
    print("训练状态快速诊断")
    print("="*80)
    
    # 如果没有Bayes误差，用动作方差粗略估计
    if bayes_error is None and action_variance is not None:
        # 粗略估计：Bayes误差约为动作方差的5-10%
        bayes_error = action_variance * 0.075
        print(f"\n⚠️ 未提供Bayes误差，使用粗略估计:")
        print(f"   动作方差 = {action_variance:.6f}")
        print(f"   估计Bayes误差 ≈ {bayes_error:.6f} (方差的7.5%)")
        print(f"   建议：运行 estimate_bayes_error.py 获取准确估计")
    
    if bayes_error is None:
        print("\n❌ 错误：需要提供 bayes_error 或 action_variance")
        return
    
    print(f"\n📊 当前指标:")
    print(f"   训练损失: {training_loss:.6f}")
    if train_loss and val_loss:
        print(f"   训练集损失: {train_loss:.6f}")
        print(f"   验证集损失: {val_loss:.6f}")
    print(f"   Bayes误差: {bayes_error:.6f}")
    
    # 计算比值
    ratio = training_loss / bayes_error
    gap = training_loss - bayes_error
    gap_percentage = (gap / bayes_error) * 100
    
    print(f"\n📈 对比分析:")
    print(f"   损失/Bayes比值: {ratio:.2f}x")
    print(f"   差距: {gap:.6f} ({gap_percentage:.1f}%)")
    
    # 检查过拟合
    if train_loss and val_loss:
        overfitting_gap = val_loss - train_loss
        overfitting_ratio = val_loss / train_loss
        print(f"   训练-验证差距: {overfitting_gap:.6f} ({(overfitting_ratio-1)*100:.1f}%)")
        
        if overfitting_ratio > 1.15:
            print(f"   ⚠️ 警告：可能存在过拟合")
    
    # 诊断
    print(f"\n{'='*80}")
    print("🔍 诊断结果:")
    print(f"{'='*80}\n")
    
    if ratio < 1.1:
        print("✅ 状态：优秀 - 已接近理论极限")
        print(f"   · 训练损失仅比Bayes误差高{gap_percentage:.0f}%")
        print(f"   · 模型已经学得很好")
        print(f"   · 继续训练提升空间极小\n")
        
        print("💡 建议：")
        print("   1. ✅ 当前模型已经很好，可以停止训练")
        print("   2. 🔍 如果任务性能仍不满意，问题在数据而非模型：")
        print("      - 检查状态表示是否充分（是否缺少关键信息）")
        print("      - 增加数据多样性和质量")
        print("      - 考虑增加时序上下文长度")
        print("   3. 📊 评估模型在真实任务上的表现")
        print("      - 损失低不一定任务成功率高")
        print("      - 可能存在causal confusion")
        
    elif ratio < 1.5:
        print("⚠️ 状态：良好 - 接近但未达极限")
        print(f"   · 训练损失比Bayes误差高{gap_percentage:.0f}%")
        print(f"   · 还有小幅提升空间（约{(1.1/ratio - 1)*100:.0f}%到最优）")
        print(f"   · 大部分优化潜力已经实现\n")
        
        print("💡 建议：")
        print("   1. 🔄 尝试继续训练（但提升可能有限）")
        print("      - 确保学习率已经衰减到足够小")
        print("      - 可以延长训练轮次")
        print("   2. 🎯 微调训练策略：")
        print("      - 使用更小的学习率（当前的1/10）")
        print("      - 检查是否正则化过强（降低dropout/weight_decay）")
        print("      - 尝试warmup + cosine衰减学习率")
        print("   3. 🚀 尝试模型集成（ensemble）")
        print("      - 训练多个模型取平均")
        print("      - 可能带来1-3%提升")
        
    elif ratio < 2.0:
        print("⚠️ 状态：中等 - 有明显优化空间")
        print(f"   · 训练损失比Bayes误差高{gap_percentage:.0f}%")
        print(f"   · 还有{(1.1/ratio - 1)*100:.0f}%的优化潜力")
        print(f"   · 模型可能欠拟合或训练不充分\n")
        
        print("💡 建议（按优先级）：")
        print("   1. 📚 增加训练轮次")
        print("      - 当前可能训练不充分")
        print("      - 将epochs增加1.5-2倍")
        print("   2. 🔧 检查学习率：")
        print("      - 学习率可能太小（收敛慢）")
        print("      - 或太大（无法细调）")
        print("      - 使用学习率调度器")
        print("   3. 🏗️ 增大模型容量：")
        print("      - 增加网络层数或宽度")
        print("      - 增加Transformer的层数和注意力头")
        print("   4. 🎛️ 减少正则化：")
        print("      - 降低dropout（如0.1→0.05）")
        print("      - 降低weight decay")
        print("   5. 🔍 检查损失函数：")
        print("      - 如果数据有多模态，MSE不合适")
        print("      - 运行多模态检测：check_multimodality.py")
        print("      - 考虑使用GMM或Diffusion")
        
    else:
        print("❌ 状态：差 - 远未达到理论极限")
        print(f"   · 训练损失是Bayes误差的{ratio:.1f}倍")
        print(f"   · 还有{gap_percentage:.0f}%的巨大提升空间")
        print(f"   · 可能存在严重问题\n")
        
        print("🚨 需要深入排查（按顺序）：")
        print("\n   第一步：排除Bug")
        print("   ----------------")
        print("   1. 检查梯度是否正常")
        print("      - 打印梯度范数，确保不是0或NaN")
        print("      - 检查是否有梯度消失/爆炸")
        print("   2. 确认损失是否在下降")
        print("      - 如果完全不降→代码有bug")
        print("      - 如果降很慢→学习率或优化器问题")
        print("   3. 验证数据加载")
        print("      - 打印几个batch，确保数据正确")
        print("      - 检查数据归一化")
        
        print("\n   第二步：检查多模态性")
        print("   ----------------")
        print("   4. 运行多模态检测：")
        print("      python agent/scripts/check_multimodality.py --dataset data.hdf5")
        print("   5. 如果检测到多模态：")
        print("      - MSE损失不适合！这是主要原因")
        print("      - 改用GMM：设置 gmm.enabled=true")
        print("      - 或使用Diffusion Policy")
        
        print("\n   第三步：增加模型容量")
        print("   ----------------")
        print("   6. 当前模型可能严重不足")
        print("      - 将网络层数/宽度翻倍")
        print("      - 增加Transformer层数（4→8）")
        print("      - 增加注意力头数（4→8）")
        
        print("\n   第四步：改进训练方法")
        print("   ----------------")
        print("   7. 使用更好的优化器")
        print("      - 尝试AdamW而不是Adam")
        print("      - 调整学习率（试试1e-4, 1e-3）")
        print("   8. 增加训练时长")
        print("      - epochs可能需要增加5-10倍")
        print("   9. 检查输入特征")
        print("      - 是否包含足够信息？")
        print("      - 是否需要增加时序上下文？")
    
    # 过拟合检查
    if train_loss and val_loss and overfitting_ratio > 1.15:
        print(f"\n{'='*80}")
        print("⚠️ 过拟合警告")
        print(f"{'='*80}")
        print(f"   验证损失比训练损失高{(overfitting_ratio-1)*100:.1f}%")
        print(f"\n   建议：")
        print(f"   1. 增加正则化（dropout, weight decay）")
        print(f"   2. 增加训练数据")
        print(f"   3. 使用数据增强")
        print(f"   4. 减小模型容量")
        print(f"   5. Early stopping")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="快速诊断训练是否接近Bayes误差"
    )
    parser.add_argument(
        "--training_loss",
        type=float,
        required=True,
        help="当前训练损失（建议使用验证集损失）"
    )
    parser.add_argument(
        "--bayes_error",
        type=float,
        default=None,
        help="估算的Bayes误差（运行estimate_bayes_error.py获取）"
    )
    parser.add_argument(
        "--action_variance",
        type=float,
        default=None,
        help="数据中动作的方差（用于粗略估计Bayes误差）"
    )
    parser.add_argument(
        "--train_loss",
        type=float,
        default=None,
        help="训练集损失（用于检查过拟合）"
    )
    parser.add_argument(
        "--val_loss",
        type=float,
        default=None,
        help="验证集损失（用于检查过拟合）"
    )
    
    args = parser.parse_args()
    
    quick_diagnosis(
        training_loss=args.training_loss,
        bayes_error=args.bayes_error,
        train_loss=args.train_loss,
        val_loss=args.val_loss,
        action_variance=args.action_variance
    )
    
    print("📖 更多信息：")
    print("   - 准确估计Bayes误差：python agent/scripts/estimate_bayes_error.py")
    print("   - 检查多模态性：python agent/scripts/check_multimodality.py")
    print("   - 查看完整文档：MULTIMODALITY_EXPLAINED.md")
    print()


if __name__ == "__main__":
    # 如果直接运行，显示使用示例
    import sys
    if len(sys.argv) == 1:
        print("\n" + "="*80)
        print("快速诊断工具 - 使用示例")
        print("="*80)
        
        print("\n示例1: 提供Bayes误差（推荐）")
        print("-"*80)
        print("python quick_diagnosis.py \\")
        print("    --training_loss 0.025 \\")
        print("    --bayes_error 0.018")
        
        print("\n示例2: 只提供损失，用动作方差估计")
        print("-"*80)
        print("python quick_diagnosis.py \\")
        print("    --training_loss 0.025 \\")
        print("    --action_variance 0.24")
        
        print("\n示例3: 同时检查过拟合")
        print("-"*80)
        print("python quick_diagnosis.py \\")
        print("    --training_loss 0.025 \\")
        print("    --bayes_error 0.018 \\")
        print("    --train_loss 0.022 \\")
        print("    --val_loss 0.028")
        
        print("\n" + "="*80)
        print("获取Bayes误差的方法：")
        print("="*80)
        print("python agent/scripts/estimate_bayes_error.py \\")
        print("    --dataset your_data.hdf5 \\")
        print("    --k 10")
        print()
        
    else:
        main()
