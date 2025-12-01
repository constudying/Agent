"""
训练时注意力监控的完整示例

展示如何在训练循环中集成注意力监控，同时保持训练性能。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agent.models.transformer import Transformer
from agent.utils.training_attention_monitor import (
    TrainingAttentionMonitor,
    LightweightAttentionMonitor
)


def example_training_with_attention_monitoring():
    """
    完整的训练示例，展示如何集成注意力监控
    """
    
    print("=" * 70)
    print("训练时注意力监控示例")
    print("=" * 70)
    
    # 1. 创建模型
    print("\n1. 创建模型...")
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
    )
    
    # 如果有GPU，使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   模型已创建，使用设备: {device}")
    
    # 2. 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 3. 创建注意力监控器
    print("\n2. 创建注意力监控器...")
    
    # 方式1: 完整的可视化监控（生成图像）
    attention_monitor = TrainingAttentionMonitor(
        save_dir='./training_attention_logs',
        save_frequency=10,  # 每10个step保存一次
        save_on_epochs=[1, 5, 10],  # 在特定epoch结束时保存
        visualization_types=['heatmap', 'statistics'],  # 只生成热力图和统计
        max_workers=2,  # 2个后台线程
        use_tensorboard=True,  # 使用TensorBoard
        batch_idx_to_visualize=0,  # 只可视化第一个样本
        layers_to_visualize=[0, -1],  # 只可视化第一层和最后一层
    )
    
    # 方式2: 轻量级监控（只记录统计，不生成图像）
    # attention_monitor = LightweightAttentionMonitor(
    #     save_dir='./training_attention_stats',
    #     save_frequency=10,
    #     use_tensorboard=True
    # )
    
    print("   ✓ 注意力监控器已创建")
    
    # 4. 模拟训练循环
    print("\n3. 开始训练...")
    num_epochs = 3
    steps_per_epoch = 20
    
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        model.train()
        epoch_loss = 0.0
        
        for step in range(steps_per_epoch):
            # 准备批次数据（模拟）
            batch_size = 4
            enc = {
                'text': torch.randn(batch_size, 10, 512).to(device),
                'robot0_eef_pos': torch.randn(batch_size, 1, 512).to(device),
                'agentview_image': torch.randn(batch_size, 9, 512).to(device),
                'agentview_depth': torch.randn(batch_size, 9, 512).to(device),
            }
            dec = {
                'text': torch.randn(batch_size, 10, 512).to(device),
                'robot0_eef_pos_past_traj': torch.randn(batch_size, 10, 512).to(device),
                'robot0_eef_pos_past_traj_delta': torch.randn(batch_size, 9, 512).to(device),
            }
            target = torch.randn(batch_size, 29, 512).to(device)
            
            # 前向传播（正常训练，不收集注意力）
            output = model(enc, dec)
            loss = criterion(output, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # ====== 关键：定期记录注意力（不影响训练） ======
            # 监控器会自动判断是否需要保存，并在后台异步处理
            attention_monitor.log_attention(
                model=model,
                enc=enc,
                dec=dec,
                step=global_step,
                loss=loss.item(),
                epoch=epoch + 1,
                metrics={'lr': optimizer.param_groups[0]['lr']}
            )
            
            # 打印进度
            if (step + 1) % 5 == 0:
                avg_loss = epoch_loss / (step + 1)
                print(f"  Step {step + 1}/{steps_per_epoch} | Loss: {avg_loss:.4f}")
        
        # Epoch结束
        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch + 1} 完成 | 平均Loss: {avg_epoch_loss:.4f}")
        
        # 在特定epoch结束时保存注意力
        if (epoch + 1) in attention_monitor.save_on_epochs:
            print(f"  → 保存Epoch {epoch + 1}的注意力可视化...")
            attention_monitor.log_attention(
                model=model,
                enc=enc,
                dec=dec,
                step=global_step,
                loss=avg_epoch_loss,
                epoch=epoch + 1
            )
    
    # 5. 训练结束，关闭监控器
    print("\n4. 训练完成，关闭监控器...")
    attention_monitor.close()
    
    print("\n" + "=" * 70)
    print("✓ 训练和注意力监控完成！")
    print("=" * 70)
    print(f"\n查看结果:")
    print(f"  - 注意力可视化: ./training_attention_logs/")
    print(f"  - TensorBoard: tensorboard --logdir=./training_attention_logs/tensorboard")
    print("=" * 70)


def example_lightweight_monitoring():
    """
    轻量级监控示例（只记录统计，不生成图像）
    适合长时间训练
    """
    
    print("\n" + "=" * 70)
    print("轻量级注意力监控示例")
    print("=" * 70)
    
    # 创建模型
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 使用轻量级监控器
    with LightweightAttentionMonitor(
        save_dir='./lightweight_attention_stats',
        save_frequency=5,
        use_tensorboard=True
    ) as monitor:
        
        print("\n使用轻量级监控器进行快速训练...")
        
        for step in range(50):
            # 准备数据
            batch_size = 4
            enc = {
                'text': torch.randn(batch_size, 10, 512).to(device),
                'robot0_eef_pos': torch.randn(batch_size, 1, 512).to(device),
                'agentview_image': torch.randn(batch_size, 9, 512).to(device),
                'agentview_depth': torch.randn(batch_size, 9, 512).to(device),
            }
            dec = {
                'text': torch.randn(batch_size, 10, 512).to(device),
                'robot0_eef_pos_past_traj': torch.randn(batch_size, 10, 512).to(device),
                'robot0_eef_pos_past_traj_delta': torch.randn(batch_size, 9, 512).to(device),
            }
            target = torch.randn(batch_size, 29, 512).to(device)
            
            # 训练
            output = model(enc, dec)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录注意力统计（非常快，几乎不影响训练）
            monitor.log_attention(model, enc, dec, step, loss.item())
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/50 | Loss: {loss.item():.4f}")
    
    print("\n✓ 轻量级监控完成！")
    print(f"  统计数据: ./lightweight_attention_stats/attention_stats.json")


def example_custom_monitoring():
    """
    自定义监控策略示例
    """
    
    print("\n" + "=" * 70)
    print("自定义监控策略示例")
    print("=" * 70)
    
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 创建多个监控器用于不同目的
    
    # 1. 详细监控（低频率，生成所有可视化）
    detailed_monitor = TrainingAttentionMonitor(
        save_dir='./detailed_attention',
        save_frequency=100,  # 每100步
        visualization_types=['heatmap', 'multi_head', 'layer_comparison', 'statistics'],
        layers_to_visualize=None,  # 所有层
    )
    
    # 2. 快速监控（高频率，只记录统计）
    quick_monitor = LightweightAttentionMonitor(
        save_dir='./quick_attention_stats',
        save_frequency=10,  # 每10步
        use_tensorboard=True
    )
    
    print("\n使用多个监控器进行训练...")
    
    try:
        for step in range(30):
            # 准备数据
            batch_size = 4
            enc = {
                'text': torch.randn(batch_size, 10, 512).to(device),
                'robot0_eef_pos': torch.randn(batch_size, 1, 512).to(device),
                'agentview_image': torch.randn(batch_size, 9, 512).to(device),
                'agentview_depth': torch.randn(batch_size, 9, 512).to(device),
            }
            dec = {
                'text': torch.randn(batch_size, 10, 512).to(device),
                'robot0_eef_pos_past_traj': torch.randn(batch_size, 10, 512).to(device),
                'robot0_eef_pos_past_traj_delta': torch.randn(batch_size, 9, 512).to(device),
            }
            target = torch.randn(batch_size, 29, 512).to(device)
            
            # 训练
            output = model(enc, dec)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 使用两个监控器
            detailed_monitor.log_attention(model, enc, dec, step, loss.item())
            quick_monitor.log_attention(model, enc, dec, step, loss.item())
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/30 | Loss: {loss.item():.4f}")
    
    finally:
        # 确保关闭
        detailed_monitor.close()
        quick_monitor.close()
    
    print("\n✓ 自定义监控完成！")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练时注意力监控示例')
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'lightweight', 'custom'],
        help='监控模式: full(完整可视化), lightweight(只统计), custom(自定义)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'full':
            example_training_with_attention_monitoring()
        elif args.mode == 'lightweight':
            example_lightweight_monitoring()
        elif args.mode == 'custom':
            example_custom_monitoring()
        
        print("\n" + "=" * 70)
        print("所有示例执行完成！")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n用户中断训练")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
