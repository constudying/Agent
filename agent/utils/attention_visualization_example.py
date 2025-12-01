"""
注意力可视化使用示例

这个文件展示了如何使用 Transformer 模型的注意力可视化功能。

依赖安装:
    pip install matplotlib seaborn

使用方法:
    python agent/utils/attention_visualization_example.py
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agent.models.transformer import Transformer, AttentionVisualizer


def example_basic_attention_visualization():
    """基础的注意力可视化示例"""
    
    print("=" * 60)
    print("示例1: 基础注意力热力图可视化")
    print("=" * 60)
    
    # 1. 创建 Transformer 模型
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
    )
    
    # 2. 准备输入数据（示例）
    batch_size = 2
    
    # 编码器输入（根据你的配置调整）
    enc = {
        'text': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos': torch.randn(batch_size, 1, 512),
        'agentview_image': torch.randn(batch_size, 9, 512),  # 3x3=9
        'agentview_depth': torch.randn(batch_size, 9, 512),
    }
    
    # 解码器输入
    dec = {
        'text': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos_past_traj': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos_past_traj_delta': torch.randn(batch_size, 9, 512),
    }
    
    # 3. 启用注意力存储
    model.enable_attention_storage()
    
    # 4. 前向传播并获取注意力权重
    with torch.no_grad():
        output, attention_weights = model(enc, dec, return_attention_weights=True)
    
    print(f"输出形状: {output.shape}")
    print(f"编码器层数: {len(attention_weights['encoder'])}")
    print(f"解码器层数: {len(attention_weights['decoder'])}")
    
    # 5. 创建可视化器
    visualizer = AttentionVisualizer()
    
    # 6. 可视化编码器第0层的自注意力
    print("\n绘制编码器第0层的自注意力热力图...")
    if attention_weights['encoder']:
        visualizer.plot_attention_heatmap(
            attention_weights['encoder'][0]['self_attention'],
            title='Encoder Layer 0 Self-Attention',
            save_path='encoder_layer0_attention.png',
            show=False  # 设置为True会弹出窗口
        )
    
    # 7. 可视化解码器第0层的交叉注意力
    print("绘制解码器第0层的交叉注意力热力图...")
    if attention_weights['decoder'] and 'cross_attention' in attention_weights['decoder'][0]:
        visualizer.plot_attention_heatmap(
            attention_weights['decoder'][0]['cross_attention'],
            title='Decoder Layer 0 Cross-Attention',
            save_path='decoder_layer0_cross_attention.png',
            show=False
        )
    
    # 8. 保存注意力统计信息
    print("保存注意力统计信息...")
    visualizer.save_attention_statistics(
        attention_weights,
        save_path='attention_statistics.json'
    )
    
    print("\n✓ 基础可视化完成！")


def example_multi_head_visualization():
    """多头注意力可视化示例"""
    
    print("\n" + "=" * 60)
    print("示例2: 多头注意力可视化")
    print("=" * 60)
    
    # 创建模型
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    
    # 准备输入
    batch_size = 1
    enc = {
        'text': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos': torch.randn(batch_size, 1, 512),
        'agentview_image': torch.randn(batch_size, 9, 512),
        'agentview_depth': torch.randn(batch_size, 9, 512),
    }
    dec = {
        'text': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos_past_traj': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos_past_traj_delta': torch.randn(batch_size, 9, 512),
    }
    
    # 启用注意力存储并前向传播
    model.enable_attention_storage()
    with torch.no_grad():
        output, attention_weights = model(enc, dec, return_attention_weights=True)
    
    # 可视化所有注意力头
    visualizer = AttentionVisualizer()
    
    print("绘制编码器第0层的所有注意力头...")
    if attention_weights['encoder']:
        visualizer.plot_multi_head_attention(
            attention_weights['encoder'][0]['self_attention'],
            title='Encoder Layer 0 - All Attention Heads',
            save_path='encoder_multi_head.png',
            show=False
        )
    
    print("\n✓ 多头可视化完成！")


def example_layer_comparison():
    """层级对比可视化示例"""
    
    print("\n" + "=" * 60)
    print("示例3: 层级注意力对比")
    print("=" * 60)
    
    # 创建模型
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
    )
    
    # 准备输入
    batch_size = 1
    enc = {
        'text': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos': torch.randn(batch_size, 1, 512),
        'agentview_image': torch.randn(batch_size, 9, 512),
        'agentview_depth': torch.randn(batch_size, 9, 512),
    }
    dec = {
        'text': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos_past_traj': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos_past_traj_delta': torch.randn(batch_size, 9, 512),
    }
    
    # 启用注意力存储并前向传播
    model.enable_attention_storage()
    with torch.no_grad():
        output, attention_weights = model(enc, dec, return_attention_weights=True)
    
    # 对比所有编码器层
    visualizer = AttentionVisualizer()
    
    print("对比所有编码器层的自注意力...")
    visualizer.plot_layer_comparison(
        attention_weights['encoder'],
        attention_type='self_attention',
        title='Encoder Self-Attention Across Layers',
        save_path='encoder_layer_comparison.png',
        show=False
    )
    
    print("对比所有解码器层的交叉注意力...")
    visualizer.plot_layer_comparison(
        attention_weights['decoder'],
        attention_type='cross_attention',
        title='Decoder Cross-Attention Across Layers',
        save_path='decoder_cross_attention_comparison.png',
        show=False
    )
    
    print("\n✓ 层级对比完成！")


def example_attention_flow():
    """完整注意力流可视化示例"""
    
    print("\n" + "=" * 60)
    print("示例4: 完整注意力流可视化")
    print("=" * 60)
    
    # 创建模型
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    
    # 准备输入
    batch_size = 1
    enc = {
        'text': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos': torch.randn(batch_size, 1, 512),
        'agentview_image': torch.randn(batch_size, 9, 512),
        'agentview_depth': torch.randn(batch_size, 9, 512),
    }
    dec = {
        'text': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos_past_traj': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos_past_traj_delta': torch.randn(batch_size, 9, 512),
    }
    
    # 启用注意力存储并前向传播
    model.enable_attention_storage()
    with torch.no_grad():
        output, attention_weights = model(enc, dec, return_attention_weights=True)
    
    # 可视化完整的注意力流
    visualizer = AttentionVisualizer()
    
    layer_idx = 0
    print(f"可视化第{layer_idx}层的注意力流...")
    
    encoder_attn = attention_weights['encoder'][layer_idx]['self_attention'] if attention_weights['encoder'] else None
    decoder_self_attn = attention_weights['decoder'][layer_idx]['self_attention'] if attention_weights['decoder'] else None
    decoder_cross_attn = attention_weights['decoder'][layer_idx].get('cross_attention', None) if attention_weights['decoder'] else None
    
    visualizer.plot_attention_flow(
        encoder_attention=encoder_attn,
        decoder_self_attention=decoder_self_attn,
        decoder_cross_attention=decoder_cross_attn,
        layer_idx=layer_idx,
        save_path=f'attention_flow_layer{layer_idx}.png',
        show=False
    )
    
    print("\n✓ 注意力流可视化完成！")


def example_custom_visualization():
    """自定义可视化参数示例"""
    
    print("\n" + "=" * 60)
    print("示例5: 自定义可视化参数")
    print("=" * 60)
    
    # 创建模型
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    
    # 准备输入
    batch_size = 2
    enc = {
        'text': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos': torch.randn(batch_size, 1, 512),
        'agentview_image': torch.randn(batch_size, 9, 512),
        'agentview_depth': torch.randn(batch_size, 9, 512),
    }
    dec = {
        'text': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos_past_traj': torch.randn(batch_size, 10, 512),
        'robot0_eef_pos_past_traj_delta': torch.randn(batch_size, 9, 512),
    }
    
    # 启用注意力存储并前向传播
    model.enable_attention_storage()
    with torch.no_grad():
        output, attention_weights = model(enc, dec, return_attention_weights=True)
    
    visualizer = AttentionVisualizer()
    
    # 自定义可视化参数
    print("使用自定义参数绘制热力图...")
    if attention_weights['encoder']:
        visualizer.plot_attention_heatmap(
            attention_weights['encoder'][0]['self_attention'],
            head_idx=3,  # 只看第3个头
            batch_idx=1,  # 看第二个样本
            title='Custom: Encoder Layer 0, Head 3, Batch 1',
            figsize=(12, 10),
            cmap='hot',  # 使用不同的颜色映射
            vmin=0.0,    # 固定颜色范围
            vmax=0.1,
            save_path='custom_attention_vis.png',
            show=False
        )
    
    print("\n✓ 自定义可视化完成！")


def print_attention_info(attention_weights):
    """打印注意力权重的详细信息"""
    
    print("\n" + "=" * 60)
    print("注意力权重信息")
    print("=" * 60)
    
    # 编码器信息
    print("\n编码器:")
    for i, layer_attn in enumerate(attention_weights['encoder']):
        if 'self_attention' in layer_attn:
            shape = layer_attn['self_attention'].shape
            print(f"  Layer {i} - Self-Attention: {shape}")
            print(f"    - Batch Size: {shape[0]}")
            print(f"    - Num Heads: {shape[1]}")
            print(f"    - Query Length: {shape[2]}")
            print(f"    - Key Length: {shape[3]}")
    
    # 解码器信息
    print("\n解码器:")
    for i, layer_attn in enumerate(attention_weights['decoder']):
        print(f"  Layer {i}:")
        if 'self_attention' in layer_attn:
            shape = layer_attn['self_attention'].shape
            print(f"    - Self-Attention: {shape}")
        if 'cross_attention' in layer_attn:
            shape = layer_attn['cross_attention'].shape
            print(f"    - Cross-Attention: {shape}")
            print(f"      (Query from decoder: {shape[2]}, Key from encoder: {shape[3]})")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Transformer 注意力可视化示例")
    print("=" * 60)
    
    try:
        # 运行所有示例
        example_basic_attention_visualization()
        example_multi_head_visualization()
        example_layer_comparison()
        example_attention_flow()
        example_custom_visualization()
        
        print("\n" + "=" * 60)
        print("所有示例执行完成！")
        print("生成的图片已保存到当前目录")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
