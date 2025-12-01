"""
如何观察编码器和解码器的注意力图 - 完整指南

这个脚本提供了多种方式来可视化和分析 Transformer 模型的注意力机制。
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agent.models.transformer import Transformer, AttentionVisualizer


def method1_basic_visualization():
    """
    方法1: 基础可视化 - 单个注意力图
    
    适用场景：快速查看某一层的注意力分布
    """
    print("=" * 70)
    print("方法1: 基础注意力可视化")
    print("=" * 70)
    
    # 1. 创建模型
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
    )
    
    # 2. 准备数据
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
    
    # 3. 启用注意力存储并获取
    model.enable_attention_storage()
    with torch.no_grad():
        output, attention_weights = model(enc, dec, return_attention_weights=True)
    model.disable_attention_storage()
    
    # 4. 创建可视化器
    visualizer = AttentionVisualizer()
    
    # 5. 可视化编码器自注意力（第0层）
    print("\n观察编码器第0层的自注意力:")
    print(f"  - 形状: {attention_weights['encoder'][0]['self_attention'].shape}")
    print(f"  - 含义: (batch_size, num_heads, seq_len, seq_len)")
    
    visualizer.plot_attention_heatmap(
        attention_weights['encoder'][0]['self_attention'],
        head_idx=None,  # 平均所有头
        batch_idx=0,    # 第一个样本
        title='编码器第0层 - 自注意力',
        save_path='encoder_layer0_self_attention.png',
        show=False
    )
    print("  ✓ 已保存: encoder_layer0_self_attention.png")
    
    # 6. 可视化解码器交叉注意力（第0层）
    print("\n观察解码器第0层的交叉注意力:")
    print(f"  - 形状: {attention_weights['decoder'][0]['cross_attention'].shape}")
    print(f"  - 含义: (batch_size, num_heads, decoder_len, encoder_len)")
    
    visualizer.plot_attention_heatmap(
        attention_weights['decoder'][0]['cross_attention'],
        head_idx=None,
        batch_idx=0,
        title='解码器第0层 - 交叉注意力（关注编码器）',
        save_path='decoder_layer0_cross_attention.png',
        show=False
    )
    print("  ✓ 已保存: decoder_layer0_cross_attention.png")
    
    # 7. 可视化解码器自注意力（第0层）
    print("\n观察解码器第0层的自注意力:")
    print(f"  - 形状: {attention_weights['decoder'][0]['self_attention'].shape}")
    
    visualizer.plot_attention_heatmap(
        attention_weights['decoder'][0]['self_attention'],
        head_idx=None,
        batch_idx=0,
        title='解码器第0层 - 自注意力（因果mask）',
        save_path='decoder_layer0_self_attention.png',
        show=False
    )
    print("  ✓ 已保存: decoder_layer0_self_attention.png")
    
    print("\n" + "=" * 70)


def method2_compare_all_layers():
    """
    方法2: 对比所有层 - 观察注意力随深度的变化
    
    适用场景：分析不同层的注意力模式
    """
    print("\n" + "=" * 70)
    print("方法2: 对比所有层的注意力")
    print("=" * 70)
    
    # 准备模型和数据
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
    )
    
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
    
    # 获取注意力
    model.enable_attention_storage()
    with torch.no_grad():
        output, attention_weights = model(enc, dec, return_attention_weights=True)
    model.disable_attention_storage()
    
    visualizer = AttentionVisualizer()
    
    # 对比编码器所有层
    print("\n生成编码器层级对比图...")
    print(f"  - 编码器层数: {len(attention_weights['encoder'])}")
    
    visualizer.plot_layer_comparison(
        attention_weights['encoder'],
        attention_type='self_attention',
        batch_idx=0,
        title='编码器自注意力 - 跨层对比',
        save_path='encoder_all_layers_comparison.png',
        show=False
    )
    print("  ✓ 已保存: encoder_all_layers_comparison.png")
    
    # 对比解码器交叉注意力
    print("\n生成解码器交叉注意力层级对比图...")
    print(f"  - 解码器层数: {len(attention_weights['decoder'])}")
    
    visualizer.plot_layer_comparison(
        attention_weights['decoder'],
        attention_type='cross_attention',
        batch_idx=0,
        title='解码器交叉注意力 - 跨层对比',
        save_path='decoder_all_layers_cross_comparison.png',
        show=False
    )
    print("  ✓ 已保存: decoder_all_layers_cross_comparison.png")
    
    print("\n" + "=" * 70)


def method3_multi_head_analysis():
    """
    方法3: 多头注意力分析 - 观察不同头的关注点
    
    适用场景：理解多头机制，每个头学到了什么
    """
    print("\n" + "=" * 70)
    print("方法3: 多头注意力分析")
    print("=" * 70)
    
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    
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
    
    model.enable_attention_storage()
    with torch.no_grad():
        output, attention_weights = model(enc, dec, return_attention_weights=True)
    model.disable_attention_storage()
    
    visualizer = AttentionVisualizer()
    
    # 可视化编码器第0层的所有8个头
    print("\n并排显示编码器第0层的所有注意力头:")
    num_heads = attention_weights['encoder'][0]['self_attention'].shape[1]
    print(f"  - 注意力头数: {num_heads}")
    
    visualizer.plot_multi_head_attention(
        attention_weights['encoder'][0]['self_attention'],
        batch_idx=0,
        title='编码器第0层 - 全部8个注意力头',
        save_path='encoder_layer0_all_heads.png',
        show=False
    )
    print("  ✓ 已保存: encoder_layer0_all_heads.png")
    
    # 可视化解码器交叉注意力的所有头
    print("\n并排显示解码器第0层交叉注意力的所有头:")
    visualizer.plot_multi_head_attention(
        attention_weights['decoder'][0]['cross_attention'],
        batch_idx=0,
        title='解码器第0层交叉注意力 - 全部8个头',
        save_path='decoder_layer0_cross_all_heads.png',
        show=False
    )
    print("  ✓ 已保存: decoder_layer0_cross_all_heads.png")
    
    print("\n" + "=" * 70)


def method4_attention_flow():
    """
    方法4: 完整注意力流 - 从编码器到解码器
    
    适用场景：理解信息如何从输入流向输出
    """
    print("\n" + "=" * 70)
    print("方法4: 完整注意力流可视化")
    print("=" * 70)
    
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    
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
    
    model.enable_attention_storage()
    with torch.no_grad():
        output, attention_weights = model(enc, dec, return_attention_weights=True)
    model.disable_attention_storage()
    
    visualizer = AttentionVisualizer()
    
    # 可视化第0层的完整注意力流
    layer_idx = 0
    print(f"\n可视化第{layer_idx}层的完整注意力流:")
    print("  编码器自注意力 → 解码器自注意力 → 解码器交叉注意力")
    
    visualizer.plot_attention_flow(
        encoder_attention=attention_weights['encoder'][layer_idx]['self_attention'],
        decoder_self_attention=attention_weights['decoder'][layer_idx]['self_attention'],
        decoder_cross_attention=attention_weights['decoder'][layer_idx]['cross_attention'],
        layer_idx=layer_idx,
        batch_idx=0,
        save_path=f'attention_flow_layer{layer_idx}.png',
        show=False
    )
    print(f"  ✓ 已保存: attention_flow_layer{layer_idx}.png")
    
    print("\n" + "=" * 70)


def method5_analyze_specific_positions():
    """
    方法5: 分析特定位置的注意力 - 查看某个token关注了什么
    
    适用场景：深入分析某个特定位置的注意力分布
    """
    print("\n" + "=" * 70)
    print("方法5: 分析特定位置的注意力")
    print("=" * 70)
    
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    
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
    
    model.enable_attention_storage()
    with torch.no_grad():
        output, attention_weights = model(enc, dec, return_attention_weights=True)
    model.disable_attention_storage()
    
    # 分析解码器交叉注意力
    cross_attn = attention_weights['decoder'][0]['cross_attention']
    batch_idx = 0
    
    print("\n解码器交叉注意力分析:")
    print(f"  形状: {cross_attn.shape}")
    print(f"  (batch_size={cross_attn.shape[0]}, "
          f"num_heads={cross_attn.shape[1]}, "
          f"decoder_len={cross_attn.shape[2]}, "
          f"encoder_len={cross_attn.shape[3]})")
    
    # 查看解码器某个位置关注编码器的哪些位置
    decoder_pos = 5  # 解码器第5个位置
    print(f"\n解码器位置 {decoder_pos} 关注编码器的注意力分布:")
    
    # 平均所有头
    attn_at_pos = cross_attn[batch_idx, :, decoder_pos, :].mean(dim=0)
    print(f"  平均注意力权重: {attn_at_pos}")
    
    # 找出最关注的top-5位置
    top_k = 5
    top_values, top_indices = torch.topk(attn_at_pos, top_k)
    print(f"\n  Top-{top_k} 最关注的编码器位置:")
    for i, (idx, val) in enumerate(zip(top_indices, top_values)):
        print(f"    {i+1}. 编码器位置 {idx.item()}: 权重 {val.item():.4f}")
    
    # 分析注意力的熵（分散程度）
    import torch.nn.functional as F
    entropy = -(attn_at_pos * torch.log(attn_at_pos + 1e-10)).sum()
    print(f"\n  注意力熵: {entropy.item():.4f}")
    print(f"    (熵越高，注意力越分散；熵越低，注意力越集中)")
    
    print("\n" + "=" * 70)


def method6_save_statistics():
    """
    方法6: 保存统计信息 - 数值分析
    
    适用场景：需要定量分析注意力分布
    """
    print("\n" + "=" * 70)
    print("方法6: 保存注意力统计信息")
    print("=" * 70)
    
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
    )
    
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
    
    model.enable_attention_storage()
    with torch.no_grad():
        output, attention_weights = model(enc, dec, return_attention_weights=True)
    model.disable_attention_storage()
    
    visualizer = AttentionVisualizer()
    
    # 保存统计信息到JSON文件
    print("\n保存注意力统计信息...")
    visualizer.save_attention_statistics(
        attention_weights,
        save_path='attention_statistics.json',
        batch_idx=0
    )
    print("  ✓ 已保存: attention_statistics.json")
    
    # 打印一些统计信息
    print("\n编码器注意力统计:")
    for i, layer_attn in enumerate(attention_weights['encoder']):
        attn = layer_attn['self_attention'][0].cpu().numpy()
        print(f"  Layer {i}:")
        print(f"    - 平均值: {attn.mean():.6f}")
        print(f"    - 标准差: {attn.std():.6f}")
        print(f"    - 最大值: {attn.max():.6f}")
    
    print("\n解码器交叉注意力统计:")
    for i, layer_attn in enumerate(attention_weights['decoder']):
        if 'cross_attention' in layer_attn:
            attn = layer_attn['cross_attention'][0].cpu().numpy()
            print(f"  Layer {i}:")
            print(f"    - 平均值: {attn.mean():.6f}")
            print(f"    - 标准差: {attn.std():.6f}")
    
    print("\n" + "=" * 70)


def quick_visualization_template():
    """
    快速可视化模板 - 复制这段代码到你的项目中
    """
    print("\n" + "=" * 70)
    print("快速可视化模板（可直接复制使用）")
    print("=" * 70)
    
    template_code = '''
# ========== 快速可视化模板 ==========
from agent.models.transformer import Transformer, AttentionVisualizer
import torch

# 1. 准备模型和数据
model = Transformer(...)  # 你的模型
enc = {...}  # 编码器输入
dec = {...}  # 解码器输入

# 2. 获取注意力
model.enable_attention_storage()
with torch.no_grad():
    output, attn = model(enc, dec, return_attention_weights=True)
model.disable_attention_storage()

# 3. 可视化
visualizer = AttentionVisualizer()

# 编码器自注意力
visualizer.plot_attention_heatmap(
    attn['encoder'][0]['self_attention'],
    title='Encoder Self-Attention',
    save_path='encoder_attn.png'
)

# 解码器交叉注意力
visualizer.plot_attention_heatmap(
    attn['decoder'][0]['cross_attention'],
    title='Decoder Cross-Attention',
    save_path='decoder_cross_attn.png'
)
'''
    
    print(template_code)
    print("\n" + "=" * 70)


def print_attention_structure_guide():
    """
    打印注意力数据结构说明
    """
    print("\n" + "=" * 70)
    print("注意力数据结构说明")
    print("=" * 70)
    
    guide = """
attention_weights = {
    'encoder': [
        {
            'self_attention': Tensor(B, NH, T, T)
            # B: batch_size
            # NH: num_heads (注意力头数)
            # T: encoder序列长度
            # 含义: 编码器中每个位置对其他位置的注意力
        },
        # ... 更多层
    ],
    'decoder': [
        {
            'self_attention': Tensor(B, NH, T, T),
            # T: decoder序列长度
            # 含义: 解码器中每个位置对之前位置的注意力（因果mask）
            
            'cross_attention': Tensor(B, NH, T_dec, T_enc)
            # T_dec: decoder序列长度
            # T_enc: encoder序列长度
            # 含义: 解码器每个位置对编码器所有位置的注意力
        },
        # ... 更多层
    ]
}

如何理解注意力矩阵:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 编码器自注意力 (T, T):
   - 行: Query位置 (问"我应该关注谁?")
   - 列: Key位置 (被关注的位置)
   - 值: 注意力权重 (0到1之间，和为1)
   
   例如: attention[5, 3] = 0.8
   → 位置5对位置3有0.8的注意力权重

2. 解码器交叉注意力 (T_dec, T_enc):
   - 行: 解码器Query位置
   - 列: 编码器Key位置
   - 含义: 解码器关注编码器的哪些位置
   
   例如: cross_attention[2, 7] = 0.6
   → 解码器位置2对编码器位置7有0.6的注意力权重

3. 解码器自注意力 (T_dec, T_dec):
   - 有因果mask: 下三角矩阵
   - 每个位置只能看到之前的位置
"""
    
    print(guide)
    print("=" * 70)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Transformer 注意力可视化完整指南")
    print("=" * 70)
    
    # 打印数据结构说明
    print_attention_structure_guide()
    
    # 运行所有方法
    try:
        method1_basic_visualization()
        method2_compare_all_layers()
        method3_multi_head_analysis()
        method4_attention_flow()
        method5_analyze_specific_positions()
        method6_save_statistics()
        quick_visualization_template()
        
        print("\n" + "=" * 70)
        print("✓ 所有可视化方法演示完成！")
        print("=" * 70)
        print("\n生成的文件:")
        print("  1. encoder_layer0_self_attention.png")
        print("  2. decoder_layer0_cross_attention.png")
        print("  3. decoder_layer0_self_attention.png")
        print("  4. encoder_all_layers_comparison.png")
        print("  5. decoder_all_layers_cross_comparison.png")
        print("  6. encoder_layer0_all_heads.png")
        print("  7. decoder_layer0_cross_all_heads.png")
        print("  8. attention_flow_layer0.png")
        print("  9. attention_statistics.json")
        print("\n推荐查看顺序:")
        print("  1. 先看基础热力图了解整体模式")
        print("  2. 再看层级对比理解深度变化")
        print("  3. 最后看多头分析和注意力流")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
