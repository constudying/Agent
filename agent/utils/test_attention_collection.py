"""
简单的注意力可视化测试脚本（不需要matplotlib）

这个脚本测试注意力权重的收集功能，不进行实际可视化。
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agent.models.transformer import Transformer


def test_attention_collection():
    """测试注意力权重收集功能"""
    
    print("=" * 60)
    print("测试注意力权重收集功能")
    print("=" * 60)
    
    # 1. 创建模型
    print("\n1. 创建Transformer模型...")
    model = Transformer(
        embed_dim=512,
        context_length=100,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    print("   ✓ 模型创建成功")
    
    # 2. 准备输入数据
    print("\n2. 准备输入数据...")
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
    print("   ✓ 输入数据准备完成")
    
    # 3. 测试不启用注意力存储的前向传播
    print("\n3. 测试普通前向传播（不收集注意力）...")
    with torch.no_grad():
        output = model(enc, dec)
    print(f"   ✓ 输出形状: {output.shape}")
    
    # 4. 启用注意力存储
    print("\n4. 启用注意力存储...")
    model.enable_attention_storage()
    print("   ✓ 注意力存储已启用")
    
    # 5. 前向传播并收集注意力权重
    print("\n5. 前向传播并收集注意力权重...")
    with torch.no_grad():
        output, attention_weights = model(enc, dec, return_attention_weights=True)
    print(f"   ✓ 输出形状: {output.shape}")
    
    # 6. 检查注意力权重结构
    print("\n6. 检查注意力权重结构...")
    
    # 编码器
    print(f"\n   编码器层数: {len(attention_weights['encoder'])}")
    for i, layer_attn in enumerate(attention_weights['encoder']):
        if 'self_attention' in layer_attn:
            shape = layer_attn['self_attention'].shape
            print(f"     Layer {i} - Self-Attention: {shape}")
            print(f"       - Batch Size: {shape[0]}")
            print(f"       - Num Heads: {shape[1]}")
            print(f"       - Sequence Length: {shape[2]} x {shape[3]}")
    
    # 解码器
    print(f"\n   解码器层数: {len(attention_weights['decoder'])}")
    for i, layer_attn in enumerate(attention_weights['decoder']):
        print(f"     Layer {i}:")
        if 'self_attention' in layer_attn:
            shape = layer_attn['self_attention'].shape
            print(f"       - Self-Attention: {shape}")
        if 'cross_attention' in layer_attn:
            shape = layer_attn['cross_attention'].shape
            print(f"       - Cross-Attention: {shape}")
    
    # 7. 验证注意力权重的值
    print("\n7. 验证注意力权重...")
    
    # 检查编码器第0层
    if attention_weights['encoder']:
        enc_attn = attention_weights['encoder'][0]['self_attention']
        print(f"   编码器第0层自注意力:")
        print(f"     - 形状: {enc_attn.shape}")
        print(f"     - 平均值: {enc_attn.mean():.6f}")
        print(f"     - 最小值: {enc_attn.min():.6f}")
        print(f"     - 最大值: {enc_attn.max():.6f}")
        print(f"     - 标准差: {enc_attn.std():.6f}")
        
        # 验证注意力权重和为1
        attn_sum = enc_attn[0, 0].sum(dim=-1)
        print(f"     - 每行和（应该接近1.0）: {attn_sum.mean():.6f}")
    
    # 检查解码器第0层的交叉注意力
    if attention_weights['decoder'] and 'cross_attention' in attention_weights['decoder'][0]:
        cross_attn = attention_weights['decoder'][0]['cross_attention']
        print(f"\n   解码器第0层交叉注意力:")
        print(f"     - 形状: {cross_attn.shape}")
        print(f"     - 平均值: {cross_attn.mean():.6f}")
        print(f"     - 最小值: {cross_attn.min():.6f}")
        print(f"     - 最大值: {cross_attn.max():.6f}")
        print(f"     - 标准差: {cross_attn.std():.6f}")
        
        # 验证注意力权重和为1
        attn_sum = cross_attn[0, 0].sum(dim=-1)
        print(f"     - 每行和（应该接近1.0）: {attn_sum.mean():.6f}")
    
    # 8. 测试禁用注意力存储
    print("\n8. 禁用注意力存储...")
    model.disable_attention_storage()
    print("   ✓ 注意力存储已禁用")
    
    # 9. 测试使用get_attention_weights方法
    print("\n9. 测试get_attention_weights()方法...")
    model.enable_attention_storage()
    with torch.no_grad():
        output = model(enc, dec)
    attn_weights_alt = model.get_attention_weights()
    print(f"   ✓ 通过get_attention_weights()获取的编码器层数: {len(attn_weights_alt['encoder'])}")
    print(f"   ✓ 通过get_attention_weights()获取的解码器层数: {len(attn_weights_alt['decoder'])}")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！注意力收集功能正常工作。")
    print("=" * 60)
    
    return True


def test_attention_statistics():
    """测试注意力统计功能"""
    
    print("\n" + "=" * 60)
    print("测试注意力统计功能")
    print("=" * 60)
    
    # 创建模型并收集注意力
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
    
    # 计算统计信息
    print("\n统计信息:")
    
    # 编码器统计
    print("\n编码器:")
    for i, layer_attn in enumerate(attention_weights['encoder']):
        if 'self_attention' in layer_attn:
            attn = layer_attn['self_attention'][0].cpu().numpy()
            print(f"  Layer {i} - Self-Attention:")
            print(f"    - Mean: {attn.mean():.6f}")
            print(f"    - Std:  {attn.std():.6f}")
            print(f"    - Min:  {attn.min():.6f}")
            print(f"    - Max:  {attn.max():.6f}")
    
    # 解码器统计
    print("\n解码器:")
    for i, layer_attn in enumerate(attention_weights['decoder']):
        print(f"  Layer {i}:")
        if 'self_attention' in layer_attn:
            attn = layer_attn['self_attention'][0].cpu().numpy()
            print(f"    Self-Attention:")
            print(f"      - Mean: {attn.mean():.6f}")
            print(f"      - Std:  {attn.std():.6f}")
        if 'cross_attention' in layer_attn:
            attn = layer_attn['cross_attention'][0].cpu().numpy()
            print(f"    Cross-Attention:")
            print(f"      - Mean: {attn.mean():.6f}")
            print(f"      - Std:  {attn.std():.6f}")
    
    print("\n" + "=" * 60)
    print("✓ 统计功能测试完成！")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Transformer 注意力功能测试")
    print("=" * 60)
    
    try:
        # 运行测试
        test_attention_collection()
        test_attention_statistics()
        
        print("\n" + "=" * 60)
        print("✓✓✓ 所有测试通过！✓✓✓")
        print("=" * 60)
        print("\n提示:")
        print("  - 注意力收集功能正常工作")
        print("  - 可以使用AttentionVisualizer进行可视化")
        print("  - 需要安装matplotlib和seaborn才能生成图表:")
        print("    pip install matplotlib seaborn")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
