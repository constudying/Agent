"""
图像序列位置编码使用示例

演示如何使用 ImageSequencePositionalEncoding 类处理 [batch, seq, c, h, w] 格式的图像数据
"""

import torch
import torch.nn as nn
from transformer import ImageSequencePositionalEncoding


def example_1_temporal_only():
    """示例1：只对时间维度编码"""
    print("=" * 60)
    print("示例1：只对时间维度编码（适合时序建模）")
    print("=" * 60)
    
    # 配置
    batch_size = 2
    seq_len = 10
    height, width = 64, 64
    embed_dim = 256
    
    # 创建位置编码器
    pos_encoder = ImageSequencePositionalEncoding(
        temporal_dim=seq_len,
        spatial_height=height,
        spatial_width=width,
        embed_dim=embed_dim,
        encoding_type='temporal_only',
        dropout=0.1
    )
    
    # 模拟图像序列输入
    x = torch.randn(batch_size, seq_len, 3, height, width)
    print(f"输入形状: {x.shape}")
    
    # 获取位置编码
    pos_encoding = pos_encoder(x)
    print(f"位置编码形状: {pos_encoding.shape}")
    print(f"说明: 只在时间维度上有变化，空间位置上相同\n")


def example_2_spatial_only():
    """示例2：只对空间维度编码"""
    print("=" * 60)
    print("示例2：只对空间维度编码（适合单帧图像理解）")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 10
    height, width = 32, 32
    embed_dim = 128
    
    pos_encoder = ImageSequencePositionalEncoding(
        temporal_dim=seq_len,
        spatial_height=height,
        spatial_width=width,
        embed_dim=embed_dim,
        encoding_type='spatial_only',
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, 3, height, width)
    print(f"输入形状: {x.shape}")
    
    pos_encoding = pos_encoder(x)
    print(f"位置编码形状: {pos_encoding.shape}")
    print(f"说明: 只在空间维度（h, w）上有变化，时间维度上相同\n")


def example_3_spatiotemporal():
    """示例3：联合时空编码"""
    print("=" * 60)
    print("示例3：联合时空编码（适合视频理解）")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 16
    height, width = 56, 56
    embed_dim = 384  # 必须能被3整除
    
    pos_encoder = ImageSequencePositionalEncoding(
        temporal_dim=seq_len,
        spatial_height=height,
        spatial_width=width,
        embed_dim=embed_dim,
        encoding_type='spatiotemporal',
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, 3, height, width)
    print(f"输入形状: {x.shape}")
    
    pos_encoding = pos_encoder(x)
    print(f"位置编码形状: {pos_encoding.shape}")
    print(f"说明: 时间、高度、宽度各分配 {embed_dim//3} 维度")
    print(f"      能够区分每个时空位置的独特性\n")


def example_4_factorized():
    """示例4：分解的时空编码"""
    print("=" * 60)
    print("示例4：分解的时空编码（时间+空间分别编码）")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 8
    height, width = 28, 28
    embed_dim = 256
    
    pos_encoder = ImageSequencePositionalEncoding(
        temporal_dim=seq_len,
        spatial_height=height,
        spatial_width=width,
        embed_dim=embed_dim,
        encoding_type='factorized',
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, 3, height, width)
    print(f"输入形状: {x.shape}")
    
    pos_encoding = pos_encoder(x)
    print(f"位置编码形状: {pos_encoding.shape}")
    print(f"说明: 时间编码和空间编码分别计算后相加")
    print(f"      计算效率高，适合长序列\n")


def example_5_learned():
    """示例5：可学习的位置编码"""
    print("=" * 60)
    print("示例5：可学习的2D位置编码（参数可训练）")
    print("=" * 60)
    
    batch_size = 3
    seq_len = 12
    height, width = 16, 16
    embed_dim = 192
    
    pos_encoder = ImageSequencePositionalEncoding(
        temporal_dim=seq_len,
        spatial_height=height,
        spatial_width=width,
        embed_dim=embed_dim,
        encoding_type='learned_2d',
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, 3, height, width)
    print(f"输入形状: {x.shape}")
    
    pos_encoding = pos_encoder(x)
    print(f"位置编码形状: {pos_encoding.shape}")
    
    # 统计可训练参数
    num_params = sum(p.numel() for p in pos_encoder.parameters() if p.requires_grad)
    print(f"可训练参数数量: {num_params:,}")
    print(f"说明: 位置编码参数在训练中优化，适应特定任务\n")


def example_6_with_transformer():
    """示例6：在Transformer中使用"""
    print("=" * 60)
    print("示例6：在Transformer中实际使用")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 8
    channels = 3
    height, width = 32, 32
    embed_dim = 256
    
    # 图像序列输入
    x = torch.randn(batch_size, seq_len, channels, height, width)
    print(f"原始图像序列: {x.shape}")
    
    # 步骤1: 将图像序列转换为特征序列
    # 通常使用CNN或ViT的patch embedding
    # 这里简化为线性投影
    spatial_features = height * width * channels
    image_encoder = nn.Linear(spatial_features, embed_dim)
    
    # 展平空间维度
    x_flat = x.reshape(batch_size, seq_len, -1)  # (B, T, C*H*W)
    features = image_encoder(x_flat)  # (B, T, D)
    print(f"提取特征后: {features.shape}")
    
    # 步骤2: 添加位置编码
    # 方式A: 如果特征已展平，使用简化的时间编码
    pos_encoder = ImageSequencePositionalEncoding(
        temporal_dim=seq_len,
        spatial_height=height,
        spatial_width=width,
        embed_dim=embed_dim,
        encoding_type='temporal_only',
        dropout=0.1
    )
    
    features_with_pos = pos_encoder(features)
    print(f"添加位置编码后: {features_with_pos.shape}")
    
    # 步骤3: 输入到Transformer
    # (这里省略Transformer的具体实现)
    print(f"\n现在可以将 {features_with_pos.shape} 的特征输入到Transformer层")
    
    print("\n方式B: 使用更精细的时空编码")
    pos_encoder_st = ImageSequencePositionalEncoding(
        temporal_dim=seq_len,
        spatial_height=height,
        spatial_width=width,
        embed_dim=embed_dim,
        encoding_type='factorized',
        dropout=0.1
    )
    
    # 获取图像序列的位置编码
    pos_encoding_2d = pos_encoder_st(x)
    print(f"2D位置编码: {pos_encoding_2d.shape}")
    print(f"说明: 这个编码可以在图像特征提取后广播相加\n")


def compare_encoding_types():
    """比较不同编码方式的特点"""
    print("=" * 60)
    print("不同位置编码方式的对比")
    print("=" * 60)
    
    config = {
        'temporal_dim': 10,
        'spatial_height': 28,
        'spatial_width': 28,
        'embed_dim': 384,
        'dropout': 0.1
    }
    
    encoding_types = [
        'temporal_only',
        'spatial_only', 
        'spatiotemporal',
        'factorized',
        'learned_2d'
    ]
    
    print(f"\n配置: T={config['temporal_dim']}, H={config['spatial_height']}, "
          f"W={config['spatial_width']}, D={config['embed_dim']}\n")
    
    print(f"{'编码类型':<20} {'参数量':<15} {'适用场景'}")
    print("-" * 70)
    
    for enc_type in encoding_types:
        try:
            pos_encoder = ImageSequencePositionalEncoding(
                temporal_dim=config['temporal_dim'],
                spatial_height=config['spatial_height'],
                spatial_width=config['spatial_width'],
                embed_dim=config['embed_dim'],
                encoding_type=enc_type,
                dropout=config['dropout']
            )
            
            num_params = sum(p.numel() for p in pos_encoder.parameters() if p.requires_grad)
            
            use_case = {
                'temporal_only': '时序建模、动作识别',
                'spatial_only': '单帧图像理解',
                'spatiotemporal': '完整视频理解',
                'factorized': '长序列视频',
                'learned_2d': '特定任务优化'
            }
            
            print(f"{enc_type:<20} {num_params:<15,} {use_case[enc_type]}")
            
        except Exception as e:
            print(f"{enc_type:<20} ERROR: {str(e)}")
    
    print("\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("图像序列位置编码 (ImageSequencePositionalEncoding) 使用示例")
    print("=" * 60 + "\n")
    
    # 运行所有示例
    example_1_temporal_only()
    example_2_spatial_only()
    example_3_spatiotemporal()
    example_4_factorized()
    example_5_learned()
    example_6_with_transformer()
    compare_encoding_types()
    
    print("=" * 60)
    print("关键要点总结:")
    print("=" * 60)
    print("""
1. 图像序列 [B, T, C, H, W] 包含时间和空间两个维度的位置信息

2. 不同编码方式的选择：
   - temporal_only: 只关注时序关系（如动作识别）
   - spatial_only: 只关注空间结构（如图像分割）
   - spatiotemporal: 完整建模时空关系（最全面，但维度分配要平衡）
   - factorized: 分解编码，计算效率高（推荐用于长序列）
   - learned_2d: 可学习，适应特定任务（需要更多训练数据）

3. 使用流程：
   a) 图像序列 -> CNN/ViT特征提取 -> 特征序列
   b) 特征序列 + 位置编码 -> Transformer输入
   c) 位置编码可以在特征提取前或后添加

4. 实际应用建议：
   - 视频理解任务: 使用 factorized 或 spatiotemporal
   - 实时处理: 使用 temporal_only (最快)
   - 高精度要求: 使用 spatiotemporal 或 learned_2d
   - 资源受限: 使用 factorized (参数少，速度快)
    """)
    print("=" * 60)
