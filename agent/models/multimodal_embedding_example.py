"""
多模态嵌入完整使用示例

展示如何组合使用：
1. 位置编码 (Position Encoding)
2. 模态类型嵌入 (Modality Type Embedding)
3. 片段嵌入 (Segment Embedding)
"""

import torch
import torch.nn as nn
from transformer import MultiModalEmbedding


def example_1_basic_usage():
    """示例1: 基础用法 - 文本 + 图像"""
    print("=" * 70)
    print("示例1: 基础用法 - 文本 + 图像双模态")
    print("=" * 70)
    
    # 配置多模态
    config = {
        'text': {
            'type': '1d',
            'max_len': 100,
            'dim': 768
        },
        'image': {
            'type': '2d',
            'height': 14,      # 224/16 for ViT patch
            'width': 14,
            'dim': 768
        }
    }
    
    # 创建嵌入层
    embedder = MultiModalEmbedding(
        modality_configs=config,
        unified_dim=768,
        dropout=0.1,
        position_encoding_type='sinusoidal'
    )
    
    # 准备输入
    batch_size = 4
    text_input = torch.randn(batch_size, 50, 768)      # [B, 50, 768]
    image_input = torch.randn(batch_size, 196, 768)    # [B, 14*14, 768]
    
    print(f"\n输入:")
    print(f"  文本: {text_input.shape}")
    print(f"  图像: {image_input.shape}")
    
    # 前向传播
    multimodal_inputs = {
        'text': text_input,
        'image': image_input
    }
    
    output, modality_masks = embedder(multimodal_inputs)
    
    print(f"\n输出:")
    print(f"  多模态嵌入: {output.shape}")  # [4, 246, 768]
    print(f"\n模态位置信息:")
    for modality, mask_info in modality_masks.items():
        print(f"  {modality}: 位置 [{mask_info['start']}:{mask_info['end']}], "
              f"长度 {mask_info['length']}")
    
    print("\n嵌入组合方式:")
    print("  最终嵌入 = 输入特征 + 位置编码 + 模态类型嵌入")
    print("  - 位置编码: 文本用1D编码，图像用2D空间编码")
    print("  - 模态类型嵌入: 文本=0, 图像=1 (可学习向量)")
    print()


def example_2_three_modalities():
    """示例2: 三模态 - 文本 + 图像 + 音频"""
    print("=" * 70)
    print("示例2: 三模态融合 - 文本 + 图像 + 音频")
    print("=" * 70)
    
    config = {
        'text': {
            'type': '1d',
            'max_len': 200,
            'dim': 512
        },
        'image': {
            'type': '2d',
            'height': 16,
            'width': 16,
            'dim': 512
        },
        'audio': {
            'type': '1d',
            'max_len': 500,
            'dim': 512
        }
    }
    
    embedder = MultiModalEmbedding(
        modality_configs=config,
        unified_dim=512,
        num_modalities=3,
        dropout=0.1
    )
    
    # 输入
    batch_size = 2
    inputs = {
        'text': torch.randn(batch_size, 80, 512),
        'image': torch.randn(batch_size, 256, 512),
        'audio': torch.randn(batch_size, 120, 512)
    }
    
    print(f"\n输入维度:")
    for modality, tensor in inputs.items():
        print(f"  {modality}: {tensor.shape}")
    
    output, masks = embedder(inputs)
    print(f"\n融合后: {output.shape}")
    print(f"总序列长度: {80 + 256 + 120} = {output.shape[1]}")
    
    print("\n每个模态的类型嵌入ID:")
    print(f"  text=0, image=1, audio=2")
    print()


def example_3_dimension_alignment():
    """示例3: 维度对齐 - 不同模态维度不同"""
    print("=" * 70)
    print("示例3: 自动维度对齐 - 不同模态有不同的原始维度")
    print("=" * 70)
    
    config = {
        'text': {
            'type': '1d',
            'max_len': 100,
            'dim': 768        # BERT维度
        },
        'image': {
            'type': '2d',
            'height': 14,
            'width': 14,
            'dim': 512        # ResNet维度
        },
        'audio': {
            'type': '1d',
            'max_len': 300,
            'dim': 256        # 音频特征维度
        }
    }
    
    # 统一到512维
    unified_dim = 512
    embedder = MultiModalEmbedding(
        modality_configs=config,
        unified_dim=unified_dim,
        dropout=0.1
    )
    
    # 输入原始维度
    batch_size = 3
    inputs = {
        'text': torch.randn(batch_size, 50, 768),    # 原始768维
        'image': torch.randn(batch_size, 196, 512),  # 原始512维
        'audio': torch.randn(batch_size, 100, 256)   # 原始256维
    }
    
    print(f"\n输入 (不同维度):")
    for modality, tensor in inputs.items():
        print(f"  {modality}: {tensor.shape}")
    
    output, _ = embedder(inputs)
    print(f"\n输出 (统一维度): {output.shape}")
    print(f"所有模态都投影到了 {unified_dim} 维")
    
    print("\n内部处理:")
    print("  1. 文本: 768 -> 512 (Linear投影)")
    print("  2. 图像: 512 -> 512 (无需投影)")
    print("  3. 音频: 256 -> 512 (Linear投影)")
    print("  4. 每个模态 += 位置编码(512维)")
    print("  5. 每个模态 += 模态类型嵌入(512维)")
    print()


def example_4_with_segment_embedding():
    """示例4: 使用片段嵌入 - 区分同一模态的不同部分"""
    print("=" * 70)
    print("示例4: 片段嵌入 - 区分同一模态内的不同片段")
    print("=" * 70)
    
    config = {
        'text': {
            'type': '1d',
            'max_len': 200,
            'dim': 512
        },
        'image': {
            'type': '2d',
            'height': 16,
            'width': 16,
            'dim': 512
        }
    }
    
    # 启用片段嵌入
    embedder = MultiModalEmbedding(
        modality_configs=config,
        unified_dim=512,
        use_segment_embedding=True,
        max_segments=5,
        dropout=0.1
    )
    
    batch_size = 2
    
    # 文本包含两个句子
    text_len = 60
    text_input = torch.randn(batch_size, text_len, 512)
    # 前30个token是句子A (segment=0)，后30个是句子B (segment=1)
    text_segments = torch.cat([
        torch.zeros(batch_size, 30, dtype=torch.long),
        torch.ones(batch_size, 30, dtype=torch.long)
    ], dim=1)
    
    # 图像包含两张图片拼接
    image_len = 128
    image_input = torch.randn(batch_size, image_len, 512)
    # 前64个patch是图片1 (segment=0)，后64个是图片2 (segment=1)
    image_segments = torch.cat([
        torch.zeros(batch_size, 64, dtype=torch.long),
        torch.ones(batch_size, 64, dtype=torch.long)
    ], dim=1)
    
    inputs = {
        'text': text_input,
        'image': image_input
    }
    
    segment_ids = {
        'text': text_segments,
        'image': image_segments
    }
    
    print(f"\n输入:")
    print(f"  文本: {text_input.shape}, 片段: [A:0-30, B:30-60]")
    print(f"  图像: {image_input.shape}, 片段: [Img1:0-64, Img2:64-128]")
    
    output, masks = embedder(inputs, segment_ids=segment_ids)
    print(f"\n输出: {output.shape}")
    
    print("\n嵌入组合:")
    print("  最终嵌入 = 输入 + 位置编码 + 模态类型嵌入 + 片段嵌入")
    print("  - 片段嵌入帮助模型区分同一模态内的不同部分")
    print("  - 例如: 区分句子A和句子B，或图片1和图片2")
    print()


def example_5_attention_mask():
    """示例5: 生成注意力掩码"""
    print("=" * 70)
    print("示例5: 多模态注意力掩码")
    print("=" * 70)
    
    config = {
        'text': {'type': '1d', 'max_len': 100, 'dim': 256},
        'image': {'type': '2d', 'height': 8, 'width': 8, 'dim': 256}
    }
    
    embedder = MultiModalEmbedding(
        modality_configs=config,
        unified_dim=256
    )
    
    inputs = {
        'text': torch.randn(2, 20, 256),
        'image': torch.randn(2, 64, 256)
    }
    
    output, masks = embedder(inputs)
    
    print(f"\n序列组成: text[0:20] + image[20:84]")
    
    # 允许跨模态注意
    mask_cross = embedder.get_modality_attention_mask(masks, allow_cross_modality=True)
    print(f"\n跨模态注意力掩码: {mask_cross.shape}")
    print(f"  全True - 所有位置可以互相注意")
    
    # 只允许模态内注意
    mask_within = embedder.get_modality_attention_mask(masks, allow_cross_modality=False)
    print(f"\n模态内注意力掩码: {mask_within.shape}")
    print(f"  文本只能注意文本，图像只能注意图像")
    print(f"  mask[0:20, 0:20] = True  (文本内部)")
    print(f"  mask[20:84, 20:84] = True  (图像内部)")
    print(f"  mask[0:20, 20:84] = False (文本->图像被mask)")
    print()


def example_6_complete_pipeline():
    """示例6: 完整流程 - 从原始数据到Transformer输入"""
    print("=" * 70)
    print("示例6: 完整的多模态Transformer输入流程")
    print("=" * 70)
    
    class MultiModalTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            
            # 1. 模态特定的特征提取器
            self.text_encoder = nn.Embedding(30000, 512)  # 词嵌入
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 512, kernel_size=16, stride=16),  # Patch embedding
                nn.Flatten(2),  # [B, 512, H*W]
            )
            
            # 2. 多模态嵌入层
            config = {
                'text': {'type': '1d', 'max_len': 100, 'dim': 512},
                'image': {'type': '2d', 'height': 14, 'width': 14, 'dim': 512}
            }
            self.embedder = MultiModalEmbedding(
                modality_configs=config,
                unified_dim=512,
                use_segment_embedding=False,
                dropout=0.1
            )
            
            # 3. Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
            
            # 4. 输出头 (例如分类)
            self.classifier = nn.Linear(512, 10)
        
        def forward(self, text_ids, image):
            """
            Args:
                text_ids: [B, L] - 文本token ids
                image: [B, 3, 224, 224] - 图像
            """
            B = text_ids.size(0)
            
            # === 步骤1: 模态特定特征提取 ===
            text_features = self.text_encoder(text_ids)  # [B, L, 512]
            
            image_features = self.image_encoder(image)  # [B, 512, 196]
            image_features = image_features.transpose(1, 2)  # [B, 196, 512]
            
            # === 步骤2: 多模态嵌入 (位置编码 + 模态类型嵌入) ===
            multimodal_input = {
                'text': text_features,
                'image': image_features
            }
            
            embedded, masks = self.embedder(multimodal_input)
            # embedded: [B, L+196, 512]
            # 包含了: 原始特征 + 位置编码 + 模态类型嵌入
            
            # === 步骤3: Transformer处理 ===
            # 注意: PyTorch Transformer需要 [seq, batch, dim] 格式
            embedded = embedded.transpose(0, 1)  # [L+196, B, 512]
            
            transformer_output = self.transformer(embedded)  # [L+196, B, 512]
            transformer_output = transformer_output.transpose(0, 1)  # [B, L+196, 512]
            
            # === 步骤4: 使用CLS token或平均池化 ===
            # 这里使用第一个token (假设是CLS)
            cls_output = transformer_output[:, 0, :]  # [B, 512]
            
            # === 步骤5: 分类 ===
            logits = self.classifier(cls_output)  # [B, 10]
            
            return logits
    
    # 使用模型
    model = MultiModalTransformer()
    
    # 输入
    batch_size = 4
    text_ids = torch.randint(0, 30000, (batch_size, 50))
    images = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\n输入:")
    print(f"  文本IDs: {text_ids.shape}")
    print(f"  图像: {images.shape}")
    
    # 前向传播
    output = model(text_ids, images)
    print(f"\n输出:")
    print(f"  分类logits: {output.shape}")
    
    print("\n完整流程:")
    print("  1. 文本: token_ids -> 词嵌入 [B,L,512]")
    print("  2. 图像: [B,3,224,224] -> patch嵌入 [B,196,512]")
    print("  3. 多模态嵌入:")
    print("     - 添加1D位置编码到文本")
    print("     - 添加2D空间位置编码到图像")
    print("     - 添加模态类型嵌入 (text=0, image=1)")
    print("     - LayerNorm + Dropout")
    print("  4. 拼接: [B, 50+196, 512]")
    print("  5. Transformer处理")
    print("  6. 分类输出")
    print()


def visualize_embedding_combination():
    """可视化嵌入组合过程"""
    print("=" * 70)
    print("嵌入组合的数学表达")
    print("=" * 70)
    
    print("""
对于第 i 个模态的第 t 个位置的token:

1. 输入特征: 
   x_i,t ∈ R^d

2. 位置编码:
   PE_i(t) ∈ R^d
   - 1D模态: PE(t) = [sin(t/10000^(2k/d)), cos(t/10000^(2k/d)), ...]
   - 2D模态: PE(h,w) = [PE_h(h), PE_w(w)]

3. 模态类型嵌入:
   M_i ∈ R^d  (可学习参数)
   - 文本: M_0
   - 图像: M_1
   - 音频: M_2
   ...

4. 片段嵌入 (可选):
   S_j ∈ R^d  (可学习参数)

5. 最终嵌入:
   
   E_i,t = LayerNorm(x_i,t + PE_i(t) + M_i + S_j) + Dropout
   
   然后拼接所有模态:
   
   E = [E_1,1, ..., E_1,L1, E_2,1, ..., E_2,L2, ...]
   
   输入到Transformer:
   
   Output = Transformer(E)

关键点:
- 所有嵌入都是 **加性组合** (element-wise addition)
- 所有项必须有 **相同的维度** d
- 位置编码是 **相对于模态内的位置**
- 模态类型嵌入是 **全局的** (同一模态所有位置共享)
- 片段嵌入用于区分 **同一模态内的不同片段**
    """)


def compare_with_without_modality_embedding():
    """对比有无模态类型嵌入的效果"""
    print("=" * 70)
    print("对比: 有无模态类型嵌入")
    print("=" * 70)
    
    config = {
        'text': {'type': '1d', 'max_len': 50, 'dim': 256},
        'image': {'type': '2d', 'height': 8, 'width': 8, 'dim': 256}
    }
    
    # 有模态类型嵌入
    embedder_with = MultiModalEmbedding(
        modality_configs=config,
        unified_dim=256
    )
    
    # 模拟：无模态类型嵌入 (仅位置编码)
    class EmbedderWithoutModality(nn.Module):
        def __init__(self, config):
            super().__init__()
            from transformer import PositionalEncoding
            self.text_pos = PositionalEncoding(256, max_len=50)
            
        def forward(self, inputs):
            # 仅添加位置编码，不添加模态类型嵌入
            text = self.text_pos(inputs['text'])
            image = inputs['image']  # 假设图像也有位置编码
            return torch.cat([text, image], dim=1)
    
    embedder_without = EmbedderWithoutModality(config)
    
    # 输入
    inputs = {
        'text': torch.randn(2, 20, 256),
        'image': torch.randn(2, 64, 256)
    }
    
    output_with, _ = embedder_with(inputs)
    output_without = embedder_without(inputs)
    
    print(f"\n有模态类型嵌入: {output_with.shape}")
    print(f"无模态类型嵌入: {output_without.shape}")
    
    print("\n差异:")
    print("  【无模态类型嵌入】")
    print("    - 模型仅依靠位置编码区分不同位置")
    print("    - 难以区分来自不同模态的信息")
    print("    - 文本token和图像patch被同等对待")
    print()
    print("  【有模态类型嵌入】")
    print("    - 每个模态有独特的类型向量")
    print("    - 模型明确知道哪些是文本，哪些是图像")
    print("    - 可以学习模态特定的处理策略")
    print("    - 性能通常更好！")
    print()
    
    print("类比: BERT的segment embedding")
    print("  BERT用segment embedding区分句子A和句子B")
    print("  这里用modality embedding区分文本、图像、音频等")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("多模态嵌入完整使用指南")
    print("位置编码 + 模态类型嵌入 + 片段嵌入")
    print("=" * 70 + "\n")
    
    example_1_basic_usage()
    example_2_three_modalities()
    example_3_dimension_alignment()
    example_4_with_segment_embedding()
    example_5_attention_mask()
    example_6_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("理论部分")
    print("=" * 70 + "\n")
    
    visualize_embedding_combination()
    compare_with_without_modality_embedding()
    
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print("""
核心要点:

1. 嵌入组合方式: **加法** (Addition)
   最终嵌入 = 输入特征 + 位置编码 + 模态类型嵌入 + [片段嵌入]

2. 为什么用加法而不是拼接?
   - 加法保持维度不变
   - 更容易优化
   - 符合Transformer的设计哲学

3. 各部分的作用:
   - 位置编码: 提供位置信息 (相对于模态内的位置)
   - 模态类型嵌入: 标识数据来源 (文本/图像/音频)
   - 片段嵌入: 区分同一模态的不同部分 (可选)

4. 实践建议:
   - 统一所有模态到相同维度
   - 使用LayerNorm稳定训练
   - 可选地使用Dropout防止过拟合
   - 为不同任务选择合适的注意力掩码

5. 应用场景:
   - 视觉问答 (VQA): 文本问题 + 图像
   - 视频字幕: 视频帧 + 文本描述
   - 多模态情感分析: 文本 + 图像 + 音频
   - 跨模态检索: 任意模态组合

参考文献:
- BERT: 提出segment embedding的概念
- ViT: 图像的patch embedding和位置编码
- CLIP/ALIGN: 视觉-语言多模态学习
- Perceiver: 通用的多模态架构
    """)
    print("=" * 70)
