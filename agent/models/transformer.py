import math
import os

import einops
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from agent.models.base_nets import get_activation
from robomimic.models.base_nets import Module


class NewGELU(nn.Module):
    """
    GELU激活函数的实现
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class GEGLU(nn.Module):
    """
    GEGLU激活函数的实现，是GLU的变体
    """
    def geglu(self, x):
        assert x.shape[-1] % 2 == 0, "@GEGLU: Input feature dimension must be even for GEGLU"
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x):
        return self.geglu(x)


class PositionalEncoding(nn.Module):
    """
    更完善的位置编码类，支持多种位置编码方式
    
    支持的编码类型:
    - 'sinusoidal': 固定的正弦-余弦位置编码 (Vaswani et al., 2017)
    - 'learned': 可学习的位置编码
    - 'learned_additive': 可学习的加性位置编码
    - 'rotary': 旋转位置编码 (RoPE, Su et al., 2021)
    
    Args:
        embed_dim (int): 嵌入维度
        max_len (int): 最大序列长度
        encoding_type (str): 位置编码类型，默认 'sinusoidal'
        dropout (float): Dropout 概率，默认 0.1
        scale (bool): 是否对嵌入进行缩放，默认 True
    """
    def __init__(
        self, 
        embed_dim, 
        max_len=5000, 
        encoding_type='sinusoidal',
        dropout=0.1,
        scale=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.encoding_type = encoding_type
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)
        
        if encoding_type == 'sinusoidal':
            # 固定的正弦-余弦位置编码
            pe = torch.zeros(max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, embed_dim, 2, dtype=torch.float) * 
                (-math.log(10000.0) / embed_dim)
            )
            
            pe[:, 0::2] = torch.sin(position * div_term)
            if embed_dim % 2 == 0:
                pe[:, 1::2] = torch.cos(position * div_term)
            else:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            
            pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
            self.register_buffer('pe', pe)
            
        elif encoding_type == 'learned':
            # 可学习的位置嵌入
            self.pos_embedding = nn.Embedding(max_len, embed_dim)
            
        elif encoding_type == 'learned_additive':
            # 可学习的加性位置编码
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
            nn.init.normal_(self.pos_embedding, std=0.02)
            
        elif encoding_type == 'rotary':
            # 旋转位置编码 (RoPE)
            # 预计算旋转矩阵的频率
            inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
            self.register_buffer('inv_freq', inv_freq)
            
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
    
    def forward(self, x, position_ids=None):
        """
        Args:
            x: 输入张量，shape (B, T, D) 或 (B, T)
            position_ids: 可选的位置索引，shape (B, T) 或 (T,)
            
        Returns:
            如果 encoding_type 是 'rotary'，返回 (cos, sin) 用于后续的旋转操作
            否则返回位置编码后的张量，shape (B, T, D)
        """
        if x.dim() == 2:
            # 如果输入是 (B, T)，假设需要位置编码
            B, T = x.shape
            D = self.embed_dim
            device = x.device
        else:
            # 如果输入是 (B, T, D)
            B, T, D = x.shape
            device = x.device
            
        assert T <= self.max_len, f"Sequence length {T} exceeds maximum length {self.max_len}"
        assert D == self.embed_dim, f"Input embedding dim {D} must equal module embedding dim {self.embed_dim}"
        
        if self.encoding_type == 'sinusoidal':
            # 固定的正弦-余弦编码
            if self.scale:
                x = x * math.sqrt(self.embed_dim)
            
            pos_encoding = self.pe[:, :T, :]
            if x.dim() == 3:
                x = x + pos_encoding
            else:
                # 如果输入是 (B, T)，只返回位置编码
                x = pos_encoding.expand(B, -1, -1)
            return self.dropout(x)
            
        elif self.encoding_type == 'learned':
            # 可学习的位置嵌入
            if position_ids is None:
                position_ids = torch.arange(T, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(B, -1)
            
            pos_encoding = self.pos_embedding(position_ids)
            
            if self.scale and x.dim() == 3:
                x = x * math.sqrt(self.embed_dim)
            
            if x.dim() == 3:
                x = x + pos_encoding
            else:
                x = pos_encoding
            return self.dropout(x)
            
        elif self.encoding_type == 'learned_additive':
            # 可学习的加性位置编码
            if self.scale and x.dim() == 3:
                x = x * math.sqrt(self.embed_dim)
            
            pos_encoding = self.pos_embedding[:, :T, :]
            if x.dim() == 3:
                x = x + pos_encoding
            else:
                # 只返回位置编码
                x = pos_encoding.expand(B, -1, -1)
            return self.dropout(x)
            
        elif self.encoding_type == 'rotary':
            # 旋转位置编码
            # 生成位置索引
            if position_ids is None:
                position_ids = torch.arange(T, dtype=torch.float, device=device)
            else:
                position_ids = position_ids.float()
            
            # 计算频率
            freqs = torch.einsum('i,j->ij', position_ids, self.inv_freq)  # (T, D/2)
            emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
            
            # 返回 cos 和 sin 用于后续的旋转操作
            cos_emb = emb.cos()[None, :, :]  # (1, T, D)
            sin_emb = emb.sin()[None, :, :]  # (1, T, D)
            
            return cos_emb, sin_emb
    
    def output_shape(self, input_shape):
        """返回输出形状"""
        if self.encoding_type == 'rotary':
            # RoPE 返回两个张量
            return [input_shape, input_shape]
        return input_shape


class ImageSequencePositionalEncoding(nn.Module):
    """
    专门用于图像序列的位置编码类，适用于 [batch, seq, c, h, w] 格式的输入
    
    对于图像序列，位置信息包含三个维度：
    1. 时间维度 (seq): 序列中的帧位置
    2. 空间维度 (h, w): 图像内的像素位置
    
    支持的编码方式：
    - 'temporal_only': 只对时间维度编码
    - 'spatial_only': 只对空间维度编码
    - 'spatiotemporal': 联合时空编码
    - 'factorized': 分解的时空编码（时间+空间分别编码后相加）
    - 'learned_2d': 可学习的二维位置编码
    
    Args:
        temporal_dim (int): 时间维度的最大长度
        spatial_height (int): 图像高度
        spatial_width (int): 图像宽度
        embed_dim (int): 嵌入维度
        encoding_type (str): 编码类型
        dropout (float): Dropout概率
        temperature (float): 温度参数，控制频率范围
    """
    def __init__(
        self,
        temporal_dim,
        spatial_height,
        spatial_width,
        embed_dim,
        encoding_type='spatiotemporal',
        dropout=0.1,
        temperature=10000.0
    ):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        self.embed_dim = embed_dim
        self.encoding_type = encoding_type
        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)
        
        if encoding_type == 'temporal_only':
            # 只对时间维度进行编码
            self._build_temporal_encoding()
            
        elif encoding_type == 'spatial_only':
            # 只对空间维度进行编码
            self._build_spatial_encoding()
            
        elif encoding_type == 'spatiotemporal':
            # 联合时空编码：时间、高度、宽度各分配一部分维度
            assert embed_dim % 3 == 0, "For spatiotemporal encoding, embed_dim should be divisible by 3"
            self.temp_dim = embed_dim // 3
            self.h_dim = embed_dim // 3
            self.w_dim = embed_dim - self.temp_dim - self.h_dim
            
            self._build_temporal_encoding(self.temp_dim)
            self._build_spatial_encoding(self.h_dim, self.w_dim)
            
        elif encoding_type == 'factorized':
            # 分解编码：时间编码和空间编码分别计算后相加
            self._build_temporal_encoding()
            self._build_spatial_encoding()
            
        elif encoding_type == 'learned_2d':
            # 可学习的位置编码
            # 时间位置编码
            self.temporal_embedding = nn.Parameter(
                torch.zeros(1, temporal_dim, 1, 1, embed_dim)
            )
            # 空间位置编码
            self.spatial_embedding = nn.Parameter(
                torch.zeros(1, 1, spatial_height, spatial_width, embed_dim)
            )
            nn.init.normal_(self.temporal_embedding, std=0.02)
            nn.init.normal_(self.spatial_embedding, std=0.02)

        elif encoding_type == 'learned':
            # 可学习的位置编码 (展平后)
            self.embedding = nn.Parameter(
                torch.zeros(1, temporal_dim * spatial_height * spatial_width, embed_dim) # +1 for cls token imitating ViT
            )
            nn.init.normal_(self.embedding, std=0.01) # 对vit中的patch使用和cls token不同标准差的初始化

        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
    
    def _build_temporal_encoding(self, dim=None):
        """构建时间维度的正弦-余弦编码"""
        if dim is None:
            dim = self.embed_dim
            
        pe = torch.zeros(self.temporal_dim, dim)
        position = torch.arange(0, self.temporal_dim, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * 
            (-math.log(self.temperature) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        self.register_buffer('temporal_pe', pe)  # (T, D)
    
    def _build_spatial_encoding(self, h_dim=None, w_dim=None):
        """构建空间维度的正弦-余弦编码"""
        if h_dim is None:
            h_dim = self.embed_dim // 2
        if w_dim is None:
            w_dim = self.embed_dim - h_dim
        
        # 高度方向的编码
        pe_h = torch.zeros(self.spatial_height, h_dim)
        position_h = torch.arange(0, self.spatial_height, dtype=torch.float).unsqueeze(1)
        div_term_h = torch.exp(
            torch.arange(0, h_dim, 2, dtype=torch.float) * 
            (-math.log(self.temperature) / h_dim)
        )
        
        pe_h[:, 0::2] = torch.sin(position_h * div_term_h)
        if h_dim % 2 == 0:
            pe_h[:, 1::2] = torch.cos(position_h * div_term_h)
        else:
            pe_h[:, 1::2] = torch.cos(position_h * div_term_h[:-1])
        
        # 宽度方向的编码
        pe_w = torch.zeros(self.spatial_width, w_dim)
        position_w = torch.arange(0, self.spatial_width, dtype=torch.float).unsqueeze(1)
        div_term_w = torch.exp(
            torch.arange(0, w_dim, 2, dtype=torch.float) * 
            (-math.log(self.temperature) / w_dim)
        )
        
        pe_w[:, 0::2] = torch.sin(position_w * div_term_w)
        if w_dim % 2 == 0:
            pe_w[:, 1::2] = torch.cos(position_w * div_term_w)
        else:
            pe_w[:, 1::2] = torch.cos(position_w * div_term_w[:-1])
        
        self.register_buffer('spatial_pe_h', pe_h)  # (H, D_h)
        self.register_buffer('spatial_pe_w', pe_w)  # (W, D_w)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量
               - 如果是图像序列: shape (B, T, C, H, W)
               - 如果是展平后的序列: shape (B, T, D)
        
        Returns:
            添加位置编码后的张量，shape与输入相同
        """
        if x.dim() == 5:
            # 输入是图像序列 (B, T, C, H, W)
            B, T, C, H, W = x.shape
            device = x.device
            
            # 需要先将图像序列转换为特征序列
            # 这里假设调用者会在外部进行转换，我们返回位置编码用于后续相加
            return self._get_position_encoding_for_image_sequence(B, T, H, W, device)
            
        elif x.dim() == 3:
            # 输入已经是展平后的特征序列 (B, T, D)
            B, T, D = x.shape
            device = x.device
            
            assert D == self.embed_dim, f"Input dim {D} != embed_dim {self.embed_dim}"
            assert T == self.temporal_dim * self.spatial_height * self.spatial_width, f"Temporal dim {T} != expected {self.temporal_dim*self.spatial_height*self.spatial_width}"
            
            return self._add_position_encoding_to_features(x, device)
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}, expected 3 or 5")
    
    def _get_position_encoding_for_image_sequence(self, B, T, H, W, device):
        """
        为图像序列生成位置编码
        返回: (B, T, H, W, D) 用于广播到图像特征
        """
        if self.encoding_type == 'temporal_only':
            # 只添加时间编码 (B, T, 1, 1, D)
            temp_pe = self.temporal_pe[:T, :].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return temp_pe.expand(B, T, H, W, self.embed_dim)
            
        elif self.encoding_type == 'spatial_only':
            # 只添加空间编码
            pe_h = self.spatial_pe_h[:H, :].unsqueeze(1)  # (H, 1, D_h)
            pe_w = self.spatial_pe_w[:W, :]  # (W, D_w)
            
            # 组合空间编码
            spatial_pe = torch.cat([
                pe_h.expand(H, W, -1),
                pe_w.unsqueeze(0).expand(H, W, -1)
            ], dim=-1)  # (H, W, D)
            
            # 扩展到批次和时间维度
            spatial_pe = spatial_pe.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
            return spatial_pe.expand(B, T, H, W, self.embed_dim)
            
        elif self.encoding_type == 'spatiotemporal':
            # 联合时空编码
            temp_pe = self.temporal_pe[:T, :self.temp_dim]  # (T, D_t)
            pe_h = self.spatial_pe_h[:H, :]  # (H, D_h)
            pe_w = self.spatial_pe_w[:W, :]  # (W, D_w)
            
            # 构建完整的时空编码
            # 时间维度: (T, D_t) -> (T, 1, 1, D_t)
            temp_pe = temp_pe.unsqueeze(1).unsqueeze(2)
            # 空间维度: (H, D_h) + (W, D_w) -> (H, W, D_h + D_w)
            spatial_pe = torch.cat([
                pe_h.unsqueeze(1).expand(H, W, self.h_dim),
                pe_w.unsqueeze(0).expand(H, W, self.w_dim)
            ], dim=-1)
            spatial_pe = spatial_pe.unsqueeze(0)  # (1, H, W, D_h + D_w)
            
            # 组合: (T, 1, 1, D_t) + (1, H, W, D_h + D_w) -> (T, H, W, D)
            temp_pe = temp_pe.expand(T, H, W, self.temp_dim)
            spatial_pe = spatial_pe.expand(T, H, W, self.h_dim + self.w_dim)
            
            full_pe = torch.cat([temp_pe, spatial_pe], dim=-1)  # (T, H, W, D)
            full_pe = full_pe.unsqueeze(0)  # (1, T, H, W, D)
            
            return full_pe.expand(B, T, H, W, self.embed_dim)
            
        elif self.encoding_type == 'factorized':
            # 分解编码：时间和空间编码相加
            temp_pe = self.temporal_pe[:T, :].unsqueeze(1).unsqueeze(2)  # (T, 1, 1, D)
            
            pe_h = self.spatial_pe_h[:H, :self.embed_dim // 2]
            pe_w = self.spatial_pe_w[:W, :self.embed_dim - self.embed_dim // 2]
            spatial_pe = torch.cat([
                pe_h.unsqueeze(1).expand(H, W, -1),
                pe_w.unsqueeze(0).expand(H, W, -1)
            ], dim=-1).unsqueeze(0)  # (1, H, W, D)
            
            # 广播相加
            full_pe = temp_pe.expand(T, H, W, self.embed_dim) + \
                      spatial_pe.expand(T, H, W, self.embed_dim)
            full_pe = full_pe.unsqueeze(0)  # (1, T, H, W, D)
            
            return full_pe.expand(B, T, H, W, self.embed_dim)
            
        elif self.encoding_type == 'learned_2d':
            # 可学习编码
            pos_encoding = self.temporal_embedding[:, :T, :, :, :] + \
                          self.spatial_embedding[:, :, :H, :W, :]
            return pos_encoding.expand(B, T, H, W, self.embed_dim)
    
    def _add_position_encoding_to_features(self, x, device):
        """
        为已展平的特征序列添加位置编码
        x: (B, T, D)
        返回: (B, T, D)
        """
        B, T, D = x.shape
        
        # 这里简化处理，只添加时间编码
        # 如果需要完整的空间信息，需要保持空间结构
        # if hasattr(self, 'temporal_pe'):
        #     temp_pe = self.temporal_pe[:T, :].unsqueeze(0)  # (1, T, D)
        #     x = x + temp_pe

        # 暂时不考虑时序编码，这里只应对展平后的情况
        if self.encoding_type == 'learned':
            pos_encoding = self.embedding[:, :T, :]
            x = x + pos_encoding
        
        return self.dropout(x)
    
    def get_encoding(self, temporal_idx=None, height_idx=None, width_idx=None):
        """
        获取特定位置的编码
        
        Args:
            temporal_idx: 时间索引，可以是标量或张量
            height_idx: 高度索引
            width_idx: 宽度索引
        
        Returns:
            对应位置的位置编码
        """
        if self.encoding_type == 'learned_2d':
            if temporal_idx is None:
                temporal_idx = torch.arange(self.temporal_dim)
            if height_idx is None:
                height_idx = torch.arange(self.spatial_height)
            if width_idx is None:
                width_idx = torch.arange(self.spatial_width)
            
            return self.temporal_embedding[:, temporal_idx, :, :, :] + \
                   self.spatial_embedding[:, :, height_idx, width_idx, :]
        else:
            # 对于固定编码，返回预计算的值
            encoding = self._get_position_encoding_for_image_sequence(
                1, self.temporal_dim, self.spatial_height, self.spatial_width,
                self.temporal_pe.device if hasattr(self, 'temporal_pe') else 'cpu'
            )
            return encoding.squeeze(0)


class MultiModalPositionalEncoding(nn.Module):
    """
    多模态位置编码 - 支持混合1D和2D数据
    """
    def __init__(
        self,
        # 1D 序列配置
        seq1d_len: int = None,
        seq1d_dim: int = None,
        # 2D 图像配置
        image_height: int = None,
        image_width: int = None,
        image_dim: int = None,
        # 共享配置
        dropout: float = 0.1
    ):
        super().__init__()

        # 1D 序列位置编码
        if seq1d_len is not None:
            self.pos_enc_1d = PositionalEncoding(
                d_model=seq1d_dim,
                dropout=dropout,
                max_len=seq1d_len
            )
        else:
            self.pos_enc_1d = None
            
        # 2D 图像位置编码
        if image_height is not None:
            self.pos_enc_2d = ImageSequencePositionalEncoding(
                temporal_dim=1,  # 单帧
                spatial_height=image_height,
                spatial_width=image_width,
                embed_dim=image_dim,
                encoding_type='spatial_only',
                dropout=dropout
            )
        else:
            self.pos_enc_2d = None
    
    def forward(self, data_1d=None, data_2d=None):
        """
        Args:
            data_1d: [B, seq_len, dim] - 1D序列 (文本、时序数据等)
            data_2d: [B, H*W, dim] - 展平的2D图像
        
        Returns:
            concatenated: [B, total_seq_len, dim] - 拼接后的多模态序列
        """
        encoded_parts = []
        
        # 编码 1D 数据
        if data_1d is not None and self.pos_enc_1d is not None:
            encoded_1d = self.pos_enc_1d(data_1d)
            encoded_parts.append(encoded_1d)
        
        # 编码 2D 数据
        if data_2d is not None and self.pos_enc_2d is not None:
            # data_2d 已经展平: [B, H*W, dim]
            encoded_2d = self.pos_enc_2d(data_2d)
            encoded_parts.append(encoded_2d)
        
        # 拼接不同模态
        if len(encoded_parts) == 0:
            raise ValueError("至少需要提供一种模态的数据")
        
        # 沿序列维度拼接
        multimodal_sequence = torch.cat(encoded_parts, dim=1)
        return multimodal_sequence


class MultiModalEmbedding(nn.Module):
    """
    完整的多模态嵌入层，组合了：
    1. 位置编码 (Position Encoding)
    2. 模态类型嵌入 (Modality Type Embedding)
    3. 可选的片段嵌入 (Segment Embedding)
    
    组合策略：
        最终嵌入 = 输入特征 + 位置编码 + 模态类型嵌入 + [片段嵌入]
    
    Args:
        modality_configs: 字典，定义每个模态的配置
        unified_dim: 统一的嵌入维度
        num_modalities: 模态数量
        use_segment_embedding: 是否使用片段嵌入
        dropout: dropout概率
        
    Example:
        ```python
        config = {
            'text': {
                'type': '1d',
                'max_len': 512,
                'dim': 768
            },
            'image': {
                'type': '2d',
                'height': 16,
                'width': 16,
                'dim': 768
            },
            'audio': {
                'type': '1d',
                'max_len': 1000,
                'dim': 768
            }
        }
        embedder = MultiModalEmbedding(config, unified_dim=768)
        ```
    """
    def __init__(
        self,
        modality_configs,
        unified_dim=512,
        num_modalities=None,
        use_segment_embedding=False,
        max_segments=10,
        dropout=0.1,
        position_encoding_type='sinusoidal'
    ):
        super().__init__()
        
        self.modality_configs = modality_configs
        self.unified_dim = unified_dim
        self.use_segment_embedding = use_segment_embedding
        self.dropout = nn.Dropout(dropout)
        
        # 自动计算模态数量
        if num_modalities is None:
            num_modalities = len(modality_configs)
        
        # 1. 模态类型嵌入 (Modality Type Embedding)
        # 为每种模态分配一个可学习的类型向量
        self.modality_type_embedding = nn.Embedding(num_modalities, unified_dim)
        nn.init.normal_(self.modality_type_embedding.weight, std=0.02)
        
        # 2. 维度投影层 (如果模态维度不一致)
        self.projection_layers = nn.ModuleDict()
        for modality_name, config in modality_configs.items():
            if config['dim'] != unified_dim:
                self.projection_layers[modality_name] = nn.Linear(
                    config['dim'], unified_dim
                )
        
        # 3. 位置编码 (Position Encoding)
        self.position_encodings = nn.ModuleDict()
        for modality_name, config in modality_configs.items():
            if config['type'] == '1d':
                # 1D序列位置编码
                self.position_encodings[modality_name] = PositionalEncoding(
                    embed_dim=unified_dim,
                    max_len=config['max_len'],
                    encoding_type=position_encoding_type,
                    dropout=0.0,  # dropout在最后统一应用
                    scale=False  # 不在位置编码内部scale
                )
            elif config['type'] == '2d':
                # 2D图像位置编码
                self.position_encodings[modality_name] = ImageSequencePositionalEncoding(
                    temporal_dim=config.get('temporal_dim', 1),
                    spatial_height=config['height'],
                    spatial_width=config['width'],
                    embed_dim=unified_dim,
                    encoding_type=config.get('spatial_encoding_type', 'spatial_only'),
                    dropout=0.0
                )
        
        # 4. 片段嵌入 (Segment Embedding) - 用于区分同一模态内的不同片段
        if use_segment_embedding:
            self.segment_embedding = nn.Embedding(max_segments, unified_dim)
            nn.init.normal_(self.segment_embedding.weight, std=0.02)
        
        # 5. 层归一化 (推荐在嵌入后进行)
        self.layer_norm = nn.LayerNorm(unified_dim)
        
        # 创建模态名称到ID的映射
        self.modality_name_to_id = {
            name: idx for idx, name in enumerate(modality_configs.keys())
        }
    
    def forward(self, modality_inputs, segment_ids=None):
        """
        Args:
            modality_inputs: 字典，key为模态名称，value为对应的输入张量
                {
                    'text': [B, L_text, D_text],
                    'image': [B, L_image, D_image],
                    ...
                }
            segment_ids: 可选，字典，标识每个token属于哪个片段
                {
                    'text': [B, L_text],
                    'image': [B, L_image],
                }
        
        Returns:
            multimodal_embedding: [B, total_length, unified_dim]
            modality_masks: 字典，记录每个模态在序列中的位置范围
        """
        batch_size = None
        embeddings_list = []
        modality_masks = {}
        current_offset = 0
        
        for modality_name, input_tensor in modality_inputs.items():
            if modality_name not in self.modality_configs:
                raise ValueError(f"Unknown modality: {modality_name}")
            
            B, L, D = input_tensor.shape
            if batch_size is None:
                batch_size = B
            assert B == batch_size, "All modalities must have the same batch size"
            
            device = input_tensor.device
            
            # ===== 步骤1: 维度投影 =====
            if modality_name in self.projection_layers:
                # 需要投影到统一维度
                projected = self.projection_layers[modality_name](input_tensor)
            else:
                # 维度已经匹配
                projected = input_tensor
            
            # ===== 步骤2: 添加位置编码 =====
            # 注意：位置编码通常是加性的
            if modality_name in self.position_encodings:
                with_position = self.position_encodings[modality_name](projected)
            else:
                with_position = projected
            
            # ===== 步骤3: 添加模态类型嵌入 =====
            # 获取该模态的ID
            modality_id = self.modality_name_to_id[modality_name]
            # 创建模态类型ID张量 [B, L]
            modality_type_ids = torch.full(
                (B, L), modality_id, dtype=torch.long, device=device
            )
            # 获取模态类型嵌入 [B, L, unified_dim]
            modality_type_emb = self.modality_type_embedding(modality_type_ids)
            
            # 加上模态类型嵌入
            with_modality_type = with_position + modality_type_emb
            
            # ===== 步骤4: 添加片段嵌入 (可选) =====
            if self.use_segment_embedding and segment_ids is not None:
                if modality_name in segment_ids:
                    seg_ids = segment_ids[modality_name]  # [B, L]
                    segment_emb = self.segment_embedding(seg_ids)  # [B, L, unified_dim]
                    final_embedding = with_modality_type + segment_emb
                else:
                    final_embedding = with_modality_type
            else:
                final_embedding = with_modality_type
            
            # ===== 步骤5: 层归一化 =====
            final_embedding = self.layer_norm(final_embedding)
            
            # ===== 步骤6: Dropout =====
            final_embedding = self.dropout(final_embedding)
            
            # 记录该模态在最终序列中的位置
            modality_masks[modality_name] = {
                'start': current_offset,
                'end': current_offset + L,
                'length': L
            }
            current_offset += L
            
            embeddings_list.append(final_embedding)
        
        # 拼接所有模态的嵌入
        multimodal_embedding = torch.cat(embeddings_list, dim=1)  # [B, total_L, unified_dim]
        
        return multimodal_embedding, modality_masks
    
    def get_modality_attention_mask(self, modality_masks, allow_cross_modality=True):
        """
        生成多模态注意力掩码
        
        Args:
            modality_masks: 从forward返回的模态位置信息
            allow_cross_modality: 是否允许跨模态attention
        
        Returns:
            attention_mask: [total_length, total_length]
                True表示允许注意，False表示mask掉
        """
        total_length = sum(m['length'] for m in modality_masks.values())
        
        if allow_cross_modality:
            # 允许所有位置互相注意
            return torch.ones(total_length, total_length, dtype=torch.bool)
        else:
            # 只允许同一模态内部注意
            mask = torch.zeros(total_length, total_length, dtype=torch.bool)
            for modality_info in modality_masks.values():
                start, end = modality_info['start'], modality_info['end']
                mask[start:end, start:end] = True
            return mask


class NormalSelfAttention(nn.Module):
    """
    自注意力机制模块
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        attn_dropout=0.1,
        output_dropout=0.1,
        store_attention=False,  # 是否存储注意力权重
    ):
        super(NormalSelfAttention, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "num_heads: {} does not divide embed_dim: {} exactly".format(num_heads, embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.store_attention = store_attention
        self.nets = nn.ModuleDict()

        self.nets["qkv"] = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.nets["attn_dropout"] = nn.Dropout(attn_dropout)
        self.nets["output_dropout"] = nn.Dropout(output_dropout)
        self.nets["output"] = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # 用于存储注意力权重
        self.attention_weights = None
   
    def forward(self, x, return_attention=False):
        """
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
        
        Args:
            x: 输入张量 (B, T, D)
            return_attention: 是否返回注意力权重
        
        Returns:
            如果 return_attention=False: 返回 y (B, T, D)
            如果 return_attention=True: 返回 (y, attention_weights)
        """
        assert x.dim() == 3, "Input x must be 3D (B, T, D)"
        B, T, D = x.shape
        assert (
            T <= self.context_length
        ), "self-attention module can only handle sequences up to {} in length but got length {}".format(
            self.context_length, T
        )
        assert D == self.embed_dim, "Input embedding dim must be equal to module embedding dim"
        NH = self.num_heads
        DH = D // NH  # dimension per head

        qkv = self.nets["qkv"](x)  # (B, T, 3 * D)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # each is (B, T, D)
        k = k.view(B, T, NH, DH).transpose(1, 2)  # (B, NH, T, DH)
        q = q.view(B, T, NH, DH).transpose(1, 2)  # (B, NH, T, DH)
        v = v.view(B, T, NH, DH).transpose(1, 2)  # (B, NH, T, DH)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, NH, T, T)
        att = F.softmax(att, dim=-1)
        
        # 存储注意力权重（在dropout之前）
        if self.store_attention or return_attention:
            self.attention_weights = att.detach().clone()
        
        att = self.nets["attn_dropout"](att)
        
        y = att @ v  # (B, NH, T, DH)
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # re-assemble all head outputs side by side (B, T, D)
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)
        
        if return_attention:
            return y, self.attention_weights
        return y

    def output_shape(self, input_shape=None):
        return list(input_shape)

class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制模块
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        attn_dropout=0.1,
        output_dropout=0.1,
        store_attention=False,  # 是否存储注意力权重
    ):
        super(CausalSelfAttention, self).__init__()

        assert embed_dim % num_heads == 0, \
          "num_heads: {} does not divide embed_dim: {} exactly".format(num_heads, embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.store_attention = store_attention
        self.nets = nn.ModuleDict()

        self.nets["qkv"] = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.nets["attn_dropout"] = nn.Dropout(self.attn_dropout)
        self.nets["output_dropout"] = nn.Dropout(self.output_dropout)
        self.nets["output"] = nn.Linear(self.embed_dim, self.embed_dim)

        mask = torch.tril(torch.ones(context_length, context_length)).view(
            1, 1, context_length, context_length
        )
        self.register_buffer("mask", mask)
        
        # 用于存储注意力权重
        self.attention_weights = None
   
    def forward(self, x, return_attention=False):
        """
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
        
        Args:
            x: 输入张量 (B, T, D)
            return_attention: 是否返回注意力权重
        
        Returns:
            如果 return_attention=False: 返回 y (B, T, D)
            如果 return_attention=True: 返回 (y, attention_weights)
        """
        assert x.dim() == 3, "Input x must be 3D (B, T, D)"
        B, T, D = x.shape
        assert (
            T <= self.context_length
        ), "self-attention module can only handle sequences up to {} in length but got length {}".format(
            self.context_length, T
        )
        assert D == self.embed_dim, "Input embedding dim must be equal to module embedding dim"
        NH = self.num_heads
        DH = D // NH  # dimension per head

        qkv = self.nets["qkv"](x)  # (B, T, 3 * D)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # each is (B, T, D)
        k = k.view(B, T, NH, DH).transpose(1, 2)  # (B, NH, T, DH)
        q = q.view(B, T, NH, DH).transpose(1, 2)  # (B, NH, T, DH)
        v = v.view(B, T, NH, DH).transpose(1, 2)  # (B, NH, T, DH)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, NH, T, T)
        att = att.masked_fill(self.mask[..., :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        
        # 存储注意力权重（在dropout之前）
        if self.store_attention or return_attention:
            self.attention_weights = att.detach().clone()
        
        att = self.nets["attn_dropout"](att)
        
        y = att @ v  # (B, NH, T, DH)
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # re-assemble all head outputs side by side (B, T, D)
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)
        
        if return_attention:
            return y, self.attention_weights
        return y

    def output_shape(self, input_shape=None):
        return list(input_shape)

class CrossAttention(nn.Module):
    """
    TODO：交叉注意力机制模块
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        attn_dropout=0.1,
        output_dropout=0.1,
        store_attention=False,  # 是否存储注意力权重
    ):
        super(CrossAttention, self).__init__()

        assert embed_dim % num_heads == 0, \
          "@CrossAttention: num_heads: {} does not divide embed_dim: {} exactly".format(num_heads, embed_dim)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.store_attention = store_attention
        self.nets = nn.ModuleDict()

        self.nets["q"] = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.nets["kv"] = nn.Linear(self.embed_dim, 2 * self.embed_dim, bias=False)
        
        self.nets["attn_dropout"] = nn.Dropout(self.attn_dropout)
        self.nets["output_dropout"] = nn.Dropout(self.output_dropout)
        self.nets["output"] = nn.Linear(self.embed_dim, self.embed_dim)
        
        # 用于存储注意力权重
        self.attention_weights = None

    def forward(self, x, context, return_attention=False):
        """
        x: 对应查询(query），来自解码器输入
        context: 对应键(key)和值(value)，来自编码器输出
        
        Args:
            x: 查询张量 (B, T, D)
            context: 上下文张量 (B, S, D)
            return_attention: 是否返回注意力权重
        
        Returns:
            如果 return_attention=False: 返回 y (B, T, D)
            如果 return_attention=True: 返回 (y, attention_weights)
        """
        assert x.dim() == 3, "@CrossAttention: Input x must be 3D (B, T, D)"
        assert context.dim() == 3, "@CrossAttention: Input context must be 3D (B, S, D)"
        B, T_q, D = x.shape
        B2, T_kv, D2 = context.shape

        assert B == B2, "@CrossAttention: Batch size of x and context must be the same"
        assert D == self.embed_dim and D2 == self.embed_dim, \
            "@CrossAttention: Input embedding dim must be equal to module embedding dim"
        assert (
            T_kv <= self.context_length
        ), "@CrossAttention: module can only handle context sequences up to {} in length but got length {}".format(
            self.context_length, T_kv
        )
        assert (
            T_q <= self.context_length
        ), "@CrossAttention: module can only handle query sequences up to {} in length but got length {}".format(
            self.context_length, T_q
        )

        NH = self.num_heads
        DH = D // NH  # dimension per head

        q = self.nets["q"](x)  # (B, T, D)
        kv = self.nets["kv"](context)  # (B, S, 2 * D)
        k, v = torch.chunk(kv, 2, dim=-1) # each is (B, S, D)

        q = q.view(B, T_q, NH, DH).transpose(1, 2)  # (B, NH, T, DH)
        k = k.view(B, T_kv, NH, DH).transpose(1, 2)  # (B, NH, S, DH)
        v = v.view(B, T_kv, NH, DH).transpose(1, 2)  # (B, NH, S, DH)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, NH, T, S)
        att = F.softmax(att, dim=-1)
        
        # 存储注意力权重（在dropout之前）
        if self.store_attention or return_attention:
            self.attention_weights = att.detach().clone()
        
        att = self.nets["attn_dropout"](att)
        
        y = att @ v  # (B, NH, T, DH)
        y = y.transpose(1, 2).contiguous().view(B, T_q, D)  # re-assemble all head outputs side by side (B, T, D)
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)
        
        if return_attention:
            return y, self.attention_weights
        return y

    def output_shape(self, input_shape=None):
        return list(input_shape)


class NormalTransformerBlock(Module):
    """
    标准Transformer块，包含自注意力和前馈网络
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        attn_dropout=0.1,
        output_dropout=0.1,
        ffw_hidden_dim=None,
        ffw_dropout=None,
        activation=nn.ReLU,
    ):
        super(NormalTransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.ffw_hidden_dim = ffw_hidden_dim if ffw_hidden_dim is not None else 4 * embed_dim
        self.ffw_dropout = ffw_dropout if ffw_dropout is not None else output_dropout
        self.activation = activation
        self.nets = nn.ModuleDict()

        self.nets["attention"] = NormalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            attn_dropout=attn_dropout,
            output_dropout=output_dropout,
        )
        self.nets["mlp"] = nn.Sequential(
            nn.Linear(embed_dim, self.ffw_hidden_dim),
            self.activation,
            nn.Linear(self.ffw_hidden_dim, embed_dim),
            nn.Dropout(self.ffw_dropout),
        )

        self.nets["ln1"] = nn.LayerNorm(embed_dim)
        self.nets["ln2"] = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
        """
        x = x + self.nets["attention"](self.nets["ln1"](x))
        x = x + self.nets["mlp"](self.nets["ln2"](x))
        return x

    def output_shape(self, input_shape=None):
        return list(input_shape)

class CausalTransformerBlock(Module):
    """
    因果Transformer块，包含因果自注意力和前馈网络
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        attn_dropout=0.1,
        output_dropout=0.1,
        ffw_hidden_dim=None,
        ffw_dropout=None,
        activation=nn.ReLU,
    ):
        super(CausalTransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.ffw_hidden_dim = ffw_hidden_dim if ffw_hidden_dim is not None else 4 * embed_dim
        self.ffw_dropout = ffw_dropout if ffw_dropout is not None else output_dropout
        self.activation = activation()
        self.nets = nn.ModuleDict()

        self.nets["attention"] = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            attn_dropout=attn_dropout,
            output_dropout=output_dropout,
        )
        self.nets["mlp"] = nn.Sequential(
            nn.Linear(embed_dim, self.ffw_hidden_dim),
            self.activation,
            nn.Linear(self.ffw_hidden_dim, embed_dim),
            nn.Dropout(self.ffw_dropout),
        )

        self.nets["ln1"] = nn.LayerNorm(embed_dim)
        self.nets["ln2"] = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
        """
        x = x + self.nets["attention"](self.nets["ln1"](x))
        x = x + self.nets["mlp"](self.nets["ln2"](x))
        return x

    def output_shape(self, input_shape=None):
        return list(input_shape)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器的单层实现
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        attn_dropout=0.1,
        output_dropout=None,
        ffw_hidden_dim=None,
        ffw_dropout=None,
        activation='relu',
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.nets = nn.ModuleDict()
        self.nets["selfattention"] = NormalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            attn_dropout=attn_dropout,
            output_dropout=output_dropout if output_dropout is not None else attn_dropout,
        ) 
        # feed-forward network (MLP)
        self.ffw_hidden_dim = ffw_hidden_dim if ffw_hidden_dim is not None else 4 * embed_dim
        # ffw dropout fallbacks to provided ffw_dropout -> output_dropout -> attn_dropout
        self.ffw_dropout = ffw_dropout if ffw_dropout is not None else (
            output_dropout if output_dropout is not None else attn_dropout
        )

        # activation may be a string or a callable; use project's get_activation helper
        self.activation = get_activation(activation)()

        self.nets["mlp"] = nn.Sequential(
            nn.Linear(embed_dim, self.ffw_hidden_dim),
            self.activation,
            nn.Linear(self.ffw_hidden_dim, embed_dim),
            nn.Dropout(self.ffw_dropout),
        )

        # layer norms for pre-norm transformer style
        self.nets["ln1"] = nn.LayerNorm(embed_dim)
        self.nets["ln2"] = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, T, D)
        Returns: (B, T, D)
        Implements pre-norm encoder layer with residual connections:
          x = x + SelfAttention(LN(x))
          x = x + MLP(LN(x))
        """
        assert x.dim() == 3, "TransformerEncoderLayer expects input of shape (B, T, D)"
        # self-attention with pre-norm and residual
        x = x + self.nets["selfattention"](self.nets["ln1"](x))
        # feed forward with pre-norm and residual
        x = x + self.nets["mlp"](self.nets["ln2"](x))
        return x

    def output_shape(self, input_shape=None):
        return list(input_shape)


class TransformerDecoderLayer(nn.Module):
    """
    Transformer 解码器的单层实现（pre-norm）

    结构：
      x = x + CausalSelfAttention(LN1(x))
      x = x + CrossAttention(LN2(x), context)   # 当 context 为 None 时跳过
      x = x + MLP(LN3(x))

    支持可选的 cross-attention（context=None 时仅做自注意力与前馈）。
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        attn_dropout=0.1,
        output_dropout=None,
        ffw_hidden_dim=None,
        ffw_dropout=None,
        activation='relu',
    ):
        super(TransformerDecoderLayer, self).__init__()

        # 输出 dropout 回退策略: output_dropout -> attn_dropout
        output_dropout = output_dropout if output_dropout is not None else attn_dropout

        self.nets = nn.ModuleDict()
        # 因果自注意力（masked self-attention）
        self.nets["selfattention"] = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            attn_dropout=attn_dropout,
            output_dropout=output_dropout,
        )

        # 交叉注意力（query 来自 decoder，kv 来自 encoder context）
        self.nets["crossattention"] = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            attn_dropout=attn_dropout,
            output_dropout=output_dropout,
        )

        # feed-forward
        self.ffw_hidden_dim = ffw_hidden_dim if ffw_hidden_dim is not None else 4 * embed_dim
        self.ffw_dropout = ffw_dropout if ffw_dropout is not None else output_dropout

        # activation helper
        self.activation = get_activation(activation)()

        self.nets["mlp"] = nn.Sequential(
            nn.Linear(embed_dim, self.ffw_hidden_dim),
            self.activation,
            nn.Linear(self.ffw_hidden_dim, embed_dim),
            nn.Dropout(self.ffw_dropout),
        )

        # three layer norms for pre-norm: self-attn, cross-attn, ffw
        self.nets["ln1"] = nn.LayerNorm(embed_dim)
        self.nets["ln2"] = nn.LayerNorm(embed_dim)
        self.nets["ln3"] = nn.LayerNorm(embed_dim)

    def forward(self, x, context=None):
        """
        x: (B, T, D)
        context: (B, S, D) or None
        返回: (B, T, D)
        """
        assert context is not None, "@TransformerDecoderLayer: context input for TransformerDecoderLayer not implemented yet"
        assert x.dim() == 3, "TransformerDecoderLayer expects input of shape (B, T, D)"

        # masked self-attention (causal)
        x = x + self.nets["selfattention"](self.nets["ln1"](x))

        # cross-attention
        assert context.dim() == 3, "context must be 3D (B, S, D)"
        x = x + self.nets["crossattention"](self.nets["ln2"](x), context)

        # feed-forward
        x = x + self.nets["mlp"](self.nets["ln3"](x))
        return x

    def output_shape(self, input_shape=None):
        return list(input_shape)


class Transformer(Module):
  """
  Transformer 编解码器实现
  """
  def __init__(
    self,
    embed_dim,
    context_length,
    attn_dropout=0.1,
    output_dropout=None,
    ffw_hidden_dim=None,
    ffw_dropout=None,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    activation='relu',
    position_encoding_type='sinusoidal',  # 位置编码方式
    use_segment_embedding=False,  # 是否使用片段嵌入
  ):
    super(Transformer, self).__init__()

    self.embed_dim = embed_dim
    self.context_length = context_length
    
    # 配置位置编码settings
    self.position_encoding_type = position_encoding_type
    self.use_segment_embedding = use_segment_embedding
    # 添加位置编码器配置：编码器融合场景特征，解码器条件特征
    self.encoder_position_config = {
        # 'text': {'type': '1d', 'max_len': 10, 'dim': 512},  # 任务文本维度
        'cls': {'type': '1d', 'max_len': 3, 'dim': 512},  # 全局信息标记维度
        # 'robot0_eef_pos': {'type': '1d', 'max_len': 1, 'dim': 512},  # 位置维度
        # 'joint': {'type': '1d', 'max_len': 1, 'dim': 512},  # 关节维度
        'agentview_image': {'type': '2d', 'temporal_dim': 1, 'height': 3, 'width': 3, 'dim': 512, 'spatial_encoding_type': 'learned'},  # 图像维度
        'agentview_depth': {'type': '2d', 'temporal_dim': 1, 'height': 3, 'width': 3, 'dim': 512, 'spatial_encoding_type': 'learned'},  # 深度维度
    }
    self.decoder_position_config = {
        # 'text': {'type': '1d', 'max_len': 10, 'dim': 512},  # 任务文本维度
        # 'cls': {'type': '1d', 'max_len': 3, 'dim': 512},  # 动作序列标记
        'robot0_eef_pos_step_traj_current': {'type': '1d', 'max_len': 10, 'dim': 512},  # 轨迹特征维度
        # 'robot0_eef_pos_past_traj_delta': {'type': '1d', 'max_len': 9, 'dim': 512},  # 轨迹变化特征维度
    }
    # self.cls_input_image_depth = nn.Embedding(1, 512)
    # cls_input_image_depth = self.cls_input_image_depth.weight
    # self.cls_token_image_depth = cls_input_image_depth.repeat(3, 1)

    # self.cls_input_pos_traj = nn.Embedding(1, 512)
    # cls_input_pos_traj = self.cls_input_pos_traj.weight
    # self.cls_token_pos_traj = cls_input_pos_traj.repeat(3, 1)
    # self.register_buffer(
    #     "image_depth_encoder_pos_enc",
    #     create_sinusoidal_pos_embedding(3, 512).unsqueeze(0),  # [1, 3, 512]
    # )
    # self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(512 // 2)
    # self.register_buffer(
    #     "pos_traj_decoder_pos_enc",
    #     create_sinusoidal_pos_embedding(3+10, 512).unsqueeze(0),
    # )
    # 创建嵌入层
    self.encoder_embedder = MultiModalEmbedding(
        modality_configs=self.encoder_position_config,
        unified_dim=self.embed_dim,
        num_modalities=len(self.encoder_position_config),
        use_segment_embedding=self.use_segment_embedding,
        max_segments=2,
        dropout=0.1,
        position_encoding_type='learned_additive'
    )
    self.decoder_embedder = MultiModalEmbedding(
        modality_configs=self.decoder_position_config,
        unified_dim=self.embed_dim,
        num_modalities=len(self.decoder_position_config),
        use_segment_embedding=self.use_segment_embedding,
        dropout=0.1,
        position_encoding_type='sinusoidal'
    )

    # 创建编码器和解码器层
    self.nets = nn.ModuleDict()
    # encoder
    self.nets["encoder"] = nn.Sequential(
      *[
        TransformerEncoderLayer(
          embed_dim=embed_dim,
          num_heads=num_heads,
          context_length=context_length,
          attn_dropout=attn_dropout,
          output_dropout=output_dropout,
          ffw_hidden_dim=ffw_hidden_dim,
          ffw_dropout=ffw_dropout,
          activation=activation,
        )
        for _ in range(num_encoder_layers)
      ]
    )
    # decoder 为了提供query功能，需要修改为ModuleList容器
    self.nets["decoder"] = nn.ModuleList(
      [
        TransformerDecoderLayer(
          embed_dim=embed_dim,
          num_heads=num_heads,
          context_length=context_length,
          attn_dropout=attn_dropout,
          output_dropout=output_dropout,
          ffw_hidden_dim=ffw_hidden_dim,
          ffw_dropout=ffw_dropout,
          activation=activation,
        )
      for _ in range(num_decoder_layers)
      ]
    )
    # 创建注意力可视化工具
    self.visualizer = AttentionVisualizer()

    self._reset_parameters()

  def _reset_parameters(self):
      for p in self.nets.parameters():
        if p.dim() > 1:
          nn.init.xavier_uniform_(p)

  def forward(self, enc, dec, return_attention_weights=False, fill_mode: str=None):
    """
    Transformer 编解码器前向计算
    Args:
        enc: 字典,编码器输入特征
        dec: 字典,解码器输入特征
        return_attention_weights: 是否返回注意力权重
        fill_mode: 补全模式，可选:
            - 'one_shot': 一次性预测所有位置（速度快）
            - 'autoregressive': 逐步自回归补全（质量高）
            - 'sliding_window': 原有的滑动窗口模式
    Returns:
        如果 return_attention_weights=False:
            decoder_output: 解码器输出特征
        如果 return_attention_weights=True:
            decoder_output: 解码器输出特征
            attention_weights: 字典,包含编码器和解码器的注意力权重
    """
    # 添加位置编码
    enc_embedd, _ = self.encoder_embedder(
        modality_inputs=enc,
        segment_ids=None
    )
    
    # 编码器前向
    encoder_output = self.nets["encoder"](enc_embedd)
    
    if fill_mode is None:
        # 非自回归模式:一次性处理整个序列
        dec_embedd, _ = self.decoder_embedder(
            modality_inputs=dec,
            segment_ids=None
        )
        decoder_output = dec_embedd
        for layer in self.nets["decoder"]:
            decoder_output = layer(decoder_output, context=encoder_output)
    else:
        if fill_mode == 'one_shot':
            # 一次性预测模式：输入有padding，一次输出所有位置
            # 适合：速度要求高，可以接受略低质量
            dec_embedd, _ = self.decoder_embedder(
                modality_inputs=dec,
                segment_ids=None
            )
            decoder_output = dec_embedd
            for layer in self.nets["decoder"]:
                decoder_output = layer(decoder_output, context=encoder_output)
            # decoder_output shape: [B, 10, hidden]
            # 所有10个位置同时预测，后面的位置无法看到前面位置的预测
        
        elif fill_mode == 'autoregressive':
            # 自回归补全模式：逐步填充padding位置
            # 适合：质量优先，输入有空缺需要逐步补全
            _, seq_len, _ = dec['robot0_eef_pos_step_traj_current'].shape
            current_seq = dec['robot0_eef_pos_step_traj_current'].clone()  # [B, seq_len, 3]

            all_outputs = []
            
            for t in range(seq_len):
                # 构造当前步的输入
                temp_dec = {'robot0_eef_pos_step_traj_current': current_seq}
                
                # Embedding + Decoder
                dec_embedd, _ = self.decoder_embedder(
                    modality_inputs=temp_dec,
                    segment_ids=None
                )

                decoder_input = dec_embedd
                
                # 通过decoder layers with causal attention
                for layer in self.nets["decoder"]:
                    decoder_input = layer(
                        decoder_input, 
                        context=encoder_output,
                    )
                
                # 取第t个位置的输出
                output_t = decoder_input[:, t:t+1, :]  # [B, 1, hidden]
                all_outputs.append(output_t)

                if t != seq_len - 1:
                    current_seq[:, t+1:t+2, :] = output_t  # 用预测值更新当前序列

            decoder_output = torch.cat(all_outputs, dim=1)  # [B, seq_len, hidden]
        
        elif fill_mode == 'sliding_window':  # fill_mode == 'sliding_window' (原始模式)
            # 滑动窗口模式：滚动预测未来
            # 适合：已有完整历史，预测未来序列
            B, window_size, D = dec['robot0_eef_pos_step_traj_current'].shape
            
            # 初始化:使用dec_embedd作为初始窗口
            current_window = dec['robot0_eef_pos_step_traj_current'].clone()  # (B, window_size, D)
            all_outputs = []
            
            # 自回归生成,最大长度与decoder输入序列长度一致
            for step in range(window_size):
                # 将当前窗口通过解码器
                temp_dec = {'robot0_eef_pos_step_traj_current': current_window}
                dec_embedd, _ = self.decoder_embedder(
                    modality_inputs=temp_dec,
                    segment_ids=None
                )
                decoder_input = dec_embedd
                for layer in self.nets["decoder"]:
                    decoder_input = layer(decoder_input, context=encoder_output)
                
                # 取最后一个位置的输出作为新生成的token
                new_token = decoder_input[:, -1:, :]  # (B, 1, hidden)
                all_outputs.append(new_token)
                
                # 滑动窗口:左移并添加新token
                if step < window_size - 1:
                    # 窗口左移:去掉第一个token,在末尾添加新生成的token
                    current_window = torch.cat([
                        current_window[:, 1:, :],  # 去掉第一个token
                        new_token                   # 添加新生成的token
                    ], dim=1)  # (B, window_size, D)
            
            # 拼接所有输出
            decoder_output = torch.cat(all_outputs, dim=1)  # (B, window_size, hidden)
        else:
            raise ValueError(f"@Transformer: Unsupported fill_mode: {fill_mode}")

    if return_attention_weights:
        attention_weights = self.get_attention_weights()
        return decoder_output, attention_weights
    
    return decoder_output

  def get_attention_weights(self):
    """
    收集所有层的注意力权重
    
    Returns:
        attention_weights: 字典，结构如下:
        {
            'encoder': [
                {'self_attention': tensor(B, NH, T, T)},  # layer 0
                {'self_attention': tensor(B, NH, T, T)},  # layer 1
                ...
            ],
            'decoder': [
                {
                    'self_attention': tensor(B, NH, T, T),
                    'cross_attention': tensor(B, NH, T, S)
                },  # layer 0
                ...
            ]
        }
    """
    attention_weights = {
        'encoder': [],
        'decoder': []
    }
    
    # 收集编码器的注意力权重
    for i, layer in enumerate(self.nets["encoder"]):
      if hasattr(layer.nets["selfattention"], 'attention_weights'):
        attention_weights['encoder'].append({
          'self_attention': layer.nets["selfattention"].attention_weights
        })
    
    # 收集解码器的注意力权重
    for i, layer in enumerate(self.nets["decoder"]):
      layer_attn = {}
      if hasattr(layer.nets["selfattention"], 'attention_weights'):
        layer_attn['self_attention'] = layer.nets["selfattention"].attention_weights
      if hasattr(layer.nets["crossattention"], 'attention_weights'):
        layer_attn['cross_attention'] = layer.nets["crossattention"].attention_weights
      attention_weights['decoder'].append(layer_attn)
    
    return attention_weights
  
  def enable_attention_storage(self):
    """启用所有注意力模块的权重存储"""
    for layer in self.nets["encoder"]:
      if hasattr(layer.nets["selfattention"], 'store_attention'):
        layer.nets["selfattention"].store_attention = True
    
    for layer in self.nets["decoder"]:
      if hasattr(layer.nets["selfattention"], 'store_attention'):
        layer.nets["selfattention"].store_attention = True
      if hasattr(layer.nets["crossattention"], 'store_attention'):
        layer.nets["crossattention"].store_attention = True
  
  def disable_attention_storage(self):
    """禁用所有注意力模块的权重存储"""
    for layer in self.nets["encoder"]:
      if hasattr(layer.nets["selfattention"], 'store_attention'):
        layer.nets["selfattention"].store_attention = False
    
    for layer in self.nets["decoder"]:
      if hasattr(layer.nets["selfattention"], 'store_attention'):
        layer.nets["selfattention"].store_attention = False
      if hasattr(layer.nets["crossattention"], 'store_attention'):
        layer.nets["crossattention"].store_attention = False
  
  def output_shape(self, input_shape=None):
    return list(input_shape)





class AttentionVisualizer:
    """
    注意力可视化工具类
    
    提供多种注意力权重的可视化方法，包括：
    1. 热力图可视化
    2. 多头注意力对比
    3. 层级注意力对比
    4. 交叉注意力可视化
    
    使用示例:
        ```python
        # 1. 启用注意力存储
        model.enable_attention_storage()
        
        # 2. 前向传播
        output, attn_weights = model(enc, dec, return_attention_weights=True)
        
        # 3. 可视化
        visualizer = AttentionVisualizer()
        
        # 可视化编码器第0层的自注意力
        visualizer.plot_attention_heatmap(
            attn_weights['encoder'][0]['self_attention'],
            title='Encoder Layer 0 Self-Attention'
        )
        
        # 可视化解码器第0层的交叉注意力
        visualizer.plot_attention_heatmap(
            attn_weights['decoder'][0]['cross_attention'],
            title='Decoder Layer 0 Cross-Attention'
        )
        
        # 对比所有编码器层
        visualizer.plot_layer_comparison(
            attn_weights['encoder'],
            attention_type='self_attention',
            title='Encoder Self-Attention Across Layers'
        )
        ```
    """
    
    def __init__(self):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.plt = plt
            self.sns = sns
            self.has_matplotlib = True
        except ImportError:
            print("Warning: matplotlib or seaborn not installed. Visualization features will be limited.")
            self.has_matplotlib = False
    
    def plot_attention_heatmap(
        self,
        attention_weights,
        head_idx=None,
        batch_idx=0,
        title='Attention Heatmap',
        figsize=(10, 8),
        cmap='viridis',
        save_path=None,
        show=True,
        vmin=None,
        vmax=None
    ):
        """
        绘制注意力热力图
        
        Args:
            attention_weights: 注意力权重张量 (B, NH, T, S) 或 (B, NH, T, T)
            head_idx: 要可视化的注意力头索引。如果为None，则平均所有头
            batch_idx: 批次索引
            title: 图表标题
            figsize: 图表大小
            cmap: 颜色映射
            save_path: 保存路径，如果为None则不保存
            show: 是否显示图表
            vmin: 颜色映射的最小值
            vmax: 颜色映射的最大值
        
        Returns:
            fig, ax: matplotlib图表对象
        """
        if not self.has_matplotlib:
            print("Cannot plot: matplotlib not available")
            return None, None
        
        # 将张量移到CPU并转换为numpy
        if isinstance(attention_weights, torch.Tensor):
            attn = attention_weights[batch_idx].detach().cpu().numpy()
        else:
            attn = attention_weights[batch_idx]
        
        # 选择特定的注意力头或平均所有头
        if head_idx is not None:
            attn = attn[head_idx]  # (T, S)
            head_info = f"Head {head_idx}"
        else:
            attn = attn.mean(axis=0)  # 平均所有头 (T, S)
            head_info = "Average of all heads"
        
        # 创建图表
        fig, ax = self.plt.subplots(figsize=figsize)
        
        # 绘制热力图
        im = ax.imshow(attn, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # 添加颜色条
        cbar = self.plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        # 设置标签和标题
        ax.set_xlabel('Key/Value Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'{title}\n{head_info}')
        
        # 调整布局
        self.plt.tight_layout()
        
        # 保存图表
        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to {save_path}")
        
        # 显示图表
        if show:
            self.plt.show()
        
        return fig, ax
    
    def plot_multi_head_attention(
        self,
        attention_weights,
        batch_idx=0,
        title='Multi-Head Attention',
        figsize=None,
        cmap='viridis',
        save_path=None,
        show=True
    ):
        """
        并排可视化所有注意力头
        
        Args:
            attention_weights: 注意力权重张量 (B, NH, T, S)
            batch_idx: 批次索引
            title: 图表标题
            figsize: 图表大小，如果为None则自动计算
            cmap: 颜色映射
            save_path: 保存路径
            show: 是否显示图表
        
        Returns:
            fig, axes: matplotlib图表对象
        """
        if not self.has_matplotlib:
            print("Cannot plot: matplotlib not available")
            return None, None
        
        # 将张量移到CPU
        if isinstance(attention_weights, torch.Tensor):
            attn = attention_weights[batch_idx].detach().cpu().numpy()
        else:
            attn = attention_weights[batch_idx]
        
        num_heads = attn.shape[0]
        
        # 自动计算图表大小
        if figsize is None:
            cols = min(4, num_heads)
            rows = (num_heads + cols - 1) // cols
            figsize = (5 * cols, 4 * rows)
        
        # 创建子图
        fig, axes = self.plt.subplots(
            nrows=(num_heads + 3) // 4,
            ncols=min(4, num_heads),
            figsize=figsize
        )
        
        if num_heads == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # 为了统一颜色范围
        vmin, vmax = attn.min(), attn.max()
        
        # 绘制每个头
        for head_idx in range(num_heads):
            ax = axes[head_idx]
            im = ax.imshow(attn[head_idx], cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title(f'Head {head_idx}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
        
        # 隐藏多余的子图
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')
        
        # 添加颜色条
        fig.colorbar(im, ax=axes, label='Attention Weight', fraction=0.046, pad=0.04)
        
        # 设置总标题
        fig.suptitle(title, fontsize=16, y=1.02)
        
        # 调整布局
        self.plt.tight_layout()
        
        # 保存图表
        if save_path is not None:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multi-head attention plot saved to {save_path}")
        
        # 显示图表
        if show:
            self.plt.show()
        
        return fig, axes
    
    def plot_layer_comparison(
        self,
        layer_attention_weights,
        attention_type='self_attention',
        batch_idx=0,
        head_idx=None,
        title='Layer-wise Attention Comparison',
        figsize=None,
        cmap='viridis',
        save_path=None,
        show=True
    ):
        """
        对比不同层的注意力模式
        
        Args:
            layer_attention_weights: 层级注意力权重列表
            attention_type: 'self_attention' 或 'cross_attention'
            batch_idx: 批次索引
            head_idx: 注意力头索引，如果为None则平均所有头
            title: 图表标题
            figsize: 图表大小
            cmap: 颜色映射
            save_path: 保存路径
            show: 是否显示图表
        
        Returns:
            fig, axes: matplotlib图表对象
        """
        if not self.has_matplotlib:
            print("Cannot plot: matplotlib not available")
            return None, None
        
        num_layers = len(layer_attention_weights)
        
        # 自动计算图表大小
        if figsize is None:
            cols = min(3, num_layers)
            rows = (num_layers + cols - 1) // cols
            figsize = (6 * cols, 5 * rows)
        
        # 创建子图
        fig, axes = self.plt.subplots(
            nrows=(num_layers + 2) // 3,
            ncols=min(3, num_layers),
            figsize=figsize
        )
        
        if num_layers == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if num_layers > 1 else [axes]
        
        # 收集所有注意力权重以统一颜色范围
        all_attn = []
        for layer_attn in layer_attention_weights:
            if attention_type in layer_attn:
                attn_tensor = layer_attn[attention_type]
                if isinstance(attn_tensor, torch.Tensor):
                    attn = attn_tensor[batch_idx].detach().cpu().numpy()
                else:
                    attn = attn_tensor[batch_idx]
                
                if head_idx is not None:
                    attn = attn[head_idx]
                else:
                    attn = attn.mean(axis=0)
                
                all_attn.append(attn)
        
        if not all_attn:
            print(f"No {attention_type} found in the provided layers")
            return None, None
        
        # 统一颜色范围
        vmin = min(a.min() for a in all_attn)
        vmax = max(a.max() for a in all_attn)
        
        # 绘制每一层
        for layer_idx, attn in enumerate(all_attn):
            ax = axes[layer_idx]
            im = ax.imshow(attn, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title(f'Layer {layer_idx}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
        
        # 隐藏多余的子图
        for idx in range(len(all_attn), len(axes)):
            axes[idx].axis('off')
        
        # 添加颜色条
        fig.colorbar(im, ax=axes, label='Attention Weight', fraction=0.046, pad=0.04)
        
        # 设置总标题
        head_info = f"Head {head_idx}" if head_idx is not None else "Average of all heads"
        fig.suptitle(f'{title}\n{attention_type.replace("_", " ").title()} - {head_info}', 
                     fontsize=16, y=1.02)
        
        # 调整布局
        self.plt.tight_layout()
        
        # 保存图表
        if save_path is not None:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layer comparison plot saved to {save_path}")
        
        # 显示图表
        if show:
            self.plt.show()
        
        return fig, axes
    
    def plot_attention_flow(
        self,
        encoder_attention,
        decoder_self_attention,
        decoder_cross_attention,
        batch_idx=0,
        layer_idx=0,
        head_idx=None,
        figsize=(18, 5),
        save_path=None,
        show=True
    ):
        """
        可视化完整的注意力流：编码器自注意力 -> 解码器自注意力 -> 交叉注意力
        
        Args:
            encoder_attention: 编码器注意力权重
            decoder_self_attention: 解码器自注意力权重
            decoder_cross_attention: 解码器交叉注意力权重
            batch_idx: 批次索引
            layer_idx: 层索引
            head_idx: 注意力头索引
            figsize: 图表大小
            save_path: 保存路径
            show: 是否显示图表
        
        Returns:
            fig, axes: matplotlib图表对象
        """
        if not self.has_matplotlib:
            print("Cannot plot: matplotlib not available")
            return None, None
        
        fig, axes = self.plt.subplots(1, 3, figsize=figsize)
        
        attention_data = [
            (encoder_attention, 'Encoder Self-Attention'),
            (decoder_self_attention, 'Decoder Self-Attention'),
            (decoder_cross_attention, 'Decoder Cross-Attention')
        ]
        
        for idx, (attn_weights, attn_title) in enumerate(attention_data):
            if attn_weights is None:
                axes[idx].axis('off')
                continue
            
            # 处理注意力权重
            if isinstance(attn_weights, torch.Tensor):
                attn = attn_weights[batch_idx].detach().cpu().numpy()
            else:
                attn = attn_weights[batch_idx]
            
            # 选择头或平均
            if head_idx is not None:
                attn = attn[head_idx]
            else:
                attn = attn.mean(axis=0)
            
            # 绘制热力图
            im = axes[idx].imshow(attn, cmap='viridis', aspect='auto')
            axes[idx].set_title(attn_title)
            axes[idx].set_xlabel('Key Position')
            axes[idx].set_ylabel('Query Position')
            
            # 添加颜色条
            self.plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        head_info = f"Head {head_idx}" if head_idx is not None else "Average of all heads"
        fig.suptitle(f'Attention Flow - Layer {layer_idx} - {head_info}', 
                     fontsize=16, y=1.02)
        
        self.plt.tight_layout()
        
        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention flow plot saved to {save_path}")
        
        if show:
            self.plt.show()
        
        # 及时释放内存
        if fig is not None:
            self.plt.close(fig)
        
        return fig, axes
    
    def save_attention_statistics(
        self,
        attention_weights,
        save_path,
        batch_idx=0
    ):
        """
        保存注意力权重的统计信息到文件
        
        Args:
            attention_weights: 注意力权重字典（来自model.get_attention_weights()）
            save_path: 保存路径
            batch_idx: 批次索引
        """
        import json
        
        stats = {
            'encoder': [],
            'decoder': []
        }
        
        # 编码器统计
        for layer_idx, layer_attn in enumerate(attention_weights['encoder']):
            if 'self_attention' in layer_attn:
                attn = layer_attn['self_attention'][batch_idx].detach().cpu().numpy()
                stats['encoder'].append({
                    'layer': layer_idx,
                    'type': 'self_attention',
                    'mean': float(attn.mean()),
                    'std': float(attn.std()),
                    'min': float(attn.min()),
                    'max': float(attn.max()),
                    'shape': list(attn.shape)
                })
        
        # 解码器统计
        for layer_idx, layer_attn in enumerate(attention_weights['decoder']):
            if 'self_attention' in layer_attn:
                attn = layer_attn['self_attention'][batch_idx].detach().cpu().numpy()
                stats['decoder'].append({
                    'layer': layer_idx,
                    'type': 'self_attention',
                    'mean': float(attn.mean()),
                    'std': float(attn.std()),
                    'min': float(attn.min()),
                    'max': float(attn.max()),
                    'shape': list(attn.shape)
                })
            
            if 'cross_attention' in layer_attn:
                attn = layer_attn['cross_attention'][batch_idx].detach().cpu().numpy()
                stats['decoder'].append({
                    'layer': layer_idx,
                    'type': 'cross_attention',
                    'mean': float(attn.mean()),
                    'std': float(attn.std()),
                    'min': float(attn.min()),
                    'max': float(attn.max()),
                    'shape': list(attn.shape)
                })
        
        # 保存到文件
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Attention statistics saved to {save_path}")