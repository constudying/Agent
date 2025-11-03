import math

import torch
import torch.nn as nn
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


class PositionEncoder(nn.Module):
    """
    传统固定位置编码器，将输入的位置信息编码为高维特征表示
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        输入形状为 BXT
        """
        position = x
        # computing 1/n^(i/d) in log space and then exponentiating and fixing the shape
        div_term = (
            torch.exp(
                torch.arange(0, self.embed_dim, 2, device=x.device)
                * (-math.log(10000.0) / self.embed_dim)
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(x.shape[0], x.shape[1], 1)
        )
        pe = torch.zeros((x.shape[0], x.shape[1], self.embed_dim), device=x.device)
        pe[:, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)
        return pe.detach()


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
        self.nets = nn.ModuleDict()

        self.nets["qkv"] = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.nets["attn_dropout"] = nn.Dropout(attn_dropout)
        self.nets["output_dropout"] = nn.Dropout(output_dropout)
        self.nets["output"] = nn.Linear(embed_dim, embed_dim, bias=False)
   
    def forward(self, x):
        """
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
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
        att = self.nets["attn_dropout"](att)
        
        y = att @ v  # (B, NH, T, DH)
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # re-assemble all head outputs side by side (B, T, D)
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)
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
    ):
        super(CausalSelfAttention, self).__init__()

        assert embed_dim % num_heads == 0, \
          "num_heads: {} does not divide embed_dim: {} exactly".format(num_heads, embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.nets = nn.ModuleDict()

        self.nets["qkv"] = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.nets["attn_dropout"] = nn.Dropout(self.attn_dropout)
        self.nets["output_dropout"] = nn.Dropout(self.output_dropout)
        self.nets["output"] = nn.Linear(self.embed_dim, self.embed_dim)

        mask = torch.tril(torch.ones(context_length, context_length)).view(
            1, 1, context_length, context_length
        )
        self.register_buffer("mask", mask)
   
    def forward(self, x):
        """
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
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
        att = self.nets["attn_dropout"](att)
        
        y = att @ v  # (B, NH, T, DH)
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # re-assemble all head outputs side by side (B, T, D)
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)
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
    ):
        super(CrossAttention, self).__init__()

        assert embed_dim % num_heads == 0, \
          "@CrossAttention: num_heads: {} does not divide embed_dim: {} exactly".format(num_heads, embed_dim)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.nets = nn.ModuleDict()

        self.nets["q"] = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.nets["kv"] = nn.Linear(self.embed_dim, 2 * self.embed_dim, bias=False)
        
        self.nets["attn_dropout"] = nn.Dropout(self.attn_dropout)
        self.nets["output_dropout"] = nn.Dropout(self.output_dropout)
        self.nets["output"] = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, context):
        """
        x: 对应查询(query）
        context: 对应键(key)和值(value)
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
        att = self.nets["attn_dropout"](att)
        
        y = att @ v  # (B, NH, T, DH)
        y = y.transpose(1, 2).contiguous().view(B, T_q, D)  # re-assemble all head outputs side by side (B, T, D)
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)
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
    

class TransformerBackbone(Module):
    """
    Transformer骨干网络，由多个Transformer块堆叠而成
    """
    def __init__(
        self,
        embed_dim,
        context_length,
        attn_dropout=0.1,
        output_dropout=0.1,
        ffw_hidden_dim=1024,
        ffw_dropout=None,
        num_heads=8,
        num_blocks=6,
        activation=nn.ReLU,
        causal=True,
    ):
        super(TransformerBackbone, self).__init__()
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.ffw_hidden_dim = ffw_hidden_dim if ffw_hidden_dim is not None else 4 * embed_dim
        self.ffw_dropout = ffw_dropout if ffw_dropout is not None else output_dropout
        self.activation = get_activation(activation)()
        self.causal = causal

        self._create_networks()

    def _create_networks(self):
        self.nets = nn.ModuleDict()
        block_class = CausalTransformerBlock if self.causal else NormalTransformerBlock
        self.nets["transformer"] = nn.Sequential(
            *[
                block_class(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    context_length=self.context_length,
                    attn_dropout=self.attn_dropout,
                    output_dropout=self.output_dropout,
                    ffw_hidden_dim=self.ffw_hidden_dim,
                    ffw_dropout=self.ffw_dropout,
                    activation=self.activation,
                )
                for _ in range(self.num_blocks)
            ]
        )
        # 方便子层连接，防止梯度爆炸
        self.nets["output_ln"] = nn.LayerNorm(self.embed_dim)

    def _init_weights(self, module):
        """
        Weight initializer. 
        TODO: 参考GPT-2初始化方法（xariver初始化方法），后续可根据需要调整
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        # TODO: 这里的context_length设定比较迷，图像存在时序维度，但是作为混合输入时，context_length能够单单描述图像的时序长度吗？
        if inputs.dim() == 2:
          inputs = inputs.unsqueeze(1)  # 添加 batch 维度，变为 (batch, context_length, embed_dim)

        assert inputs.shape[1:] == (self.context_length, self.embed_dim), \
            "the inputs of transformerbackbone dismatch the setting of programmer, which is not {}".format(inputs.shape)
        x = self.nets["transformer"](inputs)
        transformer_output = self.nets["output_ln"](x)
        return transformer_output

    def output_shape(self, input_shape=None):
        return input_shape[:-1] + [self.output_dim]


class TransformerEncoderLayer(Module):
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


class TransformerDecoderLayer(Module):
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
        num_encoder_layers=4,
        num_decoder_layers=4,
        activation='relu',
    ):
        super(Transformer, self).__init__()

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

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
              nn.init.xavier_uniform_(p)
          
    def forward(self, x):
        """
        x: (B, T, D)
        Returns: (B, T, D)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加 batch 维度，变为 (batch, context_length, embed_dim)
        assert x.dim() == 3, "Transformer expects input of shape (B, T, D)"
        obs, goal = torch.chunk(x, 2, dim=-1)  # 输入的一半是观测图像特征，一半是目标图像特征
        # encoder
        encoder_output = self.nets["encoder"](obs)
        # decoder: 逐层传递，每层都传入 context
        decoder_input = encoder_output
        for layer in self.nets["decoder"]:
            decoder_input = layer(decoder_input, context=encoder_output)
        decoder_output = decoder_input
        return decoder_output
    
    def output_shape(self, input_shape=None):
        return list(input_shape)

 