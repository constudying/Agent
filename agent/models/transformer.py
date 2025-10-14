import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.models import get_activation
from robomimic.models.base_nets import Module

class PositionEncoder(nn.Module):
    """
    位置编码器，将输入的位置信息编码为高维特征表示
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
        B, T, D = x.shape()
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
        B, T, D = x.shape()
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
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
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
    TODO：交叉注意力机制模块，初步占位，后期用于高维特征指导动作分布生成。
    """
    def __init__(
        self,
        query_dim,
        context_dim,
        embed_dim,
        num_heads,
        attn_dropout=0.1,
        output_dropout=0.1,
    ):
        super(CrossAttention, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "num_heads: {} does not divide embed_dim: {} exactly".format(num_heads, embed_dim)

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.nets = nn.ModuleDict()

        self.nets["q"] = nn.Linear(query_dim, embed_dim, bias=False)
        self.nets["kv"] = nn.Linear(context_dim, 2 * embed_dim, bias=False)
        self.nets["attn_dropout"] = nn.Dropout(attn_dropout)
        self.nets["output_dropout"] = nn.Dropout(output_dropout)
        self.nets["output"] = nn.Linear(embed_dim, embed_dim, bias=False)
   
    def forward(self, query, context):
        """
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
        """
        assert query.dim() == 3, "Input query must be 3D (B, T, D)"
        assert context.dim() == 3, "Input context must be 3D (B, T, D)"
        B, Tq, Dq = query.shape()
        Bc, Tc, Dc = context.shape()
        assert B == Bc, "Batch size of query and context must be the same"
        assert Dq == self.query_dim, "Input query dim must be equal to module query dim"
        assert Dc == self.context_dim, "Input context dim must be equal to module context dim"
        NH = self.num_heads
        DH = self.embed_dim // NH  # dimension per head

        q = self.nets["q"](query)  # (B, Tq, D)
        kv = self.nets["kv"](context)  # (B, Tc, 2 * D)
        k, v = torch.chunk(kv, 2, dim=-1) # each is (B, Tc, D)
        k = k.view(B, Tc, NH, DH).transpose(1, 2)  # (B, NH, Tc, DH)
        q = q.view(B, Tq, NH, DH).transpose(1, 2)  # (B, NH, Tq, DH)
        v = v.view(B, Tc, NH, DH).transpose(1, 2)  # (B, NH, Tc, DH)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, NH, Tq, Tc)
        att = F.softmax(att, dim=-1)
        att = self.nets["attn_dropout"](att)
        y = att @ v  # (B, NH, Tq, DH)
        y = y.transpose(1, 2).contiguous().view(B, Tq, self.embed_dim)  # re-assemble all head outputs side by side (B, Tq, D)
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)
        return y

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
        self.activation = activation()
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
        self.activation = activation()
        self.causal = causal

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
        assert inputs.shape[1:] == (self.context_length, self.embed_dim), \
            "the inputs of transformerbackbone dismatch the setting of programmer, which is not {}".format(inputs.shape)
        x = self.nets["transformer"](inputs)
        transformer_output = self.nets["output_ln"](x)
        return transformer_output

    def output_shape(self, input_shape=None):
        return input_shape[:-1] + [self.output_dim]


