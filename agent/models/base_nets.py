
import textwrap
import copy
from typing import Optional, List
from collections import OrderedDict

import torch  # 导入 PyTorch 主库
import torch.nn as nn
import torch.nn.functional as F  # 导入神经网络函数库，常用于激活函数、损失函数等
from torch import nn, Tensor  # 直接导入神经网络模块和张量类

from robomimic.models.base_nets import Module, MLP, ConvBase
from robomimic.models.obs_nets import Randomizer


def get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "linear":
        return nn.Linear
    else:
        raise ValueError(f"Unsupported activation: {activation}")


def res_args_from_config(res_config):
    """
    从配置字典中提取残差网络的参数
    """
    return dict(
        res_input_channels=res_config.input_channels,
        res_film_net_dict=res_config.film_net_enabled_dict,
        res_conv_layer_num=res_config.layer_num,
        res_conv_params_dict=res_config.conv_params_dict,
        shortcut_params_dict=res_config.shortcut_params_dict,
        res_activation=res_config.activation,
        res_film_net_class=res_config.film.net_class,
        res_film_net_kwargs=res_config.film.net_class.net_kwargs,
        res_kwargs=dict(res_config.kwargs),
    )


def transformer_args_from_config(transformer_config):
    """
    从配置字典中提取Transformer网络的参数
    """
    return dict(
        transformer_embed_dim=transformer_config.embed_dim,
        transformer_num_blocks=transformer_config.num_blocks,
        transformer_num_heads=transformer_config.num_heads,
        transformer_context_length=transformer_config.context_length,
        transformer_attn_dropout=transformer_config.attn_dropout,
        transformer_output_dropout=transformer_config.output_dropout,
        transformer_sinusoidal_embedding=transformer_config.sinusoidal_embedding,
        transformer_activation=transformer_config.activation,
        transformer_nn_parameter_for_timesteps=transformer_config.nn_parameter_for_timesteps,
        transformer_causal=transformer_config.causal,
        transformer_nn_parameter_for_timesteps=transformer_config.nn_parameter_for_timesteps,
        transformer_ffw_dim=transformer_config.ffwn_dim,
        tranformer_ffw_dropout=transformer_config.ffw_dropout,
    )


def build_film_core(net_class, **net_kwargs):
    """
        这里负责创建FiLM模块中的核心处理网络，提供多种选择。
        1. MLP网络
        2. LSTM网络
        3. GRU网络
        4. Transformer网络
        5. 自定义网络
    """
    if net_class is None:
        return None
    else:
        if net_class == "mlp":
            return FiLMMLP(**net_kwargs)
        elif net_class == "lstm":
            return FiLMLSTM(**net_kwargs)
        elif net_class == "gru":
            return FiLMGRU(**net_kwargs)
        elif net_class == "transformer":
            return FiLMTransformer(**net_kwargs)
        else:
            # 假设传入的是一个自定义的FiLM核心类
            if issubclass(net_class, FiLMCoreBase):
                return net_class(**net_kwargs)
            else:
                raise ValueError("net_class must be one of 'mlp', 'lstm', 'gru', 'transformer' or a subclass of FiLMCoreBase")


class FiLMCoreBase(Module):
    """
    线性仿射单元核心
    """
    def __init__(self):
        super(FiLMCoreBase, self).__init__()

    def output_shape(self, input_shape):
        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError


class FiLMMLP(FiLMCoreBase):
    """
    基于MLP的FiLM核心网络"""
    def __init__(
        self,
        obs_shapes,
        layer_dims,
        layer_func=nn.Linear,
        activation=nn.ReLU,
    ):
        super(FiLMCoreBase, self).__init__()

        # 参数类型检查
        assert isinstance(obs_shapes, int), "obs_shapes must be a dict"
        assert isinstance(layer_dims, (list, tuple)), "hidden_dims must be list or tuple"
        assert all(isinstance(h, int) and h > 0 for h in layer_dims), "hidden_dims must be list of positive ints"
        assert activation in ["relu", "tanh", "sigmoid", "leaky_relu"], "activation must be one of 'relu', 'tanh', 'sigmoid', 'leaky_relu'"

        self.nets = MLP(
            input_dim=obs_shapes,
            output_dim=layer_dims[-1],
            layer_dims=layer_dims[:-1],
            layer_func=layer_func,
            activation=activation,
            output_activation=activation, # make sure non-linearity is applied before decoder
        )

    def output_shape(self):
        return [self.nets.output_shape]
    
    def forward(self, inputs):
        return self.nets(inputs)
    
    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\n\nmlp={}".format(self.nets), indent)
        msg = header + '(' + msg + '\n)'
        return msg


"""
FiLM核心网络的其他变体，但是其本身FiLM作为标记量，其本身缺失现实反映和映射，
可能唯一作用在于根据其非线性能力实现记录效果，并不具备自变量作用，
因此网络结构暂时不做过多探索
"""
class FiLMLSTM(FiLMCoreBase):
    pass

class FiLMGRU(FiLMCoreBase):
    pass

class FiLMTransformer(FiLMCoreBase):
    pass   


###############################################
#
#         以下是残差块以及网络的相关实现
#
###############################################
class ResidualBlockBase(ConvBase):
    """
    定义残差块的基础类，用于确定引入FiLM的残差块结构
    """
    def __init__(self):
        super(ResidualBlockBase, self).__init__()

    def output_shape(self, input_shape):
        raise NotImplementedError
    
    def forward(self, x, *args):
        raise NotImplementedError


class ResidualBlock(ResidualBlockBase):
    def __init__(
        self,
        conv1_params: dict=None,
        conv2_params: dict=None,
        shortcut_params: dict=None,
        activation=nn.ReLU,
        **kwargs
    ): 
        super(ResidualBlock, self).__init__()
        if kwargs:
            raise TypeError(f"ResidualBlock got unexpected kwargs: {list(kwargs.keys())}")

        # 参数检查
        assert isinstance(conv1_params, dict), "conv1_params must be a dict"
        assert isinstance(conv2_params, dict), "conv2_params must be a dict"
        assert isinstance(shortcut_params, (dict, type(None))), "shortcut_params must be a dict or None"
        
        if conv1_params is None:
            conv1_params = dict(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        if conv2_params is None:
            conv2_params = dict(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 创建卷积层
        self.conv1 = nn.Conv2d(**conv1_params)
        self.conv2 = nn.Conv2d(**conv2_params)
        if shortcut_params is not None:
            self.shortcut = nn.Conv2d(**shortcut_params)
        else:
            self.shortcut = nn.Identity()
        self.activation = activation()

    def forward(self, x, *args):
        film_params = args[0] if len(args) > 0 else None
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        if film_params is not None:
            gamma, beta = film_params[0], film_params[1]
            # 假设 gamma, beta 都是Tensor，可以广播到 out 的 shape
            out = gamma * out + beta
        out += identity
        out = self.activation(out)
        return out
    
    def output_shape(self, input_shape):
        # 计算输出形状
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.forward(dummy_input)
        return list(dummy_output.shape)[1:]
    

class ResNetFiLM(ConvBase):
    """
    引入FiLM模块的残差网络
    """
    def __init__(
        self,
        input_channels: int=3,
        conv_layer_num: int=None,
        net_params: dict=None,
        activation="relu",
        film_net_class="mlp",
        film_net_kwargs=None,
        **kwargs
    ):
        super(Module, self).__init__()
        assert conv_layer_num == len(net_params), "conv_layer_num must be equal to len(net_params)"
        
        # 根据activation字符串获取对应的激活函数类(接口处将字符串转换为类，最后在底层网络类实例化处实例化)
        activation = get_activation(activation)
        film_net_kwargs["layer_func"] = get_activation(film_net_kwargs["layer_func"])

        self.resnet = nn.ModuleDict()
        for k, v in net_params.items():
            self.resnet[k] = ResidualBlock(
                conv1_params=v["conv1_params"],
                conv2_params=v["conv2_params"],
                shortcut_params=net_params[k].get("shortcut_params", None),
                activation=activation
            )
        self.resnet["block1"].conv1.in_channels = input_channels  # 设置第一个残差块的输入通道数

        self.film = build_film_core(film_net_class, **film_net_kwargs)
        self.film_dict = [net_params[k]["film_enable"] for k in net_params.keys()]

    def forward(self, x):
        if self.film is not None:
            film_out = self.film(x)
        for i in range(len(self.resnet)):
            if self.film_dict[i] is True:
                x = self.resnet[f"block{i+1}"](x, film_out[i,:])
            elif self.film_dict[i] is False:
                x = self.resnet[f"block{i+1}"](x)
            else:
                raise ValueError("film_dict values must be True or False")
        return x
    
    def output_shape(self, input_shape):
        return self.resnet[f"block{len(self.resnet)}"].output_shape(input_shape)
    
    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        for i in range(len(self.resnet)):
            msg += textwrap.indent(f"\nblock{i+1}: film={self.film_dict[i]}\n", indent)
            msg += textwrap.indent(f"{self.resnet[f'block{i+1}']}\n", indent)
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg += textwrap.indent("\noutput_shape={}".format(self.output_shape()), ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg




