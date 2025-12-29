
import numpy as np
import textwrap
import copy
from typing import Optional, List
from collections import OrderedDict

import torch  # 导入 PyTorch 主库
import torch.nn as nn
import torch.nn.functional as F  # 导入神经网络函数库，常用于激活函数、损失函数等
from torch import nn, Tensor  # 直接导入神经网络模块和张量类

import robomimic.utils.torch_utils as TorchUtils
import robomimic.models.obs_core as ObsCore
from robomimic.models.obs_nets import Randomizer
from robomimic.models.base_nets import Module, MLP

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
    elif activation == "gelu":
        return nn.GELU
    elif activation == "glu":
        return nn.GLU
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


"""
================================================
Visual Backbone Networks
================================================
"""
class ConvBase(Module):
    """
    Base class for ConvNets.
    """
    def __init__(self):
        super(ConvBase, self).__init__()

    # dirty hack - re-implement to pass the buck onto subclasses from ABC parent
    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    def forward(self, inputs):
        # x.shape: torch.Size([32, 1, 3, 84, 84])
        # 对于nn.Sequential模块，不存在显式forward方法，因此直接调用此处forward方法后启用自身Sequential的底层forward方法
        # 因此针对Sequential模块的输出形状检查在此处进行
        # import ipdb; ipdb.set_trace()
        x = self.nets(inputs)
        # if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
        #     raise ValueError('Size mismatch: expect size %s, but got size %s' % (
        #         str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
        #     )
        return x

class SpatialSoftmax(ConvBase):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape,
        num_kp=32,
        temperature=1.,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
                )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


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
        assert isinstance(layer_dims, (list, tuple)), "hidden_dims must be list or tuple"
        assert all(isinstance(h, int) and h > 0 for h in layer_dims), "hidden_dims must be list of positive ints"
        assert activation in ["relu", "tanh", "sigmoid", "leaky_relu"], "activation must be one of 'relu', 'tanh', 'sigmoid', 'leaky_relu'"

        self.nets = MLP(
            input_dim=obs_shapes,
            output_dim=layer_dims[-1],
            layer_dims=layer_dims[:-1],
            layer_func=layer_func,
            activation=get_activation(activation),
            output_activation=get_activation(activation), # make sure non-linearity is applied before decoder
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
class ResidualBlock(ConvBase):
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
        assert isinstance(shortcut_params, dict), "shortcut_params must be a dict"
        
        if conv1_params is None:
            conv1_params = dict(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        if conv2_params is None:
            conv2_params = dict(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 创建卷积层
        self.convnet = nn.ModuleList([
            nn.Conv2d(**conv1_params),
            nn.Conv2d(**conv2_params),
            nn.Conv2d(**shortcut_params)
        ])
        self.activation1 = activation()
        self.activation2 = activation()

    def forward(self, x, film_args=None):
        # 处理 FiLM 参数
        # args 可以是 None（初始化时）或包含 FiLM 参数的列表/元组
        # x.shape: torch.Size([1, 3, 84, 84])
        out = self.convnet[0](x)
        out = self.activation1(out)
        out = self.convnet[1](out)
        
        if film_args is not None:
            # film_params 形状: [batch, 2]，包含 gamma 和 beta
            # out 形状: [batch, channels, height, width]
            gamma = film_args[0]  # [batch]
            beta = film_args[1]   # [batch]
            # 将 gamma 和 beta 扩展为 [batch, 1, 1, 1] 以便广播
            # FiLM 操作: out = gamma * out + beta
            out = gamma * out + beta
        identity = self.convnet[2](x)
        out += identity
        out = self.activation2(out)
        return out
    
    def output_shape(self, input_shape=None):
        # 计算输出形状
        with torch.no_grad():
            dummy_input = torch.zeros(1, *tuple(input_shape))
            dummy_output = self.forward(dummy_input)
        return list(dummy_output.shape)[1:]
    

class ResNetFiLM(ObsCore.EncoderCore, ConvBase):
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
        self.block_nums = conv_layer_num
        for k, v in net_params.items():
            self.resnet[k] = ResidualBlock(
                conv1_params=v["conv1_params"],
                conv2_params=v["conv2_params"],
                shortcut_params=net_params[k].get("shortcut_params", None),
                activation=activation
            )

        self._film_locked = False
        self.resnet["film"] = build_film_core(film_net_class, **film_net_kwargs)
        self.film_dict = [net_params[k]["film_enabled"] for k in net_params.keys()]

    def forward(self, x):
        # x.shape: torch.Size([32, 1, 3, 84, 84])
        device = x.device # 从输入 x 获取设备，确保与模型参数一致
        film_input = torch.zeros(512, device=device)  # 假设 FiLM 网络的输入是一个固定的零向量
        film_out = self.resnet["film"](film_input)  # 输出: [blocks, 2]
        had_seq = False
        if x.dim() == 5:
            bs, seq, _, _, _ = x.shape
            x = x.view(bs * seq, *x.shape[2:])  # 合并 batch 和 sequence 维度
            had_seq = True
        for i in range(0, self.block_nums):  # 通过各个残差块
            if self.film_dict[i] is True:
                # 传递整个 film_out 给残差块，让它处理 batch 维度
                x = self.resnet[f"block{i+1}"](x, film_out)
            elif self.film_dict[i] is False:
                x = self.resnet[f"block{i+1}"](x)
            else:
                raise ValueError("film_dict values must be True or False")
        if had_seq:
            x = x.view(bs, seq, *x.shape[1:])  # 恢复 batch 和 sequence 维度，不使用提取的c、h、w，因为x的shape已经更新
        return x
    
    def output_shape(self, input_shape):
        x = copy.deepcopy(input_shape)
        for idx in range(1, self.block_nums+1):
            x = self.resnet[f"block{idx}"].output_shape(x)
        return x

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
        for k in self.resnet.keys():
            msg += textwrap.indent(f"{self.resnet[k]}\n", indent)
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg += textwrap.indent(f"\nfilm_dict: {self.film_dict}\n", indent)
        msg = header + '(' + msg + '\n)'
        return msg
