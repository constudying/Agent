
import numpy as np
import textwrap

import torch
import torch.nn as nn

import agent.models.base_nets as BaseNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict

from agent.models.base_nets import *
from robomimic.models.obs_core import EncoderCore
from robomimic.models.base_nets import ConvBase, ResNet18Conv

class AgentVisualCore(EncoderCore, BaseNets.ConvBase):
    """
    视觉观测量处理网络的基础类
    """
    def __init__(
        self,
        input_shape,
        backbone_class="ResNet18Conv", # NOTE: backbone class must be provided
        pool_class="SpatialSoftmax",
        backbone_kwargs=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=64,
    ):
        super(AgentVisualCore, self).__init__(input_shape=input_shape)
        assert backbone_class is not None and backbone_kwargs is not None, "VisualCore: @backbone has not been provided yet."
        
        self.flatten = flatten

        if backbone_kwargs is None:
            backbone_kwargs = dict()

        backbone_kwargs["input_channel"] = input_shape[0]

        backbone_kwargs = extract_class_init_kwargs_from_dict(
                cls = eval(backbone_class),
                dic=backbone_kwargs,
                copy=True
        )
        
        assert isinstance(backbone_class, str), \
            "VisualCore: @backbone_class must be a string representing the backbone class name"
        self.backbone = eval(backbone_class)(**backbone_kwargs)

        assert isinstance(self.backbone, BaseNets.ConvBase) or isinstance(self.backbone, ConvBase), \
            "VisualCore: @backbone must be instance of ConvBase class"
        
        feat_shape = self.backbone.output_shape(input_shape)
        net_list = [self.backbone]

        if pool_class is not None:
            # assert isinstance(pool_class, str)
            # # feed output shape of backbone to pool net
            # if pool_kwargs is None:
            #     pool_kwargs = dict()
            # # extract only relevant kwargs for this specific backbone
            # pool_kwargs["input_shape"] = feat_shape
            # pool_kwargs = extract_class_init_kwargs_from_dict(cls=eval(pool_class), dic=pool_kwargs, copy=True)
            # self.pool = eval(pool_class)(**pool_kwargs)
            # assert isinstance(self.pool, BaseNets.Module)

            # feat_shape = self.pool.output_shape(feat_shape)
            # net_list.append(self.pool)
            self.pool = None
        else:
            self.pool = None
        
        self.flatten = flatten
        if self.flatten:
            # net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))
            pass
        
        # maybe linear layer
        self.feature_dimension = feature_dimension
        if feature_dimension is not None:
            assert self.flatten, "VisualCore: @feature_dimension requires @flatten=True"
            linear = torch.nn.Linear(int(np.prod(feat_shape)), self.feature_dimension)
            # net_list.append(linear)
        
        self.nets = nn.Sequential(*net_list)

    def output_shape(self, input_shape=None):
        if self.feature_dimension is not None:
            # linear output
            return [self.feature_dimension]
        feat_shape = self.backbone.output_shape(input_shape)
        if self.pool is not None:
            # pool output
            feat_shape = self.pool.output_shape(feat_shape)
        # backbone + flat output
        if self.flatten:
            return [np.prod(feat_shape)]
        else:
            return feat_shape

    def forward(self, inputs):
        """
        Forward pass through visual core.
        """
        ndim = len(self.input_shape)
        # inputs.shape: torch.Size([32, 1, 3, 84, 84])
        # assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape) # 暂时不使用干扰器
        return super(AgentVisualCore, self).forward(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nbackbone_net={}".format(self.backbone), indent)
        msg += textwrap.indent("\npool_net={}".format(self.pool), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class AgentProjectCore(EncoderCore):
    
    def __init__(
        self,
        input_shape,
        output_dimension,
        activation=None,
    ):
        super(AgentProjectCore, self).__init__(input_shape=input_shape)

        self.output_dimension = output_dimension
        # 暂时不启用，避免引入不必要的非线性
        if activation is not None:
            self.activation = activation

        self.proj_net = nn.Linear(input_shape[-1], output_dimension)

    def output_shape(self, input_shape):
        assert input_shape is not None, "@ProjectCore-output_shape: input_shape must be provided to compute output shape."
        output_shape = list(input_shape)
        output_shape[-1] = self.proj_net.out_features
        return output_shape

    def forward(self, inputs):
        return self.proj_net(inputs)
    
    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nproj_net={}".format(self.proj_net), indent)
        msg = header + '(' + msg + '\n)'
        return msg