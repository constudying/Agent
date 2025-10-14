
import numpy as np
import textwrap

import torch
import torch.nn as nn


import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict
from robomimic.models.obs_core import VisualCore, EncoderCore

import agent.models.base_nets as BaseNets

class AgentVisualCore(VisualCore):
    """
    视觉观测量处理网络的基础类
    """
    def __init__(
        self,
        input_shape,
        backbone_class="ResNetFiLM",
        backbone_kwargs=None,
        pool_kwargs=None,
        flatten=None,
        feature_dim=None,
    ):
        super(AgentVisualCore, self).__init__(input_shape=input_shape)
        assert backbone_class is None, "VisualCore: @backbone has not been provided yet."
        assert pool_kwargs is None, "VisualCore: @pool has not been provided yet."
        assert flatten is None, "VisualCore: @flatten has not been provided yet."
        assert feature_dim is None, "VisualCore: @feature_dim has not been provided yet."

        if backbone_kwargs is None:
            backbone_kwargs = dict()

        backbone_kwargs["input_channel"] = input_shape[0]

        backbone_kwargs = extract_class_init_kwargs_from_dict(
                cls = ObsUtils.OBS_ENCODER_BACKBONES[backbone_class],
                dic=backbone_kwargs, copy=True)
        
        self.backbone = eval(backbone_class)(**backbone_kwargs)

        assert isinstance(self.backbone, BaseNets.ConvBase), \
            "VisualCore: @backbone must be instance of ConvBase class"
        
        feat_shape = self.backbone.output_shape(input_shape)
        net_list = [self.backbone]

        if pool_kwargs is not None:
            pass
        else:
            self.pool = None
        
        self.flatten = flatten
        if self.flatten:
            net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))
        
        # maybe linear layer
        self.feature_dim = feature_dim
        if feature_dim is not None:
            assert self.flatten, "VisualCore: @feature_dim requires @flatten=True"
            linear = torch.nn.Linear(int(np.prod(feat_shape)), feature_dim)
            net_list.append(linear)
        
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
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        return super(VisualCore, self).forward(inputs)

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


