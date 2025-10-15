
import abc

import torch
import torch.nn as nn

import agent.utils.down_utils as DownUtils

class DownModule(torch.nn.Module):
    """
    下游任务网络的基础类
    """
    def __init__(self):
        super(DownModule, self).__init__()

    def __init_subclass__(cls):
        DownUtils.register_down_decoder_network(cls)
    
    def output_shape(self, input_shape):
        raise NotImplementedError


