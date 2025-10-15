
import numpy as np
import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn

from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_nets as ObsNets
from robomimic.models.base_nets import Module

import agent.models.down_utils as DownUtils
from agent.models.transformer import PositionalEncoding, TransformerBackbone
from agent.models.base_nets import ResNetFiLM


def downstream_decoder_factory(
        input_dim,
        output_dim,
        decoder_kwargs=None
    ):
    """
    创建下游任务解码器，作为配置文件到实际解码器创建的实际执行接口函数
    """
    assert decoder_kwargs is not None, "Must provide @decoder_kwargs to create downstream decoder"
    dec_kwargs = deepcopy(decoder_kwargs)

    dec = DownstreamDecoder(
        input_dim=input_dim,
        output_dim=output_dim,
        net_class=dec_kwargs["net_class"],
        net_kwargs=dec_kwargs["net_kwargs"],
    )
    dec.make()
    return dec


class DownstreamDecoder(Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        net_class=None,
        net_kwargs=None,
    ):
        super(DownstreamDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_class = net_class
        self.net_kwargs = net_kwargs
        self.down_nets = nn.ModuleDict()
        self._locked = False

    def make(self):
        assert not self._locked, "Decoder has already been made, cannot remake"
        self._create_layers()
        self._locked = True

    def _create_layers(self):
        assert not self._locked, "Decoder has already been made, cannot remake"

        self.nets["decoder"] = DownUtils.DOWN_DECODER_CLASS(self.net_class)(**self.net_kwargs)

    def forward(self, inputs):
        assert self._locked, "Decoder must be made with .make() before calling forward"

        outputs = self.nets["decoder"](inputs)
        return outputs


class DownstreamGroupDecoder(ObsNets.ObservationDecoder):
    """
    适配下游任务产生解码器
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        decoder_kwargs=None,
    ):
        self.iput_dim = input_dim
        self.output_dim = output_dim

        self.nets = nn.ModuleDict()
        self.nets["decoder"] = downstream_decoder_factory(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            decoder_kwargs=decoder_kwargs,
        )
    
    def output_shape(self, input_shape=None):
        raise NotImplementedError
    
    def forward(self, inputs):
        raise NotImplementedError


class RESNET_MIMO_MLP(Module):
    """
    提供多输入多输出的整体网络，用于动作输出
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        layer_dims,
        activation=nn.ReLU,
        layer_func=nn.Linear,
        **kwargs,
    ):
        super(RESNET_MIMO_MLP, self).__init__()
        assert isinstance(input_obs_group_shapes, OrderedDict), "input_obs_group_shapes must be an OrderedDict"
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_dim = output_shapes

        self.nets = nn.ModuleDict()

        self.nets["image_processor"] = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            feature_activation=None,
            encoder_kwargs=kwargs,
        )

        mlp_input_dim = self.nets["image_processor"].output_dim()[0]

        self.nets["mlp"] = ObsNets.MLP(
            input_dim=mlp_input_dim,
            output_dim=layer_dims[-1],
            layer_dims=layer_dims[:-1],
            layer_func=layer_func,
            activation=activation,
            output_activation=activation, # make sure non-linearity is applied before decoder
        )

    def output_shape(self, input_shape=None):
        return { k : list(self.output_shapes[k]) for k in self.output_shapes }
    
    def forward(self, return_latent=False, **inputs):
        enc_outputs = self.nets["image_processor"].forward(**inputs)
        mlp_outputs = self.nets["mlp"].forward(enc_outputs)
        if return_latent:
            return mlp_outputs, enc_outputs
        return mlp_outputs
    
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
        msg = header + '(' + msg + '\n)'
        msg += textwrap.indent("\image_processor={}".format(self.nets["image_processor"]), indent)
        msg += textwrap.indent("\n\nmlp={}".format(self.nets["mlp"]), indent)
        return msg


class RESNET_MIMO_Transformer(Module):
    """
    提供多输入多输出的Transformer网络，用于动作输出
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        transformer_embed_dim,
        transformer_context_length,
        transformer_num_layers=None,
        transformer_num_heads=None,
        transformer_attn_dropout=None,
        transformer_output_dropout=None,
        transformer_sinusoidal_embedding=None,
        transformer_activation=None,
        transformer_nn_parameter_for_timesteps=None,
        transformer_causal=None,
        **encoder_kwargs,
    ):
        super(RESNET_MIMO_Transformer, self).__init__()

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_dim = output_shapes

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        self.nets["image_processor"] = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            feature_activation=None,
            encoder_kwargs=encoder_kwargs,
        )

        max_timestep = transformer_context_length

        if transformer_sinusoidal_embedding:
            self.nets["embed_timestep"] = PositionalEncoding(transformer_embed_dim)
        elif transformer_nn_parameter_for_timesteps:
            assert (
                not transformer_sinusoidal_embedding
            ), "nn.Parameter only works with learned embeddings"
            self.params["embed_timestep"] = nn.Parameter(
                torch.zeros(1, max_timestep, transformer_embed_dim)
            )
        else:
            self.nets["embed_timestep"] = nn.Embedding(max_timestep, transformer_embed_dim)

        self.nets["transformer_encoder"] = TransformerBackbone(
            embed_dim=transformer_embed_dim,
            num_blocks=transformer_num_layers,
            context_length=transformer_context_length,
            num_heads=transformer_num_heads,
            attn_dropout=transformer_attn_dropout,
            output_dropout=transformer_output_dropout,
            causal=transformer_causal,
            activation=transformer_activation,
        )

        self.nets["transformer_decoder"] = TransformerBackbone(
            embed_dim=transformer_embed_dim,
            num_blocks=transformer_num_layers,
            context_length=transformer_context_length,
            num_heads=transformer_num_heads,
            attn_dropout=transformer_attn_dropout,
            output_dropout=transformer_output_dropout,
            causal=transformer_causal,
            activation=transformer_activation,
        )

        self.transformer_context_length = transformer_context_length
        self.transformer_embed_dim = transformer_embed_dim
        self.transformer_sinusoidal_embedding = transformer_sinusoidal_embedding
        self.transformer_nn_parameter_for_timesteps = transformer_nn_parameter_for_timesteps

    def output_shape(self, input_shape=None):
        return { k : list(self.output_shapes[k]) for k in self.output_shapes }
    
    def forward(self, return_latent=False, **inputs):
        enc_outputs = self.nets["image_processor"].forward(**inputs)
        trans_enc_outputs = self.nets["transformer_encoder"].forward(enc_outputs)
        trans_dec_outputs = self.nets["transformer_decoder"].forward(trans_enc_outputs)
        if return_latent:
            return trans_dec_outputs, enc_outputs
        return trans_dec_outputs
    
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
        msg += textwrap.indent("\image_processor={}".format(self.nets["image_processor"]), indent)
        msg += textwrap.indent("\ntransformer_encoder={}".format(self.nets["transformer_encoder"]), indent)
        msg += textwrap.indent("\n\ntransformer_decoder={}".format(self.nets["transformer_decoder"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg
        














