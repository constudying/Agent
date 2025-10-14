
import numpy as np
import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn

import robomimic.models.obs_nets as ObsNets
from robomimic.models.base_nets import Module

from agent.models.transformer import PositionalEncoding, TransformerBackbone
from agent.models.base_nets import ResNetFiLM


class RESNET_TRANSFORMER_MIMO(Module):
    """
    提供多输入多输出的整体网络，用于动作输出
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        **kwargs,
    ):
        super(RESNET_TRANSFORMER_MIMO, self).__init__()

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_dim = output_shapes

    def output_shape(self, input_shape=None):
        return { k : list(self.output_shapes[k]) for k in self.output_shapes }
    
    def forward(self, return_latent=False, **inputs):
        return inputs
    
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
        return msg


class MIMO_Transformer(Module):
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
        super(MIMO_Transformer, self).__init__()

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















