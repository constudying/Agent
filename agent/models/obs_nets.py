
import numpy as np
import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn

from agent.models import get_activation

from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.models.base_nets import Module
from robomimic.models.obs_core import Randomizer
from robomimic.models.obs_nets import ObservationGroupEncoder, ObservationDecoder
from agent.models.transformer import PositionalEncoding, TransformerBackbone


def obs_encoder_factory(
        obs_shapes,
        feature_activation="relu",
        encoder_kwargs=None
    ):
    """
    创建观测量处理网络的工厂函数
    """
    enc = ObservationEncoder(feature_activation=get_activation(feature_activation))
    for k, obs_shape in obs_shapes.items():
        obs_modality = ObsUtils.OBS_KEYS_TO_MODALITIES[k]
        enc_kwargs = deepcopy(ObsUtils.DEFAULT_ENCODER_KWARGS[obs_modality]) if encoder_kwargs is None else \
            deepcopy(encoder_kwargs[obs_modality])
        
        for obs_module, cls_mapping in zip(("core", "obs_randomizer"),
                                            (ObsUtils.OBS_ENCODER_CORES, ObsUtils.OBS_RANDOMIZERS)):
            
            if enc_kwargs.get(f"{obs_module}_kwargs", None) is None:
                enc_kwargs[f"{obs_module}_kwargs"] = {}
            enc_kwargs[f"{obs_module}_kwargs"]["input_shape"] = obs_shape
            if enc_kwargs[f"{obs_module}_class"] is not None:
                enc_kwargs[f"{obs_module}_kwargs"] = extract_class_init_kwargs_from_dict(
                    cls=cls_mapping[enc_kwargs[f"{obs_module}_class"]],
                    dic=enc_kwargs[f"{obs_module}_kwargs"],
                    copy=True
                )
        
        randomizer = None if enc_kwargs["obs_randomizer_class"] is None else \
            ObsUtils.OBS_RANDOMIZERS[enc_kwargs["obs_randomizer_class"]](**enc_kwargs["obs_randomizer_kwargs"])

        enc.register_obs_key(
            name=k,
            shape=obs_shape,
            net_class=enc_kwargs["core_class"],
            net_kwargs=enc_kwargs["core_kwargs"],
            randomizer=randomizer,
        )
    
    enc.make()
    return enc


class ObservationEncoder(Module):
    """
    观测量处理网络的基础类
    """
    def __init__(self, feature_activation="relu"):
        super(ObservationEncoder, self).__init__()
        self.obs_shapes = OrderedDict()
        self.obs_nets_classes = OrderedDict()
        self.obs_nets_kwargs = OrderedDict()
        self.obs_nets = nn.ModuleDict()
        self.obs_randomizers = nn.ModuleDict()
        self.feature_activation = get_activation(feature_activation)
        self._locked = False
    
    def register_obs_key(
        self,
        name,
        shape,
        net_class=None,
        net_kwargs=None,
        net=None,
        randomizer=None,
    ):
        """
        注册观测量的键，建立对应的处理网络
        """
        assert not self._locked, "ObservationEncoder: @register_obs_key called after @make"
        assert name not in self.obs_shapes, "ObservationEncoder: modality {} already exists".format(name)
        
        if net is not None:
            assert isinstance(net, Module), "ObservationEncoder: @net must be instance of Module class"
            assert (net_class is None) and (net_kwargs is None), \
                "ObservationEncoder: @net provided - ignore other net creation options"
        
        net_kwargs = deepcopy(net_kwargs) if net_kwargs is not None else {}
        if randomizer is not None:
            assert isinstance(randomizer, Randomizer), "ObservationEncoder: @randomizer must be instance of Randomizer class"
            if net_kwargs is not None:
                # update input shape to visual core
                net_kwargs["input_shape"] = randomizer.output_shape_in(shape)
        
        self.obs_shapes[name] = shape
        self.obs_nets_classes[name] = net_class
        self.obs_nets_kwargs[name] = net_kwargs
        self.obs_nets[name] = net
        self.obs_randomizers[name] = randomizer

    def make(self):
        assert not self._locked, "ObservationEncoder: @make called more than once"
        self._create_layers()
        self._locked = True

    def _create_layers(self):
        assert not self._locked, "ObservationEncoder: layers have already been created"

        for k in self.obs_shapes:
            if self.obs_nets_classes[k] is not None:
                # create net to process this modality
                self.obs_nets[k] = ObsUtils.OBS_ENCODER_CORES[self.obs_nets_classes[k]](**self.obs_nets_kwargs[k])

        self.activation = None
        if self.feature_activation is not None:
            self.activation = self.feature_activation()

    def forward(self, obs_dict):
        """
        前向传播，处理观测量字典
        """
        assert self._locked, "ObservationEncoder: @forward called before @make"
        assert set(self.obs_shapes.keys()).issubset(obs_dict), "ObservationEncoder: {} does not contain all modalities {}".format(
            list(obs_dict.keys()), list(self.obs_shapes.keys())
        )

        feats = []
        for k in self.obs_shapes:
            x = obs_dict[k]
            # maybe process encoder input with randomizer
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward(x)
            if self.obs_nets[k] is not None:
                x = self.obs_nets[k].forward(x)
                if self.activation is not None:
                    x = self.activation(x)
            # maybe process encoder output with randomizer
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward_out(x)
            # flatten to [B, D]
            x = TensorUtils.flatten(x, begin_axis=1)
            feats.append(x)

        # concatenate all features together
        return torch.cat(feats, dim=-1)

    def output_shape(self, input_shape=None):
        """
        Compute the output shape of the encoder.
        """
        feat_dim = 0
        for k in self.obs_shapes:
            feat_shape = self.obs_shapes[k]
            if self.obs_randomizers[k] is not None:
                feat_shape = self.obs_randomizers[k].output_shape_in(feat_shape)
            if self.obs_nets[k] is not None:
                feat_shape = self.obs_nets[k].output_shape(feat_shape)
            if self.obs_randomizers[k] is not None:
                feat_shape = self.obs_randomizers[k].output_shape_out(feat_shape)
            feat_dim += int(np.prod(feat_shape))
        return [feat_dim]

    def __repr__(self):
        """
        Pretty print the encoder.
        """
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.obs_shapes:
            msg += textwrap.indent('\nKey(\n', ' ' * 4)
            indent = ' ' * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent)
            msg += textwrap.indent("modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent)
            msg += textwrap.indent("randomizer={}\n".format(self.obs_randomizers[k]), indent)
            msg += textwrap.indent("net={}\n".format(self.obs_nets[k]), indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg += textwrap.indent("\noutput_shape={}".format(self.output_shape()), ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg


class MIMO_NetBase(Module):
    """
    多输入多输出的网络基础类，用于处理多模态观测量输出动作预测等
    """
    def __init__(self):
        super(MIMO_NetBase, self).__init__()
    
    def output_shape(self, input_shape):
        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError


class MIMO_Transformer(MIMO_NetBase):
    """
    提供多输入多输出的Transformer网络，用于动作输出
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        transformer_embed_dim,
        tranfsormer_context_length,
        transformer_num_layers=6,
        transformer_num_heads=8,
        transformer_attn_dropout=0.1,
        transformer_output_dropout=0.1,
        transformer_sinusoidal_embedding=False,
        transformer_activation="relu",
        transformer_nn_parameter_for_timesteps=None,
        transformer_causal=True,
        encoder_kwargs=None,
    ):
        super(MIMO_Transformer, self).__init__()

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_dim = output_shapes

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        max_timestep = tranfsormer_context_length

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
            context_length=tranfsormer_context_length,
            num_heads=transformer_num_heads,
            attn_dropout=transformer_attn_dropout,
            output_dropout=transformer_output_dropout,
            causal=transformer_causal,
            activation=transformer_activation,
        )

        self.nets["transformer_decoder"] = TransformerBackbone(
            embed_dim=transformer_embed_dim,
            num_blocks=transformer_num_layers,
            context_length=tranfsormer_context_length,
            num_heads=transformer_num_heads,
            attn_dropout=transformer_attn_dropout,
            output_dropout=transformer_output_dropout,
            causal=transformer_causal,
            activation=transformer_activation,
        )

    def output_shape(self, input_shape=None):
        return { k : list(self.output_shapes[k]) for k in self.output_shapes }
    
    def forward(self, return_latent=False, **inputs):
        enc_outputs = self.nets["transformer_encoder"].forward(**inputs)
        dec_outputs = self.nets["transformer_decoder"].forward(**enc_outputs)
        if return_latent:
            return dec_outputs, enc_outputs
        return dec_outputs
    
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
        msg += textwrap.indent("\nencoder={}".format(self.nets["transformer_encoder"]), indent)
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["transformer_decoder"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg

















