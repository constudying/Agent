from __future__ import annotations

import math
import os
import numpy as np
import textwrap
from copy import deepcopy
from collections import OrderedDict
from itertools import chain
from typing import Callable, Optional

import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
# import agent.models.down_utils as DownUtils

from robomimic.models.base_nets import Module, MLP
from agent.models.transformer import Transformer
# from agent.models.coupuled_model import HumanRobotCoupledInterACT
from agent.models.obs_core import AgentVisualCore
from robomimic.models.obs_core import Randomizer
from agent.models.base_nets import get_activation

def obs_encoder_factory(
        obs_shapes,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
    ):
    """
    Utility function to create an @ObservationEncoder from kwargs specified in config.

    Args:
        obs_shapes (OrderedDict): a dictionary that maps observation key to
            expected shapes for observations.

        feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
            None to apply no activation.

        encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should be
            nested dictionary containing relevant per-modality information for encoder networks.
            Should be of form:

            obs_modality1: dict
                feature_dimension: int
                core_class: str
                core_kwargs: dict
                    ...
                    ...
                obs_randomizer_class: str
                obs_randomizer_kwargs: dict
                    ...
                    ...
            obs_modality2: dict
                ...
    """
    enc = ObservationEncoder(feature_activation=feature_activation)
    # NOTE: 需要在这里登记所有使用到的 obs key 的shape
    # obs_shapes['robot0_eef_pos_past_traj'] = [10, 3]
    # obs_shapes['robot0_eef_pos_past_traj_delta'] = [9, 3]
    obs_shapes['robot0_eef_pos_past_traj'] = [10, 3]
    # obs_shapes['robot0_eef_vel_ang_past_traj'] = [10, 3]
    # obs_shapes['robot0_eef_vel_lin_past_traj'] = [10, 3]
    obs_shapes['robot0_action_past_traj'] = [10, 7]
    obs_shapes['robot0_action_future_traj'] = [10, 7]
    obs_shapes['robot0_joint_pos_past_traj'] = [10, 7]


    for k, obs_shape in obs_shapes.items():
        obs_modality = ObsUtils.OBS_KEYS_TO_MODALITIES[k]
        enc_kwargs = deepcopy(ObsUtils.DEFAULT_ENCODER_KWARGS[obs_modality]) if encoder_kwargs is None else \
            deepcopy(encoder_kwargs[obs_modality])

        for obs_module, cls_mapping in zip(("core", "obs_randomizer"),
                                      (ObsUtils.OBS_ENCODER_CORES, ObsUtils.OBS_RANDOMIZERS)):
            # Sanity check for kwargs in case they don't exist / are None
            if enc_kwargs.get(f"{obs_module}_kwargs", None) is None:
                enc_kwargs[f"{obs_module}_kwargs"] = {}
            # Add in input shape info
            enc_kwargs[f"{obs_module}_kwargs"]["input_shape"] = obs_shape
            # If group class is specified, then make sure corresponding kwargs only contain relevant kwargs
            if enc_kwargs[f"{obs_module}_class"] is not None:
                enc_kwargs[f"{obs_module}_kwargs"] = extract_class_init_kwargs_from_dict(
                    cls=cls_mapping[enc_kwargs[f"{obs_module}_class"]],
                    dic=enc_kwargs[f"{obs_module}_kwargs"],
                    copy=False,
                )

        # Add in input shape info
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
    Module that processes inputs by observation key and then concatenates the processed
    observation keys together. Each key is processed with an encoder head network.
    Call @register_obs_key to register observation keys with the encoder and then
    finally call @make to create the encoder networks. 
    """
    def __init__(self, feature_activation=nn.ReLU):
        """
        Args:
            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation. 
        """
        super(ObservationEncoder, self).__init__()
        self.obs_shapes = OrderedDict()
        self.obs_nets_classes = OrderedDict()
        self.obs_nets_kwargs = OrderedDict()
        self.obs_share_mods = OrderedDict()
        self.obs_nets = nn.ModuleDict()
        self.obs_randomizers = nn.ModuleDict()
        self.feature_activation = feature_activation
        self._locked = False

    def register_obs_key(
        self, 
        name,
        shape, 
        net_class=None, 
        net_kwargs=None, 
        net=None, 
        randomizer=None,
        share_net_from=None,
    ):
        """
        Register an observation key that this encoder should be responsible for.

        Args:
            name (str): modality name
            shape (int tuple): shape of modality
            net_class (str): name of class in base_nets.py that should be used
                to process this observation key before concatenation. Pass None to flatten
                and concatenate the observation key directly.
            net_kwargs (dict): arguments to pass to @net_class
            net (Module instance): if provided, use this Module to process the observation key
                instead of creating a different net
            randomizer (Randomizer instance): if provided, use this Module to augment observation keys
                coming in to the encoder, and possibly augment the processed output as well
            share_net_from (str): if provided, use the same instance of @net_class 
                as another observation key. This observation key must already exist in this encoder.
                Warning: Note that this does not share the observation key randomizer
        """
        assert not self._locked, "ObservationEncoder: @register_obs_key called after @make"
        assert name not in self.obs_shapes, "ObservationEncoder: modality {} already exists".format(name)

        if net is not None:
            assert isinstance(net, Module), "ObservationEncoder: @net must be instance of Module class"
            assert (net_class is None) and (net_kwargs is None) and (share_net_from is None), \
                "ObservationEncoder: @net provided - ignore other net creation options"

        if share_net_from is not None:
            # share processing with another modality
            assert (net_class is None) and (net_kwargs is None)
            assert share_net_from in self.obs_shapes

        net_kwargs = deepcopy(net_kwargs) if net_kwargs is not None else {}
        if randomizer is not None:
            assert isinstance(randomizer, Randomizer)
            if net_kwargs is not None:
                # update input shape to visual core
                net_kwargs["input_shape"] = randomizer.output_shape_in(shape)

        self.obs_shapes[name] = shape
        self.obs_nets_classes[name] = net_class
        self.obs_nets_kwargs[name] = net_kwargs
        self.obs_nets[name] = net
        self.obs_randomizers[name] = randomizer
        self.obs_share_mods[name] = share_net_from

    def make(self):
        """
        Creates the encoder networks and locks the encoder so that more modalities cannot be added.
        """
        assert not self._locked, "ObservationEncoder: @make called more than once"
        self._create_layers()
        self._locked = True

    def _create_layers(self):
        """
        Creates all networks and layers required by this encoder using the registered modalities.
        """
        assert not self._locked, "ObservationEncoder: layers have already been created"

        for k in self.obs_shapes:
            if self.obs_nets_classes[k] is not None:
                # create net to process this modality
                self.obs_nets[k] = ObsUtils.OBS_ENCODER_CORES[self.obs_nets_classes[k]](**self.obs_nets_kwargs[k])
            elif self.obs_share_mods[k] is not None:
                # make sure net is shared with another modality
                self.obs_nets[k] = self.obs_nets[self.obs_share_mods[k]]

        self.activation = None
        if self.feature_activation is not None:
            self.activation = self.feature_activation()

    def forward(self, obs_dict):
        """
        Processes modalities according to the ordering in @self.obs_shapes. For each
        modality, it is processed with a randomizer (if present), an encoder
        network (if present), and again with the randomizer (if present), flattened,
        and then concatenated with the other processed modalities.

        Args:
            obs_dict (OrderedDict): dictionary that maps modalities to torch.Tensor
                batches that agree with @self.obs_shapes. All modalities in
                @self.obs_shapes must be present, but additional modalities
                can also be present.

        Returns:
            feats (torch.Tensor): flat features of shape [B, D]
        """

        assert self._locked, "ObservationEncoder: @make has not been called yet"

        # ensure all modalities that the encoder handles are present
        assert set(self.obs_shapes.keys()).issubset(obs_dict), "ObservationEncoder: {} does not contain all modalities {}".format(
            list(obs_dict.keys()), list(self.obs_shapes.keys())
        )
        # process modalities by order given by @self.obs_shapes
        feats_dict = OrderedDict()
        for k in self.obs_shapes:

            x = obs_dict[k]

            # maybe process encoder input with randomizer
            # depth: false; image: true
            # if self.obs_randomizers[k] is not None:
            #     x = self.obs_randomizers[k].forward_in(x)

            # maybe process with obs net
            has_extra_dim = False
            if self.obs_nets[k] is not None:
              has_extra_dim = (len(x.shape) == 5 or len(x.shape) == 3)  # [B, T, C, H, W] or [B, T, D]
              if has_extra_dim:
                # 输入含有时序维度 [B, T, C, H, W]，需要将时序维度和批次维度合并
                b, t = x.shape[0], x.shape[1]
                x = x.reshape(b * t, *x.shape[2:])

              x = self.obs_nets[k](x)
              # all is true: self.activation=ReLU
              # if self.activation is not None:
              #     x = self.activation(x)

              if has_extra_dim:
                # 恢复时序维度 [B*T, D] -> [B, T, D]
                x = x.reshape(b, t, *x.shape[1:])
                if len(x.shape) == 5:
                    # [B, T, D] -> [B, D]
                    # x = x.permute(0, 1, 3, 4, 2).contiguous()
                    x = x.reshape(b * x.shape[1], *x.shape[2:])
                    # x = x.reshape(b, x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
                has_extra_dim = False

            # maybe process encoder output with randomizer
            # depth: false; image: true
            # if self.obs_randomizers[k] is not None:
            #     x = self.obs_randomizers[k].forward_out(x)
            # flatten to [B, D]
            # x = TensorUtils.flatten(x, begin_axis=1)
            feats_dict[k] = x

        return feats_dict

    def output_shape(self, input_shape=None):
        """
        Compute the output shape of the encoder.
        """
        feat_dim = 0
        for k in self.obs_shapes:
            feat_shape = self.obs_shapes[k]
            if self.obs_nets[k] is not None:
                # agentview_imagae: 84 -> 64; robot0_eef_pos: 3 -> 3; robot0_eef_pos_future_traj: 30 ->  30
                feat_shape = self.obs_nets[k].output_shape(feat_shape)
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
            msg += textwrap.indent("sharing_from={}\n".format(self.obs_share_mods[k]), indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg += textwrap.indent("\noutput_shape={}".format(self.output_shape()), ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg


class ObservationGroupEncoder(Module):
    """
    This class allows networks to encode multiple observation dictionaries into a single
    flat, concatenated vector representation. It does this by assigning each observation
    dictionary (observation group) an @ObservationEncoder object.

    The class takes a dictionary of dictionaries, @observation_group_shapes.
    Each key corresponds to a observation group (e.g. 'obs', 'subgoal', 'goal')
    and each OrderedDict should be a map between modalities and 
    expected input shapes (e.g. { 'image' : (3, 120, 160) }).
    """
    def __init__(
        self,
        observation_group_shapes,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
    ):
        """
        Args:
            observation_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(ObservationGroupEncoder, self).__init__()

        # type checking
        assert isinstance(observation_group_shapes, OrderedDict)
        # assert np.all([isinstance(observation_group_shapes[k], OrderedDict) for k in observation_group_shapes])
        
        self.observation_group_shapes = observation_group_shapes

        # create an observation encoder per observation group
        self.nets = nn.ModuleDict()
        for obs_group in self.observation_group_shapes:
            self.nets[obs_group] = obs_encoder_factory(
                obs_shapes=self.observation_group_shapes[obs_group],
                feature_activation=feature_activation,
                encoder_kwargs=encoder_kwargs,
            )

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): dictionary that maps observation groups to observation
                dictionaries of torch.Tensor batches that agree with 
                @self.observation_group_shapes. All observation groups in
                @self.observation_group_shapes must be present, but additional
                observation groups can also be present. Note that these are specified
                as kwargs for ease of use with networks that name each observation
                stream in their forward calls.

        Returns:
            outputs (torch.Tensor): flat outputs of shape [B, D]
        """
        # ensure all observation groups we need are present
        assert set(self.observation_group_shapes.keys()).issubset(inputs), "{} does not contain all observation groups {}".format(
            list(inputs.keys()), list(self.observation_group_shapes.keys())
        )
        # 使用OrderedDict保留各个观测组的编码输出，方便后续处理，适应具体模型的接口
        outputs_dict = OrderedDict()
        for obs_group in self.observation_group_shapes:
            # pass through encoder and store features by group
            outputs_dict[obs_group] = self.nets[obs_group].forward(inputs[obs_group])
        return outputs_dict

    def output_shape(self):
        """
        Compute the output shape of this encoder.
        """
        feat_dim = 0
        for obs_group in self.observation_group_shapes:
            # get feature dimension of these keys
            feat_dim += self.nets[obs_group].output_shape()[0]
        return [feat_dim]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.observation_group_shapes:
            msg += '\n'
            indent = ' ' * 4
            msg += textwrap.indent("group={}\n{}".format(k, self.nets[k]), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class ObservationDecoder(Module):
    """
    Module that can generate observation outputs by modality. Inputs are assumed
    to be flat (usually outputs from some hidden layer). Each observation output
    is generated with a linear layer from these flat inputs. Subclass this
    module in order to implement more complex schemes for generating each
    modality.
    """
    def __init__(
        self,
        decode_shapes,
        input_feat_dim,
    ):
        """
        Args:
            decode_shapes (OrderedDict): a dictionary that maps observation key to
                expected shape. This is used to generate output modalities from the
                input features.

            input_feat_dim (int): flat input dimension size
        """
        super(ObservationDecoder, self).__init__()

        # important: sort observation keys to ensure consistent ordering of modalities
        assert isinstance(decode_shapes, OrderedDict)
        self.obs_shapes = OrderedDict()
        for k in decode_shapes:
            self.obs_shapes[k] = decode_shapes[k]

        self.input_feat_dim = input_feat_dim
        self._create_layers()

    def _create_layers(self):
        """
        Create a linear layer to predict each modality.
        """
        self.nets = nn.ModuleDict()
        for k in self.obs_shapes:
            layer_out_dim = int(np.prod(self.obs_shapes[k]))
            self.nets[k] = nn.Linear(self.input_feat_dim, layer_out_dim)
            # self.nets[k] = nn.Sequential(
            #   nn.Linear(self.input_feat_dim, 1024),
            #   nn.ReLU(),
            #   nn.Dropout(0.1),
            #   nn.Linear(1024, layer_out_dim),
            # )

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return { k : list(self.obs_shapes[k]) for k in self.obs_shapes }

    def forward(self, feats):
        """
        Predict each modality from input features, and reshape to each modality's shape.
        """
        output = {}
        for k in self.obs_shapes:
            out = self.nets[k](feats)
            output[k] = out  # .reshape(-1, *self.obs_shapes[k])
        return output

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.obs_shapes:
            msg += textwrap.indent('\nKey(\n', ' ' * 4)
            indent = ' ' * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent)
            msg += textwrap.indent("modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent)
            msg += textwrap.indent("net=({})\n".format(self.nets[k]), indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg


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


class DownstreamGroupDecoder(ObservationDecoder):
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
        self.nets["image_processor"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            feature_activation=None,
            encoder_kwargs=kwargs,
        )

        mlp_input_dim = self.nets["image_processor"].output_shape()[0]

        self.nets["mlp"] = MLP(
            input_dim=mlp_input_dim,
            output_dim=layer_dims[-1],
            layer_dims=layer_dims[:-1],
            layer_func=layer_func,
            activation=activation,
            output_activation=activation, # make sure non-linearity is applied before decoder
        )

    def output_shape(self, input_shape):
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
        encoder_kwargs=None, # NOTE: 不能使用可扩展关键字参数传递，其会将传入参数打包成字典，键为encoder_kwargs，值为传入的参数值，导致无法正确解析
    ):
        super(RESNET_MIMO_Transformer, self).__init__()

        self.input_obs_group_shapes = input_obs_group_shapes
        del self.input_obs_group_shapes['goal']  # 删除goal输入，底层网络不需要goal输入
        self.output_dim = output_shapes

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()
        self.nets["image_processor"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            feature_activation=None,
            encoder_kwargs=encoder_kwargs,
        )

        max_timestep = transformer_context_length

        if transformer_sinusoidal_embedding:
            self.nets["embed_timestep"] = PositionEncoder(transformer_embed_dim)
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

        self.nets["mlp_decoder"] = MLP(
            input_dim=transformer_embed_dim,
            output_dim=output_shapes['action'][0],
            layer_dims=[256, 128],
            output_activation=get_activation(transformer_activation),
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
        mlp_dec_outputs = self.nets["mlp_decoder"].forward(trans_dec_outputs)
        if return_latent:
            return mlp_dec_outputs, enc_outputs
        return mlp_dec_outputs

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
        

class MIMO_MLP(Module):
    """
    提供多输入多输出的整体网络，用于动作输出
    """
    def __init__(
        self,
        device,
        input_obs_group_shapes,
        output_shapes,
        layer_dims,
        layer_func="linear", 
        activation="relu",
        encoder_kwargs=None,
    ):
        super(MIMO_MLP, self).__init__()
        assert isinstance(output_shapes, OrderedDict), "output_shapes must be an OrderedDict"

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        # self.nets = nn.ModuleDict()

        self.pre_encoder = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # self.past_traj_encoder = PastTrajEncoder(
        #     input_dim=9,
        #     dim_feedforward=[256],
        #     output_dim=512,
        #     dropout=[0.1],
        #     feedforward_activation="relu",
        # )

        # self.past_traj_proj = PastTrajEncoder(
        #     input_dim=3,
        #     dim_feedforward=[],
        #     output_dim=512,
        #     # dropout=[0.1],
        #     feedforward_activation="relu",
        # )

        # self.traj_query_encoder = TrajQueryEncoder(
        #     in_channels=512+128,
        #     out_channels=1,
        #     activation="relu",
        # )

        # self.sub_target_decoder_coarse = CoarseTargetDecoder(
        #     input_dim = 128,
        #     dim_feedforward = [1024],
        #     output_dim = 512,
        #     dropout = [0.1],
        #     feedforward_activation = "relu",
        # )

        # self.sub_target_decoder_residual = ResidualTargetDecoder(
        #     n_decoder_layers = 3,
        #     dim_model = 512,
        #     n_heads = 8,
        #     dim_feedforward = 2048,
        #     dropout = 0.1,
        #     feedforward_activation = "relu",
        #     pre_norm = True,
        # )
        # self.sub_target_decoder = SubTargetDecoder(
        #     n_decoder_layers=3,
        #     dim_model=512,
        #     n_heads=8,
        #     dim_feedforward=2048,
        #     dropout=0.1,
        #     feedforward_activation="relu",
        #     pre_norm=True,
        # )

        # self.sub_segment_decoder = SubSegmentDecoder(
        #     n_decoder_layers=3,
        #     dim_model=512,
        #     n_heads=8,
        #     dim_feedforward=2048,
        #     dropout=0.1,
        #     feedforward_activation="relu",
        #     pre_norm=True,
        # )

        # self.fusion_proj = nn.Linear(512 * 2, 512)

        # self.gmm_head = ObservationDecoder(
        #     decode_shapes=self.output_shapes,
        #     input_feat_dim=512  # layer_dims[-1],
        # )

        # 设置词表和词嵌入
        # self.cls_input_seg1 = nn.Embedding(1, 512).to(device)
        # self.cls_input_seg2 = nn.Embedding(1, 128).to(device)
        # self.cls_input_seg3 = nn.Embedding(1, 128).to(device)
        # self.cls_input_seg4 = nn.Embedding(1, 128).to(device)
        # self.cls_input_seg5 = nn.Embedding(1, 128).to(device)
        
        # cls_token_seg1 = self.cls_input_seg1.weight
        # self.cls_token_seg1 = cls_token_seg1.repeat(32, 1) # [B, D]
        # cls_token_seg2 = self.cls_input_seg2.weight
        # self.cls_token_seg2 = cls_token_seg2.repeat(32, 1) # [B, D]
        # cls_token_seg3 = self.cls_input_seg3.weight
        # self.cls_token_seg3 = cls_token_seg3.repeat(32, 1) # [B, D]
        # cls_token_seg4 = self.cls_input_seg4.weight
        # self.cls_token_seg4 = cls_token_seg4.repeat(32, 1) # [B, D]
        # cls_token_seg5 = self.cls_input_seg5.weight
        # self.cls_token_seg5 = cls_token_seg5.repeat(32, 1) # [B, D]

        # self.register_buffer('time_scale', torch.tensor([0.5, 1.0]))
        self.query_input_num = 7
        self.register_buffer(
            'sub_target_pos_enc',
            create_sinusoidal_pos_embedding(self.query_input_num, 512).unsqueeze(1)
        )
        self.register_buffer(
            'state_joint_pos_enc',
            create_sinusoidal_pos_embedding(1, 512).unsqueeze(1)
        )
        # self.condition_mlp = MLP(
        #     input_dim=9,  #mlp_input_dim,
        #     output_dim=512,
        #     layer_dims=[256],
        #     layer_func=nn.Linear,
        #     dropouts=[0.1],
        #     activation=get_activation_fn("relu"),
        #     output_activation=get_activation_fn("relu"), # make sure non-linearity is applied before decoder
        # )

        # self.query_embedding = nn.Embedding(3, 512)
        
        # self.query_to_history = CrossAttentionLayer(
        #     n_layers = 3,
        #     dim_model=512,
        #     n_heads=8,
        #     dropout=0.1,
        #     pre_norm=True,
        # )
        # self.register_buffer(
        #     'sub_segment_pos_enc',
        #     create_sinusoidal_pos_embedding(3, 512).unsqueeze(1)
        # )
        self.encoder_image_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(512 // 2)

        # 深度图patch到像素中心点坐标的映射
        original_h, original_w = 240, 240  # 原始深度图尺寸
        patch_h, patch_w = 8, 8  # patch特征图尺寸
        # 计算每个patch对应的原始图像区域大小
        stride_h = original_h / patch_h  # 10.5
        stride_w = original_w / patch_w  # 10.5
        # 为每个patch计算其中心点在原始图像中的(u, v)坐标
        patch_centers = []
        for i in range(patch_h):
            for j in range(patch_w):
                # 计算patch中心点坐标 (u, v)
                center_v = (i + 0.5) * stride_h  # 行坐标 (v)
                center_u = (j + 0.5) * stride_w  # 列坐标 (u)
                patch_centers.append([center_u, center_v])
        # 转换为tensor并注册为buffer (不参与梯度计算)
        self.register_buffer(
            'depth_patch_centers',
            torch.tensor(patch_centers, dtype=torch.float32)  # [64, 2]
        )

        # 相机内参矩阵
        self.fx = 289.7056274847714
        self.fy = 289.7056274847714
        self.cx = 120.0
        self.cy = 120.0

        # 相机外参矩阵 (World to Camera transformation)
        self.register_buffer(
            'extrinsics_matrix',
            torch.tensor([
          [0.0, -0.6282662749116676, 0.777998385479441, -1.2528497000568177],
          [1.0, 0.0, 0.0, -0.658613],
          [0.0, 0.777998385479441, 0.6282662749116676, -1.0117285958040039],
          [0.0, 0.0, 0.0, 1.0]
            ], dtype=torch.float32)
        )
        
        # 相机位置 (World coordinates)
        self.register_buffer(
            'camera_position',
            torch.tensor([0.658613, 0.0, 1.61035], dtype=torch.float32)
        )
        
        # 相机旋转矩阵 (World to Camera rotation)
        self.register_buffer(
            'camera_rotation',
            torch.tensor([
          [0.0, -0.6282662749116676, 0.777998385479441],
          [1.0, 0.0, 0.0],
          [0.0, 0.777998385479441, 0.6282662749116676]
            ], dtype=torch.float32)
        )

        self.rgb_film_mlp = MLP(
            input_dim=3,  #mlp_input_dim,
            output_dim=1024,
            layer_dims=[512],
            layer_func=nn.Linear,
            dropouts=[0.1],
            activation=get_activation_fn("relu"),
            output_activation=get_activation_fn("relu"), # make sure non-linearity is applied before decoder
        )
        self.image_to_depth = CrossAttentionLayer(
            n_layers = 3,
            dim_model=512,
            n_heads=8,
            dropout=0.1,
            pre_norm=True,
        )
        self.depth_to_image = CrossAttentionLayer(
            n_layers = 3,
            dim_model=512,
            n_heads=8,
            dropout=0.1,
            pre_norm=True,
        )

        self.joint_pos_proj = nn.Linear(7, 512)
        self.query_to_patch = nn.ModuleList(
            SubTargetDecoderLayer(
                dim_model = 512,
                n_heads = 8,
                dim_feedforward = 3200,
                dropout = 0.1,
                feedforward_activation = "relu",
                pre_norm = True,
            ) for _ in range(6)
        )

        self.image_norm_center = TokenNormCenter(512)
        self.depth_norm_center = TokenNormCenter(512)
        self.traj_q_norm_center = TokenNormCenter(512)
        self.traj_kv_norm_center = TokenNormCenter(512)

        # 门控系数
        self.image_to_depth_a = nn.Parameter(torch.tensor(float(-6.0)))
        self.depth_to_image_a = nn.Parameter(torch.tensor(float(-6.0)))
        self.query_to_history_a = nn.Parameter(torch.tensor(float(-6.0)))
        self.query_to_patch_a = nn.Parameter(torch.tensor(float(-6.0)))

        self.action_proj = nn.Linear(512, 7)

        self._reset_parameters()

        # 用于可视化注意力权重的计数器
        self.counter = 0
        self.label = 0
        self.save_attention = False
        self.attention_maps = {
            'encoder_attn': [],
            'decoder_attn': []
        }

    def _reset_parameters(self):
        # for p in chain(self.sub_target_decoder_coarse.parameters(), self.sub_target_decoder_residual.parameters(), self.sub_segment_decoder.parameters()):
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        # for p in chain(self.past_traj_encoder.parameters(), self.traj_query_encoder.parameters()):
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        #         nn.init.zeros_(p)
        for p in chain(self.image_to_depth.parameters(), self.depth_to_image.parameters(), self.query_to_patch.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                nn.init.zeros_(p)

        if hasattr(self, 'fusion_proj'):
            nn.init.xavier_uniform_(self.fusion_proj.weight)
            nn.init.zeros_(self.fusion_proj.bias)
        # if hasattr(self.pre_encoder.nets.obs.obs_nets, 'robot0_eef_pos_step_traj_current'):
        #     nn.init.xavier_uniform_(self.pre_encoder.nets.obs.obs_nets['robot0_eef_pos_step_traj_current'].proj_net.weight)
        #     nn.init.zeros_(self.pre_encoder.nets.obs.obs_nets['robot0_eef_pos_step_traj_current'].proj_net.bias)
        # if hasattr(self.gmm_head.nets, 'mean'):
        #     nn.init.xavier_uniform_(self.gmm_head.nets['mean'].weight)
        #     nn.init.zeros_(self.gmm_head.nets['mean'].bias)
        # if hasattr(self.gmm_head.nets, 'scale'):
        #     nn.init.xavier_uniform_(self.gmm_head.nets['scale'].weight)
        #     nn.init.zeros_(self.gmm_head.nets['scale'].bias)
        # if hasattr(self.gmm_head.nets, 'logits'):
        #     nn.init.xavier_uniform_(self.gmm_head.nets['logits'].weight)
            # nn.init.zeros_(self.gmm_head.nets['logits'].bias)
        if hasattr(self, 'action_proj'):
            nn.init.xavier_uniform_(self.action_proj.weight)
            nn.init.zeros_(self.action_proj.bias)


    def forward(self, return_latent=False, return_attention_weights=False, fill_mode: str=None, stage: str=None, **inputs):
        """
        模型的底层前向传播函数
        """
        # depth_feat shape: [B, C, H, W]
        # depth_patch_centers shape: [64, 2] (u, v coordinates)
        # 将像素坐标归一化到 [-1, 1] 范围，用于 grid_sample
        # 原始图像尺寸
        original_h, original_w = 240, 240
        batch_size = inputs['obs']['agentview_depth'].shape[0]
        # 归一化坐标: x = 2 * u / (W - 1) - 1, y = 2 * v / (H - 1) - 1
        normalized_coords = self.depth_patch_centers.clone()
        normalized_coords[:, 0] = 2.0 * normalized_coords[:, 0] / (original_w - 1) - 1.0  # u -> x
        normalized_coords[:, 1] = 2.0 * normalized_coords[:, 1] / (original_h - 1) - 1.0  # v -> y
        # 扩展到 batch 维度: [B, 64, 2]
        grid = normalized_coords.unsqueeze(0).expand(batch_size, -1, -1)
        # grid_sample 需要 [B, H, W, 2] 格式的 grid，这里我们只有 64 个点
        # 因此需要 reshape: [B, 64, 2] -> [B, 8, 8, 2]
        grid = grid.view(batch_size, 8, 8, 2)
        # 对 depth_feat 进行双线性采样
        # mode='bilinear', padding_mode='border', align_corners=True
        sampled_depth = F.grid_sample(
            inputs['obs']['agentview_depth'], 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )  # [B, C, 8, 8]
        # 使用采样的深度值进行3D抬升
        # sampled_depth shape: [B, 1, 8, 8]
        depth_values = sampled_depth.squeeze(1)  # [B, 8, 8]
        # 获取每个patch中心点的像素坐标
        patch_centers_batch = self.depth_patch_centers.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 64, 2]
        u_coords = patch_centers_batch[:, :, 0].view(batch_size, 8, 8)  # [B, 8, 8]
        v_coords = patch_centers_batch[:, :, 1].view(batch_size, 8, 8)  # [B, 8, 8]
        # 3D抬升公式: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy (相机坐标系)
        Z_cam = depth_values  # [B, 8, 8]
        X_cam = (u_coords - self.cx) * Z_cam / self.fx  # [B, 8, 8]
        Y_cam = (v_coords - self.cy) * Z_cam / self.fy  # [B, 8, 8]
        # 将相机坐标转换为世界坐标
        # 构建相机坐标齐次坐标 [B, 64, 4]
        points_cam = torch.stack([X_cam, Y_cam, Z_cam], dim=1).view(batch_size, 3, -1)  # [B, 3, 64]
        ones = torch.ones(batch_size, 1, 64, device=points_cam.device)
        points_cam_homo = torch.cat([points_cam, ones], dim=1)  # [B, 4, 64]
        # 使用相机外参的逆矩阵进行坐标变换: P_world = T^(-1) * P_cam
        extrinsics_inv = torch.inverse(self.extrinsics_matrix)  # [4, 4]
        extrinsics_inv_batch = extrinsics_inv.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 4, 4]
        points_world_homo = torch.bmm(extrinsics_inv_batch, points_cam_homo)  # [B, 4, 64]
        
        # 提取世界坐标系下的3D坐标
        X = points_world_homo[:, 0, :].view(batch_size, 8, 8)  # [B, 8, 8]
        Y = points_world_homo[:, 1, :].view(batch_size, 8, 8)  # [B, 8, 8]
        Z = points_world_homo[:, 2, :].view(batch_size, 8, 8)  # [B, 8, 8]
        # 将3D坐标堆叠并展平为token序列
        # points_3d shape: [B, 3, 8, 8] -> [B, 64, 3]
        points_3d = torch.stack([X, Y, Z], dim=1)  # [B, 3, 8, 8]
        points_3d = points_3d.view(batch_size, 3, -1).permute(0, 2, 1)  # [B, 64, 3]

        # 图像和深度信息编码，并已经z-scores归一化所有输入
        enc_outputs_dict = self.pre_encoder.forward(**inputs)
        # 处理图像特征，获取位置编码
        image_feat = enc_outputs_dict['obs']['agentview_image']
        image_feat_pos_embed = self.encoder_image_feat_pos_embed(image_feat).to(dtype=image_feat.dtype)
        # FiLM调制图像特征
        gamma_raw, beta = self.rgb_film_mlp(points_3d).chunk(2, dim=-1)  # [B, 64, 512] each
        gamma = 1.0 + 0.1 * F.tanh(gamma_raw)  # [B, 64, 512]
        # 对image_feat进行FiLM调制
        C, H, W = image_feat.shape[1], image_feat.shape[2], image_feat.shape[3]
        image_feat = image_feat.view(batch_size, C, H * W).permute(0, 2, 1)  # [B, 64, C]
        image_feat = image_feat * gamma + beta  # FiLM调制


        # 计算token-token之间的3D欧氏距离矩阵
        # points_3d: [B, 64, 3]
        dist_3d = torch.cdist(points_3d, points_3d, p=2)  # [B, 64, 64]

        # 构建KNN掩码 (例如k=8的最近邻)
        k = 8
        dist_masked = dist_3d.clone()
        idx = torch.arange(64)
        dist_masked[:, idx, idx] = float('inf')  # 排除自身距离
        _, knn_indices = dist_masked.topk(k, dim=-1, largest=False)

        knn_mask = torch.zeros_like(dist_3d, dtype=torch.float32)  # [B, 64, 64]
        knn_mask.scatter_(dim=2, index=knn_indices, src=torch.ones_like(knn_indices, dtype=torch.float32))  # 标记KNN位置为True
        # knn_mask = torch.maximum(knn_mask, knn_mask.transpose(1, 2))  # 对称掩码
        # knn_mask[:, idx, idx] = 0.0  # 对角线位置清零

        # 几何正则
        knn_d = dist_3d.gather(2, knn_indices)  # [B, 64, k]
        sigma = knn_d.median(dim=-1).values.median(dim=-1).values  # [B]
        sigma = sigma.clamp_min(1e-6)  # 防止除零
        W = torch.exp(-(dist_3d**2) / (sigma[:, None, None]**2)) * knn_mask  # [B, 64, 64]
        W[:, idx, idx] = 0.0  # 对角线位置清零

        image_feat_norm = torch.nn.functional.layer_norm(image_feat, (512,))
        diff2 = (image_feat_norm[:, :, None, :] - image_feat_norm[:, None, :, :]).pow(2).mean(dim=-1)  # [B, 64, 64]
        L_geo = (W * diff2).sum() / W.sum().clamp_min(1)
        # L_geo = (W[..., None] * (image_feat[:, :, None, :] - image_feat[:, None, :, :]).pow(2)).sum()

        # 处理深度特征，实现跨模态注意力融合
        depth_feat = enc_outputs_dict['obs']['agentview_depth']
        # 图像和深度特征位置编码
        depth_feat_pos_embed = self.encoder_image_feat_pos_embed(depth_feat).to(dtype=depth_feat.dtype)
        depth_feat_pos_embed = einops.rearrange(depth_feat_pos_embed, 'B C H W -> (H W) B C')  # [N, B, D]
        depth_feat = einops.rearrange(depth_feat, 'B C H W -> B (H W) C')  # [N, B, D]
        # # token 去中心化
        # image_feat = self.image_norm_center(image_feat)
        # depth_feat = self.depth_norm_center(depth_feat)

        image_feat = einops.rearrange(image_feat, 'B S C -> S B C')  # [N, B, D]
        depth_feat = einops.rearrange(depth_feat, 'B S C -> S B C')  # [N, B, D]
        image_feat_pos_embed = einops.rearrange(image_feat_pos_embed, 'B C H W -> (H W) B C')  # [N, B, D]

        cam_feat = torch.cat([image_feat, depth_feat], dim=-1)  # [N, B, 2D]
        cam_feat = einops.rearrange(cam_feat, 'S B C -> B S C')  # [B, N, 2D]

        image_to_depth_feat, _ = self.image_to_depth(
            image_feat, depth_feat, image_feat_pos_embed, depth_feat_pos_embed
        )
        depth_to_image_feat, _ = self.depth_to_image(
            depth_feat, image_feat, depth_feat_pos_embed, image_feat_pos_embed
        )

        image_feat_fused = torch.sigmoid(self.image_to_depth_a) * image_to_depth_feat + image_feat
        depth_feat_fused = torch.sigmoid(self.depth_to_image_a) * depth_to_image_feat + depth_feat

        # 提取当前机器人状态
        joint_pos = enc_outputs_dict['obs']['robot0_joint_pos_past_traj'][:, -1, :]  # [B, 7]
        joint_pos_feat = self.joint_pos_proj(joint_pos)  # [B, 512]
        joint_pos_feat = joint_pos_feat.unsqueeze(0)  # [1, B, 512]
        # 融合图像和深度特征，带机器人自感知
        fused_feat = torch.cat([image_feat_fused, depth_feat_fused, joint_pos_feat], dim=0)  # [128+1, B, D]
        fused_feat_pos_embed = torch.cat([image_feat_pos_embed, depth_feat_pos_embed, self.state_joint_pos_enc], dim=0)  # [128+1, B, D]
        # 轨迹查询
        query_output =  torch.zeros(7, 32, 512).to(fused_feat.device)  # [query_num, batch_size, dim_model]
        for layer in self.query_to_patch:
            query_output = layer(
                query_output, fused_feat, self.sub_target_pos_enc, fused_feat_pos_embed
            )
            query_output = query_output + torch.sigmoid(self.query_to_patch_a) * query_output
        
        
        # 融合图像和深度特征
        # fused_feat = torch.cat([image_feat_fused, depth_feat_fused], dim=0)  # [128, B, D]
        # fused_feat_pos_embed = torch.cat([image_feat_pos_embed, depth_feat_pos_embed], dim=0)  # [128, B, D]
        # fused_points3d = torch.cat([points_3d, points_3d], dim=1)  # [B, 128, 3]
        
        # 轨迹查询
        # pos_traj = enc_outputs_dict['obs']['robot0_eef_pos_past_traj']  # ['robot0_eef_pos_step_traj_current']
        # vel_ang_traj = enc_outputs_dict['obs']['robot0_eef_vel_ang_past_traj']
        # vel_lin_traj = enc_outputs_dict['obs']['robot0_eef_vel_lin_past_traj']

        # traj = torch.cat([pos_traj, vel_lin_traj, vel_ang_traj], dim=-1)  # [B, S, 9]
        # traj_feat = self.past_traj_encoder(traj)  # [B, S, D] -> [32, 10, 512]

        # pos_traj_last = pos_traj[:, -1:, :]  # [B, 1, 3]
        # vel_ang_traj_last = vel_ang_traj[:, -1:, :]  # [B, 1, 3]
        # vel_lin_traj_last = vel_lin_traj[:, -1:, :]  # [B, 1, 3]
        # traj_feat_last = torch.cat([pos_traj_last, vel_lin_traj_last, vel_ang_traj_last], dim=-1)  # [B, 1, 9]

        # traj_feat_last = self.condition_mlp(traj_feat_last)  # [B, 1, D]，可能需要使用不同的投影网络

        # traj_feat_last_query = traj_feat_last.repeat(1, 3, 1)  # [B, 3, D]
        # query_embedding = self.query_embedding.weight  # [3, D]
        # query_embedding = query_embedding.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, D]
        # traj_feat_last_query = traj_feat_last_query + query_embedding  # [B, 3, D]
        
        # traj_feat = einops.rearrange(traj_feat, 'B S D -> S B D')  # [S, B, D]
        # traj_feat_last_query = einops.rearrange(traj_feat_last_query, 'B S D -> S B D')  # [S, B, D]

        # traj_dquery_output, _ = self.query_to_history(
        #     traj_feat_last_query, traj_feat, self.sub_segment_pos_enc, self.sub_target_pos_enc
        # )

        # traj_query_output = traj_feat_last_query + torch.sigmoid(self.query_to_history_a) * traj_dquery_output  # [S, B, D]

        # _, attn0 = self.query_to_patch(
        #    traj_query_output, fused_feat, self.sub_segment_pos_enc, fused_feat_pos_embed, return_attention=True
        # )

        # # 使用attn0对fused_points3d进行加权平均
        # # attn0: [B, S, N] 其中S=20是查询数量,N=128是patch数量
        # # fused_points3d: [B, 128, 3]
        # # 使用注意力权重对3D点进行加权平均
        # # attn_weights: [B, S, N], fused_points3d: [B, N, 3]
        # points_ref = torch.matmul(attn0, fused_points3d)  # [B, S, 3]
        
        # D = torch.norm(points_ref[:, :, None, :] - fused_points3d[:, None, :, :], dim=-1)  # [B, S, N]

        # D_med = D.median(dim=-1).values.clamp_min(1e-6)  # [B, S]
        # D_norm = D / D_med[:, :, None]  # [B, S, N]
        # lam = 1.0
        # B_seg = -lam * D_norm  # [B, S, N]
        # B_seg = B_seg.unsqueeze(1).repeat(1, 8, 1, 1)  # [B, 8, S, N]，扩展为4D以匹配注意力机制的bias输入要求
        # B_seg = B_seg.view(batch_size * 8, traj_query_output.shape[0], fused_feat.shape[0])  # [B*8, S, N]

        # z_output, _ = self.query_to_patch(
        #     traj_query_output, fused_feat, self.sub_segment_pos_enc, fused_feat_pos_embed, bias=B_seg
        # )

        # traj_query_output = einops.rearrange(traj_query_output, 'S B D -> B S D')  # [B, S, D]
        # z_output = einops.rearrange(z_output, 'S B D -> B S D')  # [B, S, D]
        # z_output = traj_query_output + torch.sigmoid(self.query_to_patch_a) * z_output  # [B, S, D]

        # q = traj_feat_last.repeat(1, self.query_input_num, 1)  # [B, S, D]
        # q = q + self.step_embedding.unsqueeze(0)
        # score = torch.einsum('btd,bkd->btk', q, z_output) / math.sqrt(512)  # [B, S, S]
        # w = F.softmax(score, dim=-1)  # [B, S, S]
        # z_step = torch.einsum('btk,bkd->btd', w, z_output)  # [B, S, D]

        # H = q + z_step  # [B, S, D]

        # traj_feat = enc_outputs_dict['obs']['robot0_eef_pos_past_traj']  # ['robot0_eef_pos_step_traj_current']
        # traj_feat_flat = traj_feat.clone()
        # if traj_feat_flat.dim() == 3:
        #     traj_feat_flat = traj_feat_flat.flatten(start_dim=1)  # [B, N, D] -> [B, N*D]
        # mlp_test_output   = torch.cat([image_feat, depth_feat, traj_feat], dim=-1)  # [B, D]
        # past_traj_encoder_output = self.past_traj_encoder(traj_feat_flat)  # [B, D]

        # past_traj_condition = past_traj_encoder_output.view(batch_size, -1, 1, 1).repeat(1, 1, 8, 8) # [B, D, H, W]
        # F_cat = torch.cat([image_feat, past_traj_condition], dim=1)  # [B, C+D, H, W]
        # logits_goal = self.traj_query_encoder(F_cat)  # [B, 1, H, W]
        # goal_feat = self.traj_query_encoder.weighted_pool(image_feat, logits_goal)  # [B, C]

        # sub_target_input = past_traj_encoder_output# torch.cat([past_traj_encoder_output, goal_feat], dim=-1)  # [B, D+C]
        # past_traj_query = self.past_traj_proj(traj_feat)
        # past_traj_query = einops.rearrange(past_traj_query, 'B S D -> S B D')
        # image_pos_embed = self.encoder_image_feat_pos_embed(image_feat).to(dtype=image_feat.dtype)
        # image_feat = einops.rearrange(image_feat, 'B C H W -> (H W) B C')  # [N, B, D]
        # image_pos_embed = einops.rearrange(image_pos_embed, 'B C H W -> (H W) B C')  # [N, B, D]

        # sub_target_output_coarse = self.sub_target_decoder_coarse(sub_target_input)  # [B, D]
        # sub_target_output_residual = self.sub_target_decoder_residual(past_traj_query, self.sub_target_pos_enc, image_feat, image_pos_embed)  # [S, B, D]
        # sub_target_output_coarse = sub_target_output_coarse.unsqueeze(0).repeat(20, 1, 1)  # [S, B, D]
        # sub_target_output_coarse = sub_target_output_coarse + self.sub_target_pos_enc
        # sub_target_output = sub_target_output_coarse #  + sub_target_output_residual  # [S, B, D]
        # sub_target_output = einops.rearrange(
        #     sub_target_output,
        #     'S B D -> B S D'
        # )  # [B, S, D]

        # sub_target_output = self.sub_target_decoder(sub_target_input, past_traj_query, self.sub_target_pos_enc, image_feat, image_pos_embed)

        # traj_query_input_seg1 = torch.cat([past_traj_encoder_output, cls_token_seg1, self.time_scale[0].unsqueeze(0).repeat(batch_size, 1)], dim=-1)  # [B, D+1]
        # traj_query_input_seg2 = torch.cat([past_traj_encoder_output, cls_token_seg2, self.time_scale[1].unsqueeze(0).repeat(batch_size, 1)], dim=-1)  # [B, D+1]
        # traj_query_input_seg3 = torch.cat([past_traj_encoder_output, cls_token_seg3, self.time_scale[2].unsqueeze(0).repeat(batch_size, 1)], dim=-1)  # [B, D+1]
        # traj_query_input_seg4 = torch.cat([past_traj_encoder_output, cls_token_seg4, self.time_scale[3].unsqueeze(0).repeat(batch_size, 1)], dim=-1)  # [B, D+1]
        # traj_query_input_seg5 = torch.cat([past_traj_encoder_output, cls_token_seg5, self.time_scale[4].unsqueeze(0).repeat(batch_size, 1)], dim=-1)  # [B, D+1]
        # traj_query_input_seg5 = torch.cat([past_traj_encoder_output, self.cls_token_seg5], dim=-1)  # [B, D+1]

        # traj_query_output_seg1 = self.traj_query_encoder(traj_query_input_seg1)  # [B, D]
        # traj_query_output_seg2 = self.traj_query_encoder(traj_query_input_seg2)  # [B, D]
        # traj_query_output_seg3 = self.traj_query_encoder(traj_query_input_seg3)  # [B, D]
        # traj_query_output_seg4 = self.traj_query_encoder(traj_query_input_seg4)  # [B, D]
        # traj_query_output_seg5 = self.traj_query_encoder(traj_query_input_seg5)  # [B, D]
        # traj_query_output_seg5 = self.traj_query_encoder(traj_query_input_seg5)  # [B, D]

        # sub_target_input = torch.stack([
        #     traj_query_output_seg1,
        #     traj_query_output_seg2,
        #     traj_query_output_seg3,
        #     traj_query_output_seg4,
        #     traj_query_output_seg5,
        # ], dim=0)  # [S, B, D]
        # sub_target_input = traj_query_output_seg5.unsqueeze(0)  # [1, B, D]

        # image_pos_embed = self.encoder_image_feat_pos_embed(image_feat).to(dtype=image_feat.dtype)
        # image_feat = einops.rearrange(image_feat, 'B C H W -> (H W) B C')  # [N, B, D]
        # image_pos_embed = einops.rearrange(image_pos_embed, 'B C H W -> (H W) B C')  # [N, B, D]
        
        # sub_target_pos_enc = self.sub_target_pos_enc.unsqueeze(1).to(dtype=image_feat.dtype)

        # sub_target_output = self.sub_target_decoder.forward(
        #     sub_target_input,
        #     sub_target_pos_enc,
        #     image_feat,
        #     image_pos_embed,
        # )  # [S, B, D]

        # sub_target_output = einops.rearrange(
        #     sub_target_output,
        #     'S B D -> B S D'
        # )  # [B, S, D]

        # if stage != 'A':
        #     # 仅阶段A不引入深度解码器
        #     depth_pos_embed = self.encoder_image_feat_pos_embed(depth_feat).to(dtype=depth_feat.dtype)
        #     depth_feat = einops.rearrange(depth_feat, 'B C H W -> (H W) B C')  # [N, B, D]
        #     depth_pos_embed = einops.rearrange(depth_pos_embed, 'B C H W -> (H W) B C')  # [N, B, D]
            
        #     sub_segment_output = self.sub_segment_decoder.forward(
        #         sub_target_input,
        #         sub_target_pos_enc,
        #         depth_feat,
        #         depth_pos_embed,
        #     )  # [S, B, D]

        #     sub_segment_output = einops.rearrange(
        #         sub_segment_output,
        #         'S B D -> B S D'
        #     )  # [B, S, D]

        #     # 融合两部分输出
        #     fused_output = torch.cat([sub_target_output, sub_segment_output], dim=-1)  # [B, S, D*2]
        #     fused_output = self.fusion_proj(fused_output)  # [B, S, D]

        # else:
        #     fused_output = sub_target_output

        # gmm_head_output = self.gmm_head.forward(segment_output)  # [B, S, D]
        query_output = einops.rearrange(query_output, 'S B D -> B S D')  # [B, S, D]
        action_output = self.action_proj(query_output)  # [B, S, 7]

        if self.save_attention:
            self.attention_maps['encoder_attn'] = self.nets['backbone_encoder'].get_attention_weights()
            self.attention_maps['decoder_attn'] = self.nets['backbone_decoder'].get_attention_weights()
        # depth_norm = F.normalize(depth_input, p=2, dim=-1)

        # attn_bias = torch.bmm(depth_norm, depth_norm.transpose(1, 2))  # [B, N, N]
        # attn_bias_scale = 0.05 * attn_bias  # scale the bias

        # attn_bias_scale_multihead = attn_bias_scale.unsqueeze(1)

        if return_latent and return_attention_weights:
            return action_output, L_geo, cam_feat.detach(), joint_pos_feat.detach()
        elif return_latent and not return_attention_weights:
            return action_output, L_geo, joint_pos_feat.detach()
        elif not return_latent and return_attention_weights:
            return action_output, L_geo, cam_feat.detach()
        else:
            return action_output, L_geo
        # enc_inputs = {
        #     # 'robot0_eef_pos': obs['robot0_eef_pos'],
        #     'cls': cls_embed,
        #     'agentview_image': obs['agentview_image'],
        #     'agentview_depth': obs['agentview_depth']
        # }

        # img_feat = torch.cat([obs['agentview_image'], obs['agentview_depth']], dim=-1)

        # dec_inputs = {
        #     # 'text': cls_embed,
        #     'robot0_eef_pos_step_traj_current': obs['robot0_eef_pos_step_traj_current'],
        #     # 'robot0_eef_pos_past_traj_delta': obs['robot0_eef_pos_past_traj_delta']
        # }

        # self.nets['backbone'].enable_attention_storage()

        # if return_attention_weights:
        #     backbone_outputs, attention_weights = self.nets['backbone'].forward(enc_inputs, dec_inputs, return_attention_weights=return_attention_weights, fill_mode=fill_mode)
        # else:
        #     backbone_outputs = self.nets['backbone'].forward(enc_inputs, dec_inputs, return_attention_weights=return_attention_weights, fill_mode=fill_mode)
        
        # self.counter += 1
        # if self.counter >= 1000 and return_attention_weights:
        #     self.counter = 0
        #     self.label += 1

        #     layer_idx = 0
        #     print("\n")
        #     print(f"可视化第{layer_idx}层的注意力流...")
        #     encoder_attn = attention_weights['encoder'][layer_idx]['self_attention'] if attention_weights['encoder'] else None
        #     decoder_self_attn = attention_weights['decoder'][layer_idx]['self_attention'] if attention_weights['decoder'] else None
        #     decoder_cross_attn = attention_weights['decoder'][layer_idx].get('cross_attention', None) if attention_weights['decoder'] else None

        #     self.nets['backbone'].visualizer.plot_attention_flow(
        #         encoder_attention=encoder_attn,
        #         decoder_self_attention=decoder_self_attn,
        #         decoder_cross_attention=decoder_cross_attn,
        #         layer_idx=layer_idx,
        #         save_path=f'./transformer/attention{self.label}/attention_flow_layer{layer_idx}.png',
        #         show=False
        #     )
        #     self.nets['backbone'].visualizer.save_attention_statistics(
        #         attention_weights,
        #         save_path=f'./transformer/attention{self.label}/attention_statistics{layer_idx}.json'
        #     )
        # self.nets['backbone'].disable_attention_storage()

        # backbone_outputs = backbone_outputs.flatten(start_dim=1)

        # dec_outputs = self.nets["decoder"].forward(backbone_outputs)

        # if return_latent and not return_attention_weights:
        #     return dec_outputs, backbone_outputs.detach()
        # elif return_latent and return_attention_weights:
        #     return dec_outputs, backbone_outputs.detach(), attention_weights
        # elif not return_latent and return_attention_weights:
        #     return dec_outputs, img_feat.detach(), attention_weights

        # return dec_outputs

    def output_shape(self, input_shape=None):
        return { k : list(self.output_shapes[k]) for k in self.output_shapes }
    
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
        msg += textwrap.indent("\npre_encoder={}".format(self.pre_encoder), indent)
        # msg += textwrap.indent("\npast_traj_encoder={}".format(self.past_traj_encoder), indent)
        # msg += textwrap.indent("\ntraj_query_encoder={}".format(self.traj_query_encoder), indent)
        # msg += textwrap.indent("\nsub_target_decoder_coarse={}".format(self.sub_target_decoder_coarse), indent)
        # msg += textwrap.indent("\nsub_target_decoder_residual={}".format(self.sub_target_decoder_residual), indent)
        # msg += textwrap.indent("\ngmm_head={}".format(self.gmm_head), indent)
        # msg += textwrap.indent("\n\nmlp={}".format(self.nets["mlp"]), indent)
        # msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg
    
    def enable_attention_saving(self):
        self.save_attention = True
        for layer in self.nets['backbone_encoder'].layers:
            layer.save_attention = True
        for layer in self.nets['backbone_decoder'].layers:
            layer.save_attention = True
    
    def disable_attention_saving(self):
        self.save_attention = False
        for layer in self.nets['backbone_encoder'].layers:
            layer.save_attention = False
        for layer in self.nets['backbone_decoder'].layers:
            layer.save_attention = False
    
    def visualize_attention_maps(self, save_dir='./transformer', batch_idx=0):
        
        os.makedirs(save_dir, exist_ok=True)

        for layer_idx, attn_weights in enumerate(self.attention_maps['encoder_self_attn']):
            fig = self._visualize_single_attention(
                attn_weights, batch_idx, save_dir,
                f'encoder_layer{layer_idx}_self_attn',
                'Encoder Self-Attention',
                layer_idx,
            )
        
        for layer_idx, attn_weights in enumerate(self.attention_maps['decoder_self_attn']):
            fig = self._visualize_single_attention(
                attn_weights, batch_idx, save_dir,
                f'decoder_layer{layer_idx}_self_attn',
                'Decoder Self-Attention',
                layer_idx,
            )
        
        for layer_idx, attn_weights in enumerate(self.attention_maps['decoder_cross_attn']):
            fig = self._visualize_single_attention(
                attn_weights, batch_idx, save_dir,
                f'decoder_layer{layer_idx}_cross_attn',
                'Decoder Cross-Attention',
                layer_idx,
            )
        
    def _visualize_single_attention(self, attn_weights, batch_idx, save_dir, filename, title, layer_idx):
        """
        可视化单个注意力权重矩阵
        """
        import matplotlib.pyplot as plt

        if attn_weights is None:
            return None

        attn_np = attn_weights[batch_idx].detach().cpu().numpy()  # [num_heads, N, N]
        num_heads = attn_np.shape[0]

        attn_avg = attn_np.mean(axis=0)  # [N, N]

        fig = plt.figure(figsize=(20, 4 * ((num_heads + 1) // 4 + 1)))

        for head_idx in range(num_heads):
            plt.subplot((num_heads + 1) // 4 + 1, 4, head_idx + 1)
            plt.imshow(attn_np[head_idx], cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f'{title} - Head {head_idx} - Layer {layer_idx}')
            plt.tight_layout()
        
        # Plot average attention
        plt.subplot((num_heads + 1) // 4 + 1, 4, num_heads + 1)
        plt.imshow(attn_avg, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Average Attention over all heads')
        plt.suptitle(f'{title} - Layer {layer_idx}', fontsize=14, y=1.0)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{filename}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        print(f"Saved: {save_path}")
        plt.close(fig)

        return None


class TokenNormCenter(nn.Module):
    def __init__(self, d=512):
      super().__init__()

      self.ln = nn.LayerNorm(d)

    def forward(self, x):
      x = self.ln(x)
      # x = x - x.mean(dim=1, keepdim=True)
      return x

class PastTrajEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int=30,
        dim_feedforward: Optional[list[int]]=None,
        output_dim: int=512,
        dropout: Optional[list[float]]=None,
        feedforward_activation: str="relu",
    ):
        super(PastTrajEncoder, self).__init__()

        self.traj_encoder = MLP(
            input_dim=input_dim,  #mlp_input_dim,
            output_dim=output_dim,
            layer_dims=dim_feedforward,
            layer_func=nn.Linear,
            dropouts=dropout,
            activation=get_activation_fn(feedforward_activation),
            output_activation=get_activation_fn(feedforward_activation), # make sure non-linearity is applied before decoder
        )

    def forward(self, traj_seg):
        return self.traj_encoder(traj_seg)
    

class TrajQueryEncoder(nn.Module):
  """
  接收 ResNet / 图像特征图 (B, C, H, W)，依次经过 3x3 卷积、激活、1x1 卷积，
  输出 heatmap (B, out_channels, H, W)。

  Args:
    in_channels (int): 输入特征图通道数（ResNet 输出通道数）。
    mid_channels (int): 中间卷积通道数（3x3 的输出通道）。
    out_channels (int): heatmap 输出通道数。
    activation (str|Callable): 激活函数名称 ('relu','gelu','glu') 或 nn.Module 类/实例。
  """
  def __init__(
    self,
    in_channels: int = 512,
    out_channels: int = 1,
    activation: str | Callable = "relu",
  ):
    super(TrajQueryEncoder, self).__init__()
    mid_channels = in_channels // 2  # 中间通道数为输入通道数的一半
    # 3x3 conv 保持空间尺寸，1x1 conv 投影到 heatmap 通道
    self.conv3x3 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True)
    self.conv1x1 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True)

    # 构造激活函数：接受字符串或直接传入的可调用
    if isinstance(activation, str):
      act_cls = get_activation_fn(activation)
      self.act = act_cls()
    elif isinstance(activation, type) and issubclass(activation, nn.Module):
      self.act = activation()
    else:
      # 允许传入已经实例化的激活函数
      self.act = activation

    # 权重初始化：与文件中其它模块风格一致
    nn.init.kaiming_uniform_(self.conv3x3.weight, a=math.sqrt(5))
    if self.conv3x3.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv3x3.weight)
      bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
      nn.init.uniform_(self.conv3x3.bias, -bound, bound)
    nn.init.kaiming_uniform_(self.conv1x1.weight, a=math.sqrt(5))
    if self.conv1x1.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv1x1.weight)
      bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
      nn.init.uniform_(self.conv1x1.bias, -bound, bound)

  def forward(self, feat_map: Tensor) -> Tensor:
    """
    Args:
      feat_map: (B, C, H, W) 输入特征图
    Returns:
      heatmap: (B, out_channels, H, W)
    """
    x = self.conv3x3(feat_map)
    x = self.act(x)
    heatmap = self.conv1x1(x)
    return heatmap

  def weighted_pool(self, feat_map, heatmap):
    """
    使用 heatmap 对特征图进行加权池化。

    Args:
      feat_map: (B, C, H, W) 输入特征图
      heatmap: (B, out_channels, H, W) 权重热图
    Returns:
      pooled: (B, C) 加权池化后的特征向量
    """
    # 对 heatmap 进行 softmax 归一化，在空间维度 (H, W) 上
    B, out_ch, H, W = heatmap.shape
    heatmap_flat = heatmap.view(B, out_ch, -1)  # (B, out_channels, H*W)
    weights = F.softmax(heatmap_flat, dim=-1)  # (B, out_channels, H*W)

    # 如果 out_channels > 1，对所有通道的权重取平均
    if out_ch > 1:
        weights = weights.mean(dim=1, keepdim=True)  # (B, 1, H*W)

    # 将特征图展平
    C = feat_map.size(1)
    feat_flat = feat_map.view(B, C, -1)  # (B, C, H*W)

    # 广播权重到所有通道，并进行加权求和
    # weights: (B, 1, H*W) -> broadcast to (B, C, H*W)
    pooled = (feat_flat * weights).sum(dim=-1)  # (B, C)

    return pooled


class SelfAttentionBlock(nn.Module):
  """自注意力模块"""
  def __init__(
    self,
    dim_model: int = 512,
    n_heads: int = 8,
    dropout: float = 0.1,
    pre_norm: bool = True,
  ):
    super(SelfAttentionBlock, self).__init__()
    self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
    self.norm = nn.LayerNorm(dim_model)
    self.dropout = nn.Dropout(dropout)
    self.pre_norm = pre_norm
    
    self.save_attention = False
    self.last_attn_weights = None

  def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Optional[Tensor] = None) -> Tensor:
    return tensor if pos_embed is None else tensor + pos_embed

  def forward(
    self,
    x: Tensor,
    pos_embed: Optional[Tensor] = None,
  ) -> Tensor:
    skip = x
    if self.pre_norm:
      x = self.norm(x)
    
    q = k = self.maybe_add_pos_embed(x, pos_embed)
    if self.save_attention:
      x, attn_weights = self.self_attn(q, k, value=x, need_weights=True)
      self.last_attn_weights = attn_weights.detach()
    else:
      x = self.self_attn(q, k, value=x)[0]
    
    x = skip + self.dropout(x)
    if not self.pre_norm:
      x = self.norm(x)
    return x


class CrossAttentionBlock(nn.Module):
  """交叉注意力模块"""
  def __init__(
    self,
    dim_model: int = 512,
    n_heads: int = 8,
    dropout: float = 0.1,
    pre_norm: bool = True,
  ):
    super(CrossAttentionBlock, self).__init__()
    self.multihead_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
    self.norm = nn.LayerNorm(dim_model)
    self.dropout = nn.Dropout(dropout)
    self.pre_norm = pre_norm
    
    self.save_attention = False
    self.last_attn_weights = None

  def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Optional[Tensor] = None) -> Tensor:
    return tensor if pos_embed is None else tensor + pos_embed

  def forward(
    self,
    x: Tensor,
    encoder_out: Tensor,
    decoder_pos_embed: Optional[Tensor] = None,
    encoder_pos_embed: Optional[Tensor] = None,
    return_attention: bool = True,
    bias: Optional[Tensor] = None,
  ) -> Tensor:
    skip = x
    if self.pre_norm:
      x = self.norm(x)

    x, attn_weights = self.multihead_attn(
      query=self.maybe_add_pos_embed(x, decoder_pos_embed),
      key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
      value=encoder_out,
      need_weights=True,
      attn_mask=bias,
      # average_attn_weights=False,
    )

    if self.save_attention:
      self.last_attn_weights = attn_weights.detach()
    
    x = skip + self.dropout(x)
    if not self.pre_norm:
      x = self.norm(x)
    
    # if return_attention:
    #   return x, attn_weights
    
    return x, attn_weights

class CrossAttentionLayer(nn.Module):
    """交叉注意力层，仅包含交叉注意力模块"""
    def __init__(
        self,
        n_layers: int = 3,
        dim_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ):
      super(CrossAttentionLayer, self).__init__()
      
      self.cross_attn_layers = nn.ModuleList([
        CrossAttentionBlock(
          dim_model=dim_model,
          n_heads=n_heads,
          dropout=dropout,
          pre_norm=pre_norm,
        )
        for _ in range(n_layers)
      ])

      # 添加最后的输出归一化
      self.norm = nn.LayerNorm(dim_model)

    def forward(
      self,
      x: Tensor,
      encoder_out: Tensor,
      decoder_pos_embed: Optional[Tensor] = None,
      encoder_pos_embed: Optional[Tensor] = None,
      return_attention: bool = True,
      bias: Optional[Tensor] = None,
    ) -> Tensor:
      """
      Args:
        x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
        encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder.
        decoder_pos_embed: (DS, 1, C) positional embedding for decoder queries.
        encoder_pos_embed: (ES, 1, C) positional embedding for encoder keys.
      Returns:
        (DS, B, C) tensor of decoder output features.
      """
      for layer in self.cross_attn_layers:
        x, attn = layer(
          x,
          encoder_out,
          decoder_pos_embed,
          encoder_pos_embed,
          return_attention=return_attention,
          bias=bias,
        )

      # 在返回前进行归一化
      x = self.norm(x)
    
      return x, attn

class SubTargetDecoderLayer(nn.Module):
  """完整的解码器层，结合自注意力、交叉注意力和前馈网络"""
  def __init__(
    self,
    dim_model: int = 512,
    n_heads: int = 8,
    dim_feedforward: int = 3200,
    dropout: float = 0.1,
    feedforward_activation: str = "relu",
    pre_norm: bool = True,
  ):
    super(SubTargetDecoderLayer, self).__init__()
    
    # 自注意力模块
    self.self_attn_block = SelfAttentionBlock(
      dim_model=dim_model,
      n_heads=n_heads,
      dropout=dropout,
      pre_norm=pre_norm,
    )
    
    # 交叉注意力模块
    self.cross_attn_block = CrossAttentionBlock(
      dim_model=dim_model,
      n_heads=n_heads,
      dropout=dropout,
      pre_norm=pre_norm,
    )
    
    # 前馈网络
    self.linear1 = nn.Linear(dim_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, dim_model)
    self.norm3 = nn.LayerNorm(dim_model)
    self.dropout3 = nn.Dropout(dropout)
    self.activation = get_activation_fn(feedforward_activation)()
    self.pre_norm = pre_norm
    
    self.save_attention = False

  @property
  def last_self_attn_weights(self):
    return self.self_attn_block.last_attn_weights
  
  @property
  def last_cross_attn_weights(self):
    return self.cross_attn_block.last_attn_weights

  def forward(
    self,
    x: Tensor,
    encoder_out: Tensor,
    decoder_pos_embed: Optional[Tensor] = None,
    encoder_pos_embed: Optional[Tensor] = None,
    return_attention: bool = True,
    bias: Optional[Tensor] = None,
  ) -> Tensor:
    """
    Args:
      x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
      encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder.
      decoder_pos_embed: (DS, 1, C) positional embedding for decoder queries.
      encoder_pos_embed: (ES, 1, C) positional embedding for encoder keys.
    Returns:
      (DS, B, C) tensor of decoder output features.
    """
    # 设置子模块的注意力保存状态
    self.self_attn_block.save_attention = self.save_attention
    self.cross_attn_block.save_attention = self.save_attention
    
    # 自注意力
    x = self.self_attn_block(x, decoder_pos_embed)
    
    # 交叉注意力
    x, _ = self.cross_attn_block(x, encoder_out, decoder_pos_embed, encoder_pos_embed, return_attention=return_attention, bias=bias)
    
    # 前馈网络
    skip = x
    if self.pre_norm:
      x = self.norm3(x)
    x = self.linear2(self.dropout(self.activation(self.linear1(x))))
    x = skip + self.dropout3(x)
    if not self.pre_norm:
      x = self.norm3(x)
    
    return x

class CoarseTargetDecoder(nn.Module):
  """MLP主干网络，用于粗粒度目标解码"""
  def __init__(
    self,
    input_dim: int = 512 + 128,
    dim_feedforward: Optional[list[int]] = None,
    output_dim: int = 512,
    dropout: Optional[list[float]] = None,
    feedforward_activation: str = "relu",
  ):
    super(CoarseTargetDecoder, self).__init__()
    
    self.mlp = MLP(
      input_dim=input_dim,
      output_dim=output_dim,
      layer_dims=dim_feedforward,
      layer_func=nn.Linear,
      dropouts=dropout,
      activation=get_activation_fn(feedforward_activation),
      output_activation=get_activation_fn(feedforward_activation),
    )
  
  def forward(self, x: Tensor) -> Tensor:
    """
    Args:
      x: (B, input_dim) 输入特征
    Returns:
      (B, output_dim) 粗粒度目标特征
    """
    return self.mlp(x)


class ResidualTargetDecoder(nn.Module):
  """Transformer解码器，用于细粒度残差修正"""
  def __init__(
    self,
    n_decoder_layers: int = 3,
    dim_model: int = 512,
    n_heads: int = 8,
    dim_feedforward: int = 3200,
    dropout: float = 0.1,
    feedforward_activation: str = "relu",
    pre_norm: bool = True,
  ):
    super(ResidualTargetDecoder, self).__init__()
    
    self.decoder_layers = nn.ModuleList([
      SubTargetDecoderLayer(
        dim_model=dim_model,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        feedforward_activation=feedforward_activation,
        pre_norm=pre_norm,
      )
      for _ in range(n_decoder_layers)
    ])
    
    self.norm = nn.LayerNorm(dim_model)
  
  def forward(
    self,
    decoder_input: Tensor,
    decoder_pos_embed: Optional[Tensor] = None,
    encoder_context: Optional[Tensor] = None,
    encoder_pos_embed: Optional[Tensor] = None,
  ) -> Tensor:
    """
    Args:
      decoder_input: (S, B, D) 解码器输入
      decoder_pos_embed: (S, 1, D) 解码器位置编码
      encoder_context: (N, B, D) 编码器上下文
      encoder_pos_embed: (N, 1, D) 编码器位置编码
    Returns:
      (S, B, D) 细粒度残差特征
    """
    x = decoder_input
    for layer in self.decoder_layers:
      x = layer(
        x,
        encoder_context,
        decoder_pos_embed=decoder_pos_embed,
        encoder_pos_embed=encoder_pos_embed,
      )
    
    return self.norm(x)


class SubTargetDecoder(nn.Module):
  """组合粗粒度和细粒度解码器，通过残差连接输出最终结果"""
  def __init__(
    self,
    input_dim: int = 512 + 128,
    dim_feedforward: int = 1024,
    output_dim: int = 512,
    n_decoder_layers: int = 3,
    dim_model: int = 512,
    n_heads: int = 8,
    trans_dim_feedforward: int = 3200,
    dropout: float = 0.1,
    feedforward_activation: str = "relu",
    pre_norm: bool = True,
  ):
    super(SubTargetDecoder, self).__init__()
    
    # 粗粒度MLP解码器
    self.coarse_decoder = CoarseTargetDecoder(
      input_dim=input_dim,
      dim_feedforward=dim_feedforward,
      output_dim=output_dim,
      dropout=dropout,
      feedforward_activation=feedforward_activation,
    )
    
    # 细粒度Transformer解码器
    self.residual_decoder = ResidualTargetDecoder(
      n_decoder_layers=n_decoder_layers,
      dim_model=dim_model,
      n_heads=n_heads,
      dim_feedforward=trans_dim_feedforward,
      dropout=dropout,
      feedforward_activation=feedforward_activation,
      pre_norm=pre_norm,
    )
  
  def forward(
    self,
    coarse_input: Tensor,
    decoder_input: Tensor,
    decoder_pos_embed: Optional[Tensor] = None,
    encoder_context: Optional[Tensor] = None,
    encoder_pos_embed: Optional[Tensor] = None,
  ) -> Tensor:
    """
    Args:
      coarse_input: (B, input_dim) MLP输入
      decoder_input: (S, B, D) Transformer解码器输入
      decoder_pos_embed: (S, 1, D) 解码器位置编码
      encoder_context: (N, B, D) 编码器上下文
      encoder_pos_embed: (N, 1, D) 编码器位置编码
    Returns:
      (S, B, D) 最终输出特征
    """
    # 粗粒度分支
    coarse_output = self.coarse_decoder(coarse_input)  # (B, output_dim)
    
    # 细粒度分支
    residual_output = self.residual_decoder(
      decoder_input,
      decoder_pos_embed=decoder_pos_embed,
      encoder_context=encoder_context,
      encoder_pos_embed=encoder_pos_embed,
    )  # (S, B, D)
    
    # 残差连接：需要将coarse_output扩展到序列维度
    S = residual_output.shape[0]
    coarse_output_expanded = coarse_output.unsqueeze(0).expand(S, -1, -1)  # (S, B, D)
    
    output = coarse_output_expanded + residual_output
    
    return output


class SubSegmentDecoder(nn.Module):
    def __init__(
      self,
      n_decoder_layers: int=3,
      dim_model: int=512,
      n_heads: int=8,
      dim_feedforward: int=3200,
      dropout: float=0.1,
      feedforward_activation: str="relu",
      pre_norm: bool=True,
    ):
        super(SubSegmentDecoder, self).__init__()

        self.sub_target_decoder = nn.ModuleList([
            SubTargetDecoderLayer(
                dim_model=dim_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                feedforward_activation=feedforward_activation,
                pre_norm=pre_norm,
            )
            for _ in range(n_decoder_layers)
        ])

        # 输出归一化
        self.norm = nn.LayerNorm(dim_model)
    
    def forward(
        self,
        decoder_input,
        decoder_pos_embed,
        encoder_context,
        encoder_pos_embed,
    ):

        for layer in self.sub_target_decoder:
            decoder_input = layer(
                decoder_input,
                encoder_context,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        
        z_subtraj_output = self.norm(decoder_input)

        return z_subtraj_output

def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()




class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed
    

def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
