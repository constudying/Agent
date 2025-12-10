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
    obs_shapes['robot0_eef_pos_step_traj_current'] = [10, 3]


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

            # # maybe process encoder input with randomizer
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

            # # maybe process encoder output with randomizer
            # if self.obs_randomizers[k] is not None:
            #     x = self.obs_randomizers[k].forward_out(x)
            # # flatten to [B, D]
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
            output[k] = out.reshape(-1, *self.obs_shapes[k])
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

        self.nets = nn.ModuleDict()
        self.nets["pre_encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # self.nets['backbone'] = Transformer(#HumanRobotCoupledInterACT(
        #     embed_dim=512,
        #     context_length=30,
        #     attn_dropout=0.1,
        #     output_dropout=0.1,
        #     ffw_hidden_dim=2048,
        #     ffw_dropout=0.1,
        #     num_heads=8,
        #     num_encoder_layers=6,
        #     num_decoder_layers=6,
        #     activation='relu',
        # )

        self.nets['backbone_encoder'] = IDCEncoder(
            num_blocks=3,
            num_cls_tokens_traj=3,
            num_cls_tokens_image=3,
            num_cls_tokens_depth=3,
            dim_model=512,
            n_heads=8,
            dim_feedforward=3200,
            dropout=0.1,
            feedforward_activation="relu",
            pre_norm=True,
        )
        self.nets['backbone_decoder'] = TrajectoryDecoder(
            num_cls_tokens_traj=3,
            num_cls_tokens_image=3,
            num_cls_tokens_depth=3,
            n_pre_decoder_layers=2,
            n_post_decoder_layers=2,
            n_sync_decoder_layers=1,
            dim_model=512,
            n_heads=8,
            dim_feedforward=3200,
            dropout=0.1,
            feedforward_activation="relu",
            pre_norm=True,
        )
        # self.nets["mlp"] = MLP(
        #     input_dim=backbone_output_dim,
        #     output_dim=layer_dims[-1],
        #     layer_dims=layer_dims[:-1],
        #     layer_func=get_activation(layer_func),
        #     activation=get_activation(activation),
        #     output_activation=get_activation(activation), # make sure non-linearity is applied before decoder
        # )
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=layer_dims[-1]*10,
        )

        # 设置词表和词嵌入
        self.cls_input_traj = nn.Embedding(1, 512).to(device)
        cls_input_traj = self.cls_input_traj.weight
        self.cls_token_traj = cls_input_traj.repeat(3, 1)
        num_robot_input_token_encoder = 3 + 10
        self.register_buffer(
            'traj_encoder_pos_enc',
            create_sinusoidal_pos_embedding(
                num_robot_input_token_encoder,
                512
            ),
        )

        self.cls_input_image = nn.Embedding(1, 512).to(device)
        cls_input_image = self.cls_input_image.weight
        self.cls_token_image = cls_input_image.repeat(3, 1)
        self.register_buffer(
            "image_encoder_pos_enc",
            create_sinusoidal_pos_embedding(
                3,
                512
            ),
        )
        self.cls_input_depth = nn.Embedding(1, 512).to(device)
        cls_input_depth = self.cls_input_depth.weight
        self.cls_token_depth = cls_input_depth.repeat(3, 1)
        self.register_buffer(
            "depth_encoder_pos_enc",
            create_sinusoidal_pos_embedding(
                3,
                512
            ),
        )
        # 用于cls聚合的位置嵌入
        self.register_buffer(
            'cls_encoder_pos_enc',
            create_sinusoidal_pos_embedding(
                3*3,
                512
            ),
        )
        # 2d数据的位置嵌入
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(512 // 2)
        # 用于未来轨迹解码器的位置嵌入，可学习的位置嵌入
        self.decoder_pos_embed = nn.Embedding(10, 512).to(device)

        # 用于可视化注意力权重的计数器
        self.counter = 0
        self.label = 0
        self.save_attention = False
        self.attention_maps = {
            'encoder_attn': [],
            'decoder_attn': []
        }

    def forward(self, return_latent=False, return_attention_weights=False, fill_mode: str=None, **inputs):
        """
        模型的底层前向传播函数
        """

        enc_outputs_dict = self.nets["pre_encoder"].forward(**inputs)

        batch_size = enc_outputs_dict['obs']['agentview_image'].shape[0]

        image_feat = enc_outputs_dict['obs']['agentview_image'] # 没有归一化
        depth_feat = enc_outputs_dict['obs']['agentview_depth'] # 没有归一化
        traj_feat = enc_outputs_dict['obs']['robot0_eef_pos_step_traj_current'] # 没有归一化

        # # 利用深度图特征连续性，计算attention bias，引导注意力机制关注image图像特征区域
        # depth_pos_embed = self.encoder_cam_feat_pos_embed(depth_feat)
        # # 基于余弦相似度计算patch-level的depth特征attention bias
        # _, _, h, w = depth_feat.shape
        # num_patches = h * w
        # depth_feat_flat = depth_feat.flatten(2).permute(0, 2, 1)  # [B, N, D]
        
        cls_token_traj = self.cls_token_traj.unsqueeze(0).repeat(batch_size, 1, 1).to(traj_feat.device)  # [B, S, D]

        encoder_in_token_traj = torch.cat([cls_token_traj, traj_feat], dim=1)  # [B, S+N, D]

        traj_pos_embed = self.traj_encoder_pos_enc.unsqueeze(1)

        cls_token_image = self.cls_token_image.unsqueeze(0).repeat(batch_size, 1, 1).to(image_feat.device)  # [B, S, D]
        image_pos_embed = self.encoder_cam_feat_pos_embed(image_feat).to(dtype=image_feat.dtype)
        all_image_pos_embed = torch.cat([self.image_encoder_pos_enc, einops.rearrange(image_pos_embed, 'b c h w -> b (h w) c')[0]], dim=0)
        all_image_pos_embed = all_image_pos_embed.unsqueeze(1)
        encoder_in_token_image = torch.cat([cls_token_image, einops.rearrange(image_feat, 'b c h w -> b (h w) c')], dim=1)  # [B, S+N, D]


        cls_token_depth = self.cls_token_depth.unsqueeze(0).repeat(batch_size, 1, 1).to(depth_feat.device)  # [B, S, D]
        depth_pos_embed = self.encoder_cam_feat_pos_embed(depth_feat).to(dtype=depth_feat.dtype)
        all_depth_pos_embed = torch.cat([self.depth_encoder_pos_enc, einops.rearrange(depth_pos_embed, 'b c h w -> b (h w) c')[0]], dim=0)
        all_depth_pos_embed = all_depth_pos_embed.unsqueeze(1)
        encoder_in_token_depth = torch.cat([cls_token_depth, einops.rearrange(depth_feat, 'b c h w -> b (h w) c')], dim=1)  # [B, S+N, D]
        
        encoder_in_cls_pos_embed = self.cls_encoder_pos_enc.unsqueeze(1)

        encoder_out_image, encoder_out_depth, encoder_out_traj = self.nets['backbone_encoder'].forward(
            segments_image=encoder_in_token_image,
            pos_embed_image=all_image_pos_embed,
            segments_depth=encoder_in_token_depth,
            pos_embed_depth=all_depth_pos_embed,
            segments_traj=encoder_in_token_traj,
            pos_embed_traj=traj_pos_embed,
            pos_embed_cls=encoder_in_cls_pos_embed
        )  # [B, S_total, D]

        encoder_out_image_cls = encoder_out_image[:3, :, :]  # [B, 3, D]
        encoder_out_depth_cls = encoder_out_depth[:3, :, :]  # [B, 3, D]

        encoder_in_pos_embed_image_cls = all_image_pos_embed[:3, :, :]  # [B, 3, D]
        encoder_in_pos_embed_depth_cls = all_depth_pos_embed[:3, :, :]  # [B, 3, D]

        encoder_out_traj_feat = encoder_out_traj[3:, :, :]  # [B, N, D]

        image_encoder_context = torch.cat([
            encoder_out_image,
            encoder_out_depth_cls,
            # encoder_out_traj_feat
        ], dim=0)  # [B, S_image+S_depth+S_traj, D]

        image_encoder_pos = torch.cat([
            all_image_pos_embed,
            encoder_in_pos_embed_depth_cls,
            # traj_pos_embed
        ], dim=0)  # [B, S_image+S_depth+S_traj, D]

        depth_encoder_context = torch.cat([
            encoder_out_depth,
            encoder_out_image_cls,
            # encoder_out_traj
        ], dim=0)  # [B, S_depth+S_image+S_traj, D]

        depth_encoder_pos = torch.cat([
            all_depth_pos_embed,
            encoder_in_pos_embed_image_cls,
            # traj_pos_embed
        ], dim=0)  # [B, S_depth+S_image+S_traj, D]

        decoder_in = torch.zeros(
            (10, batch_size, 512),
            dtype=encoder_out_image.dtype,
            device=encoder_out_image.device,
        )

        back_output = self.nets['backbone_decoder'].forward(
            encoder_out_traj_feat,
            image_encoder_context,
            depth_encoder_context,
            image_encoder_pos,
            depth_encoder_pos,
            self.decoder_pos_embed.weight.unsqueeze(1),
        )  # [chunk_size, B, D]

        back_output = einops.rearrange(back_output, 's b d -> b (s d)')

        traj_output = self.nets['decoder'].forward(back_output)

        if self.save_attention:
            self.attention_maps['encoder_attn'] = self.nets['backbone_encoder'].get_attention_weights()
            self.attention_maps['decoder_attn'] = self.nets['backbone_decoder'].get_attention_weights()
        # depth_norm = F.normalize(depth_input, p=2, dim=-1)

        # attn_bias = torch.bmm(depth_norm, depth_norm.transpose(1, 2))  # [B, N, N]
        # attn_bias_scale = 0.05 * attn_bias  # scale the bias

        # attn_bias_scale_multihead = attn_bias_scale.unsqueeze(1)

        if return_latent and return_attention_weights:
            return traj_output, image_feat.detach(), back_output.detach()
        elif return_latent and not return_attention_weights:
            return traj_output, back_output.detach()
        elif not return_latent and return_attention_weights:
            return traj_output, image_feat.detach()
        else:
            return traj_output
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
        msg += textwrap.indent("\npre_encoder={}".format(self.nets["pre_encoder"]), indent)
        msg += textwrap.indent("\n\nbackbone_encoder={}".format(self.nets["backbone_encoder"]), indent)
        msg += textwrap.indent("\n\nbackbone_decoder={}".format(self.nets["backbone_decoder"]), indent)
        # msg += textwrap.indent("\n\nmlp={}".format(self.nets["mlp"]), indent)
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
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


class IDCEncoder(nn.Module):
    def __init__(
        self,
        num_blocks: int=3,
        num_cls_tokens_traj: int=3,
        num_cls_tokens_image: int=3,
        num_cls_tokens_depth: int=3,
        dim_model: int=512,
        n_heads: int=8,
        dim_feedforward: int=3200,
        dropout: float=0.1,
        feedforward_activation: str="relu",
        pre_norm: bool=True,
    ):
        super(IDCEncoder, self).__init__()
        
        self.num_blocks = num_blocks

        self.segment_wise_encoder_image = nn.ModuleList([
            IDCEncoderLayer(
                dim_model=dim_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                feedforward_activation=feedforward_activation,
                pre_norm=pre_norm,
            )
            for _ in range(self.num_blocks)
        ])
        self.segment_wise_encoder_depth = nn.ModuleList([
            IDCEncoderLayer(
                dim_model=dim_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                feedforward_activation=feedforward_activation,
                pre_norm=pre_norm,
            )
            for _ in range(self.num_blocks)
        ])
        self.segment_wise_encoder_traj = nn.ModuleList([
            IDCEncoderLayer(
                dim_model=dim_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                feedforward_activation=feedforward_activation,
                pre_norm=pre_norm,
            )
            for _ in range(self.num_blocks)
        ])

        self.cross_segment_encoder = nn.ModuleList([
            IDCEncoderLayer(
                dim_model=dim_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                feedforward_activation=feedforward_activation,
                pre_norm=pre_norm,
            )
            for _ in range(self.num_blocks)
        ])

        self.traj_cls = num_cls_tokens_traj
        self.image_cls = num_cls_tokens_image
        self.depth_cls = num_cls_tokens_depth

    def forward(
        self,
        segments_image,
        pos_embed_image,
        segments_depth,
        pos_embed_depth,
        segments_traj,
        pos_embed_traj,
        pos_embed_cls
    ):
        segments_image = einops.rearrange(segments_image, 'b s d -> s b d')  # [S, B, D]
        segments_depth = einops.rearrange(segments_depth, 'b s d -> s b d')  # [S, B, D]
        segments_traj = einops.rearrange(segments_traj, 'b s d -> s b d')  # [S, B, D]

        for i in range(self.num_blocks):
            # segment-wise encoding
            # NOTE: 暂时使用共享的segment-wise encoder
            update_segment_image = self.segment_wise_encoder_image[i](segments_image, pos_embed=pos_embed_image)
            update_segment_depth = self.segment_wise_encoder_depth[i](segments_depth, pos_embed=pos_embed_depth)
            update_segment_traj = self.segment_wise_encoder_traj[i](segments_traj, pos_embed=pos_embed_traj)

            update_cls_tokens = self.cross_segment_encoder[i](
                torch.cat([
                    update_segment_image[:self.image_cls],
                    update_segment_depth[:self.depth_cls],
                    update_segment_traj[:self.traj_cls],
                ], dim=0),
                pos_embed=pos_embed_cls
            )

            segments_image = torch.cat([update_cls_tokens[:self.image_cls], update_segment_image[self.image_cls:]], dim=0)
            segments_depth = torch.cat([update_cls_tokens[self.image_cls:self.image_cls + self.depth_cls], update_segment_depth[self.depth_cls:]], dim=0)
            segments_traj = torch.cat([update_cls_tokens[self.image_cls + self.depth_cls:], update_segment_traj[self.traj_cls:]], dim=0)
    
        # segments = torch.cat([segments_image, segments_depth, segments_traj], dim=0)  # [S_total, B, D]

        # return segments
        return segments_image, segments_depth, segments_traj
    
    def get_attention_weights(self):
        encoder_self_attn_weights = []
        for layer in self.segment_wise_encoder:
            encoder_self_attn_weights.append(layer.last_self_attn_weights)
        for layer in self.cross_segment_encoder:
            encoder_self_attn_weights.append(layer.last_self_attn_weights)
        return encoder_self_attn_weights

class IDCEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int=512,
        n_heads: int=8,
        dim_feedforward: int=3200,
        dropout: float=0.1,
        feedforward_activation: str="relu",
        pre_norm: bool=True,
    ):
        super(IDCEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(feedforward_activation)
        self.pre_norm = pre_norm

        self.save_attention = False
        self.last_self_attn_weights = None

    def forward(self, x, pos_embed: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        if self.save_attention:
            x, attn_weights = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=True)
            self.last_self_attn_weights = attn_weights.detach()
        else:
            x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]
        # x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class IDCDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int=512,
        n_heads: int=8,
        dim_feedforward: int=3200,
        dropout: float=0.1,
        feedforward_activation: str="relu",
        pre_norm: bool=True,
    ):
        super(IDCDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = get_activation_fn(feedforward_activation)
        self.pre_norm = pre_norm

        self.save_attention = False
        self.last_self_attn_weights = None
        self.last_cross_attn_weights = None

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Optional[Tensor] = None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Optional[Tensor] = None,
        encoder_pos_embed: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        if self.save_attention:
            x, self_attn_weights = self.self_attn(q, k, value=x, need_weights=True)
            self.last_self_attn_weights = self_attn_weights.detach()
        else:
            x = self.self_attn(q, k, value=x)[0]
        # x = x[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        if self.save_attention:
            x, cross_attn_weights = self.multihead_attn(
                query=self.maybe_add_pos_embed(x, decoder_pos_embed),
                key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
                value=encoder_out,
                need_weights=True,
            )
            self.last_cross_attn_weights = cross_attn_weights.detach()
        else:
            x = self.multihead_attn(
                query=self.maybe_add_pos_embed(x, decoder_pos_embed),
                key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
                value=encoder_out,
            )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x

class TrajectoryDecoder(nn.Module):
    def __init__(
      self,
      num_cls_tokens_traj: int=3,
      num_cls_tokens_image: int=3,
      num_cls_tokens_depth: int=3,
      n_pre_decoder_layers: int=2,
      n_post_decoder_layers: int=2,
      n_sync_decoder_layers: int=1,
      dim_model: int=512,
      n_heads: int=8,
      dim_feedforward: int=3200,
      dropout: float=0.1,
      feedforward_activation: str="relu",
      pre_norm: bool=True,
    ):
        super(TrajectoryDecoder, self).__init__()

        self.image_pre_decoder = nn.ModuleList([
            IDCDecoderLayer(
                dim_model=dim_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                feedforward_activation=feedforward_activation,
                pre_norm=pre_norm,
            )
            for _ in range(n_pre_decoder_layers)
        ])
        self.depth_pre_decoder = nn.ModuleList([
            IDCDecoderLayer(
                dim_model=dim_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                feedforward_activation=feedforward_activation,
                pre_norm=pre_norm,
            )
            for _ in range(n_pre_decoder_layers)
        ])

        self.sync_block = nn.ModuleList([
            nn.MultiheadAttention(dim_model, n_heads, dropout=dropout, batch_first=False)
            for _ in range(n_sync_decoder_layers)
        ])
        self.sync_attn_weight = None

        self.traj_post_decoder = nn.ModuleList([
            IDCDecoderLayer(
                dim_model=dim_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                feedforward_activation=feedforward_activation,
                pre_norm=pre_norm,
            )
            for _ in range(n_post_decoder_layers)
        ])

        # 输出归一化
        self.norm = nn.LayerNorm(dim_model)

        self.num_cls_tokens_traj = num_cls_tokens_traj
        self.num_cls_tokens_image = num_cls_tokens_image
        self.num_cls_tokens_depth = num_cls_tokens_depth
    
    def forward(
        self,
        decoder_input,
        image_encoder_context,
        depth_encoder_context,
        image_encoder_pos,
        depth_encoder_pos,
        decoder_pos_embed,
    ):
        # 基于depth特征，通过余弦相似度计算获取反映depth连续性的attention bias
        # depth_norm = F.normalize(depth_input, p=2, dim=-1)

        # attn_bias = torch.bmm(depth_norm, depth_norm.transpose(1, 2))  # [B, N, N]
        # attn_bias_scale = 0.05 * attn_bias  # scale the bias

        # attn_bias_scale_multihead = attn_bias_scale.unsqueeze(1)

        # num_head = 8
        # attn_bias_scale_multihead = attn_bias_scale.unsqueeze(1).repeat(1, num_head, 1, 1)  # [B, num_head, N, N]

        image_output = decoder_input.clone()
        depth_output = decoder_input.clone()

        for layer in self.image_pre_decoder:
            image_output = layer(
                image_output,
                image_encoder_context,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=image_encoder_pos,
            )

        for layer in self.depth_pre_decoder:
            depth_output = layer(
                depth_output,
                depth_encoder_context,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=depth_encoder_pos,
            )
        
        
        # concatenated = torch.cat([image_output, depth_output], dim=0)  # [2*chunk_size, B, D]
        # concatenated_with_pos = torch.cat([image_output + decoder_pos_embed, depth_output + decoder_pos_embed], dim=0)
        # concatenated = image_output + depth_output  # [chunk_size, B, D]

        for sync_layer in self.sync_block:
            # concatenated_with_pos = concatenated + decoder_pos_embed
            # concatenated_with_pos = torch.cat([image_output + decoder_pos_embed, depth_output + decoder_pos_embed], dim=0)
            # concatenated_with_pos = concatenated + torch.cat([decoder_pos_embed, decoder_pos_embed], dim=0)
            synchronized, sync_attn_weights = sync_layer(
                image_output + decoder_pos_embed,
                depth_output + decoder_pos_embed,
                depth_output,
                # attn_mask=attn_bias_scale_multihead,
            )
            depth_output = depth_output + synchronized
            self.sync_attn_weight = sync_attn_weights.detach()

        for layer in self.traj_post_decoder:
            traj_output = layer(
                depth_output,
                depth_output,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=decoder_pos_embed
            )

        traj_output = self.norm(traj_output)

        return traj_output

    def get_attention_weights(self):
        decoder_attn_weights = []
        for layer in self.image_pre_decoder:
            decoder_attn_weights.append(layer.last_self_attn_weights)
            decoder_attn_weights.append(layer.last_cross_attn_weights)
        for layer in self.depth_pre_decoder:
            decoder_attn_weights.append(layer.last_self_attn_weights)
            decoder_attn_weights.append(layer.last_cross_attn_weights)
        
        for layer in self.sync_block:
            decoder_attn_weights.append(layer.last_attn_weights)

        for layer in self.traj_post_decoder:
            decoder_attn_weights.append(layer.last_self_attn_weights)
            decoder_attn_weights.append(layer.last_cross_attn_weights)
        return decoder_attn_weights


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
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
