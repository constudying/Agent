
import numpy as np
import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn

from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
# import agent.models.down_utils as DownUtils

from robomimic.models.base_nets import Module, MLP
from agent.models.obs_core import AgentVisualCore
from robomimic.models.obs_core import Randomizer
from agent.models.transformer import PositionEncoder, TransformerBackbone
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
        feats = []
        for k in self.obs_shapes:
            x = obs_dict[k]
            # maybe process encoder input with randomizer
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward_in(x)
            # maybe process with obs net
            if self.obs_nets[k] is not None:
                # 这里会调用不同的编码器网络，编码不同模态
                # agentview_image输入形状： 32*1*3*84*84
                # if len(x.shape) == 5:
                #     bs, seq, c, h, w = x.shape
                #     x = x.view(bs * seq, c, h, w)
                # 暂时不在此处处理时序维度，交给各个编码器网络自己处理，避免重复代码，保持高层代码接口统一
                x = self.obs_nets[k](x)
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
        outputs = []
        # Deterministic order since self.observation_group_shapes is OrderedDict
        for obs_group in self.observation_group_shapes:
            # pass through encoder
            outputs.append(
                self.nets[obs_group].forward(inputs[obs_group])
            )

        return torch.cat(outputs, dim=-1)

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
        














