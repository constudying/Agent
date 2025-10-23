
from collections import OrderedDict

import torch

import robomimic.utils.tensor_utils as TensorUtils
from agent.models.obs_nets import RESNET_MIMO_Transformer, RESNET_MIMO_MLP


class MlpActorNetwork(RESNET_MIMO_MLP):
    """
    基于MLP结构，根据输入的观察模态预测动作的基础策略网络
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes
        self.ac_dim = ac_dim

        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()
        super(MlpActorNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            **encoder_kwargs,
        )

    def _get_output_shapes(self):
        return OrderedDict(action=(self.ac_dim,))
    
    def output_shape(self, input_shape):
        return [self.ac_dim]

    def forward_train(self, obs_dict, goal_dict=None, return_latent=False):
        return super(MlpActorNetwork, self).forward(obs=obs_dict, goal=goal_dict, return_latent=return_latent)
    
    def forward(self, obs_dict, goal_dict=None):
        actions = super(MlpActorNetwork, self).forward(obs=obs_dict, goal=goal_dict)
        return actions
    
    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim)


class GMMMlpActorNetwork(MlpActorNetwork):
    pass


class TransformerActorNetwork(RESNET_MIMO_Transformer):
    """
    基于transformer结构，根据输入的观察模态预测动作的基础策略网络
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_attn_dropout=None,
        transformer_output_dropout=None,
        transformer_sinusoidal_embedding=None,
        transformer_activation=None,
        transformer_causal=None,
        transformer_nn_parameter_for_timesteps=None,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:

            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.
            
            ac_dim (int): dimension of action space.

            transformer_embed_dim (int): dimension for embeddings used by transformer

            transformer_num_layers (int): number of transformer blocks to stack

            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is 
                computed over this many partitions of the embedding dimension separately.
            
            transformer_context_length (int): expected length of input sequences

            transformer_embedding_dropout (float): dropout probability for embedding inputs in transformer

            transformer_attn_dropout (float): dropout probability for attention outputs for each transformer block

            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block
            
            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.
            
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
        self.ac_dim = ac_dim

        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes

        self.transformer_nn_parameter_for_timesteps = transformer_nn_parameter_for_timesteps

        # set up different observation groups for @RNN_MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()
        super(TransformerActorNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            transformer_embed_dim=transformer_embed_dim,
            transformer_context_length=transformer_context_length,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_attn_dropout=transformer_attn_dropout,
            transformer_output_dropout=transformer_output_dropout,
            transformer_sinusoidal_embedding=transformer_sinusoidal_embedding,
            transformer_activation=transformer_activation,
            transformer_nn_parameter_for_timesteps=transformer_nn_parameter_for_timesteps,
            transformer_causal=transformer_causal,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @MIMO_Transformer, since we won't
        always directly predict actions, but may instead predict the parameters
        of a action distribution.
        """
        output_shapes = OrderedDict(action=(self.ac_dim,))
        return output_shapes

    def output_shape(self, input_shape):
        # note: @input_shape should be dictionary (key: mod)
        # infers temporal dimension from input shape
        mod = list(self.obs_shapes.keys())[0]
        T = input_shape[mod][0]
        TensorUtils.assert_size_at_dim(input_shape, size=T, dim=0, 
                msg="TransformerActorNetwork: input_shape inconsistent in temporal dimension")
        return [T, self.ac_dim]

    def forward_train(self, obs_dict, goal_dict=None):

        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(goal_dict, size=obs_dict[mod].shape[1], dim=1)

        forward_kwargs = dict(obs=obs_dict, goal=goal_dict)
        outputs = super(TransformerActorNetwork, self).forward(**forward_kwargs)

        return outputs

    def forward(self, obs_dict, actions=None, goal_dict=None):
        """
        Forward a sequence of inputs through the Transformer.
        Args:
            obs_dict (dict): batch of observations - each tensor in the dictionary
                should have leading dimensions batch and time [B, T, ...]
            actions (torch.Tensor): batch of actions of shape [B, T, D]
            goal_dict (dict): if not None, batch of goal observations
        Returns:
            outputs (torch.Tensor or dict): contains predicted action sequence, or dictionary
                with predicted action sequence and predicted observation sequences
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(goal_dict, size=obs_dict[mod].shape[1], dim=1)

        forward_kwargs = dict(obs=obs_dict, goal=goal_dict)
        outputs = super(TransformerActorNetwork, self).forward(**forward_kwargs)

        return outputs["action"]
    
    def forward_step(self, obs):
        forward_kwargs = dict(obs=obs)
        outputs = super(TransformerActorNetwork, self).forward(**forward_kwargs)
        return outputs

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim) 


class ImageActorNetwork(TransformerActorNetwork):
    """
    基于transformer结构，根据输入的图像观察预测动作的策略网络
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        transformer_embed_dim,
        transformer_context_length,
        transformer_num_layers=6,
        transformer_num_heads=8,
        transformer_attn_dropout=0.1,
        transformer_output_dropout=0.1,
        transformer_sinusoidal_embedding=False,
        transformer_activation="relu",
        transformer_causal=True,
        transformer_nn_parameter_for_timesteps=False,
        use_tanh=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        
        
        self.use_tanh = use_tanh

        super(ImageActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            transformer_embed_dim=transformer_embed_dim,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_context_length=transformer_context_length,
            transformer_attn_dropout=transformer_attn_dropout,
            transformer_output_dropout=transformer_output_dropout,
            transformer_sinusoidal_embedding=transformer_sinusoidal_embedding,
            transformer_activation=transformer_activation,
            transformer_causal=transformer_causal,
            transformer_nn_parameter_for_timesteps=transformer_nn_parameter_for_timesteps,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )


    def _get_output_shapes(self):
        raise NotImplementedError
    
    def forward_train(self, obs_dict, goal_dict=None, return_latent=False):
        if return_latent:
            dec_outputs, enc_outputs = RESNET_MIMO_Transformer.forward(self, return_latent=return_latent, obs=obs_dict, goal=goal_dict)
            return dict(action=dec_outputs["action"], latent=enc_outputs["latent"])
        else:
            dec_outputs = RESNET_MIMO_Transformer.forward(self, return_latent=return_latent, obs=obs_dict, goal=goal_dict)
            return dict(action=dec_outputs["action"])

    def forward(self, obs_dict, actions=None, goal_dict=None):
        
        forward_kwargs = dict(obs=obs_dict, goal=goal_dict)
        image_ouputs = self.nets["image_processor"](forward_kwargs)
        outputs = super(TransformerActorNetwork, self).forward(**forward_kwargs)

        return outputs["action"]
    
    def output_shape(self, input_shape):
        raise NotImplementedError

    def _to_string(self):
        raise NotImplementedError
 


        
