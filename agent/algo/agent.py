from collections import OrderedDict

import copy
import h5py
import torch
import torch.nn as nn

import agent.models.policy_nets as PolicyNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from algo.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("agent")
def algo_config_to_class(algo_config):
    """
    向算法工厂函数中返回函数映射，建立算法的类实例
    """

    if algo_config.highlevel.enable:
        if algo_config.lowlevel.enable:
            return Image_Action_agent()
        else:
            return Image_Single_pretrain()
    else:
        if algo_config.lowlevel.enable:
            return Action_Single_pretrain() # TODO：实际类未编写
        else:
            return None # TODO：实际类未编写

class Image_Action_agent(PolicyAlgo):
    """
    训练模型的感知模块，使其中间量输出关注操作有关物体细节。
    TODO：未来融入全卷积层做物体提取。
    """
    def _create_networks(self):
        """
        根据算法配置创建网络，并放入@self.nets中。
        """
        assert self.algo_config.highlevel.enabled
        assert self.algo_config.lowlevel.enabled

        # TODO: 删去self.obs_shaps中不需要的键
        del self.obs_shapes['robot_camera_image_bias_traj']
        self.ac_dim = self.algo_config.highlevel.ac_dim

        self.nets = nn.ModuleDict()
        # TODO: 根据算法配置，传入参数创建实际网络
        self.nets["policy"] = PolicyNets.MlpActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=self.algo_config.encoder_kwargs,
        )
        
        self.save_count = 0

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        TODO:占位实现，直接返回输入 batch，需要后续补充具体逻辑。
        """
        batch["goal_obs"]["agentview_image"] = batch["goal_obs"]["agentview_image"][:, 0]

        return TensorUtils.to_device(TensorUtils.to_float(batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(Image_Action_agent, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)
        
        return info
    
    def _get_latent_plan(self, obs, goal):
        assert 'agentview_image' in obs.keys(), "obs中必须包含agentview_image"

        if len(obs['agentview_image'].size) == 5:
            bs, seq, c, h, w = obs['agentview_image'].size()

            for item in ['agentview_image']:
                obs[item] = obs[item].view(bs * seq, c, h, w)
                goal[item] = goal[item].view(bs * seq, c, h, w)
            
            dec_out, enc_out = self.nets["policy"].forward_train(
                obs_dict=obs,
                goal_dict=goal,
                return_latent=True
            )

        return dec_out, enc_out

    def _forward_training(self, batch):
        """
        模型训练时使用，通过字典，返回网络输出
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"]
        )

        action_out = dists["action"]

        predictions = OrderedDict(
            action_out=action_out,
        )
        return predictions
    
    def _compute_losses(self, predictions, batch):
        """
        TODO：可用损失还有待补充，如果没有，考虑和_forward_training方法合并
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["action_out"]
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)

        action_losses = [
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.l2_weight * losses["l2_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses
    
    def _train_step(self, losses):
        """
        根据损失进行反向传播和优化
        """
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info
    
    def log_info(self, info):
        """
        总结训练信息，用于tensorboard输出日志
        """
        log = super(Image_Single_pretrain, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class Action_Single_pretrain(PolicyAlgo):
    """
    TODO：动作模块的预训练
    """

    pass


class Image_Single_pretrain(PolicyAlgo):
    """
    TODO：完整的模型训练
    """
    """
    训练模型的感知模块，使其中间量输出关注操作有关物体细节。
    TODO：未来融入全卷积层做物体提取。
    """
    def _create_networks(self):
        """
        根据算法配置创建网络，并放入@self.nets中。
        """
        assert self.algo_config.highlevel.enabled
        assert not self.algo_config.lowlevel.enabled

        # TODO: 删去self.obs_shaps中不需要的键
        del self.obs_shapes['robot_camera_image_bias_traj']
        self.ac_dim = self.algo_config.highlevel.ac_dim

        self.nets = nn.ModuleDict()
        # TODO: 根据算法配置，传入参数创建实际网络
        self.nets["policy"] = PolicyNets.ImageTransformerActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            transfor_embed_dim=self.algo_config.transformer.embed_dim,
            transformer_num_layers=self.algo_config.transformer.num_layers,
            transformer_num_heads=self.algo_config.transformer.num_heads,
            transformer_context_length=self.algo_config.transformer.context_length,
            transformer_attn_dropout=self.algo_config.transformer.attn_dropout,
            transformer_output_dropout=self.algo_config.transformer.output_dropout,
            transformer_sinusoidal_embedding=self.algo_config.transformer.transformer_sinusoidal_embedding,
            transformer_activation=self.algo_config.transformer.activation,
            transformer_causal=self.algo_config.transformer.causal,
            transformer_nn_parameter_for_timesteps=self.algo_config.transformer.transformer_nn_parameter_for_timesteps,
            encoder_kwargs=self.algo_config.encoder_kwargs,
        )
        
        self.save_count = 0

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        TODO:占位实现，直接返回输入 batch，需要后续补充具体逻辑。
        """
        batch["goal_obs"]["agentview_image"] = batch["goal_obs"]["agentview_image"][:, 0]

        return TensorUtils.to_device(TensorUtils.to_float(batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(Image_Single_pretrain, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)
        
        return info
    
    def _get_latent_plan(self, obs, goal):
        assert 'agentview_image' in obs.keys(), "obs中必须包含agentview_image"

        if len(obs['agentview_image'].size) == 5:
            bs, seq, c, h, w = obs['agentview_image'].size()

            for item in ['agentview_image']:
                obs[item] = obs[item].view(bs * seq, c, h, w)
                goal[item] = goal[item].view(bs * seq, c, h, w)
            
            dec_out, enc_out = self.nets["policy"].forward_train(
                obs_dict=obs,
                goal_dict=goal,
                return_latent=True
            )

        return dec_out, enc_out

    def _forward_training(self, batch):
        """
        模型训练时使用，通过字典，返回网络输出
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"]
        )

        action_out = dists["action"]

        predictions = OrderedDict(
            action_out=action_out,
        )
        return predictions
    
    def _compute_losses(self, predictions, batch):
        """
        TODO：可用损失还有待补充，如果没有，考虑和_forward_training方法合并
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["action_out"]
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)

        action_losses = [
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.l2_weight * losses["l2_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses
    
    def _train_step(self, losses):
        """
        根据损失进行反向传播和优化
        """
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info
    
    def log_info(self, info):
        """
        总结训练信息，用于tensorboard输出日志
        """
        log = super(Image_Single_pretrain, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log