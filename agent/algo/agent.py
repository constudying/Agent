from collections import OrderedDict

import copy
import h5py
import torch
import torch.nn as nn

import agent.models.policy_nets as PolicyNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from agent.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("agent")
def algo_config_to_class(algo_config):
    """
    向算法工厂函数中返回函数映射，建立算法的类实例
    """

    if algo_config.highlevel.enabled:
        if algo_config.lowlevel.enabled:
            return Image_Action_agent, {}
        else:
            return Image_Single_agent, {}
    else:
        if algo_config.lowlevel.enabled:
            return Action_Single_pretrain, {} # TODO：实际类未编写
        else:
            return None # TODO：实际类未编写

class Image_Single_agent(PolicyAlgo):
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
        # del self.obs_shapes['robot_camera_image_bias_traj']
        self.ac_dim = self.algo_config.highlevel.ac_dim

        self.nets = nn.ModuleDict()
        # TODO: 根据算法配置，传入参数创建实际网络
        self.nets["policy"] = PolicyNets.MlpActorNetwork(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        
        self.save_count = 0

        self.nets = self.nets.float().to(self.device)

    def find_nearest_index(self, ee_pos, current_id):
        distances = torch.norm(self.goal_ee_traj[current_id : (current_id + self.eval_goal_img_window)] - ee_pos, dim=1)
        nearest_index = distances.argmin().item()
        if nearest_index == 0:
            self.zero_count += 1
        if self.zero_count > self.eval_max_goal_img_iter:
            nearest_index += 1
            self.zero_count = 0

        return min(nearest_index + current_id, self.goal_image_length - 1)

    def load_eval_video_prompt(self, video_path):
        self.goal_image = h5py.File(video_path, 'r')['data']['demo_1']['obs']['agentview_image'][:]
        self.goal_ee_traj = h5py.File(video_path, 'r')['data']['demo_1']['obs']['robot0_eef_pos'][:]
        self.goal_image = torch.from_numpy(self.goal_image).cuda().float()
        self.goal_ee_traj = torch.from_numpy(self.goal_ee_traj).cuda().float()
        self.goal_image = self.goal_image.permute(0, 3, 1, 2)
        self.goal_image = self.goal_image / 255.
        self.goal_image_length = len(self.goal_image)

    def postprocess_batch_for_training(self, batch, obs_normalization_stats):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """

        # ensure obs_normalization_stats are torch Tensors on proper device
        obs_normalization_stats = TensorUtils.to_float(
            TensorUtils.to_device(TensorUtils.to_tensor(obs_normalization_stats), self.device))

        # we will search the nested batch dictionary for the following special batch dict keys
        # and apply the processing function to their values (which correspond to observations)
        obs_keys = ["obs", "next_obs", "goal_obs"]

        def recurse_helper(d):
            """
            Apply process_obs_dict to values in nested dictionary d that match a key in obs_keys.
            """
            for k in d:
                if k in obs_keys:
                    # found key - stop search and process observation
                    if d[k] is not None:
                        d[k] = ObsUtils.process_obs_dict(d[k])
                        if obs_normalization_stats is not None:
                            d[k] = ObsUtils.normalize_obs(d[k], obs_normalization_stats=obs_normalization_stats)
                elif isinstance(d[k], dict):
                    # search down into dictionary
                    recurse_helper(d[k])

        recurse_helper(batch)

        batch["goal_obs"]["agentview_image"] = batch["goal_obs"]["agentview_image"][:, 0]

        return TensorUtils.to_device(TensorUtils.to_float(batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(Image_Single_agent, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

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
        a_target = a_target.squeeze(1) # 消去预测动作的多余维度，后续可以匹配动作分块操作

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
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)

    def log_info(self, info):
        """
        总结训练信息，用于tensorboard输出日志
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class Action_Single_pretrain(PolicyAlgo):
    """
    TODO：动作模块的预训练
    """

    pass


class Image_Action_agent(PolicyAlgo):
    """
    TODO：完整的模型训练
    """
    def _create_networks(self):
        """
        根据算法配置创建网络，并放入@self.nets中。
        """
        assert self.algo_config.highlevel.enabled
        assert self.algo_config.lowlevel.enabled

        # TODO: 删去self.obs_shaps中不需要的键
        # del self.obs_shapes['robot_camera_image_bias_traj']
        self.ac_dim = self.algo_config.highlevel.ac_dim

        self.nets = nn.ModuleDict()
        # TODO: 根据算法配置，传入参数创建实际网络
        self.nets["policy"] = PolicyNets.RESNET_MIMO_Transformer(
            input_obs_group_shapes=self.obs_shapes,
            output_shapes=self.goal_shapes,
            transformer_embed_dim=self.algo_config.transformer.embed_dim,
            transformer_num_layers=self.algo_config.transformer.num_layers,
            transformer_num_heads=self.algo_config.transformer.num_heads,
            transformer_context_length=self.algo_config.transformer.context_length,
            transformer_attn_dropout=self.algo_config.transformer.attn_dropout,
            transformer_output_dropout=self.algo_config.transformer.output_dropout,
            transformer_sinusoidal_embedding=self.algo_config.transformer.transformer_sinusoidal_embedding,
            transformer_activation=self.algo_config.transformer.activation,
            transformer_causal=self.algo_config.transformer.causal,
            transformer_nn_parameter_for_timesteps=self.algo_config.transformer.transformer_nn_parameter_for_timesteps,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        
        self.save_count = 0

        self.nets = self.nets.float().to(self.device)

    def find_nearest_index(self):
        pass
    
    def load_eval_video_prompt(self):
        pass

    def process_batch_for_training(self, batch):
        """
        TODO:占位实现，直接返回输入 batch，需要后续补充具体逻辑。
        """
        batch["goal_obs"]["agentview_image"] = batch["goal_obs"]["agentview_image"][:, 0]

        return TensorUtils.to_device(TensorUtils.to_float(batch), self.device)

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
    
    def log_info(self, info):
        """
        总结训练信息，用于tensorboard输出日志
        """
        log = super(Image_Action_agent, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def get_action(self, obs_dict, goal_dict=None):
        pass

    def reset(self):
        pass