"""
Config for MimicPlay algorithm.
"""

from agent.configs.base_config import BaseConfig


class AgentConfig(BaseConfig):
    ALGO_NAME = "agent"

    def train_config(self):
        """
        MimicPlay doesn't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(AgentConfig, self).train_config()
        self.train.hdf5_load_next_obs = False

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.learning_rate.scheduler_type = "multistep" # learning rate scheduler ("multistep", "linear", etc)
        self.algo.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength

        # loss weights
        self.algo.loss.l2_weight = 1.0      # L2 loss weight
        self.algo.loss.l1_weight = 0.0      # L1 loss weight
        self.algo.loss.cos_weight = 0.0     # cosine loss weight

        # actor network layer dimensions
        self.algo.actor_layer_dims = (1024, 1024)

        # stochastic GMM policy settings
        self.algo.gmm.enabled = False                   # whether to train a GMM policy
        self.algo.gmm.num_modes = 5                     # number of GMM modes
        self.algo.gmm.min_std = 0.0001                  # minimum std output from network
        self.algo.gmm.std_activation = "softplus"       # activation to use for std output from policy net
        self.algo.gmm.low_noise_eval = True             # low-std at test-time 

        # transformer settings
        self.algo.transformer.enabled = True                         # whether to use transformer for sequence modeling
        self.algo.transformer.embed_dim = None
        self.algo.transformer.num_layers = None
        self.algo.transformer.num_heads = None
        self.algo.transformer.context_length = None
        self.algo.transformer.attn_dropout = 0.1
        self.algo.transformer.output_dropout = 0.1
        self.algo.transformer.transformer_sinusoidal_embedding = False
        self.algo.transformer.activation = "relu"
        self.algo.transformer.causal = False
        self.algo.transformer.transformer_nn_parameter_for_timesteps = False

        # highlevel polict settings
        self.algo.highlevel.enabled = False                     # whether to train the highlevel planner of MimicPlay
        self.algo.highlevel.ac_dim = 30                         # 3D trajectory output dimension (3 x 10 points = 30)
        self.algo.highlevel.latent_plan_dim = 400               # latent plan embedding size
        self.algo.highlevel.do_not_lock_keys()

        # lowlevel policy settings
        self.algo.lowlevel.enabled = False                      # whether to train the lowlevel guided policy of MimicPlay (if highlevel is not enabled, an end-to-end lowlevel policy will be trained (BC-transformer baseline))
        self.algo.lowlevel.feat_dim = 656                       # feature dimansion of transformer
        self.algo.lowlevel.n_layer = 4                          # number of layers in transformer
        self.algo.lowlevel.n_head = 8                           # number of heads in transformer
        self.algo.lowlevel.block_size = 10                      # sequence block size, which should be same as train.seq_length in json config file
        self.algo.lowlevel.gmm_modes = 5                        # number of gmm modes for action output
        self.algo.lowlevel.action_dim = 7                       # robot action dimension
        self.algo.lowlevel.proprio_dim = 7                      # input robot's proprioception dimension (end-effector 3D location + end-effector orientation in quaternion)
        self.algo.lowlevel.spatial_softmax_num_kp = 64          # number of keypoints in the spatial softmax pooling layer
        self.algo.lowlevel.gmm_min_std = 0.0001                 # std of gmm output during inference
        self.algo.lowlevel.dropout = 0.1                        # training dropout rate
        self.algo.lowlevel.trained_highlevel_planner = None     # load trained highlevel latent planner (set to None when learning highlevel planner or other baselines)
        self.algo.lowlevel.eval_goal_img_window = 30            # goal image sampling window during evaluation rollouts
        self.algo.lowlevel.eval_max_goal_img_iter = 5           # maximum idling steps of sampled goal image during evaluation rollouts
        self.algo.lowlevel.do_not_lock_keys()

        # Playdata training/inference settings
        self.algo.playdata.enabled = False                       # whether to train with plan data (unlabeled, no-cut)
        self.algo.playdata.goal_image_range = [100, 200]        # goal image sampling range during training
        self.algo.playdata.eval_goal_gap = 150                  # goal image sampling gap during evaluation rollouts (mid of training goal_image_range)
        self.algo.playdata.do_not_lock_keys()



