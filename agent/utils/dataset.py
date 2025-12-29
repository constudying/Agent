
from __future__ import annotations

import os, glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import random
import numpy as np
import h5py
import torch
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils

from robomimic.utils.dataset import SequenceDataset

from torch.utils.data import IterableDataset, DataLoader, get_worker_info, Dataset

@dataclass
class ValidDatasetConfig:
    
    # 路径参数，可以是目录或文件
    path: str

    # 滑窗参数
    history_len: int = 30  # H
    pred_len: int = 30  # T
    past_downsample: int = 10
    future_downsample: int = 10
    stride: int = 1  # 滑窗步长, 默认1表示每帧采样一个窗口
    max_windows_per_demo: Optional[int] = None  # 每个demo的最大窗口数，None表示不限制

    keys: Dict[str, str] = field(
        default_factory=lambda: {
            "rgb": "/data/demo/obs/agentview_image",
            "depth": "/data/demo/obs/agentview_depth",
            "action": "/data/demo/actions",
            "joint_pos": "/data/demo/obs/robot0_joint_pos",
            "eef_pos": "/data/demo/obs/robot0_eef_pos",
        }
    )

    use_rgb: bool = True
    use_depth: bool = True

    rgb_channel_first: bool = True  # RGB图像是否为通道优先格式
    depth_channel_first: bool = True  # 深度图像是否为通道优先格式

    rgb_to_float: bool = True  # 是否将RGB图像转换为浮点数
    rgb_div_255: bool = True  # 是否将RGB图像除以255进行归一化
    depth_to_float: bool = True  # 是否将深度图像转换为浮点数


class H5ValidWindowDataset(Dataset):
    """
    固定枚举验证窗口（可复现），返回 t 用于对齐回原始时间轴做可视化。
    支持 valid 目录包含一个或多个 demo .hdf5 文件。

    每个样本返回：
      - past   : [H, Dtraj]
      - future : [T, Dtraj]
      - t      : int（窗口起点在原序列中的索引）
      - rgb/depth（可选）
    """

    def __init__(self, cfg: ValidDatasetConfig):
        super().__init__()
        self.cfg = cfg

        # 收集文件
        if os.path.isdir(cfg.path):
            self.files = sorted(glob.glob(os.path.join(cfg.path, "*.hdf5")))
        else:
            self.files = [cfg.path]

        if not self.files:
            raise FileNotFoundError(f"No .hdf5 files found under: {cfg.path}")

        # 每个进程/worker独立打开文件句柄：用 pid 绑定，fork 后自动重开
        self._pid: Optional[int] = None
        self._handles: Dict[str, h5py.File] = {}

        # 构造固定索引表 (file_path, t)
        self.index: List[Tuple[str, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        H, T, ds_past, ds_future, stride = (
            self.cfg.history_len,
            self.cfg.pred_len,
            self.cfg.past_downsample,
            self.cfg.future_downsample,
            self.cfg.stride
        )
        traj_key = self.cfg.keys["eef_pos"]

        for fp in self.files:
            with h5py.File(fp, "r") as f:
                if traj_key not in f:
                    raise KeyError(f"Key '{traj_key}' not found in {fp}")
                L = int(f[traj_key].shape[0])

            t_min = (H - 1) * ds_past
            t_max = L - 1 - (T * ds_future)
            if t_max <= t_min:
                continue

            ts = list(range(t_min, t_max + 1, stride))

            # 可选：每条demo最多取 max_windows_per_demo 个点（均匀下采样）
            if self.cfg.max_windows_per_demo is not None and len(ts) > self.cfg.max_windows_per_demo:
                idx = np.linspace(0, len(ts) - 1, self.cfg.max_windows_per_demo).round().astype(int)
                ts = [ts[i] for i in idx]

            for t in ts:
                self.index.append((fp, int(t)))

        if len(self.index) == 0:
            raise RuntimeError("No valid windows constructed. Check (H, T, ds) and trajectory lengths.")

    def __len__(self) -> int:
        return len(self.index)

    def _get_handle(self, fp: str) -> h5py.File:
        pid = os.getpid()
        # fork 后 pid 改变，需要关闭旧句柄并重开
        if self._pid is None or self._pid != pid:
            self._close_all()
            self._pid = pid

        h = self._handles.get(fp)
        if h is None:
            h = h5py.File(fp, "r")
            self._handles[fp] = h
        return h

    def _close_all(self) -> None:
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self._close_all()

    # @staticmethod
    # def _ensure_rgb_tensor(x: np.ndarray, cfg: ValidDatasetConfig) -> torch.Tensor:
    #     # 支持 [H,W,3] 或 [3,H,W]
    #     if x.ndim == 3 and x.shape[-1] == 3 and cfg.rgb_channel_first:
    #         x = np.transpose(x, (2, 0, 1))  # -> [3,H,W]
    #     elif x.ndim == 3 and x.shape[0] == 3 and (not cfg.rgb_channel_first):
    #         x = np.transpose(x, (1, 2, 0))  # -> [H,W,3]

    #     t = torch.from_numpy(np.ascontiguousarray(x))
    #     if cfg.rgb_to_float:
    #         t = t.float()
    #         if cfg.rgb_div_255:
    #             t = t / 255.0
    #     return t

    # @staticmethod
    # def _ensure_depth_tensor(x: np.ndarray, cfg: ValidDatasetConfig) -> torch.Tensor:
    #     # 支持 [H,W]、[1,H,W]、[H,W,1]
    #     if x.ndim == 2:
    #         if cfg.depth_channel_first:
    #             x = x[None, :, :]  # -> [1,H,W]
    #         else:
    #             x = x[:, :, None]  # -> [H,W,1]
    #     elif x.ndim == 3 and x.shape[-1] == 1 and cfg.depth_channel_first:
    #         x = np.transpose(x, (2, 0, 1))  # -> [1,H,W]

    #     t = torch.from_numpy(np.ascontiguousarray(x))
    #     if cfg.depth_to_float:
    #         t = t.float()
    #     return t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fp, t = self.index[idx]
        f = self._get_handle(fp)

        H, T, ds_past, ds_future = (
            self.cfg.history_len,
            self.cfg.pred_len,
            self.cfg.past_downsample,
            self.cfg.future_downsample
        )
        keys = self.cfg.keys

        eef_pos = f[keys["eef_pos"]]  # [L, Dtraj]
        # vel_ang = f[keys["vel_ang"]]
        # vel_lin = f[keys["vel_lin"]]
        action = f[keys["action"]]
        joint_pos = f[keys["joint_pos"]]
        past_idx = np.arange(t - (H - 1) * ds_past, t + 1, ds_past)
        fut_idx = np.arange(t + ds_future, t + (T + 1) * ds_future, ds_future)

        past = np.array(eef_pos[past_idx])   # [H, Dtraj]
        future = np.array(eef_pos[fut_idx])  # [T, Dtraj]
        action_past = np.array(action[past_idx])  # [H, Daction]
        action_future = np.array(action[fut_idx])  # [T, Daction]
        joint_pos_past = np.array(joint_pos[past_idx])  # [H, Djoint]
        # vel_ang_past = np.array(vel_ang[past_idx])
        # vel_lin_past = np.array(vel_lin[past_idx])

        out: Dict[str, Any] = {
            "obs": {
                "robot0_eef_pos_past_traj": torch.from_numpy(past).float(),
                "robot0_eef_pos_future_traj": torch.from_numpy(future).float(),
                "robot0_action_past_traj": torch.from_numpy(np.array(action_past)).float(),
                "robot0_action_future_traj": torch.from_numpy(np.array(action_future)).float(),
                "robot0_joint_pos_past_traj": torch.from_numpy(np.array(joint_pos_past)).float(),
                # "robot0_eef_vel_ang_past_traj": torch.from_numpy(vel_ang_past).float(),
                # "robot0_eef_vel_lin_past_traj": torch.from_numpy(vel_lin_past).float(),
                # "t": int(t),
                # "file": fp,
            }
        }

        if self.cfg.use_rgb:
            rgb = f[keys["rgb"]]
            rgb_t = np.array(rgb[t])
            out["obs"]["agentview_image"] = torch.from_numpy(rgb_t).float()  # self._ensure_rgb_tensor(rgb_t, self.cfg)

        if self.cfg.use_depth and ("depth" in keys) and (keys["depth"] in f):
            depth = f[keys["depth"]]
            depth_t = np.array(depth[t])
            out["obs"]["agentview_depth"] = torch.from_numpy(depth_t).float()  # self._ensure_depth_tensor(depth_t, self.cfg)

        return out


class H5WindowSamgpler(IterableDataset):
    def __init__(self, folder, history_len=30, pred_len=30, 
                 past_downsample=10, future_downsample=10,
                 samples_per_epoch=100000, min_gap=None, keys=None):
        """
        Args:
            folder: HDF5 文件所在目录
            history_len: 历史窗口长度（帧数）
            pred_len: 预测窗口长度（帧数）
            past_downsample: 历史轨迹的降采样率（每隔 past_downsample 帧采样一次）
            future_downsample: 未来轨迹的降采样率（每隔 future_downsample 帧采样一次）
            samples_per_epoch: 每个 epoch 的采样数量
            min_gap: 同一文件内相邻采样点的最小时间间隔（单位：帧）
            keys: 数据键映射字典
        """
        self.files = sorted(glob.glob(os.path.join(folder, '*.hdf5')))
        assert len(self.files) > 0, f"No hdf5 files found in {folder}"
        self.H = history_len
        self.T = pred_len
        self.ds_past = past_downsample  # 历史轨迹降采样率
        self.ds_future = future_downsample  # 未来轨迹降采样率
        self.samples_per_epoch = samples_per_epoch
        self.min_gap = min_gap  # optional: 用于限制同一条轨迹内相邻采样点间隔
        self.keys = keys or {
            "rgb": "/data/demo/obs/agentview_image",
            "depth": "/data/demo/obs/agentview_depth",
            "action": "/data/demo/actions",
            "joint_pos": "/data/demo/obs/robot0_joint_pos",
            "eef_pos": "/data/demo/obs/robot0_eef_pos",
        }

        # 读取obs的轨迹长度信息
        self.lengths = []
        for fp in self.files:
            with h5py.File(fp, 'r') as f:
                L = f[self.keys['rgb']].shape[0]
            self.lengths.append(L)
        
        # 用长度加权，避免过多采样短轨迹
        self.p = np.array(self.lengths, dtype=np.float64)
        self.p = self.p / self.p.sum()

        # 每个worker的文件句柄缓冲，避免反复打开关闭
        self._handles = {}
    
    def _get_handle(self, fp):
        h = self._handles.get(fp, None)
        if h is None:
            h = h5py.File(fp, 'r')  # 每个worker进程各自打开，不要跨进程共享
            self._handles[fp] = h
        return h
    
    def __iter__(self):
        wi = get_worker_info()
        # 让不同worker有不同随机种子
        seed = (os.getpid() * 9973 + (wi.id if wi else 0) * 10007) & 0xFFFFFFFF
        rng = random.Random(seed)

        # 每个worker分担samples_per_epoch
        n = self.samples_per_epoch
        if wi:
            n = int(np.ceil(n / wi.num_workers))

        last_samples = {}  # 存储多个时间步，实现更强的间隔约束，记录每个文件上次采样的时间步，避免过近采样，用于min_gap

        for _ in range(n):
          max_file_retries = len(self.files) * 2

          for file_retry in range(max_file_retries):
              # 按权重随机选择文件
              file_idx = int(np.searchsorted(np.cumsum(self.p), rng.random(), side='right'))
              file_idx = min(file_idx, len(self.files) - 1)  # 防止越界
              fp = self.files[file_idx]
              L = self.lengths[file_idx]

              # 合法 t 范围：需要 past H 步 + future T 步
              # 修改：分别考虑过去和未来的降采样跨度
              # past 需要: t - (H-1) * ds_past >= 0
              # future 需要: t + T * ds_future < L
              t_min = (self.H - 1) * self.ds_past
              t_max = L - 1 - (self.T * self.ds_future)
              if t_max <= t_min:
                  continue  # 轨迹太短，跳过
              
              max_t_retries = 50
              found = False
              
              # 随机采样 t，附加 min_gap 限制
              for _try in range(max_t_retries):
                  t = rng.randint(t_min, t_max)

                  if self.min_gap is None:
                      found = True
                      break

                  prev_samples = last_samples.get(fp, [])
                  if not prev_samples or all(abs(t - pt) >= self.min_gap for pt in prev_samples):
                      found = True
                      # 更新记录（保留最近10个采样点，防止内存占用过多）
                      if fp not in last_samples:
                          last_samples[fp] = []
                      last_samples[fp].append(t)
                      if len(last_samples[fp]) > 10:  # 只保留最近10个采样点，防止内存占用过多
                          last_samples[fp].pop(0)
                      break 
              if found:
                  # 成功找到合法的 t，开始提取数据
                  f = self._get_handle(fp)
                  action = f[self.keys['action']]
                  joint_pos = f[self.keys['joint_pos']]
                  eef_pos = f[self.keys['eef_pos']]
                  # traj = f[self.keys['traj']]
                  # vel_ang = f[self.keys['vel_ang']]
                  # vel_lin = f[self.keys['vel_lin']]
                  rgb = f[self.keys['rgb']]
                  depth = f[self.keys['depth']] if "depth" in self.keys and self.keys['depth'] in f else None

                  # 按 ds 采样窗口
                  past_idx = np.arange(t - (self.H - 1) * self.ds_past, t + 1, self.ds_past)
                  fut_idx = np.arange(t + self.ds_future, t + self.T * self.ds_future + 1, self.ds_future)

                  # 数据验证（可选，调试时开启）
                  assert len(past_idx) == self.H, f"Past length mismatch: expected {self.H}, got {len(past_idx)}"
                  assert len(fut_idx) == self.T, f"Future length mismatch: expected {self.T}, got {len(fut_idx)}"

                  img = rgb[t]
                  action_past = action[past_idx]
                  action_future = action[fut_idx]
                  joint_pos_past = joint_pos[past_idx]
                  past = eef_pos[past_idx]
                  future = eef_pos[fut_idx]
                  # vel_ang_past = vel_ang[past_idx]
                  # vel_lin_past = vel_lin[past_idx]

                  sample = {
                      "obs": {
                          "robot0_eef_pos_past_traj": torch.from_numpy(np.array(past)).float(),
                          "robot0_eef_pos_future_traj": torch.from_numpy(np.array(future)).float(),
                          "robot0_action_past_traj": torch.from_numpy(np.array(action_past)).float(),
                          "robot0_action_future_traj": torch.from_numpy(np.array(action_future)).float(),
                          "robot0_joint_pos_past_traj": torch.from_numpy(np.array(joint_pos_past)).float(),
                          # "robot0_eef_vel_ang_past_traj": torch.from_numpy(np.array(vel_ang_past)).float(),
                          # "robot0_eef_vel_lin_past_traj": torch.from_numpy(np.array(vel_lin_past)).float(),
                          "agentview_image": torch.from_numpy(np.array(img)),
                      }

                  }
                  if depth is not None:
                      sample["obs"]["agentview_depth"] = torch.from_numpy(np.array(depth[t]))

                  yield sample
                  break # 成功采样，跳出文件重试循环
          else:
              # 所有文件都尝试过，无法采样，发出警告
              import warnings
              warnings.warn(f"Failed to sample from any file after {max_file_retries} attempts."
                            f"Consider relaxing min_gap={self.min_gap} or using more/longer trajectories.")

    def __del__(self):
        """清理文件句柄"""
        for h in self._handles.values():
            try:
                h.close()
            except:
                pass




class PlaydataSequenceDataset(SequenceDataset):
    def __init__(
            self,
            hdf5_path,
            obs_keys,
            dataset_keys,
            goal_obs_gap,
            frame_stack=1,
            seq_length=1,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode=None,
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=None,
            load_next_obs=True,
    ):
        """
        Dataset class for fetching sequences of experience.
        Length of the fetched sequence is equal to (@frame_stack - 1 + @seq_length)

        Args:
            hdf5_path (str): path to hdf5

            obs_keys (tuple, list): keys to observation items (image, object, etc) to be fetched from the dataset

            dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset

            frame_stack (int): numbers of stacked frames to fetch. Defaults to 1 (single frame).

            seq_length (int): length of sequences to sample. Defaults to 1 (single frame).

            pad_frame_stack (int): whether to pad sequence for frame stacking at the beginning of a demo. This
                ensures that partial frame stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise, the
                first frame stacked observation would be (s_0, s_1, s_2, s_3).

            pad_seq_length (int): whether to pad sequence for sequence fetching at the end of a demo. This
                ensures that partial sequences at the end of a demonstration are observed, such as
                (s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence provided would be
                (s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).

            get_pad_mask (bool): if True, also provide padding masks as part of the batch. This can be
                useful for masking loss functions on padded parts of the data.

            goal_mode (str): either "last" or None. Defaults to None, which is to not fetch goals

            hdf5_cache_mode (str): one of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5
                in memory - this is by far the fastest for data loading. Set to "low_dim" to cache all
                non-image data. Set to None to use no caching - in this case, every batch sample is
                retrieved via file i/o. You should almost never set this to None, even for large
                image datasets.

            hdf5_use_swmr (bool): whether to use swmr feature when opening the hdf5 file. This ensures
                that multiple Dataset instances can all access the same hdf5 file without problems.

            hdf5_normalize_obs (bool): if True, normalize observations by computing the mean observation
                and std of each observation (in each dimension and modality), and normalizing to unit
                mean and variance in each dimension.

            filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
                demonstrations to load

            load_next_obs (bool): whether to load next_obs from the dataset
        """

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_normalize_obs = hdf5_normalize_obs
        self._hdf5_file = None

        self.goal_obs_gap = goal_obs_gap

        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode

        self.load_next_obs = load_next_obs
        self.filter_by_attribute = filter_by_attribute

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["nstep"]

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.load_demo_info(filter_by_attribute=self.filter_by_attribute)

        # maybe prepare for observation normalization
        self.obs_normalization_stats = None
        if self.hdf5_normalize_obs:
            self.obs_normalization_stats = self.normalize_obs()

        # maybe store dataset in memory for fast access
        if self.hdf5_cache_mode in ["all", "low_dim"]:
            obs_keys_in_memory = self.obs_keys
            if self.hdf5_cache_mode == "low_dim":
                # only store low-dim observations
                obs_keys_in_memory = []
                for k in self.obs_keys:
                    if ObsUtils.key_is_obs_modality(k, "low_dim"):
                        obs_keys_in_memory.append(k)
            self.obs_keys_in_memory = obs_keys_in_memory

            self.hdf5_cache = self.load_dataset_in_memory(
                demo_list=self.demos,
                hdf5_file=self.hdf5_file,
                obs_keys=self.obs_keys_in_memory,
                dataset_keys=self.dataset_keys,
                load_next_obs=self.load_next_obs
            )

            if self.hdf5_cache_mode == "all":
                # cache getitem calls for even more speedup. We don't do this for
                # "low-dim" since image observations require calls to getitem anyways.
                print("SequenceDataset: caching get_item calls...")
                self.getitem_cache = [self.get_item(i) for i in LogUtils.custom_tqdm(range(len(self)))]

                # don't need the previous cache anymore
                del self.hdf5_cache
                self.hdf5_cache = None
        else:
            self.hdf5_cache = None

        self.close_and_delete_hdf5_handle()


    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack - 1, # note: need to decrement self.n_frame_stack by one
            seq_length=self.seq_length
        )

        # determine goal index
        goal_index = None
        if self.goal_mode == "nstep":
            goal_index = min(index_in_demo + random.randint(self.goal_obs_gap[0], self.goal_obs_gap[1]) , demo_length) - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs"
        )

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs"
            )

        if goal_index is not None:
            meta["goal_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="obs",
            )

        return meta