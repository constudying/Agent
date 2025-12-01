
from email import parser
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase

def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
):

    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    env.reset()
    obs = env.reset_to(initial_state)

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        states=np.array(states),
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]

    for t in range(1, traj_len + 1):
        if t == traj_len:
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            next_obs = env.reset_to({'states' : states[t]})
        
        r = env.get_reward()
        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)
    
    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    # 将列表的字典转换为字典的列表，优化hdf5存储，其更适合存储字典的列表格式，即同类型数据的数组
    # 同时每个key对应的数据类型也更统一，支持高效压缩和读取
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    # numpy数组操作速度更快，列表操作慢
    # numpy存储效率比python列表更高
    # 许多ML框架需要Numpy数组作为输入
    # numpy还能提供丰富的数值计算功能
    # gotcha在于其会修改原始字典的值
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def dataset_add_depth(args):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env_meta['env_kwargs']['bddl_file_name'] = './agent/scripts/bddl_files/KITCHEN_SCENE9_playdata.bddl'
    env_meta['env_kwargs']['camera_depths'] = [True] * len(args.camera_names)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("=======================================================")

    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)
    f = h5py.File(args.dataset, 'r')
    demos = list(f['data'].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if args.n is not None:
        demos = demos[:args.n]

    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    f_out = h5py.File(output_path, 'w')
    data_grp = f_out.create_group('data')
    print('input file: {}'.format(args.dataset))
    print('output file: {}'.format(output_path))

    total_samples = 0
    for ind in range(len(demos)):
        ep = demos[ind]

        states = f['data/{}/states'.format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state['model'] = f['data/{}'.format(ep)].attrs['model_file']
        
        actions = f['data/{}/actions'.format(ep)][()]
        traj = extract_trajectory(
            env=env, 
            initial_state=initial_state, 
            states=states, 
            actions=actions,
            done_mode=args.done_mode,
        )

        if args.copy_rewards:
            traj["rewards"] = f["data/{}/rewards".format(ep)][()]
        if args.copy_dones:
            traj["dones"] = f["data/{}/dones".format(ep)][()]

        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset('actions', data=np.array(traj['actions']))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        for k in traj["obs"]:
            if args.compress:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
            else:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            if not args.exclude_next_obs:
                if args.compress:
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

        # episode metadata
        if is_robosuite_env:
            ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
        total_samples += traj["actions"].shape[0]
        print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f_out.close()
    f.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='path to the dataset hdf5 dataset',
    )
    parser.add_argument(
        '--output_name',
        type=str,
        required=True,
        help='name of output hdf5 dataset',
    )

    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectory points are processed",
    )

    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )
    parser.add_argument(
        '--camera_names',
        type=str,
        nargs='+',
        default=[],
        help='name of cameras to add depth channel',
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )
    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to exclude next obs in dataset
    parser.add_argument(
        "--exclude-next-obs", 
        action='store_true',
        help="(optional) exclude next obs in dataset",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--compress", 
        action='store_true',
        help="(optional) compress observations with gzip option in hdf5",
    )

    args = parser.parse_args()
    dataset_add_depth(args)