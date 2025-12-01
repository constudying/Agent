

import h5py
import numpy as np
import argparse

# Define constants
POINT_GAP = 15
FUTURE_POINTS_COUNT = 10
PAST_POINTS_COUNT = 10

def get_future_points(arr):
    future_traj = []

    for i in range(POINT_GAP, (FUTURE_POINTS_COUNT + 1) * POINT_GAP, POINT_GAP):
      # Identify the indices for the current and prior points
      index_current = min(len(arr) - 1, i)

      current_point = arr[index_current]
      future_traj.append(current_point)

    return future_traj

def get_past_points(arr):
    past_traj = []

    for i in range(POINT_GAP, (PAST_POINTS_COUNT + 1) * POINT_GAP, POINT_GAP):
      # Identify the indices for the current and prior points
      index_prior = max(0, len(arr) - 1 - i)

      prior_point = arr[index_prior]
      past_traj.append(prior_point)

    return past_traj[::-1]

def get_past_trajectory_delta(past_traj):
    # Calculate the deltas between consecutive points
    deltas = np.diff(past_traj, axis=0)
    return deltas

def get_step_trajectory(pos, vel):
    """
    实现滑动窗口切片操作，每10个点作为一个时间步长
    """
    # step_traj_past = []
    step_traj_current = []
    step_traj_future = []
    total_points = len(pos)
    time_gap = 10  # 每个时间步长包含的点数
    point_gap = 1  # 点之间的间隔

    for step in range(total_points):
        # begin_index = step - time_gap * point_gap
        begin_index = step
        end_index = begin_index + 2 * time_gap * point_gap
        if begin_index >= 0 and end_index <= total_points:
            # step_traj_past.append(arr[begin_index : mid_index : point_gap])
                  pos_current = pos[begin_index : begin_index + time_gap * point_gap : point_gap]
                  vel_current = vel[begin_index : begin_index + time_gap * point_gap : point_gap]
                  pos_future = pos[begin_index + time_gap * point_gap : end_index : point_gap]
                  vel_future = vel[begin_index + time_gap * point_gap : end_index : point_gap]
                  
                  step_traj_current.append(pos_current)
                  step_traj_future.append(pos_future)  # elif begin_index < 0 and end_index <= total_points:
        #     step_temp = arr[0 : mid_index]
        #     # 长度没有达到 time_gap，则复制开头的点进行补齐，直到达到 time_gap
        #     if len(step_temp) <  time_gap:
        #       first_point = step_temp[0] if len(step_temp) > 0 else arr[0]
        #       padding = np.tile(first_point, (time_gap - len(step_temp), 1))
        #       step_temp = np.vstack([padding, step_temp])
        #     step_traj_past.append(step_temp[:time_gap])
        #     step_traj_current.append(arr[mid_index : mid_index + time_gap : point_gap])
        #     step_traj_future.append(arr[mid_index + time_gap : end_index : point_gap])
        elif begin_index >= 0 and end_index > total_points:
            pos_step_temp = pos[begin_index : total_points : point_gap]
            vel_step_temp = vel[begin_index : total_points : point_gap]
            # 长度没有达到 time_gap，则复制末尾的点进行补齐，直到达到 time_gap
            if len(pos_step_temp) < 2 * time_gap:
              last_point = pos_step_temp[-1] if len(pos_step_temp) > 0 else pos[-1]
              padding = np.tile(last_point, (2 * time_gap - len(pos_step_temp), 1))
              pos_step_temp = np.vstack([pos_step_temp, padding])
            if len(vel_step_temp) < 2 * time_gap:
              last_point = vel_step_temp[-1] if len(vel_step_temp) > 0 else vel[-1]
              padding = np.tile(last_point, (2 * time_gap - len(vel_step_temp), 1))
              vel_step_temp = np.vstack([vel_step_temp, padding])
            # step_traj_past.append(arr[begin_index : mid_index : point_gap])
            # step_temp = np.concatenate([pos_step_temp, vel_step_temp], axis=-1)
            
            step_traj_current.append(pos_step_temp[0 : time_gap])
            step_traj_future.append(pos_step_temp[time_gap :])
        else:
            raise ValueError("Unexpected case in trajectory extraction.")

    # step_traj_past = np.array(step_traj_past)
    step_traj_current = np.array(step_traj_current)
    step_traj_future = np.array(step_traj_future)

    return step_traj_current.reshape(step_traj_current.shape[0], -1), step_traj_future.reshape(step_traj_future.shape[0], -1)

def process_dataset(dataset_file):
    # Open the HDF5 file in read+ mode (allows reading and writing)
    with h5py.File(dataset_file, 'r+') as f:
        demo_keys = [key for key in f['data'].keys() if 'demo_' in key]
        DEMO_COUNT = len(demo_keys)

        for i in range(0, DEMO_COUNT):
            # Extract the robot0_eef_pos data
            eef_pos = f[f'data/demo_{i}/obs/robot0_eef_pos'][...]
            eef_vel = f[f'data/demo_{i}/obs/robot0_eef_vel_lin'][...]
            # [...]和[()]的区别：
            # [...]：是python中的省略号对象，返回一个 NumPy 数组，包含数据集对应键的所有数据，一般是多维数组（会将该数据集的所有数据读取到内存中，转换为 NumPy 数组）
            # [()]：对于标量返回python原生类型，对于数组返回 NumPy 数组，一般读取单个值或小数组（其中无多个维度）

            # Calculate the future trajectory for each data point
            future_traj_data = np.array([get_future_points(eef_pos[j:]) for j in range(len(eef_pos))])
            past_traj_data = np.array([get_past_points(eef_pos[:j+1]) for j in range(len(eef_pos))])
            past_traj_data_delta = np.array([get_past_trajectory_delta(past_traj_data[j]) for j in range(len(past_traj_data))])

            # Reshape from [seq, gap, 3] to [seq, gap*3] by flattening the last two dimensions
            future_traj_data = future_traj_data.reshape(future_traj_data.shape[0], -1)
            past_traj_data = past_traj_data.reshape(past_traj_data.shape[0], -1)
            past_traj_data_delta = past_traj_data_delta.reshape(past_traj_data_delta.shape[0], -1)

            step_traj_current, step_traj_future = get_step_trajectory(eef_pos, eef_vel)

            # Create or overwrite datasets
            datasets = {
              # f'data/demo_{i}/obs/robot0_eef_pos_future_traj': future_traj_data,
              # f'data/demo_{i}/obs/robot0_eef_pos_past_traj': past_traj_data,
              # f'data/demo_{i}/obs/robot0_eef_pos_step_traj_past': step_traj_past,
              f'data/demo_{i}/obs/robot0_eef_pos_step_traj_current': step_traj_current,
              f'data/demo_{i}/obs/robot0_eef_pos_step_traj_future': step_traj_future,
            }

            # datasets_del = {
            #   f'data/demo_{i}/obs/robot0_eef_pos_step_traj': None,
            #   f'data/demo_{i}/obs/robot0_eef_pos_past_traj_delta': None,
            #   f'data/demo_{i}/obs/robot0_eef_pos_past_traj': None,
            #   f'data/demo_{i}/obs/robot0_eef_pos_future_traj': None,
            #   f'data/demo_{i}/obs/robot0_eye_in_hand_image': None,
            #   f'data/demo_{i}/obs/robot0_eye_in_hand_depth': None,
            # }

            # for path in datasets_del.keys():
            #   if path in f:
            #     del f[path]
            
            for path, data in datasets.items():
              if path in f:
                del f[path]
              f.create_dataset(path, data=data)

    print(f"Processed {DEMO_COUNT} demos!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process hdf5 dataset to generate future and past trajectory data.')
    parser.add_argument('--dataset', required=True, help='Path to the hdf5 dataset file.')

    args = parser.parse_args()
    process_dataset(args.dataset)