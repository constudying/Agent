"""
计算HDF5数据集中各模态观测数据的均值和标准差统计信息

用法:
    python compute_obs_stats.py --dataset <path_to_hdf5> --output <output_json_path>
    
示例:
    python compute_obs_stats.py \
        --dataset agent/datasets/playdata/image_demo_local_depth_step.hdf5 \
        --output agent/datasets/playdata/obs_normalization_stats.json
"""

import h5py
import numpy as np
import json
import argparse
from tqdm import tqdm


def compute_obs_normalization_stats(hdf5_path, obs_keys=None, exclude_image_keys=True):
    """
    计算HDF5数据集中各模态数据的均值和标准差
    
    参数:
        hdf5_path (str): HDF5文件路径
        obs_keys (list): 要计算统计信息的观测键列表。如果为None，则使用数据集中的所有观测键
        exclude_image_keys (bool): 是否排除图像模态。如果设为False，将计算图像的统计信息
                                   （注意：应该在图像已经normalize到[0,1]之后计算统计，
                                   即对process_obs处理后的图像计算，而不是原始uint8图像）。
                                   默认为True。
    返回:
        dict: 包含每个观测键的均值和标准差的字典
              格式: {obs_key: {"mean": array, "std": array}}
    """
    
    print(f"正在打开数据集: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        # 获取所有demo
        demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
        print(f"找到 {len(demo_keys)} 个demonstrations")
        
        # 如果未指定obs_keys，从第一个demo中获取
        if obs_keys is None:
            first_demo = f[f'data/{demo_keys[0]}/obs']
            obs_keys = list(first_demo.keys())
            print(f"观测键: {obs_keys}")
        
        # 过滤掉图像模态（可选）
        if exclude_image_keys:
            # 通常包含 'image' 或 'depth' 关键字的是图像数据
            original_keys = obs_keys.copy()
            obs_keys = [k for k in obs_keys if 'image' not in k.lower() and 'depth' not in k.lower()]
            excluded = set(original_keys) - set(obs_keys)
            if excluded:
                print(f"排除图像模态: {excluded}")
        
        print(f"将计算以下模态的统计信息: {obs_keys}")
        
        # 辅助函数：计算单条轨迹的统计信息
        def _compute_traj_stats(traj_obs_dict):
            """
            计算单条轨迹观测数据的统计信息
            
            对于图像数据（3维或4维），按通道计算全局统计量
            对于低维数据（1维或2维），按特征维度计算
            
            返回:
                dict: {obs_key: {"n": count, "mean": mean_array, "sqdiff": sqdiff_array}}
            """
            traj_stats = {}
            for k in traj_obs_dict:
                data = traj_obs_dict[k]
                
                # 判断是否为图像数据（shape: [T, H, W, C] 或 [T, H, W]）
                is_image = data.ndim >= 3
                
                if is_image:
                    # 图像数据：沿 T×H×W 计算，保留通道维度
                    # 例如 [T, H, W, C] -> 沿 (0,1,2) 计算 -> mean/std shape: [C]
                    axes = tuple(range(data.ndim - 1))  # (0, 1, 2) for 4D, (0, 1) for 3D
                    n_samples = np.prod([data.shape[i] for i in axes])  # T*H*W
                    
                    mean = data.mean(axis=axes)  # shape: [C] or scalar
                    sqdiff = ((data - mean) ** 2).sum(axis=axes)  # shape: [C] or scalar
                    
                    traj_stats[k] = {
                        "n": n_samples,
                        "mean": mean.reshape(-1) if mean.ndim == 0 else mean,  # 确保至少是1D
                        "sqdiff": sqdiff.reshape(-1) if sqdiff.ndim == 0 else sqdiff
                    }
                else:
                    # 低维数据：沿时间轴（axis=0）计算
                    traj_stats[k] = {
                        "n": data.shape[0],
                        "mean": data.mean(axis=0, keepdims=True),  # shape: [1, ...]
                        "sqdiff": ((data - data.mean(axis=0, keepdims=True)) ** 2).sum(axis=0, keepdims=True)  # shape: [1, ...]
                    }
            return traj_stats
        
        # 辅助函数：聚合两组统计信息
        def _aggregate_traj_stats(stats_a, stats_b):
            """
            聚合两组轨迹统计信息
            使用Welford's online algorithm的并行版本
            参考: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            
            返回:
                dict: 聚合后的统计信息
            """
            merged_stats = {}
            for k in stats_a:
                n_a, avg_a, M2_a = stats_a[k]["n"], stats_a[k]["mean"], stats_a[k]["sqdiff"]
                n_b, avg_b, M2_b = stats_b[k]["n"], stats_b[k]["mean"], stats_b[k]["sqdiff"]
                
                n = n_a + n_b
                mean = (n_a * avg_a + n_b * avg_b) / n
                delta = (avg_b - avg_a)
                M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
                
                merged_stats[k] = {
                    "n": n,
                    "mean": mean,
                    "sqdiff": M2
                }
            return merged_stats

        # 辅助函数：处理观测数据（对图像进行归一化）
        def _process_obs_data(obs_array, obs_key):
            """
            处理观测数据，对图像/深度数据进行归一化到[0,1]
            
            参数:
                obs_array: 原始观测数据数组
                obs_key: 观测键名称
            
            返回:
                处理后的float32数组
            """
            data = obs_array.astype('float32')
            
            # 检查是否是图像模态（包含'image'关键字且通常是uint8类型）
            is_image = 'image' in obs_key.lower() and obs_array.dtype == np.uint8
            # 检查是否是深度图（包含'depth'关键字）
            is_depth = 'depth' in obs_key.lower()
            
            if is_image:
                # RGB图像: 从[0,255]归一化到[0,1]
                data = data / 255.0
            elif is_depth:
                # 深度图通常已经是float，范围在[0,1]或其他，这里假设已归一化
                # 如果你的深度图不在[0,1]范围，需要根据实际情况调整
                pass
            
            return data

        # 处理第一个demo作为初始统计信息
        first_demo_path = f'data/{demo_keys[0]}/obs'
        obs_traj = {k: _process_obs_data(f[f'{first_demo_path}/{k}'][()], k) for k in obs_keys}
        merged_stats = _compute_traj_stats(obs_traj)
        
        # 遍历其余的demos并聚合统计信息
        print("\n正在计算统计信息...")
        for demo_key in tqdm(demo_keys[1:], desc="处理demonstrations"):
            demo_path = f'data/{demo_key}/obs'
            obs_traj = {k: _process_obs_data(f[f'{demo_path}/{k}'][()], k) for k in obs_keys}
            traj_stats = _compute_traj_stats(obs_traj)
            merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)
        
        # 计算最终的均值和标准差
        obs_normalization_stats = {}
        for k in merged_stats:
            # 添加小的容差值1e-3防止除零
            mean = merged_stats[k]["mean"].astype(np.float32)
            std = (np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3).astype(np.float32)
            
            obs_normalization_stats[k] = {
                "mean": mean,
                "std": std
            }
            
            # 打印统计信息
            print(f"\n{k}:")
            print(f"  Shape: {mean.shape}")
            print(f"  样本数: {merged_stats[k]['n']}")
            if mean.size <= 10:  # 对于低维数据，打印具体值
                print(f"  Mean: {mean}")
                print(f"  Std:  {std}")
            else:  # 对于高维数据（如图像），打印范围
                print(f"  Mean范围: [{mean.min():.6f}, {mean.max():.6f}]")
                print(f"  Std范围:  [{std.min():.6f}, {std.max():.6f}]")
    
    return obs_normalization_stats


def save_stats_to_json(stats, output_path):
    """
    将统计信息保存为JSON文件
    
    参数:
        stats (dict): 统计信息字典
        output_path (str): 输出JSON文件路径
    """
    # 将numpy数组转换为列表以便JSON序列化
    json_stats = {}
    for k, v in stats.items():
        json_stats[k] = {
            "mean": v["mean"].tolist(),
            "std": v["std"].tolist()
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print(f"\n统计信息已保存到: {output_path}")


def save_stats_to_npz(stats, output_path):
    """
    将统计信息保存为NPZ文件（NumPy格式）
    
    参数:
        stats (dict): 统计信息字典
        output_path (str): 输出NPZ文件路径
    """
    # 展平字典结构
    save_dict = {}
    for k, v in stats.items():
        save_dict[f"{k}_mean"] = v["mean"]
        save_dict[f"{k}_std"] = v["std"]
    
    np.savez(output_path, **save_dict)
    print(f"\n统计信息已保存到: {output_path}")


def load_stats_from_json(json_path):
    """
    从JSON文件加载统计信息
    
    参数:
        json_path (str): JSON文件路径
    
    返回:
        dict: 统计信息字典，numpy数组格式
    """
    with open(json_path, 'r') as f:
        json_stats = json.load(f)
    
    stats = {}
    for k, v in json_stats.items():
        stats[k] = {
            "mean": np.array(v["mean"], dtype=np.float32),
            "std": np.array(v["std"], dtype=np.float32)
        }
    
    return stats


def load_stats_from_npz(npz_path):
    """
    从NPZ文件加载统计信息
    
    参数:
        npz_path (str): NPZ文件路径
    
    返回:
        dict: 统计信息字典
    """
    data = np.load(npz_path)
    
    stats = {}
    # 从展平的结构重建字典
    keys = set([k.replace('_mean', '').replace('_std', '') for k in data.keys()])
    for k in keys:
        stats[k] = {
            "mean": data[f"{k}_mean"],
            "std": data[f"{k}_std"]
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='计算HDF5数据集的观测数据归一化统计信息'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='HDF5数据集路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径（.json或.npz）。如果未指定，将在数据集同目录下生成'
    )
    parser.add_argument(
        '--include-images',
        action='store_true',
        help='是否包含图像模态的统计信息（默认排除）'
    )
    parser.add_argument(
        '--obs-keys',
        type=str,
        nargs='+',
        default=None,
        help='要计算统计信息的观测键列表（默认为所有非图像模态）'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'npz', 'both'],
        default='json',
        help='输出格式：json, npz, 或 both（默认: json）'
    )
    
    args = parser.parse_args()
    
    # 计算统计信息
    stats = compute_obs_normalization_stats(
        hdf5_path=args.dataset,
        obs_keys=args.obs_keys,
        exclude_image_keys=not args.include_images
    )
    
    # 确定输出路径
    if args.output is None:
        import os
        base_path = os.path.splitext(args.dataset)[0]
        json_output = f"{base_path}_obs_stats.json"
        npz_output = f"{base_path}_obs_stats.npz"
    else:
        import os
        base_path = os.path.splitext(args.output)[0]
        json_output = f"{base_path}.json"
        npz_output = f"{base_path}.npz"
    
    # 保存统计信息
    if args.format in ['json', 'both']:
        save_stats_to_json(stats, json_output)
    
    if args.format in ['npz', 'both']:
        save_stats_to_npz(stats, npz_output)
    
    print("\n✓ 完成!")


if __name__ == "__main__":
    main()
