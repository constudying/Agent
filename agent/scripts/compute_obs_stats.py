"""
计算HDF5数据集中各模态观测数据的均值和标准差统计信息

用法:
    # 单个文件模式
    python compute_obs_stats.py --dataset <path_to_hdf5> --output <output_json_path>
    
    # 多文件模式（目录）
    python compute_obs_stats.py --dataset <path_to_dataset_dir> --output <output_json_path>
    
示例:
    # 单个文件
    python compute_obs_stats.py \
        --dataset agent/datasets/playdata/image_demo_local_depth_step.hdf5 \
        --output agent/datasets/playdata/obs_normalization_stats.json
    
    # 多文件目录（自动扫描train和valid子目录）
    python compute_obs_stats.py \
        --dataset agent/datasets/playdata/image_demo_local_depth_step_image_heatmap_per_episode \
        --output agent/datasets/playdata/obs_normalization_stats.json
"""

import h5py
import numpy as np
import json
import argparse
import os
import glob
import gc
from tqdm import tqdm


def get_hdf5_files(dataset_path):
    """
    获取数据集文件列表
    
    参数:
        dataset_path (str): 数据集文件或目录路径
    
    返回:
        list: HDF5文件路径列表
    """
    if os.path.isfile(dataset_path):
        # 单个文件模式
        return [dataset_path]
    elif os.path.isdir(dataset_path):
        # 目录模式：扫描train和valid子目录
        hdf5_files = []
        
        # 扫描train目录
        train_dir = os.path.join(dataset_path, 'train')
        if os.path.exists(train_dir):
            train_files = sorted(glob.glob(os.path.join(train_dir, '*.hdf5')))
            hdf5_files.extend(train_files)
            print(f"在train目录找到 {len(train_files)} 个文件")
        
        # 扫描valid目录
        valid_dir = os.path.join(dataset_path, 'valid')
        if os.path.exists(valid_dir):
            valid_files = sorted(glob.glob(os.path.join(valid_dir, '*.hdf5')))
            hdf5_files.extend(valid_files)
            print(f"在valid目录找到 {len(valid_files)} 个文件")
        
        # 也扫描根目录下的hdf5文件，根目录下只有masked数据
        # root_files = sorted(glob.glob(os.path.join(dataset_path, '*.hdf5')))
        # if root_files:
        #     hdf5_files.extend(root_files)
        #     print(f"在根目录找到 {len(root_files)} 个文件")
        
        if not hdf5_files:
            raise ValueError(f"在 {dataset_path} 及其子目录中未找到任何HDF5文件")
        
        return hdf5_files
    else:
        raise ValueError(f"路径不存在: {dataset_path}")


def compute_obs_normalization_stats(hdf5_path_or_dir, obs_keys=None, exclude_image_keys=True, verbose=False, chunk_size=10):
    """
    计算HDF5数据集中各模态数据的均值和标准差
    支持单个文件或目录（包含多个子文件）
    
    参数:
        hdf5_path_or_dir (str): HDF5文件路径或包含多个HDF5文件的目录路径
        obs_keys (list): 要计算统计信息的观测键列表。如果为None，则使用数据集中的所有观测键
        exclude_image_keys (bool): 是否排除图像模态。如果设为False，将计算图像的统计信息
                                   （注意：应该在图像已经normalize到[0,1]之后计算统计，
                                   即对process_obs处理后的图像计算，而不是原始uint8图像）。
                                   默认为True。
        verbose (bool): 是否显示详细的处理信息，包括内存使用情况。默认为False。
        chunk_size (int): 对于大型图像数据，每次处理的帧数。默认为10帧。
    返回:
        dict: 包含每个观测键的均值和标准差的字典
              格式: {obs_key: {"mean": array, "std": array}}
    """
    
    # 获取所有HDF5文件
    hdf5_files = get_hdf5_files(hdf5_path_or_dir)
    print(f"\n总共找到 {len(hdf5_files)} 个HDF5文件")
    
    # 从第一个文件获取obs_keys
    if obs_keys is None:
        with h5py.File(hdf5_files[0], 'r') as f:
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo')]
            if demo_keys:
                first_demo = f[f'data/{demo_keys[0]}/obs']
                obs_keys = list(first_demo.keys())
                print(f"观测键: {obs_keys}")
            else:
                raise ValueError(f"在 {hdf5_files[0]} 中未找到demo数据")
    
    # 过滤掉图像模态（可选）
    if exclude_image_keys:
        original_keys = obs_keys.copy()
        obs_keys = [k for k in obs_keys if 'image' not in k.lower() and 'depth' not in k.lower()]
        excluded = set(original_keys) - set(obs_keys)
        if excluded:
            print(f"排除图像模态: {excluded}")
    
    print(f"将计算以下模态的统计信息: {obs_keys}")
    
    # 辅助函数：分块计算图像统计（用于超大图像数据）
    def _compute_image_stats_chunked(dataset, obs_key, chunk_size=10):
        """
        对超大图像数据集使用分块方式计算统计信息
        
        参数:
            dataset: HDF5数据集对象
            obs_key: 观测键名称
            chunk_size: 每次读取的帧数
        
        返回:
            dict: {"n": count, "mean": mean_array, "sqdiff": sqdiff_array}
        """
        shape = dataset.shape
        dtype = dataset.dtype
        
        if verbose:
            print(f"    {obs_key}: shape={shape}, dtype={dtype}, 采用分块处理")
        
        if len(shape) == 4:  # [T, H, W, C]
            T, H, W, C = shape
            n_samples = T * H * W
            
            # 第一遍：计算均值
            mean = np.zeros(C, dtype=np.float64)
            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                chunk = dataset[start:end]
                # 处理数据（归一化）
                if 'image' in obs_key.lower() and dtype == np.uint8:
                    chunk = chunk.astype(np.float32) / 255.0
                else:
                    chunk = chunk.astype(np.float32)
                mean += chunk.sum(axis=(0, 1, 2), dtype=np.float64)
                del chunk
                gc.collect()
            mean = mean / n_samples
            
            # 第二遍：计算方差
            sqdiff = np.zeros(C, dtype=np.float64)
            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                chunk = dataset[start:end]
                # 处理数据（归一化）
                if 'image' in obs_key.lower() and dtype == np.uint8:
                    chunk = chunk.astype(np.float64) / 255.0
                else:
                    chunk = chunk.astype(np.float64)
                sqdiff += ((chunk - mean) ** 2).sum(axis=(0, 1, 2))
                del chunk
                gc.collect()
            
            return {
                "n": n_samples,
                "mean": mean,
                "sqdiff": sqdiff
            }
        elif len(shape) == 3:  # [T, H, W]
            T, H, W = shape
            n_samples = T * H * W
            
            # 第一遍：计算均值
            mean = 0.0
            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                chunk = dataset[start:end]
                if 'depth' in obs_key.lower():
                    chunk = chunk.astype(np.float32)
                else:
                    chunk = chunk.astype(np.float32)
                mean += chunk.sum(dtype=np.float64)
                del chunk
                gc.collect()
            mean = mean / n_samples
            
            # 第二遍：计算方差
            sqdiff = 0.0
            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                chunk = dataset[start:end].astype(np.float64)
                sqdiff += ((chunk - mean) ** 2).sum()
                del chunk
                gc.collect()
            
            return {
                "n": n_samples,
                "mean": np.array([mean]),
                "sqdiff": np.array([sqdiff])
            }
        else:
            raise ValueError(f"Unexpected shape for image data: {shape}")
    
    # 辅助函数：计算单条轨迹的统计信息（内存优化版本 - 支持超大图像数据）
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
                if data.ndim == 4:  # [T, H, W, C]
                    T, H, W, C = data.shape
                    n_samples = T * H * W
                    
                    # 使用在线算法逐帧计算，避免一次性计算占用过多内存
                    # 第一步：计算全局均值
                    mean = np.zeros(C, dtype=np.float64)
                    for t in range(T):
                        mean += data[t].sum(axis=(0, 1), dtype=np.float64)
                    mean = mean / n_samples
                    
                    # 第二步：逐帧计算方差
                    sqdiff = np.zeros(C, dtype=np.float64)
                    for t in range(T):
                        # 分块处理每一帧以进一步节省内存
                        frame = data[t].astype(np.float64)
                        sqdiff += ((frame - mean) ** 2).sum(axis=(0, 1))
                        del frame  # 立即释放
                    
                    traj_stats[k] = {
                        "n": n_samples,
                        "mean": mean,
                        "sqdiff": sqdiff
                    }
                elif data.ndim == 3:  # [T, H, W]
                    T, H, W = data.shape
                    n_samples = T * H * W
                    
                    # 逐帧计算均值
                    mean = 0.0
                    for t in range(T):
                        mean += data[t].sum(dtype=np.float64)
                    mean = mean / n_samples
                    
                    # 逐帧计算方差
                    sqdiff = 0.0
                    for t in range(T):
                        frame = data[t].astype(np.float64)
                        sqdiff += ((frame - mean) ** 2).sum()
                        del frame
                    
                    traj_stats[k] = {
                        "n": n_samples,
                        "mean": np.array([mean]),
                        "sqdiff": np.array([sqdiff])
                    }
            else:
                # 低维数据：沿时间轴（axis=0）计算
                mean_val = data.mean(axis=0, keepdims=True, dtype=np.float64)
                sqdiff_val = ((data - mean_val) ** 2).sum(axis=0, keepdims=True, dtype=np.float64)
                
                traj_stats[k] = {
                    "n": data.shape[0],
                    "mean": mean_val,  # shape: [1, ...]
                    "sqdiff": sqdiff_val  # shape: [1, ...]
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
            
            # 确保使用numpy数组进行运算
            avg_a = np.asarray(avg_a, dtype=np.float64)
            avg_b = np.asarray(avg_b, dtype=np.float64)
            M2_a = np.asarray(M2_a, dtype=np.float64)
            M2_b = np.asarray(M2_b, dtype=np.float64)
            
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
    
    # 初始化merged_stats为None
    merged_stats = None
    total_demos = 0
    
    # 遍历所有HDF5文件（流式处理，一次处理一个文件）
    print("\n正在计算统计信息...")
    for file_idx, hdf5_file in enumerate(tqdm(hdf5_files, desc="处理文件")):
        if verbose:
            print(f"\n处理文件 {file_idx + 1}/{len(hdf5_files)}: {os.path.basename(hdf5_file)}")
        
        with h5py.File(hdf5_file, 'r') as f:
            # 获取当前文件中的所有demo
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo')]
            
            # 遍历当前文件中的每个demo（流式处理每个demo）
            for demo_key in demo_keys:
                demo_path = f'data/{demo_key}/obs'
                
                # 检查数据大小，决定使用哪种处理方式
                # 对于超大图像数据（>1GB），使用分块处理
                use_chunked = {}
                for k in obs_keys:
                    dataset = f[f'{demo_path}/{k}']
                    data_size_gb = dataset.size * dataset.dtype.itemsize / (1024**3)
                    # 如果数据大于1GB或者是4D图像数据，使用分块处理
                    is_large_image = (data_size_gb > 1.0 or dataset.ndim >= 3) and ('image' in k.lower() or 'depth' in k.lower())
                    use_chunked[k] = is_large_image
                    if verbose and is_large_image:
                        print(f"  {k}: {data_size_gb:.2f}GB, 将使用分块处理")
                
                # 分别处理：分块处理的数据 vs 一次性加载的数据
                traj_stats = {}
                
                # 1. 处理需要分块的大型图像数据
                for k in obs_keys:
                    if use_chunked[k]:
                        dataset = f[f'{demo_path}/{k}']
                        traj_stats[k] = _compute_image_stats_chunked(dataset, k, chunk_size)
                
                # 2. 处理可以一次性加载的小型数据
                obs_traj = {}
                for k in obs_keys:
                    if not use_chunked[k]:
                        # 读取数据
                        obs_data = f[f'{demo_path}/{k}'][()]
                        # 处理数据
                        obs_traj[k] = _process_obs_data(obs_data, k)
                        # 删除原始数据引用，帮助垃圾回收
                        del obs_data
                
                # 计算小型数据的统计信息
                if obs_traj:
                    small_data_stats = _compute_traj_stats(obs_traj)
                    traj_stats.update(small_data_stats)
                    del obs_traj, small_data_stats
                
                # 强制垃圾回收
                gc.collect()
                
                # 第一个demo初始化统计信息
                if merged_stats is None:
                    merged_stats = traj_stats
                else:
                    # 聚合统计信息
                    merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)
                    # 删除当前轨迹统计，释放内存
                    del traj_stats
                
                total_demos += 1
                
                if verbose and total_demos % 10 == 0:
                    print(f"  已处理 {total_demos} 条轨迹")
        
        # 每处理完一个文件，强制进行垃圾回收
        gc.collect()
        
        if verbose:
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                print(f"  当前内存使用: {mem_info.rss / 1024 / 1024:.1f} MB")
            except ImportError:
                pass
    
    print(f"\n总共处理了 {total_demos} 条轨迹")
    
    # 计算最终的均值和标准差
    obs_normalization_stats = {}
    for k in merged_stats:
        # 确保所有值都是numpy数组
        mean_val = merged_stats[k]["mean"]
        sqdiff_val = merged_stats[k]["sqdiff"]
        n_val = merged_stats[k]["n"]
        
        # 转换为numpy数组（如果不是的话）
        if not isinstance(mean_val, np.ndarray):
            mean_val = np.array([mean_val], dtype=np.float64)
        if not isinstance(sqdiff_val, np.ndarray):
            sqdiff_val = np.array([sqdiff_val], dtype=np.float64)
        
        # 确保是numpy数组类型后再进行计算
        mean_val = np.asarray(mean_val, dtype=np.float64)
        sqdiff_val = np.asarray(sqdiff_val, dtype=np.float64)
        
        # 计算标准差（添加小的容差值1e-3防止除零）
        variance = sqdiff_val / n_val
        std_val = np.sqrt(variance) + 1e-3
        
        # 转换为float32
        mean = mean_val.astype(np.float32)
        std = std_val.astype(np.float32)
        
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
        description='计算HDF5数据集的观测数据归一化统计信息（支持单文件或多文件目录）'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='HDF5数据集文件路径或包含train/valid子目录的数据集目录路径'
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
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细的处理信息，包括内存使用情况'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10,
        help='处理大型图像数据时每次读取的帧数（默认: 10）。减小此值可以降低内存使用'
    )
    
    args = parser.parse_args()
    
    # 计算统计信息
    stats = compute_obs_normalization_stats(
        hdf5_path_or_dir=args.dataset,
        obs_keys=args.obs_keys,
        exclude_image_keys=not args.include_images,
        verbose=args.verbose,
        chunk_size=args.chunk_size
    )
    
    # 确定输出路径
    if args.output is None:
        if os.path.isfile(args.dataset):
            # 单个文件模式
            base_path = os.path.splitext(args.dataset)[0]
        else:
            # 目录模式
            base_path = os.path.join(args.dataset, 'obs_stats')
        json_output = f"{base_path}.json"
        npz_output = f"{base_path}.npz"
    else:
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
