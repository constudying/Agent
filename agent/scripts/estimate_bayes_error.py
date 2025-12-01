"""
估算Bayes误差（理论最小损失）的工具

Bayes误差是由数据本身的不确定性决定的理论下界，
无论模型多强大都无法低于这个值。

判断训练是否接近Bayes误差可以帮助诊断：
- 是否还有优化空间？
- 是否已经达到数据质量极限？
- 是否应该改进数据而不是模型？
"""

import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BayesErrorEstimator:
    """Bayes误差估算器"""
    
    def __init__(self, states, actions):
        """
        Args:
            states: 状态数组 [N, state_dim]
            actions: 动作数组 [N, action_dim]
        """
        self.states = states
        self.actions = actions
        self.n_samples = len(states)
        self.action_dim = actions.shape[1] if actions.ndim > 1 else 1
        
        # 标准化状态（用于相似度计算）
        self.states_mean = states.mean(axis=0)
        self.states_std = states.std(axis=0) + 1e-8
        self.states_normalized = (states - self.states_mean) / self.states_std
        
    def estimate_by_nearest_neighbors(self, k=10, n_queries=1000):
        """
        方法1：最近邻方法
        
        原理：对于每个状态，找到K个最相似的状态，
        计算它们动作的方差作为该状态的不确定性。
        
        Args:
            k: 最近邻数量
            n_queries: 查询的样本数（太多会很慢）
        
        Returns:
            dict: 包含各种误差估计
        """
        print(f"\n{'='*80}")
        print(f"方法1: 基于K近邻的Bayes误差估计 (K={k})")
        print(f"{'='*80}")
        
        # 随机选择查询点
        if n_queries < self.n_samples:
            query_indices = np.random.choice(self.n_samples, n_queries, replace=False)
        else:
            query_indices = np.arange(self.n_samples)
        
        # 构建KNN模型
        print(f"构建KNN模型...")
        knn = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
        knn.fit(self.states_normalized)
        
        # 查找最近邻
        print(f"查找最近邻...")
        distances, indices = knn.kneighbors(self.states_normalized[query_indices])
        
        # 计算每个查询点的动作方差
        mse_errors = []
        l1_errors = []
        local_variances = []
        
        for i, idx in enumerate(query_indices):
            # 获取最近邻的动作（排除自己）
            neighbor_actions = self.actions[indices[i, 1:]]  # 跳过第一个（自己）
            
            # 计算这些动作的均值（最优预测）
            mean_action = neighbor_actions.mean(axis=0)
            
            # 计算MSE（这就是这个状态的Bayes误差）
            mse = np.mean((neighbor_actions - mean_action) ** 2)
            l1 = np.mean(np.abs(neighbor_actions - mean_action))
            variance = neighbor_actions.var(axis=0).mean()
            
            mse_errors.append(mse)
            l1_errors.append(l1)
            local_variances.append(variance)
        
        # 统计结果
        results = {
            'method': 'K-Nearest Neighbors',
            'k': k,
            'n_queries': len(query_indices),
            'mse_bayes_error': np.mean(mse_errors),
            'l1_bayes_error': np.mean(l1_errors),
            'avg_local_variance': np.mean(local_variances),
            'mse_std': np.std(mse_errors),
            'mse_median': np.median(mse_errors),
            'mse_percentiles': {
                '25%': np.percentile(mse_errors, 25),
                '75%': np.percentile(mse_errors, 75),
                '90%': np.percentile(mse_errors, 90),
                '95%': np.percentile(mse_errors, 95),
            },
            'mse_distribution': mse_errors,
            'neighbor_distances': distances[:, 1:].mean()  # 平均邻居距离
        }
        
        self._print_results(results)
        return results
    
    def estimate_by_clustering(self, distance_threshold=0.1, min_cluster_size=5):
        """
        方法2：聚类方法
        
        原理：将相似状态聚类，计算每个聚类内部的动作方差
        
        Args:
            distance_threshold: 状态相似度阈值
            min_cluster_size: 最小聚类大小
        
        Returns:
            dict: 包含各种误差估计
        """
        print(f"\n{'='*80}")
        print(f"方法2: 基于聚类的Bayes误差估计")
        print(f"{'='*80}")
        
        # 计算距离矩阵（采样以加速）
        max_samples = min(2000, self.n_samples)
        sample_indices = np.random.choice(self.n_samples, max_samples, replace=False)
        
        print(f"计算距离矩阵 (采样{max_samples}个点)...")
        distances = squareform(pdist(self.states_normalized[sample_indices], metric='euclidean'))
        
        # 简单聚类：基于距离阈值
        print(f"聚类...")
        visited = np.zeros(max_samples, dtype=bool)
        clusters = []
        
        for i in range(max_samples):
            if visited[i]:
                continue
            
            # 找到距离小于阈值的所有点
            cluster_mask = distances[i] < distance_threshold
            cluster_indices = sample_indices[np.where(cluster_mask)[0]]
            
            if len(cluster_indices) >= min_cluster_size:
                clusters.append(cluster_indices)
                visited[cluster_mask] = True
        
        print(f"找到 {len(clusters)} 个聚类")
        
        if len(clusters) == 0:
            print("❌ 未找到足够的聚类，尝试增大distance_threshold或减小min_cluster_size")
            return None
        
        # 计算每个聚类的内部方差
        cluster_mse_errors = []
        cluster_l1_errors = []
        cluster_sizes = []
        
        for cluster_idx in clusters:
            cluster_actions = self.actions[cluster_idx]
            mean_action = cluster_actions.mean(axis=0)
            
            mse = np.mean((cluster_actions - mean_action) ** 2)
            l1 = np.mean(np.abs(cluster_actions - mean_action))
            
            cluster_mse_errors.append(mse)
            cluster_l1_errors.append(l1)
            cluster_sizes.append(len(cluster_idx))
        
        # 加权平均（按聚类大小加权）
        weights = np.array(cluster_sizes) / sum(cluster_sizes)
        
        results = {
            'method': 'Clustering',
            'n_clusters': len(clusters),
            'avg_cluster_size': np.mean(cluster_sizes),
            'mse_bayes_error': np.average(cluster_mse_errors, weights=weights),
            'l1_bayes_error': np.average(cluster_l1_errors, weights=weights),
            'mse_std': np.std(cluster_mse_errors),
            'mse_median': np.median(cluster_mse_errors),
            'cluster_sizes': cluster_sizes,
            'mse_distribution': cluster_mse_errors,
        }
        
        self._print_results(results)
        return results
    
    def estimate_by_local_variance(self, window_size=50):
        """
        方法3：局部方差方法（适用于时序数据）
        
        原理：假设数据是按轨迹组织的，在局部窗口内计算方差
        
        Args:
            window_size: 窗口大小
        
        Returns:
            dict: 包含各种误差估计
        """
        print(f"\n{'='*80}")
        print(f"方法3: 基于局部方差的Bayes误差估计")
        print(f"{'='*80}")
        
        # 计算每个动作维度的方差
        action_variances = []
        
        for i in range(0, self.n_samples - window_size, window_size // 2):
            window_actions = self.actions[i:i+window_size]
            variance = window_actions.var(axis=0).mean()
            action_variances.append(variance)
        
        results = {
            'method': 'Local Variance',
            'window_size': window_size,
            'n_windows': len(action_variances),
            'mse_bayes_error': np.mean(action_variances),
            'mse_std': np.std(action_variances),
            'mse_median': np.median(action_variances),
        }
        
        self._print_results(results)
        return results
    
    def estimate_by_noise_level(self):
        """
        方法4：基于噪声水平的简单估计
        
        原理：假设数据中的高频变化主要是噪声
        
        Returns:
            dict: 包含各种误差估计
        """
        print(f"\n{'='*80}")
        print(f"方法4: 基于噪声水平的简单估计")
        print(f"{'='*80}")
        
        # 计算相邻动作的差异（一阶差分）
        action_diffs = np.diff(self.actions, axis=0)
        noise_variance = np.mean(action_diffs ** 2) / 2  # 除以2是因为差分放大了方差
        
        results = {
            'method': 'Noise Level',
            'mse_bayes_error': noise_variance,
            'note': '这是一个粗略估计，假设相邻样本的差异主要来自噪声',
        }
        
        self._print_results(results)
        return results
    
    def _print_results(self, results):
        """打印结果"""
        print(f"\n估计结果:")
        print(f"  MSE Bayes误差: {results['mse_bayes_error']:.6f}")
        if 'l1_bayes_error' in results:
            print(f"  L1 Bayes误差:  {results['l1_bayes_error']:.6f}")
        if 'mse_median' in results:
            print(f"  中位数:        {results['mse_median']:.6f}")
        if 'mse_std' in results:
            print(f"  标准差:        {results['mse_std']:.6f}")
    
    def compare_with_training_loss(self, training_loss, loss_type='mse'):
        """
        比较训练损失与Bayes误差
        
        Args:
            training_loss: 当前训练损失
            loss_type: 'mse' 或 'l1'
        
        Returns:
            dict: 诊断信息
        """
        print(f"\n{'='*80}")
        print(f"训练损失 vs Bayes误差 比较")
        print(f"{'='*80}")
        
        # 获取最可靠的Bayes误差估计（KNN方法）
        bayes_results = self.estimate_by_nearest_neighbors(k=10, n_queries=500)
        
        if loss_type == 'mse':
            bayes_error = bayes_results['mse_bayes_error']
        else:
            bayes_error = bayes_results['l1_bayes_error']
        
        ratio = training_loss / bayes_error
        gap = training_loss - bayes_error
        gap_percentage = (gap / bayes_error) * 100
        
        print(f"\n当前训练{loss_type.upper()}损失: {training_loss:.6f}")
        print(f"估计的Bayes误差:        {bayes_error:.6f}")
        print(f"比值 (loss/bayes):      {ratio:.2f}x")
        print(f"差距:                    {gap:.6f} ({gap_percentage:.1f}%)")
        
        # 诊断
        print(f"\n{'='*80}")
        print("诊断结果:")
        print(f"{'='*80}")
        
        if ratio < 1.1:  # 在10%以内
            print("✅ 训练损失已经非常接近Bayes误差！")
            print("   分析：")
            print("   - 模型已经学得很好，接近理论极限")
            print("   - 继续训练提升空间很小")
            print("   - 建议：")
            print("     1. 检查任务性能（损失低不一定任务好）")
            print("     2. 如果任务性能不好，问题在数据质量而非模型")
            print("     3. 考虑改进数据采集或增加数据多样性")
            
        elif ratio < 1.5:  # 在50%以内
            print("⚠️ 训练损失接近Bayes误差，但还有一定空间")
            print("   分析：")
            print(f"   - 还有约{gap_percentage:.0f}%的优化空间")
            print("   - 模型基本收敛，但可能未达到最优")
            print("   - 建议：")
            print("     1. 尝试继续训练（但提升可能有限）")
            print("     2. 微调学习率（使用更小的学习率）")
            print("     3. 检查是否有正则化过强")
            print("     4. 尝试模型集成（ensemble）")
            
        elif ratio < 2.0:  # 在2倍以内
            print("⚠️ 训练损失明显高于Bayes误差")
            print("   分析：")
            print(f"   - 还有{gap_percentage:.0f}%的优化空间")
            print("   - 模型可能欠拟合或训练不充分")
            print("   - 建议：")
            print("     1. 增加训练轮次")
            print("     2. 检查学习率（可能太小）")
            print("     3. 增大模型容量")
            print("     4. 减少正则化")
            print("     5. 检查损失函数是否合适（MSE vs 似然）")
            
        else:  # 2倍以上
            print("❌ 训练损失远高于Bayes误差")
            print("   分析：")
            print(f"   - 还有{gap_percentage:.0f}%的巨大优化空间")
            print("   - 可能的问题：")
            print("     1. 模型容量严重不足")
            print("     2. 训练方法有问题（学习率、优化器等）")
            print("     3. 数据有多模态性但使用了MSE损失")
            print("     4. 训练轮次远远不够")
            print("     5. 存在bug（梯度消失、数值问题等）")
            print("   - 建议：")
            print("     1. 首先检查是否有bug")
            print("     2. 检查多模态性（如果有，改用GMM/Diffusion）")
            print("     3. 大幅增加模型容量")
            print("     4. 调整学习率和优化器")
        
        print(f"{'='*80}\n")
        
        return {
            'training_loss': training_loss,
            'bayes_error': bayes_error,
            'ratio': ratio,
            'gap': gap,
            'gap_percentage': gap_percentage,
            'status': 'optimal' if ratio < 1.1 else 'good' if ratio < 1.5 else 'suboptimal' if ratio < 2.0 else 'poor'
        }
    
    def visualize_bayes_error(self, results_list, output_path='bayes_error_analysis.png'):
        """可视化Bayes误差分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 不同方法的估计对比
        ax1 = axes[0, 0]
        methods = [r['method'] for r in results_list if r is not None]
        estimates = [r['mse_bayes_error'] for r in results_list if r is not None]
        
        bars = ax1.bar(methods, estimates, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('MSE Bayes误差估计', fontsize=11)
        ax1.set_title('不同方法的Bayes误差估计', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, est in zip(bars, estimates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{est:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Bayes误差分布（如果有）
        ax2 = axes[0, 1]
        for result in results_list:
            if result is not None and 'mse_distribution' in result:
                ax2.hist(result['mse_distribution'], bins=30, alpha=0.5, 
                        label=result['method'], density=True)
        ax2.set_xlabel('局部MSE误差', fontsize=11)
        ax2.set_ylabel('密度', fontsize=11)
        ax2.set_title('Bayes误差分布', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 动作方差分析
        ax3 = axes[1, 0]
        action_vars = self.actions.var(axis=0)
        ax3.bar(range(len(action_vars)), action_vars, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('动作维度', fontsize=11)
        ax3.set_ylabel('方差', fontsize=11)
        ax3.set_title('各动作维度的方差', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. 总结信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        avg_estimate = np.mean([r['mse_bayes_error'] for r in results_list if r is not None])
        std_estimate = np.std([r['mse_bayes_error'] for r in results_list if r is not None])
        
        summary = f"""
        Bayes误差估计总结
        {'='*50}
        
        数据统计：
        · 样本数量: {self.n_samples}
        · 状态维度: {self.states.shape[1]}
        · 动作维度: {self.action_dim}
        
        Bayes误差估计：
        · 平均值: {avg_estimate:.6f}
        · 标准差: {std_estimate:.6f}
        · 范围: [{min(estimates):.6f}, {max(estimates):.6f}]
        
        动作统计：
        · 总体方差: {self.actions.var():.6f}
        · 平均绝对值: {np.abs(self.actions).mean():.4f}
        · 范围: [{self.actions.min():.4f}, {self.actions.max():.4f}]
        
        {'='*50}
        解释：
        
        Bayes误差 = 数据固有的不确定性
        
        即使完美的模型也无法低于这个值，
        因为数据本身就包含：
        · 传感器噪声
        · 人类示教的随机性
        · 环境的随机性
        · 多模态选择
        
        如果训练损失 ≈ Bayes误差：
        ✅ 模型已经很好，达到理论极限
        ✅ 进一步提升需要改进数据质量
        """
        
        ax4.text(0.05, 0.95, summary,
                transform=ax4.transAxes,
                fontsize=9,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n可视化结果已保存到: {output_path}")
        plt.close()


def load_dataset(dataset_path, n_demos=None, state_keys=None):
    """加载HDF5数据集"""
    print(f"正在加载数据集: {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as f:
        demos = list(f['data'].keys())
        if n_demos:
            demos = demos[:n_demos]
        
        print(f"加载 {len(demos)} 个演示")
        
        all_states = []
        all_actions = []
        
        for demo in demos:
            demo_grp = f[f'data/{demo}']
            
            # 提取动作
            actions = demo_grp['actions'][:]
            all_actions.append(actions)
            
            # 提取状态
            if 'states' in demo_grp:
                states = demo_grp['states'][:]
            elif 'obs' in demo_grp:
                obs_group = demo_grp['obs']
                state_parts = []
                
                if state_keys:
                    keys_to_use = state_keys
                else:
                    keys_to_use = [k for k in obs_group.keys() 
                                  if 'image' not in k and 'rgb' not in k]
                
                for key in keys_to_use:
                    if key in obs_group:
                        obs_data = obs_group[key][:]
                        if obs_data.ndim > 2:
                            obs_data = obs_data.reshape(obs_data.shape[0], -1)
                        state_parts.append(obs_data)
                
                if state_parts:
                    states = np.concatenate(state_parts, axis=1)
                else:
                    print(f"警告: demo {demo} 没有找到合适的状态数据")
                    continue
            else:
                print(f"警告: demo {demo} 没有状态信息")
                continue
            
            all_states.append(states)
        
        all_states = np.vstack(all_states)
        all_actions = np.vstack(all_actions)
        
        print(f"\n数据集统计:")
        print(f"  总样本数: {len(all_states)}")
        print(f"  状态维度: {all_states.shape[1]}")
        print(f"  动作维度: {all_actions.shape[1]}")
        
        return all_states, all_actions


def main():
    parser = argparse.ArgumentParser(
        description="估算Bayes误差并与训练损失对比"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HDF5数据集路径"
    )
    parser.add_argument(
        "--training_loss",
        type=float,
        default=None,
        help="当前训练损失（用于对比）"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default='mse',
        choices=['mse', 'l1'],
        help="损失类型"
    )
    parser.add_argument(
        "--n_demos",
        type=int,
        default=None,
        help="使用前N个演示"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="KNN的K值"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图片路径"
    )
    
    args = parser.parse_args()
    
    # 加载数据
    states, actions = load_dataset(args.dataset, n_demos=args.n_demos)
    
    # 采样（如果数据太大）
    max_samples = 10000
    if len(states) > max_samples:
        print(f"\n数据量较大，随机采样{max_samples}个样本...")
        indices = np.random.choice(len(states), max_samples, replace=False)
        states = states[indices]
        actions = actions[indices]
    
    # 创建估算器
    estimator = BayesErrorEstimator(states, actions)
    
    # 使用多种方法估算
    results = []
    
    # 方法1: KNN
    knn_results = estimator.estimate_by_nearest_neighbors(k=args.k, n_queries=min(1000, len(states)))
    results.append(knn_results)
    
    # 方法2: 聚类
    cluster_results = estimator.estimate_by_clustering(distance_threshold=0.2, min_cluster_size=5)
    if cluster_results:
        results.append(cluster_results)
    
    # 方法3: 局部方差
    if len(states) > 100:
        local_var_results = estimator.estimate_by_local_variance(window_size=50)
        results.append(local_var_results)
    
    # 方法4: 噪声水平
    noise_results = estimator.estimate_by_noise_level()
    results.append(noise_results)
    
    # 可视化
    output_path = args.output or args.dataset.replace('.hdf5', '_bayes_error.png')
    estimator.visualize_bayes_error(results, output_path)
    
    # 如果提供了训练损失，进行对比
    if args.training_loss is not None:
        comparison = estimator.compare_with_training_loss(args.training_loss, args.loss_type)
        
        # 保存诊断结果
        diagnosis_path = args.dataset.replace('.hdf5', '_diagnosis.json')
        with open(diagnosis_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n诊断结果已保存到: {diagnosis_path}")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print("\n关键要点:")
    print("1. Bayes误差 = 数据固有的不确定性，无法低于此值")
    print("2. 如果训练损失 ≈ Bayes误差 → 模型已经很好")
    print("3. 如果训练损失 >> Bayes误差 → 还有很大优化空间")
    print("4. 提升方向取决于比值：")
    print("   - 比值 < 1.5x: 主要靠数据质量提升")
    print("   - 比值 > 2x:  主要靠模型和训练方法改进")
    print("="*80)


if __name__ == "__main__":
    main()
