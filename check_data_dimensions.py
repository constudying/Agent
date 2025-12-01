"""
快速检查数据维度和训练目标的脚本
运行此脚本来确认问题
"""

import sys
import json

# 1. 读取配置文件
config_path = "/home/lsy/cjh/project1/Agent/agent/configs/stage2_actionpre.json"
print("=" * 80)
print("检查配置文件")
print("=" * 80)

with open(config_path, 'r') as f:
    config = json.load(f)

print(f"\n配置文件路径: {config_path}")

# 检查关键参数
if 'algo' in config and 'highlevel' in config['algo']:
    highlevel = config['algo']['highlevel']
    print(f"\n高层配置:")
    for key, value in highlevel.items():
        print(f"  {key}: {value}")
    
    if 'ac_dim' in highlevel:
        ac_dim = highlevel['ac_dim']
        print(f"\n⚠️ 网络输出维度 (ac_dim): {ac_dim}")
    else:
        print("\n❌ 配置中没有找到 ac_dim!")

# 检查GMM配置
if 'algo' in config and 'gmm' in config['algo']:
    gmm = config['algo']['gmm']
    print(f"\nGMM配置:")
    for key, value in gmm.items():
        print(f"  {key}: {value}")

# 2. 检查数据集
print("\n" + "=" * 80)
print("检查数据集")
print("=" * 80)

if 'train' in config and 'data' in config['train']:
    data_path = config['train']['data']
    if data_path:
        print(f"\n数据集路径: {data_path}")
        
        try:
            import h5py
            import numpy as np
            
            with h5py.File(data_path, 'r') as f:
                # 获取第一个demo
                demo_keys = list(f['data'].keys())
                demo = f['data'][demo_keys[0]]
                
                print(f"\n第一个demo: {demo_keys[0]}")
                
                # 检查动作
                if 'actions' in demo:
                    actions = demo['actions'][:]
                    print(f"\n✓ actions.shape: {actions.shape}")
                    print(f"  - 序列长度: {actions.shape[0]}")
                    print(f"  - 动作维度: {actions.shape[1]}")
                    print(f"  - 值范围: [{actions.min():.4f}, {actions.max():.4f}]")
                    action_dim = actions.shape[1]
                else:
                    print("\n❌ 没有找到 actions!")
                    action_dim = None
                
                # 检查观测
                obs_keys = list(demo['obs'].keys())
                print(f"\n可用的观测键: {obs_keys}")
                
                # 检查未来轨迹
                if 'robot0_eef_pos_future_traj' in demo['obs']:
                    future_traj = demo['obs/robot0_eef_pos_future_traj'][:]
                    print(f"\n✓ robot0_eef_pos_future_traj.shape: {future_traj.shape}")
                    print(f"  - 序列长度: {future_traj.shape[0]}")
                    if len(future_traj.shape) == 2:
                        print(f"  - 特征维度: {future_traj.shape[1]}")
                        traj_dim = future_traj.shape[1]
                    elif len(future_traj.shape) == 3:
                        print(f"  - 轨迹长度: {future_traj.shape[1]}")
                        print(f"  - 位置维度: {future_traj.shape[2]}")
                        traj_dim = future_traj.shape[1] * future_traj.shape[2]
                    print(f"  - 值范围: [{future_traj.min():.4f}, {future_traj.max():.4f}]")
                else:
                    print("\n⚠️ 没有找到 robot0_eef_pos_future_traj")
                    traj_dim = None
                
                # 检查当前位置
                if 'robot0_eef_pos' in demo['obs']:
                    eef_pos = demo['obs/robot0_eef_pos'][:]
                    print(f"\n✓ robot0_eef_pos.shape: {eef_pos.shape}")
                    print(f"  - 位置维度: {eef_pos.shape[1]}")
                    print(f"  - 值范围: [{eef_pos.min():.4f}, {eef_pos.max():.4f}]")
                
                # 比较维度
                print("\n" + "=" * 80)
                print("维度匹配检查")
                print("=" * 80)
                
                if action_dim is not None and traj_dim is not None:
                    print(f"\n动作维度 (actions):           {action_dim}")
                    print(f"轨迹维度 (future_traj):       {traj_dim}")
                    print(f"网络输出维度 (ac_dim):        {ac_dim if 'ac_dim' in locals() else '未配置'}")
                    
                    if action_dim == traj_dim:
                        print("\n⚠️ 动作维度 == 轨迹维度 (可能是巧合，但语义不同!)")
                    else:
                        print(f"\n❌ 动作维度 != 轨迹维度 (相差 {abs(action_dim - traj_dim)})")
                    
                    if 'ac_dim' in locals():
                        if ac_dim == action_dim:
                            print(f"✓ 网络输出维度 == 动作维度 (正确)")
                        elif ac_dim == traj_dim:
                            print(f"⚠️ 网络输出维度 == 轨迹维度 (可能配置错误)")
                        else:
                            print(f"❌ 网络输出维度既不等于动作维度也不等于轨迹维度!")
                
                # 检查一个样本的实际值
                print("\n" + "=" * 80)
                print("样本数据检查 (第一个时间步)")
                print("=" * 80)
                
                if 'actions' in demo:
                    print(f"\nactions[0]: {actions[0]}")
                
                if 'robot0_eef_pos_future_traj' in demo['obs']:
                    traj_sample = future_traj[0]
                    if len(future_traj.shape) == 3:
                        traj_sample = traj_sample.flatten()
                    print(f"future_traj[0]: {traj_sample}")
                
                if 'robot0_eef_pos' in demo['obs']:
                    print(f"eef_pos[0]: {eef_pos[0]}")
                
        except Exception as e:
            print(f"\n❌ 读取数据集时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️ 配置中的 data 字段为空，请手动指定数据集路径")
        print("\n请手动运行:")
        print("  python check_data_dimensions.py <数据集路径.hdf5>")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)

print("""
请根据上面的输出检查:

1. ac_dim 的值是多少？
2. actions.shape[1] 是多少？
3. future_traj 的展平后维度是多少？
4. 它们是否匹配？

如果 ac_dim == traj_dim 但 != action_dim:
  → 你在用轨迹训练，但应该用动作！
  → 解决方案: 修改 agent.py line 662
     log_probs = dists.log_prob(batch["actions"])

如果维度都匹配，但loss仍然不降:
  → 可能是输入特征不足
  → 建议增加 robot0_eef_pos 到输入观测中
  → 或者运行 estimate_bayes_error.py 检查数据质量
""")
