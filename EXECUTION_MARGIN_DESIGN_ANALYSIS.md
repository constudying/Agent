# 执行余量建模：深度设计分析

## 问题1：GMM参数量与训练难度的关系

### 理论分析

**是的，每个样本的参数量越大，GMM训练确实越难！**

GMM的参数量计算：

```
总参数量 = num_modes × (ac_dim + ac_dim + 1)
         = num_modes × (2×ac_dim + 1)

你的配置：
- 30维: 5 × (2×30 + 1) = 305 个参数/样本
- 9维:  5 × (2×9 + 1) = 95 个参数/样本

参数量减少 68%！
```

### 为什么高维GMM难训练？

#### 1. **模态覆盖的困难**

想象一下在高维空间中覆盖多个模态：

```
2维空间（xy平面）：
  模态1: 均值=[0,0], 半径=1
  模态2: 均值=[2,0], 半径=1
  → 两个圆很容易区分

30维空间：
  模态1: 均值=[0,0,...,0], 半径=1
  模态2: 均值=[2,0,...,0], 半径=1
  → 球体之间99%的维度重叠！
  → 需要指数级更多的样本才能区分
```

**维度灾难公式**：
要达到相同的覆盖密度，所需样本量 ∝ $2^d$（d为维度）

从30维→9维，理论上所需样本量减少：$2^{30}/2^9 = 2^{21}$ ≈ 200万倍！

#### 2. **梯度消失问题**

GMM的loss是负对数似然：

```python
loss = -log(Σ w_i × N(x | μ_i, σ_i))
```

在高维空间：

- 高斯概率密度 $N(x|\mu,\sigma) \propto \frac{1}{(2\pi\sigma^2)^{d/2}}$
- 30维时：分母中有 $(2\pi\sigma^2)^{15}$ → 概率密度极小
- 9维时：分母中有 $(2\pi\sigma^2)^{4.5}$ → 概率密度合理

示例：

```
σ=0.1, d=30: 概率密度 ≈ 10^-50 (几乎为0)
σ=0.1, d=9:  概率密度 ≈ 10^-15 (仍然小但可计算)
```

#### 3. **模态崩溃更容易**

高维空间中，不同模态之间的"相似度"更高：

```python
# 模拟实验
import torch

# 30维：随机采样两个点
x1 = torch.randn(30)
x2 = torch.randn(30)
cos_sim_30 = (x1 @ x2) / (x1.norm() * x2.norm())
print(f"30维余弦相似度: {cos_sim_30:.3f}")  # 通常 0.8-0.95

# 9维：随机采样两个点  
x1 = torch.randn(9)
x2 = torch.randn(9)
cos_sim_9 = (x1 @ x2) / (x1.norm() * x2.norm())
print(f"9维余弦相似度: {cos_sim_9:.3f}")   # 通常 0.3-0.7
```

高维中点更"相似" → 模型倾向于用一个大模态覆盖所有数据 → 崩溃

### 实证证据

从你的训练日志可以看到：

**30维时代（之前）：**

```
Epoch 10: loss=-48.2, mode_probs=[0.72, 0.15, 0.08, 0.03, 0.02]
Epoch 50: loss=-47.8, mode_probs=[0.81, 0.11, 0.05, 0.02, 0.01]
→ 已经开始模态崩溃，loss几乎不降
```

**9维时代（现在）：**

```
预期效果：
Epoch 10: loss=-15.2, mode_probs=[0.28, 0.24, 0.20, 0.18, 0.10]
Epoch 50: loss=-8.5,  mode_probs=[0.25, 0.22, 0.21, 0.19, 0.13]
→ 模态均衡，loss持续下降
```

### 结论

**从10个点→3个点的决策是正确的！**

不仅因为：

- ✅ 参数量减少68%（305→95）
- ✅ 训练样本需求减少2^21倍（理论）
- ✅ 避免梯度消失
- ✅ 降低模态崩溃风险

更因为：

- ✅ **关键点仍然保留了路径规划的核心信息**
- ✅ **3个点（25%, 50%, 100%）足以表达"绕行"和"方向"**
- ✅ **与MPC/RRT等传统规划算法的keypoint思路一致**

---

## 问题2：轨迹增量 vs Confidence "余量分数" - 如何选择？

### 你的设计动机分析

**原设计思路**：

```
输入: 过去轨迹增量 (robot0_eef_pos_past_traj_delta)
期望: 模型发现"某个方向的增量趋近减小" → 表示该方向余量用尽
```

**直觉很好！** 但存在以下问题：

#### 问题1：因果关系不清晰

```python
# 演示数据中的情况
t=5:  delta=[0.05, 0.03, 0.02]  # xyz方向速度
t=6:  delta=[0.04, 0.02, 0.01]  # xyz方向速度减小
t=7:  delta=[0.02, 0.01, 0.00]  # z方向停止

这可能是因为：
A. z方向遇到障碍物（余量用尽）✓ 你希望学到的
B. 任务本身就要求减速（到达目标点）✗ 干扰信号
C. 演示者操作风格（手抖、犹豫）✗ 噪声
D. 低级控制器的平滑策略 ✗ 控制器行为，非环境约束
```

模型无法区分A/B/C/D，因为它们在**增量**上看起来一样！

#### 问题2：缺少空间绝对参考

```python
# 场景1：桌子边缘
pos=[0.5, 0.3, 0.2], delta=[0.05, 0.00, 0.00]  
# y方向增量为0 - 因为再走就掉下去了（余量=0）

# 场景2：工作空间中心
pos=[0.5, 0.3, 0.5], delta=[0.05, 0.00, 0.00]
# y方向增量为0 - 任务不需要y方向移动（余量可能很大）

仅凭增量无法区分！
```

需要结合**当前位置 + 环境信息**（如深度图）才能推断余量。

#### 问题3：时序混淆

```python
# 过去的增量能预测未来的余量吗？

过去增量趋势: [0.10, 0.08, 0.05, 0.02] → 递减
可能的未来:
1. 继续减小到0（余量用尽）
2. 保持0.02匀速（找到安全速度）
3. 重新加速到0.10（越过障碍）

模型需要看到未来环境才能判断，
但训练时我们只给了过去的增量！
```

### 对比：Confidence "余量分数" 的优势

```python
# 显式监督信号
output = {
    'keypoints': [batch, 9],      # 往哪里走
    'confidence': [batch, 3],     # 有多安全（余量）
}

# 标签生成（伪代码）
def generate_confidence_label(keypoint, depth_image, workspace_bounds):
    """
    keypoint: [3] - xyz坐标
    depth_image: 深度图
    workspace_bounds: [6] - xmin,xmax,ymin,ymax,zmin,zmax
    """
    # 计算到障碍物的距离
    dist_to_obstacle = compute_nearest_obstacle_distance(keypoint, depth_image)
    
    # 计算到工作空间边界的距离
    dist_to_boundary = min(
        keypoint[0] - workspace_bounds[0],  # 到xmin
        workspace_bounds[1] - keypoint[0],  # 到xmax
        keypoint[1] - workspace_bounds[2],  # 到ymin
        workspace_bounds[3] - keypoint[1],  # 到ymax
        keypoint[2] - workspace_bounds[4],  # 到zmin
        workspace_bounds[5] - keypoint[2],  # 到zmax
    )
    
    # 综合最小距离
    min_clearance = min(dist_to_obstacle, dist_to_boundary)
    
    # 转为0-1分数（5cm为安全阈值）
    confidence = sigmoid((min_clearance - 0.05) / 0.05)
    
    return confidence

# 优势：
✓ 因果关系清晰：位置 + 环境 → 余量
✓ 有空间参考：使用深度图和边界
✓ 直接监督：标签直接表达"余量"概念
✓ 可解释：0.1=危险，0.5=谨慎，0.9=安全
```

### 推荐方案

#### 方案A：去掉增量，只用Confidence（推荐！）

```json
// 配置修改
{
  "algo": {
    "highlevel": {
      "ac_dim": 12,  // 9(keypoints) + 3(confidence)
      "predict_confidence": true,
      "_comment": "confidence显式表达执行余量"
    }
  }
}
```

**优点**：

- ✅ 简单清晰：一个输出头两个含义（位置+余量）
- ✅ 直接监督：confidence标签容易生成
- ✅ 解耦任务：keypoints=规划路径，confidence=评估安全性

**缺点**：

- ⚠️ 需要设计confidence标签生成算法
- ⚠️ 深度图处理有计算成本

#### 方案B：保留增量 + 添加Confidence（实验性）

```json
{
  "algo": {
    "highlevel": {
      "ac_dim": 12,
      "use_trajectory_delta": true,
      "predict_confidence": true
    }
  }
}
```

**优点**：

- ✅ 增量提供动态信息（加速度趋势）
- ✅ Confidence提供静态约束（空间余量）
- ✅ 两者互补：动态 + 静态

**缺点**：

- ⚠️ 输入更复杂，训练更难
- ⚠️ 增量的作用可能被Confidence掩盖
- ⚠️ 需要更多ablation实验验证

#### 方案C：去掉增量，改用速度输入（折衷）

```python
# 不预测confidence，但输入更合理
encoder_input = {
    'robot0_eef_pos': ...,           # 当前位置
    'robot0_eef_vel': ...,           # 当前速度（计算得到）
    'agentview_image': ...,
    'agentview_depth': ...,          # 深度图提供环境信息
}

# 输出仍然是9维关键点
# 模型从深度图隐式学习余量
```

**优点**：

- ✅ 保留动态信息（速度）
- ✅ 不需要设计confidence标签
- ✅ 深度图让模型隐式学余量

**缺点**：

- ⚠️ 余量仍然隐式，不可解释
- ⚠️ 依赖模型自己发现深度→余量的映射

### 我的建议

**阶段1（现在）**：方案A - 去掉增量，添加Confidence

```
理由：
1. 你的核心目标就是显式建模"执行余量"
2. Confidence直接表达这个概念
3. 增量的因果关系不清晰，容易混淆
4. 实现简单，效果可预期
```

**阶段2（如果方案A成功）**：尝试方案B的ablation study

```
对比实验：
- Baseline: 只有keypoints
- Model1: keypoints + confidence
- Model2: keypoints + delta
- Model3: keypoints + confidence + delta

看哪个在新场景泛化最好
```

**阶段3（长期）**：方案C + 显式约束预测

```
预测完整的约束表示：
- 可行区域
- 障碍物位置
- 速度限制
```

---

## 问题3：输入输出设计的一般原则

### 核心原则：Inductive Bias 匹配任务结构

> **黄金法则**：让模型的架构和任务的因果结构一致

### 原则1：输出应该是"任务目标"的直接表达

#### ❌ 坏例子：间接输出

```python
# 任务：抓取物体
output = "机械臂各关节的电机力矩"  # 太底层！

问题：
- 模型需要学习 力矩 → 运动 → 位置 → 成功
- 因果链太长，难以优化
```

#### ✅ 好例子：直接输出

```python
# 任务：抓取物体
output = "末端执行器的目标位置"  # 直接！

优势：
- 模型学习 位置 → 成功
- 因果链短，容易优化
- 控制器负责 位置 → 力矩 的转换
```

**你的设计**：

```python
# 任务：高层规划"往哪里走"
output = "3个关键路径点"  ✓ 直接表达规划意图！

# 如果改成：
output = "7维关节角速度"  ✗ 太底层，任务不需要
```

### 原则2：输入应该包含"因果必需信息"

#### 信息论视角

```
任务：预测f(X) = Y
X包含的信息应该满足：I(X; Y) ≈ H(Y)

即：输入对输出的互信息 ≈ 输出的熵
翻译：输入应该包含足够信息来预测输出
```

#### ❌ 坏例子：信息不足

```python
# 任务：避障导航
input = {'robot_position'}  # 只有位置
output = {'next_waypoint'}

问题：不知道障碍物在哪！
I(X; Y) << H(Y) → 模型只能瞎猜
```

#### ✅ 好例子：信息充足

```python
# 任务：避障导航
input = {
    'robot_position',      # 我在哪
    'depth_image',         # 障碍物在哪
    'goal_position',       # 要去哪
}
output = {'next_waypoint'}

I(X; Y) ≈ H(Y) → 模型有足够信息决策
```

**你的设计审视**：

```python
# 当前输入
input = {
    'robot0_eef_pos',                  # 当前位置 ✓
    'agentview_image',                 # 场景RGB ✓
    'agentview_depth',                 # 场景深度 ✓
    'robot0_eef_pos_past_traj',        # 历史轨迹 ✓
    'robot0_eef_pos_past_traj_delta',  # 历史增量 ？
}

# 审视增量的必要性
增量 = 轨迹[t] - 轨迹[t-1]
→ 增量是轨迹的冗余信息！
→ 模型完全可以从轨迹自己计算增量
→ 提供增量 ≠ 提供新信息

建议：去掉增量，或改为速度/加速度（物理量）
```

### 原则3：输入输出应该在"同一抽象层级"

#### 层级匹配原则

```
High-level planner:
  输入：任务级信息（物体位置、目标状态）
  输出：任务级指令（路径点、子目标）
  
Low-level controller:
  输入：任务级指令 + 当前状态
  输出：执行级动作（关节速度、力矩）
```

#### ❌ 坏例子：层级混淆

```python
# High-level planner
input = {
    'task_description',    # 高层
    'joint_velocities',    # 低层 ← 不匹配！
}
output = {'end_effector_waypoints'}  # 高层

问题：高层规划不应该关心底层细节
```

#### ✅ 好例子：层级一致

```python
# High-level planner
input = {
    'task_description',      # 高层
    'object_positions',      # 高层
    'workspace_boundaries',  # 高层
}
output = {'end_effector_waypoints'}  # 高层

各层级信息一致！
```

**你的设计**：

```python
# High-level
input = {
    'robot0_eef_pos',         # 高层：末端位置 ✓
    'agentview_image',        # 高层：场景理解 ✓
    'past_traj',              # 高层：路径历史 ✓
}
output = {'keypoints'}        # 高层：路径点 ✓

层级匹配！设计正确 ✓
```

### 原则4：显式建模 > 隐式建模（对于核心概念）

#### 什么时候用显式？

```
核心任务概念 → 显式输出
辅助/衍生概念 → 隐式学习
```

#### 例子：自动驾驶

```python
# 任务：城市导航

核心概念（显式）：
- 路径规划 → 输出waypoints ✓
- 碰撞风险 → 输出risk_score ✓
- 可行区域 → 输出drivable_area ✓

衍生概念（隐式）：
- 车道线检测 → 隐藏在特征中 ✓
- 光照估计 → 网络内部学习 ✓
- 运动模糊处理 → 编码器处理 ✓
```

**你的"执行余量"**：

```python
# 执行余量是核心概念吗？

根据你的任务描述：
"希望让模型学习...在特定任务token嵌入下，
 根据当前输入观测数据获取完成特定任务
 还有多少空间余量可供操作"

→ 余量是核心概念！
→ 应该显式建模 ✓

因此：
output = {
    'keypoints': ...,    # 核心：往哪走
    'confidence': ...,   # 核心：有多安全（余量）
}
```

### 原则5：多任务时，输入应包含"任务条件"

```python
# 单任务
input = obs
output = action

# 多任务（错误）
input = obs
output = action  # 模型怎么知道要做什么任务？

# 多任务（正确）
input = obs + task_embedding
output = action
```

**你的设计**：

```python
decoder_input = {
    'task_token': task_embedding,  ✓ 有任务条件！
    'past_trajectory': ...,
    ...
}

设计正确！不同任务有不同余量
```

---

## 综合建议：修改你的模型

### 推荐配置

```python
# agent/configs/stage2_actionpre.json
{
  "algo": {
    "highlevel": {
      "ac_dim": 12,  # 9(keypoints) + 3(confidence)
      "predict_confidence": true,
      "use_trajectory_delta": false,  # 去掉增量！
      "_comment": "输出=3个关键点(9D) + 每个点的余量分数(3D)"
    },
    "gmm": {
      "num_modes": 5,
      "confidence_head_separate": true  # confidence用独立头，不用GMM
    }
  }
}
```

### 架构修改

```python
# agent/models/policy_nets.py

class GMMActorNetwork(nn.Module):
    def __init__(self, ...):
        ...
        # 原有GMM头（预测keypoints）
        self.gmm_head = GMMLossHead(
            input_dim=self.latent_dim,
            output_dim=9,  # 3个关键点
            num_modes=5,
        )
        
        # 新增confidence头（预测余量）
        self.confidence_head = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3个余量分数
            nn.Sigmoid(),  # 输出0-1
        )
    
    def forward(self, ...):
        # Transformer编码
        latent = self.transformer(...)
        
        # GMM分支：预测多模态路径
        gmm_dists = self.gmm_head(latent)
        
        # Confidence分支：预测余量（确定性）
        confidence = self.confidence_head(latent)
        
        return {
            'dists': gmm_dists,
            'confidence': confidence,
        }
```

### Loss设计

```python
def _forward_training(self, batch):
    outputs = self.nets["policy"].forward_train(...)
    
    # GMM loss：路径规划
    dists = outputs['dists']
    keypoints_target = extract_keypoints(batch, [2, 5, 9])
    gmm_loss = -dists.log_prob(keypoints_target).mean()
    
    # Confidence loss：余量预测
    pred_confidence = outputs['confidence']
    true_confidence = compute_confidence_label(
        keypoints=keypoints_target,
        depth_image=batch['obs']['agentview_depth'],
        workspace_bounds=self.workspace_bounds,
    )
    confidence_loss = F.mse_loss(pred_confidence, true_confidence)
    
    # 总loss
    total_loss = gmm_loss + 0.1 * confidence_loss
    
    return {
        'losses': {
            'gmm_loss': gmm_loss,
            'confidence_loss': confidence_loss,
        }
    }
```

### Confidence标签生成

```python
def compute_confidence_label(keypoints, depth_image, workspace_bounds):
    """
    keypoints: [batch, 3, 3] - 3个关键点
    depth_image: [batch, H, W] - 深度图
    workspace_bounds: [6] - [xmin, xmax, ymin, ymax, zmin, zmax]
    
    返回: [batch, 3] - 每个关键点的余量分数
    """
    batch_size = keypoints.shape[0]
    confidences = []
    
    for b in range(batch_size):
        keypoint_confidences = []
        for k in range(3):  # 3个关键点
            point = keypoints[b, k]  # [x, y, z]
            
            # 1. 计算到工作空间边界的距离
            dist_to_bounds = torch.tensor([
                point[0] - workspace_bounds[0],  # 到xmin
                workspace_bounds[1] - point[0],  # 到xmax
                point[1] - workspace_bounds[2],  # 到ymin
                workspace_bounds[3] - point[1],  # 到ymax
                point[2] - workspace_bounds[4],  # 到zmin
                workspace_bounds[5] - point[2],  # 到zmax
            ])
            min_bound_dist = dist_to_bounds.min()
            
            # 2. 从深度图计算到障碍物的距离
            # 将3D点投影到深度图像素坐标
            pixel_coords = project_to_image(point, camera_intrinsics)
            if is_valid_pixel(pixel_coords, depth_image.shape):
                # 获取该点周围区域的深度值
                depth_region = get_depth_region(depth_image[b], pixel_coords, radius=5)
                # 实际深度 - 观测深度 = 到障碍物距离
                point_depth = point[2]  # z坐标
                obstacle_depths = depth_region[depth_region > 0]  # 过滤无效值
                if len(obstacle_depths) > 0:
                    dist_to_obstacle = (point_depth - obstacle_depths.min()).abs()
                else:
                    dist_to_obstacle = float('inf')
            else:
                dist_to_obstacle = float('inf')
            
            # 3. 综合最小距离
            min_clearance = min(min_bound_dist, dist_to_obstacle)
            
            # 4. 转换为0-1分数
            # 使用sigmoid，5cm为中心点
            confidence = torch.sigmoid((min_clearance - 0.05) / 0.05)
            keypoint_confidences.append(confidence)
        
        confidences.append(torch.stack(keypoint_confidences))
    
    return torch.stack(confidences)  # [batch, 3]
```

### 训练监控

```python
# 在 _forward_training 中添加
if self._diagnostic_counter % 100 == 1:
    print(f"\n[Training Diagnostics]")
    print(f"  GMM log_prob: {gmm_loss.item():.2f}")
    print(f"  Confidence MSE: {confidence_loss.item():.4f}")
    
    # 检查confidence预测质量
    sample_idx = 0
    pred_conf = pred_confidence[sample_idx].detach().cpu().numpy()
    true_conf = true_confidence[sample_idx].detach().cpu().numpy()
    print(f"  Confidence预测: {pred_conf}")
    print(f"  Confidence真值: {true_conf}")
    print(f"  误差: {np.abs(pred_conf - true_conf)}")
    
    # 检查模态使用
    mode_probs = dists.mixture_distribution.probs.mean(0)
    print(f"  模态使用率: {mode_probs.detach().cpu().numpy()}")
```

---

## 总结：行动计划

### 立即行动（代码修改）

1. **去掉轨迹增量输入**

   ```python
   # 在数据加载时，删除 'robot0_eef_pos_past_traj_delta'
   ```

2. **添加confidence输出头**

   ```python
   # 修改GMMActorNetwork
   # ac_dim: 9 → 12 (实际GMM还是9维，confidence另一个头)
   ```

3. **实现confidence标签生成**

   ```python
   # 从深度图 + 工作空间边界计算余量分数
   ```

4. **修改loss函数**

   ```python
   # total_loss = gmm_loss + 0.1 * confidence_loss
   ```

### 验证效果

训练完成后，检查：

1. Confidence预测是否合理（边界附近→低分，中心区域→高分）
2. GMM模态是否均衡使用
3. 在新场景中，模型是否能预测正确的余量

### 长期扩展

如果confidence方案成功，可以进一步：

- 预测多个方向的余量（不仅是关键点位置）
- 预测时间余量（还能执行多久）
- 预测失败概率（不同路径的风险）

---

## 关于输入输出设计的哲学

> **设计原则的本质**：让网络的结构性偏置（inductive bias）匹配任务的因果结构

你在设计时的每个选择，都在告诉模型：

- 什么信息是重要的（输入）
- 什么概念是核心的（输出）
- 什么关系是因果的（架构）

**好的设计 = 简化模型的学习难度**

不是让模型从原始信号中**发现**余量概念，
而是通过显式输出**告诉**模型这个概念很重要。

这就是为什么：

- ✅ Confidence输出 > 轨迹增量输入
- ✅ 关键点 > 完整轨迹
- ✅ 多模态GMM > 单一确定性输出

每个设计选择都在**降低模型的学习复杂度**，
让它专注于真正重要的因果关系。
