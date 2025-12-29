import os
os.environ['MUJOCO_GL'] = 'glfw'

import pickle
import mujoco
import mujoco.viewer

# 加载数据
with open('/home/lsy/cjh/project1/Agent/agent/datasets/playdata/env_visualization_data.pkl', 'rb') as f:
    env_data = pickle.load(f)

# 创建模型和数据
model = mujoco.MjModel.from_xml_string(env_data['model_xml'])
data = mujoco.MjData(model)

# 恢复状态
data.qpos[:] = env_data['qpos']
data.qvel[:] = env_data['qvel']
mujoco.mj_forward(model, data)

# 启动可视化
mujoco.viewer.launch(model, data)