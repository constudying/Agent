import os
os.environ['MUJOCO_GL'] = 'glfw'

import pickle
import numpy as np
import mujoco
import mujoco.viewer
import json

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

def get_camera_intrinsics(model, camera_name, width=None, height=None):
  """
  获取相机的内参矩阵
  
  Returns:
    K: 3x3 内参矩阵
    width: 图像宽度
    height: 图像高度
  """

  camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
  
  width = 240
  height = 240
  
  fovy = model.cam_fovy[camera_id]
  
  # 计算焦距
  f_y = height / (2 * np.tan(np.radians(fovy) / 2))
  f_x = f_y * (width / height)
  
  # 主点
  c_x = width / 2.0
  c_y = height / 2.0
  
  # 内参矩阵
  K = np.array([
    [f_x, 0, c_x],
    [0, f_y, c_y],
    [0, 0, 1]
  ])
  
  return K, width, height

def get_camera_extrinsics(model, data, camera_name):
  """
  获取相机的外参矩阵（世界坐标系到相机坐标系）
  
  Returns:
    extrinsics: 4x4 变换矩阵 [R | t; 0 0 0 1]
    cam_pos: 相机在世界坐标系中的位置
    cam_rot: 相机的旋转矩阵
  """
  camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
  
  # 相机位置（世界坐标系）
  cam_pos = data.cam_xpos[camera_id].copy()  # shape: (3,)
  
  # 相机旋转矩阵
  cam_rot = data.cam_xmat[camera_id].reshape(3, 3).copy()  # shape: (3, 3)
  
  # 构建 4x4 外参矩阵
  extrinsics = np.eye(4)
  extrinsics[:3, :3] = cam_rot
  extrinsics[:3, 3] = -cam_rot @ cam_pos
  
  return extrinsics, cam_pos, cam_rot

# 打印所有相机名称
print("Available cameras:")
for i in range(model.ncam):
  print(f"  Camera {i}: {model.camera(i).name}")

# 收集所有相机的参数
camera_params = {}

for i in range(model.ncam):
  camera_name = model.camera(i).name
  print(f"\nProcessing camera: {camera_name}")
  
  # 获取内参
  K, width, height = get_camera_intrinsics(model, camera_name, width=640, height=480)
  
  # 获取外参
  extrinsics, cam_pos, cam_rot = get_camera_extrinsics(model, data, camera_name)
  
  # 保存到字典（转换为列表以便JSON序列化）
  camera_params[camera_name] = {
    "intrinsics": {
      "K": K.tolist(),
      "width": int(width),
      "height": int(height)
    },
    "extrinsics": {
      "matrix": extrinsics.tolist(),
      "position": cam_pos.tolist(),
      "rotation": cam_rot.tolist()
    }
  }
  
  print(f"  Intrinsic Matrix K:")
  print(K)
  print(f"  Camera Position: {cam_pos}")

# 保存到JSON文件
output_path = '/home/lsy/cjh/project1/Agent/agent/datasets/playdata/camera_parameters.json'
with open(output_path, 'w') as f:
  json.dump(camera_params, f, indent=2)

print(f"\nCamera parameters saved to: {output_path}")
