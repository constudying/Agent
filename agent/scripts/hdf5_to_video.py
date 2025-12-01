"""
从HDF5文件中提取图片数据并生成视频
"""
import h5py
import numpy as np
import cv2
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def extract_images_from_hdf5(hdf5_path, demo_key='demo_0', image_key='agentview_image', 
                              output_path=None, fps=30, frame_skip=1, use_ffmpeg=True):
    """
    从HDF5文件中提取图片并生成视频
    
    Args:
        hdf5_path: HDF5文件路径
        demo_key: demo键名，例如 'demo_0'
        image_key: 图片数据键名，例如 'agentview_image' 或 'robot0_eye_in_hand_image'
        output_path: 输出视频路径，如果为None则自动生成
        fps: 视频帧率
        frame_skip: 跳帧数量（例如frame_skip=2表示每2帧取1帧）
        use_ffmpeg: 是否使用FFmpeg（推荐，更可靠）
    """
    # 打开HDF5文件
    print(f"正在打开HDF5文件: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        # 检查文件结构
        if 'data' not in f:
            raise ValueError("HDF5文件中没有找到'data'组")
        
        if demo_key not in f['data']:
            available_demos = list(f['data'].keys())
            raise ValueError(f"找不到demo键'{demo_key}'。可用的demo: {available_demos}")
        
        demo = f['data'][demo_key]
        
        if 'obs' not in demo:
            raise ValueError(f"Demo '{demo_key}' 中没有'obs'组")
        
        if image_key not in demo['obs']:
            available_keys = list(demo['obs'].keys())
            raise ValueError(f"找不到图片键'{image_key}'。可用的键: {available_keys}")
        
        # 读取图片数据
        print(f"正在读取图片数据: {demo_key}/obs/{image_key}")
        images = demo['obs'][image_key][:]
        print(f"图片数据形状: {images.shape}, 数据类型: {images.dtype}")
        
        # 应用跳帧
        if frame_skip > 1:
            images = images[::frame_skip]
            print(f"跳帧后的帧数: {len(images)}")
        
        # 确定输出路径
        if output_path is None:
            hdf5_dir = Path(hdf5_path).parent
            output_path = hdf5_dir / f"{demo_key}_{image_key}.mp4"
        else:
            output_path = Path(output_path)
        
        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 获取图片尺寸
        num_frames, height, width, channels = images.shape
        
        print(f"正在创建视频: {output_path}")
        print(f"视频参数: 分辨率={width}x{height}, 帧率={fps}, 总帧数={num_frames}")
        
        # 使用FFmpeg生成视频（更可靠）
        if use_ffmpeg:
            try:
                create_video_with_ffmpeg(images, output_path, fps)
                print(f"视频已保存到: {output_path}")
                print(f"视频时长: {num_frames/fps:.2f} 秒")
                return str(output_path)
            except Exception as e:
                print(f"FFmpeg失败: {e}")
                print("回退到OpenCV方法...")
        
        # 使用OpenCV生成视频（备用方案）
        create_video_with_opencv(images, output_path, fps, width, height)
        
        print(f"视频已保存到: {output_path}")
        print(f"视频时长: {num_frames/fps:.2f} 秒")
    
    return str(output_path)


def create_video_with_ffmpeg(images, output_path, fps):
    """
    使用FFmpeg创建视频（更可靠的方法）
    """
    print("使用FFmpeg创建视频...")
    
    # 创建临时目录保存帧
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # 保存所有帧为PNG文件
        print("正在保存帧到临时文件...")
        for i, img in enumerate(images):
            if i % 100 == 0:
                print(f"保存进度: {i}/{len(images)}")
            frame_path = tmpdir_path / f"frame_{i:06d}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # 使用FFmpeg生成视频
        print("正在使用FFmpeg编码视频...")
        cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-framerate', str(fps),
            '-i', str(tmpdir_path / 'frame_%06d.png'),
            '-c:v', 'libx264',  # 使用H.264编码
            '-pix_fmt', 'yuv420p',  # 确保兼容性
            '-preset', 'medium',  # 编码速度/质量平衡
            '-crf', '23',  # 质量控制（18-28，越小质量越高）
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg失败: {result.stderr}")
        
        print("FFmpeg编码完成")


def create_video_with_opencv(images, output_path, fps, width, height):
    """
    使用OpenCV创建视频（备用方案）
    """
    print("使用OpenCV创建视频...")
    
    # 创建视频写入器 - 使用更兼容的编码器
    # 尝试使用H264编码器（更通用）
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # 如果H264不可用，尝试其他编码器
    if not video_writer.isOpened():
        print("警告: avc1编码器不可用，尝试使用X264...")
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("警告: X264编码器不可用，尝试使用XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.with_suffix('.avi')  # XVID通常用于AVI
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise RuntimeError("无法创建视频写入器，请检查OpenCV和编码器安装")
    
    # 写入每一帧
    print("正在写入视频帧...")
    for i, img in enumerate(images):
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(images)}")
        
        # OpenCV使用BGR格式，而numpy数组通常是RGB格式
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)
    
    video_writer.release()
    print("OpenCV视频创建完成")


def list_demos_and_images(hdf5_path):
    """
    列出HDF5文件中所有可用的demos和图片类型
    """
    print(f"\n正在分析HDF5文件: {hdf5_path}\n")
    
    with h5py.File(hdf5_path, 'r') as f:
        # 列出所有demos
        if 'data' in f:
            demos = list(f['data'].keys())
            print(f"可用的demos数量: {len(demos)}")
            print(f"前10个demos: {demos[:10]}")
            
            # 查看第一个demo的图片信息
            if demos:
                first_demo = f['data'][demos[0]]
                if 'obs' in first_demo:
                    obs_keys = list(first_demo['obs'].keys())
                    print(f"\n观测数据键: {obs_keys}")
                    
                    # 找出所有图片键
                    image_keys = [key for key in obs_keys if 'image' in key.lower()]
                    print(f"\n可用的图片键: {image_keys}")
                    
                    # 显示每个图片的详细信息
                    for img_key in image_keys:
                        img_data = first_demo['obs'][img_key]
                        print(f"  - {img_key}: shape={img_data.shape}, dtype={img_data.dtype}")
        else:
            print("警告: HDF5文件中没有'data'组")


def main():
    parser = argparse.ArgumentParser(description='从HDF5文件提取图片并生成视频')
    parser.add_argument('--hdf5', type=str, required=True,
                        help='HDF5文件路径')
    parser.add_argument('--demo', type=str, default='demo_0',
                        help='要提取的demo键名 (默认: demo_0)')
    parser.add_argument('--image_key', type=str, default='agentview_image',
                        help='图片数据键名 (默认: agentview_image)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出视频路径 (默认: 自动生成)')
    parser.add_argument('--fps', type=int, default=30,
                        help='视频帧率 (默认: 30)')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='跳帧数量，用于加速视频 (默认: 1，不跳帧)')
    parser.add_argument('--use_opencv', action='store_true', # 该参数不推荐使用
                        help='使用OpenCV而不是FFmpeg（FFmpeg是默认且推荐的）')
    parser.add_argument('--list', action='store_true',
                        help='列出HDF5文件中所有可用的demos和图片类型')
    parser.add_argument('--all_demos', action='store_true',
                        help='提取所有demos的视频')
    parser.add_argument('--all_views', action='store_true',
                        help='提取所有视图的视频')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.hdf5):
        print(f"错误: 文件不存在: {args.hdf5}")
        return
    
    # 如果是列出模式
    if args.list:
        list_demos_and_images(args.hdf5)
        return
    
    # 提取所有demos
    if args.all_demos:
        with h5py.File(args.hdf5, 'r') as f:
            if 'data' in f:
                demos = list(f['data'].keys())
                print(f"将提取 {len(demos)} 个demos的视频")
                for demo in demos:
                    try:
                        print(f"\n{'='*60}")
                        output = extract_images_from_hdf5(
                            args.hdf5, demo, args.image_key, 
                            None, args.fps, args.frame_skip, not args.use_opencv
                        )
                        print(f"成功: {demo} -> {output}")
                    except Exception as e:
                        print(f"失败: {demo} - {str(e)}")
        return
    
    # 提取所有视图
    if args.all_views:
        with h5py.File(args.hdf5, 'r') as f:
            if 'data' in f and args.demo in f['data']:
                demo = f['data'][args.demo]
                if 'obs' in demo:
                    image_keys = [key for key in demo['obs'].keys() if 'image' in key.lower()]
                    print(f"将提取 {len(image_keys)} 个视图的视频: {image_keys}")
                    for img_key in image_keys:
                        try:
                            print(f"\n{'='*60}")
                            output = extract_images_from_hdf5(
                                args.hdf5, args.demo, img_key, 
                                None, args.fps, args.frame_skip, not args.use_opencv
                            )
                            print(f"成功: {img_key} -> {output}")
                        except Exception as e:
                            print(f"失败: {img_key} - {str(e)}")
        return
    
    # 提取单个视频
    try:
        output_path = extract_images_from_hdf5(
            args.hdf5, args.demo, args.image_key, 
            args.output, args.fps, args.frame_skip, not args.use_opencv
        )
        print(f"\n成功！视频已保存到: {output_path}")
    except Exception as e:
        print(f"\n错误: {str(e)}")


if __name__ == '__main__':
    main()
