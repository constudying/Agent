"""
训练过程中的注意力监控工具

这个模块提供了在训练过程中定期保存和可视化注意力图的功能，
同时确保不影响训练性能。

主要特点：
1. 异步保存：使用线程池避免阻塞训练
2. 内存高效：及时释放注意力权重
3. 可配置：控制保存频率和可视化类型
4. TensorBoard集成：可选地将注意力图记录到TensorBoard

使用示例：
    ```python
    from agent.utils.training_attention_monitor import TrainingAttentionMonitor
    
    # 创建监控器
    monitor = TrainingAttentionMonitor(
        save_dir='./attention_logs',
        save_frequency=100,  # 每100个step保存一次
        use_tensorboard=True
    )
    
    # 在训练循环中
    for step, batch in enumerate(dataloader):
        output = model(enc, dec)
        loss = criterion(output, target)
        
        # 定期保存注意力图
        if monitor.should_visualize(step):
            monitor.log_attention(model, enc, dec, step, loss.item())
        
        loss.backward()
        optimizer.step()
    ```
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from threading import Thread
from queue import Queue
from typing import Dict, Optional, List, Union
import time


class TrainingAttentionMonitor:
    """
    训练过程中的注意力监控器
    
    这个类负责在训练过程中定期保存和可视化注意力图，
    同时确保不影响训练速度。
    """
    
    def __init__(
        self,
        save_dir: str = './attention_logs',
        save_frequency: int = 100,
        save_on_epochs: Optional[List[int]] = None,
        visualization_types: List[str] = ['heatmap', 'statistics'],
        max_workers: int = 2,
        use_tensorboard: bool = False,
        tensorboard_log_dir: Optional[str] = None,
        save_raw_weights: bool = False,
        batch_idx_to_visualize: int = 0,
        layers_to_visualize: Optional[List[int]] = None,
        heads_to_visualize: Optional[Union[int, str]] = 'average',
    ):
        """
        初始化注意力监控器
        
        Args:
            save_dir: 保存注意力图的目录
            save_frequency: 保存频率（每N个step）
            save_on_epochs: 在特定epoch结束时保存，例如 [1, 5, 10, 20]
            visualization_types: 要生成的可视化类型
                - 'heatmap': 热力图
                - 'multi_head': 多头对比
                - 'layer_comparison': 层级对比
                - 'statistics': 统计信息
            max_workers: 最大后台工作线程数
            use_tensorboard: 是否使用TensorBoard记录
            tensorboard_log_dir: TensorBoard日志目录
            save_raw_weights: 是否保存原始注意力权重（会占用较多空间）
            batch_idx_to_visualize: 要可视化的批次索引
            layers_to_visualize: 要可视化的层索引列表，None表示所有层
            heads_to_visualize: 要可视化的头索引，'average'表示平均所有头
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_frequency = save_frequency
        self.save_on_epochs = save_on_epochs or []
        self.visualization_types = visualization_types
        self.batch_idx = batch_idx_to_visualize
        self.layers_to_visualize = layers_to_visualize
        self.heads_to_visualize = heads_to_visualize
        self.save_raw_weights = save_raw_weights
        
        # TensorBoard支持
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = tensorboard_log_dir or str(self.save_dir / 'tensorboard')
                self.tb_writer = SummaryWriter(tb_dir)
                print(f"TensorBoard logging enabled: {tb_dir}")
            except ImportError:
                print("Warning: tensorboard not installed. TensorBoard logging disabled.")
                self.use_tensorboard = False
        
        # 异步保存队列
        self.save_queue = Queue(maxsize=max_workers * 2)
        self.workers = []
        for _ in range(max_workers):
            worker = Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        # 统计信息
        self.last_save_time = time.time()
        self.save_count = 0
        
        print(f"TrainingAttentionMonitor initialized:")
        print(f"  - Save directory: {self.save_dir}")
        print(f"  - Save frequency: every {save_frequency} steps")
        print(f"  - Visualization types: {visualization_types}")
        print(f"  - Background workers: {max_workers}")
    
    def should_visualize(self, step: int, epoch: Optional[int] = None) -> bool:
        """
        判断是否应该在当前step可视化注意力
        
        Args:
            step: 当前训练步数
            epoch: 当前epoch（可选）
        
        Returns:
            是否应该可视化
        """
        # 基于step的频率
        if step > 0 and step % self.save_frequency == 0:
            return True
        
        # 基于epoch的特定保存点
        if epoch is not None and epoch in self.save_on_epochs:
            return True
        
        return False
    
    def log_attention(
        self,
        model,
        enc: Dict[str, torch.Tensor],
        dec: Dict[str, torch.Tensor],
        step: int,
        loss: Optional[float] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict] = None
    ):
        """
        记录注意力权重（非阻塞）
        
        Args:
            model: Transformer模型
            enc: 编码器输入
            dec: 解码器输入
            step: 当前训练步数
            loss: 当前损失值
            epoch: 当前epoch
            metrics: 其他要记录的指标
        """
        # 快速检查
        if not self.should_visualize(step, epoch):
            return
        
        # 临时启用注意力存储
        was_training = model.training
        model.eval()
        
        # 保存原始的store_attention状态
        original_states = []
        for layer in model.nets["encoder"]:
            if hasattr(layer.nets["selfattention"], 'store_attention'):
                original_states.append(layer.nets["selfattention"].store_attention)
                layer.nets["selfattention"].store_attention = True
        
        for layer in model.nets["decoder"]:
            if hasattr(layer.nets["selfattention"], 'store_attention'):
                original_states.append(layer.nets["selfattention"].store_attention)
                layer.nets["selfattention"].store_attention = True
            if hasattr(layer.nets["crossattention"], 'store_attention'):
                original_states.append(layer.nets["crossattention"].store_attention)
                layer.nets["crossattention"].store_attention = True
        
        # 前向传播（不计算梯度）
        with torch.no_grad():
            try:
                _, attention_weights = model(enc, dec, return_attention_weights=True)
            except Exception as e:
                print(f"Warning: Failed to collect attention weights: {e}")
                # 恢复状态
                if was_training:
                    model.train()
                return
        
        # 恢复训练状态
        if was_training:
            model.train()
        
        # 将注意力权重移到CPU（释放GPU内存）
        attention_cpu = self._move_to_cpu(attention_weights)
        
        # 准备元数据
        metadata = {
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'timestamp': time.time(),
        }
        if metrics:
            metadata.update(metrics)
        
        # 异步保存
        save_task = {
            'attention_weights': attention_cpu,
            'metadata': metadata,
            'step': step,
            'epoch': epoch,
        }
        
        try:
            self.save_queue.put(save_task, block=False)
            self.save_count += 1
        except:
            print(f"Warning: Save queue full, skipping attention save at step {step}")
        
        # 立即释放GPU上的注意力权重
        del attention_weights
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _move_to_cpu(self, attention_weights: Dict) -> Dict:
        """将注意力权重移动到CPU"""
        cpu_weights = {
            'encoder': [],
            'decoder': []
        }
        
        for layer_attn in attention_weights['encoder']:
            cpu_layer = {}
            for key, tensor in layer_attn.items():
                if tensor is not None:
                    cpu_layer[key] = tensor.detach().cpu()
            cpu_weights['encoder'].append(cpu_layer)
        
        for layer_attn in attention_weights['decoder']:
            cpu_layer = {}
            for key, tensor in layer_attn.items():
                if tensor is not None:
                    cpu_layer[key] = tensor.detach().cpu()
            cpu_weights['decoder'].append(cpu_layer)
        
        return cpu_weights
    
    def _worker_loop(self):
        """后台工作线程循环"""
        while True:
            try:
                task = self.save_queue.get(timeout=1.0)
                if task is None:  # 停止信号
                    break
                
                self._process_save_task(task)
                self.save_queue.task_done()
                
            except:
                continue
    
    def _process_save_task(self, task: Dict):
        """处理保存任务"""
        attention_weights = task['attention_weights']
        metadata = task['metadata']
        step = task['step']
        epoch = task['epoch']
        
        # 创建保存目录
        if epoch is not None:
            step_dir = self.save_dir / f'epoch_{epoch}_step_{step}'
        else:
            step_dir = self.save_dir / f'step_{step}'
        step_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 保存统计信息
            if 'statistics' in self.visualization_types:
                self._save_statistics(attention_weights, metadata, step_dir)
            
            # 2. 生成可视化（如果安装了matplotlib）
            try:
                from agent.models.transformer import AttentionVisualizer
                visualizer = AttentionVisualizer()
                
                # 确定要可视化的层
                encoder_layers = self.layers_to_visualize or list(range(len(attention_weights['encoder'])))
                decoder_layers = self.layers_to_visualize or list(range(len(attention_weights['decoder'])))
                
                # 热力图
                if 'heatmap' in self.visualization_types:
                    self._generate_heatmaps(
                        visualizer, attention_weights, 
                        encoder_layers, decoder_layers, step_dir
                    )
                
                # 多头对比
                if 'multi_head' in self.visualization_types:
                    self._generate_multi_head_plots(
                        visualizer, attention_weights,
                        encoder_layers, decoder_layers, step_dir
                    )
                
                # 层级对比
                if 'layer_comparison' in self.visualization_types:
                    self._generate_layer_comparison(
                        visualizer, attention_weights, step_dir
                    )
                
            except ImportError:
                pass  # matplotlib未安装，跳过可视化
            
            # 3. 保存原始权重（可选）
            if self.save_raw_weights:
                self._save_raw_weights(attention_weights, step_dir)
            
            # 4. TensorBoard记录
            if self.use_tensorboard and self.tb_writer is not None:
                self._log_to_tensorboard(attention_weights, metadata, step)
            
            print(f"✓ Attention visualizations saved to {step_dir}")
            
        except Exception as e:
            print(f"Error processing save task for step {step}: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_statistics(self, attention_weights: Dict, metadata: Dict, save_dir: Path):
        """保存注意力统计信息"""
        stats = {
            'metadata': metadata,
            'encoder': [],
            'decoder': []
        }
        
        # 编码器统计
        for layer_idx, layer_attn in enumerate(attention_weights['encoder']):
            if 'self_attention' in layer_attn:
                attn = layer_attn['self_attention'][self.batch_idx].numpy()
                stats['encoder'].append({
                    'layer': layer_idx,
                    'type': 'self_attention',
                    'mean': float(attn.mean()),
                    'std': float(attn.std()),
                    'min': float(attn.min()),
                    'max': float(attn.max()),
                    'shape': list(attn.shape)
                })
        
        # 解码器统计
        for layer_idx, layer_attn in enumerate(attention_weights['decoder']):
            if 'self_attention' in layer_attn:
                attn = layer_attn['self_attention'][self.batch_idx].numpy()
                stats['decoder'].append({
                    'layer': layer_idx,
                    'type': 'self_attention',
                    'mean': float(attn.mean()),
                    'std': float(attn.std()),
                    'min': float(attn.min()),
                    'max': float(attn.max()),
                    'shape': list(attn.shape)
                })
            
            if 'cross_attention' in layer_attn:
                attn = layer_attn['cross_attention'][self.batch_idx].numpy()
                stats['decoder'].append({
                    'layer': layer_idx,
                    'type': 'cross_attention',
                    'mean': float(attn.mean()),
                    'std': float(attn.std()),
                    'min': float(attn.min()),
                    'max': float(attn.max()),
                    'shape': list(attn.shape)
                })
        
        # 保存到文件
        with open(save_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _generate_heatmaps(
        self, visualizer, attention_weights, 
        encoder_layers, decoder_layers, save_dir
    ):
        """生成热力图"""
        # 编码器自注意力
        for layer_idx in encoder_layers:
            if layer_idx < len(attention_weights['encoder']):
                layer_attn = attention_weights['encoder'][layer_idx]
                if 'self_attention' in layer_attn:
                    visualizer.plot_attention_heatmap(
                        layer_attn['self_attention'],
                        head_idx=None if self.heads_to_visualize == 'average' else self.heads_to_visualize,
                        batch_idx=self.batch_idx,
                        title=f'Encoder Layer {layer_idx} Self-Attention',
                        save_path=str(save_dir / f'encoder_layer{layer_idx}_self_attn.png'),
                        show=False
                    )
        
        # 解码器注意力
        for layer_idx in decoder_layers:
            if layer_idx < len(attention_weights['decoder']):
                layer_attn = attention_weights['decoder'][layer_idx]
                
                # 自注意力
                if 'self_attention' in layer_attn:
                    visualizer.plot_attention_heatmap(
                        layer_attn['self_attention'],
                        head_idx=None if self.heads_to_visualize == 'average' else self.heads_to_visualize,
                        batch_idx=self.batch_idx,
                        title=f'Decoder Layer {layer_idx} Self-Attention',
                        save_path=str(save_dir / f'decoder_layer{layer_idx}_self_attn.png'),
                        show=False
                    )
                
                # 交叉注意力
                if 'cross_attention' in layer_attn:
                    visualizer.plot_attention_heatmap(
                        layer_attn['cross_attention'],
                        head_idx=None if self.heads_to_visualize == 'average' else self.heads_to_visualize,
                        batch_idx=self.batch_idx,
                        title=f'Decoder Layer {layer_idx} Cross-Attention',
                        save_path=str(save_dir / f'decoder_layer{layer_idx}_cross_attn.png'),
                        show=False
                    )
    
    def _generate_multi_head_plots(
        self, visualizer, attention_weights,
        encoder_layers, decoder_layers, save_dir
    ):
        """生成多头对比图"""
        # 只为第一层生成（避免生成太多图）
        if encoder_layers and 0 in encoder_layers:
            layer_attn = attention_weights['encoder'][0]
            if 'self_attention' in layer_attn:
                visualizer.plot_multi_head_attention(
                    layer_attn['self_attention'],
                    batch_idx=self.batch_idx,
                    title='Encoder Layer 0 - All Heads',
                    save_path=str(save_dir / 'encoder_layer0_all_heads.png'),
                    show=False
                )
    
    def _generate_layer_comparison(self, visualizer, attention_weights, save_dir):
        """生成层级对比图"""
        # 编码器层级对比
        if attention_weights['encoder']:
            visualizer.plot_layer_comparison(
                attention_weights['encoder'],
                attention_type='self_attention',
                batch_idx=self.batch_idx,
                head_idx=None if self.heads_to_visualize == 'average' else self.heads_to_visualize,
                title='Encoder Self-Attention Across Layers',
                save_path=str(save_dir / 'encoder_layer_comparison.png'),
                show=False
            )
        
        # 解码器交叉注意力层级对比
        if attention_weights['decoder']:
            visualizer.plot_layer_comparison(
                attention_weights['decoder'],
                attention_type='cross_attention',
                batch_idx=self.batch_idx,
                head_idx=None if self.heads_to_visualize == 'average' else self.heads_to_visualize,
                title='Decoder Cross-Attention Across Layers',
                save_path=str(save_dir / 'decoder_cross_layer_comparison.png'),
                show=False
            )
    
    def _save_raw_weights(self, attention_weights: Dict, save_dir: Path):
        """保存原始注意力权重"""
        torch.save(attention_weights, save_dir / 'attention_weights.pt')
    
    def _log_to_tensorboard(self, attention_weights: Dict, metadata: Dict, step: int):
        """记录到TensorBoard"""
        if self.tb_writer is None:
            return
        
        # 记录标量统计信息
        for layer_idx, layer_attn in enumerate(attention_weights['encoder']):
            if 'self_attention' in layer_attn:
                attn = layer_attn['self_attention'][self.batch_idx].numpy()
                self.tb_writer.add_scalar(
                    f'attention/encoder_layer{layer_idx}_mean',
                    attn.mean(),
                    step
                )
                self.tb_writer.add_scalar(
                    f'attention/encoder_layer{layer_idx}_std',
                    attn.std(),
                    step
                )
        
        for layer_idx, layer_attn in enumerate(attention_weights['decoder']):
            if 'cross_attention' in layer_attn:
                attn = layer_attn['cross_attention'][self.batch_idx].numpy()
                self.tb_writer.add_scalar(
                    f'attention/decoder_layer{layer_idx}_cross_mean',
                    attn.mean(),
                    step
                )
        
        # 记录注意力图像（可选）
        try:
            import matplotlib.pyplot as plt
            from agent.models.transformer import AttentionVisualizer
            
            visualizer = AttentionVisualizer()
            
            # 只记录第一层以节省空间
            if attention_weights['encoder']:
                layer_attn = attention_weights['encoder'][0]
                if 'self_attention' in layer_attn:
                    # 创建图像
                    fig, ax = visualizer.plot_attention_heatmap(
                        layer_attn['self_attention'],
                        batch_idx=self.batch_idx,
                        show=False
                    )
                    
                    # 转换为numpy数组
                    fig.canvas.draw()
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    
                    # 记录到TensorBoard (HWC -> CHW)
                    self.tb_writer.add_image(
                        'attention/encoder_layer0_self',
                        image.transpose(2, 0, 1),
                        step
                    )
                    
                    plt.close(fig)
        except:
            pass  # 如果失败就跳过图像记录
    
    def close(self):
        """关闭监控器，等待所有任务完成"""
        print(f"\nClosing TrainingAttentionMonitor...")
        print(f"  - Total saves: {self.save_count}")
        
        # 等待队列清空
        self.save_queue.join()
        
        # 停止工作线程
        for _ in self.workers:
            self.save_queue.put(None)
        
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        # 关闭TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        print("TrainingAttentionMonitor closed.")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LightweightAttentionMonitor:
    """
    轻量级注意力监控器
    
    只保存统计信息，不生成可视化图像，更适合长时间训练。
    """
    
    def __init__(
        self,
        save_dir: str = './attention_stats',
        save_frequency: int = 100,
        use_tensorboard: bool = False,
        tensorboard_log_dir: Optional[str] = None,
    ):
        """
        初始化轻量级监控器
        
        Args:
            save_dir: 保存目录
            save_frequency: 保存频率
            use_tensorboard: 是否使用TensorBoard
            tensorboard_log_dir: TensorBoard日志目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_frequency = save_frequency
        self.stats_history = []
        
        # TensorBoard支持
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = tensorboard_log_dir or str(self.save_dir / 'tensorboard')
                self.tb_writer = SummaryWriter(tb_dir)
            except ImportError:
                self.use_tensorboard = False
        
        print(f"LightweightAttentionMonitor initialized: {self.save_dir}")
    
    def should_visualize(self, step: int) -> bool:
        """判断是否应该记录"""
        return step > 0 and step % self.save_frequency == 0
    
    def log_attention(
        self,
        model,
        enc: Dict[str, torch.Tensor],
        dec: Dict[str, torch.Tensor],
        step: int,
        loss: Optional[float] = None
    ):
        """记录注意力统计信息（非阻塞，快速）"""
        if not self.should_visualize(step):
            return
        
        was_training = model.training
        model.eval()
        
        # 快速收集注意力
        model.enable_attention_storage()
        with torch.no_grad():
            try:
                _, attention_weights = model(enc, dec, return_attention_weights=True)
            except:
                model.disable_attention_storage()
                if was_training:
                    model.train()
                return
        
        model.disable_attention_storage()
        if was_training:
            model.train()
        
        # 快速计算统计信息（在GPU上）
        stats = {
            'step': step,
            'loss': loss,
            'encoder': {},
            'decoder': {}
        }
        
        for layer_idx, layer_attn in enumerate(attention_weights['encoder']):
            if 'self_attention' in layer_attn:
                attn = layer_attn['self_attention'][0]  # 第一个样本
                stats['encoder'][f'layer_{layer_idx}'] = {
                    'mean': float(attn.mean().cpu()),
                    'std': float(attn.std().cpu()),
                }
        
        for layer_idx, layer_attn in enumerate(attention_weights['decoder']):
            if 'cross_attention' in layer_attn:
                attn = layer_attn['cross_attention'][0]
                stats['decoder'][f'layer_{layer_idx}_cross'] = {
                    'mean': float(attn.mean().cpu()),
                    'std': float(attn.std().cpu()),
                }
        
        # 记录到历史
        self.stats_history.append(stats)
        
        # TensorBoard记录
        if self.use_tensorboard and self.tb_writer is not None:
            for key, value in stats['encoder'].items():
                self.tb_writer.add_scalar(f'attention/encoder_{key}_mean', value['mean'], step)
            for key, value in stats['decoder'].items():
                self.tb_writer.add_scalar(f'attention/decoder_{key}_mean', value['mean'], step)
        
        # 定期保存到文件
        if len(self.stats_history) % 10 == 0:
            self._save_history()
        
        # 立即释放
        del attention_weights
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _save_history(self):
        """保存历史统计"""
        with open(self.save_dir / 'attention_stats.json', 'w') as f:
            json.dump(self.stats_history, f, indent=2)
    
    def close(self):
        """关闭监控器"""
        self._save_history()
        if self.tb_writer is not None:
            self.tb_writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
