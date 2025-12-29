"""
训练函数包装器，用于在封装的训练函数中集成Monitor
"""

class MonitoredTrainer:
    """
    包装器类，用于在训练函数中集成Monitor设置
    """

    def __init__(self, monitor):
        self.monitor = monitor
        self.epoch_callback = monitor.create_epoch_callback()
        self.batch_callback = monitor.create_batch_callback()

    def wrap_training_function(self, original_train_fn):
        """
        包装训练函数，自动注入Monitor回调

        Args:
            original_train_fn: 原始训练函数，签名应为 train_fn(config, device, **kwargs)

        Returns:
            包装后的训练函数
        """
        def monitored_train_fn(config, device, **kwargs):
            # 在训练开始前可以做一些准备工作
            print("Starting monitored training...")

            # 调用原始训练函数，传入回调
            result = original_train_fn(
                config,
                device,
                epoch_callback=self.epoch_callback,
                batch_callback=self.batch_callback,
                **kwargs
            )

            # 在训练结束后清理
            self.monitor.remove_hooks()
            print("Monitored training completed.")

            return result

        return monitored_train_fn


def patch_training_function(original_train_fn):
    """
    修补现有训练函数，添加Monitor支持

    这是一个装饰器版本，可以直接应用于训练函数
    """
    def decorator(monitor):
        trainer = MonitoredTrainer(monitor)

        def wrapper(config, device, **kwargs):
            return trainer.wrap_training_function(original_train_fn)(config, device, **kwargs)

        return wrapper

    return decorator