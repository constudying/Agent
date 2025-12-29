"""
Monitor集成示例：如何在封装的训练函数中使用Monitor
"""

# 示例1：修改训练函数接受回调参数
def example_modified_train_function(config, device, **kwargs):
    """
    修改后的训练函数，接受回调参数
    """
    # 从kwargs中获取回调函数
    epoch_callback = kwargs.get('epoch_callback')
    batch_callback = kwargs.get('batch_callback')

    # 训练循环
    for epoch in range(1, config.train.num_epochs + 1):
        # 调用epoch回调
        if epoch_callback:
            epoch_callback(epoch)

        for batch_idx, batch in enumerate(config.dataloader):
            # 调用batch回调
            if batch_callback:
                batch_callback(batch_idx)

            # 正常的训练逻辑
            # ...

        # epoch结束后的Monitor处理
        if hasattr(config, 'monitor') and config.monitor:
            # 获取数据进行自定义处理
            single_data = config.monitor.get_single_data()
            epoch_data = config.monitor.get_epoch_data()

            # 您的自定义处理逻辑
            # process_monitor_data(single_data, epoch_data)

            # 保存数据
            config.monitor.save_single_outputs()
            config.monitor.save_epoch_outputs()

            # 清理内存
            config.monitor.clear_current_epoch_data()


# 示例2：使用包装器
def example_with_wrapper():
    """
    使用MonitoredTrainer包装器的示例
    """
    from agent.scripts.monitor import Monitor
    from agent.scripts.training_wrapper import MonitoredTrainer

    # 创建Monitor
    monitor = Monitor(
        model=config.model,
        layers_to_monitor=config.monitor_layers,
        epochs_to_record=config.monitor_epochs,
        output_dir=config.monitor_output_dir,
        batch_indices=config.monitor_batch_indices
    )

    # 创建包装器
    trainer = MonitoredTrainer(monitor)

    # 包装原始训练函数
    original_train = lambda config, device, **kwargs: train(config, device, **kwargs)
    monitored_train = trainer.wrap_training_function(original_train)

    # 调用包装后的训练函数
    monitored_train(config, device)


# 示例3：使用装饰器
def example_with_decorator():
    """
    使用装饰器的示例
    """
    from agent.scripts.monitor import Monitor
    from agent.scripts.training_wrapper import patch_training_function

    # 创建Monitor
    monitor = Monitor(...)

    # 使用装饰器包装训练函数
    @patch_training_function(monitor)
    def my_training_function(config, device, **kwargs):
        # 正常的训练逻辑
        for epoch in range(1, config.train.num_epochs + 1):
            # Monitor会自动设置epoch（通过回调）
            for batch_idx, batch in enumerate(config.dataloader):
                # Monitor会自动设置batch_idx（通过回调）
                # 训练逻辑...
                pass

            # Monitor数据处理会自动进行（如果在包装器中实现）

    # 调用包装后的函数
    my_training_function(config, device)


# 示例4：最简单的集成方式 - 直接修改调用点
def example_simple_integration():
    """
    最简单的集成方式：在训练脚本的调用点添加Monitor
    """
    from agent.scripts.monitor import Monitor

    # 在训练前创建Monitor
    monitor = Monitor(
        model=model,
        layers_to_monitor=monitor_config,
        epochs_to_record=epochs_to_record,
        output_dir=output_dir,
        batch_indices=batch_indices
    )

    # 修改训练循环调用
    try:
        for epoch in range(1, num_epochs + 1):
            monitor.set_epoch(epoch)

            # 如果训练函数是封装的，但您可以控制循环
            for batch_idx in range(num_batches):
                monitor.set_batch_idx(batch_idx)

                # 调用封装的训练函数的一个batch
                train_one_batch(batch_idx)

            # epoch结束处理
            single_data = monitor.get_single_data()
            epoch_data = monitor.get_epoch_data()
            # 自定义处理...
            monitor.save_single_outputs()
            monitor.save_epoch_outputs()
            monitor.clear_current_epoch_data()

    finally:
        monitor.remove_hooks()


if __name__ == "__main__":
    print("Monitor集成示例")
    print("请选择合适的集成方式根据您的训练函数封装情况")