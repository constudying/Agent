import os
import torch
import json
from collections import defaultdict


class Monitor:
    """
    A monitor class to capture intermediate outputs from specified layers in a PyTorch model
    during training, without modifying the model or training script core logic.

    Features:
    - Aliases for layers to improve readability.
    - Support for 'single' (specific batches) and 'epoch_full' (entire epoch) recording.
    - Data is stored in memory during epoch, saved manually after processing.

    Data Format:
    - single_data/epoch_data: {alias: {'layer': str, 'alias': str, 'output_shape': list, 'data': {(epoch, batch_idx): output, ...}}}

    Usage:
    - Initialize with model, layers to monitor, epochs to record, output directory, and optional batch_indices.
    - Call set_epoch(epoch) at the start of each epoch.
    - Call set_batch_idx(batch_idx) before each batch.
    - After epoch, use get_single_data() and get_epoch_data() for custom processing (data keyed by alias).
    - Call save_single_outputs() and save_epoch_outputs() to save data.
    - Call clear_current_epoch_data() to free memory.
    - Call remove_hooks() when done.
    """

    def __init__(self, model, layers_to_monitor, epochs_to_record, output_dir, batch_indices=None):
        """
        Args:
            model: The PyTorch model to monitor.
            layers_to_monitor: List of layer names or dict with details.
                - If list: ['layer1', 'layer2'] (aliases default to layer names)
                - If dict: {'layer1': {'alias': 'attention_output', 'record_type': 'single'}, 'layer2': {'alias': None, 'record_type': 'epoch_full'}}
                  Aliases must be unique; if None, layer name is used as alias.
            epochs_to_record: List of epoch numbers to record outputs for.
            output_dir: Directory to save the recorded outputs.
            batch_indices: Optional list of batch indices to record (if batches are sequential).
        """
        self.model = model
        self.epochs_to_record = set(epochs_to_record) if epochs_to_record else set()
        self.output_dir = output_dir
        self.batch_indices = set(batch_indices) if batch_indices else None
        self.current_epoch = None
        self.current_batch_idx = 0
        self.hooks = []
        self.single_data = {}  # {alias: {'layer': str, 'alias': str, 'output_shape': list, 'data': {(epoch, batch_idx): output, ...}}}
        self.epoch_data = {}  # {alias: {'layer': str, 'alias': str, 'output_shape': list, 'data': {(epoch, batch_idx): output, ...}}}

        # Parse layers_to_monitor
        self.layer_configs = self._parse_layers(layers_to_monitor)

        # Validate configuration
        self._validate_config()

        # Validate that all specified layers exist in the model
        self._validate_layers_exist()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Register hooks
        self._register_hooks()

    def _parse_layers(self, layers_to_monitor):
        """Parse layers_to_monitor into a dict of configs."""
        configs = {}
        aliases = set()
        if isinstance(layers_to_monitor, list):
            for layer in layers_to_monitor:
                alias = None
                if alias in aliases:
                    raise ValueError(f"Duplicate alias '{alias}' in layers_to_monitor.")
                aliases.add(alias)
                configs[layer] = {'alias': alias, 'record_type': 'single'}
        elif isinstance(layers_to_monitor, dict):
            for layer, config in layers_to_monitor.items():
                alias = config.get('alias', None)
                if alias in aliases:
                    raise ValueError(f"Duplicate alias '{alias}' in layers_to_monitor.")
                aliases.add(alias)
                record_type = config.get('record_type', 'single')
                if record_type not in ['single', 'epoch_full']:
                    raise ValueError(f"Invalid record_type: {record_type}. Must be 'single' or 'epoch_full'.")
                configs[layer] = {'alias': alias, 'record_type': record_type}
        else:
            raise ValueError("layers_to_monitor must be a list or dict.")
        return configs

    def _validate_config(self):
        """Validate the configuration based on record_types."""
        if not self.epochs_to_record:
            raise ValueError("epochs_to_record cannot be empty. At least one epoch must be specified for recording.")

        has_single = any(config['record_type'] == 'single' for config in self.layer_configs.values())
        has_epoch_full = any(config['record_type'] == 'epoch_full' for config in self.layer_configs.values())

        if has_single:
            if self.batch_indices is None or not self.batch_indices:
                raise ValueError("When any layer has record_type='single', batch_indices must be specified and non-empty.")

    def _validate_layers_exist(self):
        """Validate that all specified layers exist in the model."""
        all_modules = set(dict(self.model.named_modules()).keys())
        missing_layers = []
        for layer_name in self.layer_configs.keys():
            if layer_name not in all_modules:
                missing_layers.append(layer_name)
        
        if missing_layers:
            available_layers = sorted(all_modules)
            available_str = "\n".join(f"  - {layer}" for layer in available_layers)
            raise ValueError(
                f"Available layers in the model:\n{available_str}\n"
                f"The following layers specified in 'layers_to_monitor' do not exist in the model: {missing_layers}."
            )

    def _register_hooks(self):
        """Register forward hooks on the specified layers."""
        named_modules = dict(self.model.named_modules())
        for layer_name, config in self.layer_configs.items():
            module = named_modules[layer_name]
            hook = module.register_forward_hook(self._hook_fn(layer_name, config))
            self.hooks.append(hook)

    def _hook_fn(self, layer_name, config):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            if self.current_epoch in self.epochs_to_record:
                record_type = config['record_type']
                alias = config['alias'] or layer_name
                # Handle tuple outputs (e.g., MultiheadAttention returns (attn_output, attn_output_weights))
                if isinstance(output, tuple):
                    output_shape = [list(o.shape) if hasattr(o, 'shape') else None for o in output]
                    processed_output = tuple(o.detach().cpu() for o in output)
                else:
                    output_shape = list(output.shape) if hasattr(output, 'shape') else None
                    processed_output = output.detach().cpu()
                
                key = (self.current_epoch, self.current_batch_idx)
                
                if record_type == 'single':
                    if self.batch_indices is None or self.current_batch_idx in self.batch_indices:
                        if alias not in self.single_data:
                            self.single_data[alias] = {
                                'layer': layer_name,
                                'alias': config['alias'],
                                'output_shape': output_shape,
                                'data': {}
                            }
                        self.single_data[alias]['data'][key] = processed_output
                elif record_type == 'epoch_full':
                    if alias not in self.epoch_data:
                        self.epoch_data[alias] = {
                            'layer': layer_name,
                            'alias': config['alias'],
                            'output_shape': output_shape,
                            'data': {}
                        }
                    self.epoch_data[alias]['data'][key] = processed_output
        return hook

    def _save_output(self, output_data):
        """Save a single output to disk."""
        epoch = output_data['epoch']
        batch_idx = output_data['batch_idx']
        layer = output_data['layer']
        alias = output_data['alias'] or layer
        filename = f"epoch_{epoch}_batch_{batch_idx}_{alias}.pt"
        filepath = os.path.join(self.output_dir, filename)
        torch.save(output_data, filepath)

    def set_epoch(self, epoch):
        """Set the current epoch and clear previous data."""
        self.current_epoch = epoch
        self.current_batch_idx = 0
        # Clear data dicts
        for alias in self.single_data:
            self.single_data[alias]['data'].clear()
        for alias in self.epoch_data:
            self.epoch_data[alias]['data'].clear()

    def set_batch_idx(self, batch_idx):
        """Set the current batch index."""
        self.current_batch_idx = batch_idx

    def get_single_data(self, layers=None):
        """Get a copy of the single data for specified layers (by alias, or all if None)."""
        if layers is None:
            layers = self.single_data.keys()
        return {layer: dict(self.single_data[layer]) for layer in layers if layer in self.single_data}

    def get_epoch_data(self, layers=None):
        """Get a copy of the epoch data for specified layers (by alias, or all if None)."""
        if layers is None:
            layers = self.epoch_data.keys()
        return {layer: dict(self.epoch_data[layer]) for layer in layers if layer in self.epoch_data}

    def save_single_outputs(self, layers=None):
        """Save single outputs for specified layers (by alias, or all if None)."""
        if layers is None:
            layers = self.single_data.keys()
        for layer in layers:
            if layer in self.single_data:
                layer_info = self.single_data[layer]
                for (epoch, batch_idx), output in layer_info['data'].items():
                    output_data = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'layer': layer_info['layer'],
                        'alias': layer_info['alias'],
                        'output_shape': layer_info['output_shape'],
                        'output': output
                    }
                    self._save_output(output_data)

    def save_epoch_outputs(self, layers=None):
        """Save epoch outputs for specified layers (by alias, or all if None)."""
        if layers is None:
            layers = self.epoch_data.keys()
        for layer in layers:
            if layer in self.epoch_data:
                layer_info = self.epoch_data[layer]
                for (epoch, batch_idx), output in layer_info['data'].items():
                    output_data = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'layer': layer_info['layer'],
                        'alias': layer_info['alias'],
                        'output_shape': layer_info['output_shape'],
                        'output': output
                    }
                    self._save_output(output_data)

    def clear_current_epoch_data(self):
        """Clear current epoch data to free memory."""
        for alias in self.single_data:
            self.single_data[alias]['data'].clear()
        for alias in self.epoch_data:
            self.epoch_data[alias]['data'].clear()

    def create_epoch_callback(self):
        """Create a callback function for epoch setting."""
        def epoch_callback(epoch):
            self.set_epoch(epoch)
        return epoch_callback

    def create_batch_callback(self):
        """Create a callback function for batch setting."""
        def batch_callback(batch_idx):
            self.set_batch_idx(batch_idx)
        return batch_callback

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []