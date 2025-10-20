import torch
from typing import Any, Dict, List, Tuple


def _iter_tensors(x: Any, prefix: str = ""):
    """
    Recursively iterate through nested structures and yield (path, tensor).

    Supports dict, list, tuple. Other objects are ignored.
    """
    if isinstance(x, torch.Tensor):
        yield prefix.rstrip("."), x
        return
    if isinstance(x, dict):
        for k, v in x.items():
            key = str(k)
            yield from _iter_tensors(v, f"{prefix}{key}.")
        return
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _iter_tensors(v, f"{prefix}{i}.")


def summarize_batch_devices(batch: Any) -> Dict[str, List[str]]:
    """
    Return a mapping: device_str -> list of tensor paths found in the batch.
    """
    by_device: Dict[str, List[str]] = {}
    for path, t in _iter_tensors(batch):
        dev = str(t.device)
        by_device.setdefault(dev, []).append(path)
    return by_device


def print_batch_device_report(batch: Any, header: str = "Batch device report", max_items_per_device: int = 20):
    """
    Print a concise device report for a nested batch structure.
    """
    by_device = summarize_batch_devices(batch)
    total = sum(len(v) for v in by_device.values())
    print(f"\n===== {header} =====")
    print(f"Total tensors found: {total}")
    if not by_device:
        print("(no tensors found)")
        return
    for dev, paths in sorted(by_device.items()):
        print(f"- Device {dev}: {len(paths)} tensors")
        for p in paths[:max_items_per_device]:
            print(f"    • {p}")
        if len(paths) > max_items_per_device:
            print(f"    … (+{len(paths) - max_items_per_device} more)")


def assert_batch_on_device(batch: Any, device: torch.device) -> List[Tuple[str, str]]:
    """
    Check that all tensors in the batch are on the provided device.
    Returns a list of (path, device_str) for offending tensors. Does not raise.
    """
    mismatches: List[Tuple[str, str]] = []
    want = str(device)
    for path, t in _iter_tensors(batch):
        have = str(t.device)
        if have != want:
            mismatches.append((path, have))
    return mismatches


def print_model_parameter_devices(module: torch.nn.Module, header: str = "Model parameter devices", max_items_per_device: int = 30):
    """
    Print device distribution of model parameters.
    """
    by_device: Dict[str, List[str]] = {}
    total = 0
    for name, p in module.named_parameters(recurse=True):
        dev = str(p.device)
        by_device.setdefault(dev, []).append(name)
        total += 1

    print(f"\n===== {header} =====")
    print(f"Total parameters: {total}")
    for dev, names in sorted(by_device.items()):
        print(f"- Device {dev}: {len(names)} params")
        for n in names[:max_items_per_device]:
            print(f"    • {n}")
        if len(names) > max_items_per_device:
            print(f"    … (+{len(names) - max_items_per_device} more)")


def print_optimizer_state_devices(optimizer: torch.optim.Optimizer, header: str = "Optimizer state devices"):
    """
    Inspect optimizer state tensors as they can also live on CPU by default.
    """
    by_device: Dict[str, int] = {}
    for group in optimizer.param_groups:
        for p in group.get("params", []):
            state = optimizer.state.get(p, {})
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    by_device[str(v.device)] = by_device.get(str(v.device), 0) + 1
    print(f"\n===== {header} =====")
    if not by_device:
        print("(no optimizer tensor state found)")
        return
    for dev, count in sorted(by_device.items()):
        print(f"- Device {dev}: {count} state tensors")
