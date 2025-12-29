#!/usr/bin/env python3
"""
HDF5 文件检查和操作工具

支持列出结构、检查属性、获取数据集和属性值，并输出到终端或文件。

使用示例：
  # 列出文件结构
  python hdf5_check.py -f file.h5 -ls

  # 检查属性
  python hdf5_check.py -f file.h5 -ca

  # 获取数据集值
  python hdf5_check.py -f file.h5 -gd dataset/path

  # 获取切片数据
  python hdf5_check.py -f file.h5 -gd dataset/path -s "0:10"

  # 获取属性值
  python hdf5_check.py -f file.h5 -ga path attr_name

  # 输出到文件
  python hdf5_check.py -f file.h5 -ls -tf output.txt

  # 同时输出到终端和文件
  python hdf5_check.py -f file.h5 -ca -tb results.txt
"""
# 另外，还通过 python hdf5_check.py -h 查看在命令行中使用hdf5_check.py的选项帮助信息

import h5py
import argparse
import sys
import os
from typing import List, Dict, Any


def parse_slice(slice_str: str):
    """解析切片字符串，支持一维和多维切片"""
    slice_str = slice_str.strip()
    if ',' in slice_str:
        # 多维切片，如 "0:10, :, 2:5"
        dims = slice_str.split(',')
        slices = []
        for dim in dims:
            dim = dim.strip()
            if dim == ':':
                slices.append(slice(None))
            else:
                parts = dim.split(':')
                if len(parts) > 3:
                    raise ValueError(f"切片格式错误: {dim}，应为 start:stop:step")
                start = int(parts[0]) if parts[0] else None
                stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
                step = int(parts[2]) if len(parts) > 2 and parts[2] else None
                slices.append(slice(start, stop, step))
        return tuple(slices)
    else:
        # 一维切片，如 "0:10" 或 "0:10:2"
        if slice_str == ':':
            return slice(None)
        parts = slice_str.split(':')
        if len(parts) > 3:
            raise ValueError(f"切片格式错误: {slice_str}，应为 start:stop:step")
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
        step = int(parts[2]) if len(parts) > 2 and parts[2] else None
        return slice(start, stop, step)


def list_structure(f: h5py.File, indent: int = 0) -> List[str]:
    """递归列出HDF5文件的结构索引"""
    structure = []
    for name, obj in f.items():
        prefix = "  " * indent
        if isinstance(obj, h5py.Group):
            structure.append(f"{prefix}Group: {name}")
            structure.extend(list_structure(obj, indent + 1))
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape if obj.shape else "scalar"
            dtype = obj.dtype
            structure.append(f"{prefix}Dataset: {name} (shape: {shape}, dtype: {dtype})")
    return structure


def check_attributes(f: h5py.File) -> Dict[str, List[str]]:
    """检查并列出所有对象的属性"""
    attributes = {}

    def visit_func(name, obj):
        attr_list = []
        if hasattr(obj, 'attrs') and obj.attrs:
            for attr_name in obj.attrs.keys():
                attr_list.append(attr_name)
        if attr_list:
            obj_type = "File" if name == "/" else ("Group" if isinstance(obj, h5py.Group) else "Dataset")
            attributes[f"{obj_type}: {name}"] = attr_list

    f.visititems(visit_func)
    # 检查文件级属性
    if f.attrs:
        attributes["File: /"] = list(f.attrs.keys())

    return attributes


def get_dataset_value(f: h5py.File, path: str, slice_obj=None) -> Any:
    """获取指定数据集的值，支持切片"""
    if path not in f:
        raise ValueError(f"数据集路径 '{path}' 不存在")
    obj = f[path]
    if not isinstance(obj, h5py.Dataset):
        raise ValueError(f"'{path}' 不是数据集")
    try:
        if slice_obj is not None:
            return obj[slice_obj]
        else:
            return obj[()]
    except (IndexError, TypeError) as e:
        raise ValueError(f"切片操作失败: {e}. 请检查切片格式是否与数据集维度匹配。")


def get_attribute_value(f: h5py.File, path: str, attr_name: str) -> Any:
    """获取指定对象的属性值"""
    if path == "/":
        obj = f
    elif path in f:
        obj = f[path]
    else:
        raise ValueError(f"对象路径 '{path}' 不存在")

    if attr_name not in obj.attrs:
        raise ValueError(f"属性 '{attr_name}' 在 '{path}' 中不存在")
    return obj.attrs[attr_name]


def output_result(result: str, output_mode: str, output_file: str = None):
    """根据选项输出结果"""
    if output_mode in ["terminal", "both"]:
        print(result)
    if output_mode in ["file", "both"]:
        if not output_file:
            print("错误: 指定输出到文件但未提供文件路径", file=sys.stderr)
            return
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"结果已写入文件: {output_file}")
        except Exception as e:
            print(f"写入文件失败: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="HDF5 文件检查和操作工具")
    parser.add_argument('-f', '--file', required=True, help="HDF5 文件路径")
    parser.add_argument('-ls', '--list-structure', action='store_true', help="列出文件结构索引")
    parser.add_argument('-ca', '--check-attributes', action='store_true', help="检查并列出所有属性")
    parser.add_argument('-gd', '--get-dataset', help="获取指定数据集的值 (提供数据集路径)")
    parser.add_argument('-s', '--slice', help="数据集切片 (与 -gd 一起使用，如 '0:10' 或 '0:10, :, 2:5')")
    parser.add_argument('-ga', '--get-attribute', nargs=2, metavar=('PATH', 'ATTR_NAME'),
                       help="获取指定对象的属性值 (提供对象路径和属性名)")
    parser.add_argument('-ts', '--stdout', action='store_true', help="输出到终端 (默认)")
    parser.add_argument('-tf', '--to-file', metavar='FILE', help="输出到文件 (提供文件路径)")
    parser.add_argument('-tb', '--both', metavar='FILE', help="输出到终端和文件 (提供文件路径)")

    args = parser.parse_args()

    # 检查逻辑错误
    if args.slice and not args.get_dataset:
        print("错误: 指定了切片选项 (-s/--slice) 但未指定获取数据集操作 (-gd/--get-dataset)。切片选项只能与获取数据集操作一起使用。", file=sys.stderr)
        sys.exit(1)

    # 检查输出选项
    if args.to_file and not args.to_file:
        print("错误: -tf/--to-file 需要提供文件路径。", file=sys.stderr)
        sys.exit(1)
    if args.both and not args.both:
        print("错误: -tb/--both 需要提供文件路径。", file=sys.stderr)
        sys.exit(1)

    # 检查文件是否存在
    if not os.path.exists(args.file):
        print(f"错误: 文件 '{args.file}' 不存在", file=sys.stderr)
        sys.exit(1)

    try:
        with h5py.File(args.file, 'r') as f:
            results = []

            if args.list_structure:
                structure = list_structure(f)
                results.append("=== HDF5 文件结构 ===\n" + "\n".join(structure))

            if args.check_attributes:
                attrs = check_attributes(f)
                if attrs:
                    attr_lines = []
                    for location, attr_list in attrs.items():
                        attr_lines.append(f"{location}: {', '.join(attr_list)}")
                    results.append("=== 属性检查 ===\n" + "\n".join(attr_lines))
                else:
                    results.append("=== 属性检查 ===\n未找到任何属性")

            if args.get_dataset:
                slice_obj = None
                if args.slice:
                    try:
                        slice_obj = parse_slice(args.slice)
                    except ValueError as e:
                        results.append(f"错误: 切片解析失败 - {e}")
                        slice_obj = None
                try:
                    value = get_dataset_value(f, args.get_dataset, slice_obj)
                    slice_desc = f" (切片: {args.slice})" if args.slice else ""
                    results.append(f"=== 数据集 '{args.get_dataset}' 的值{slice_desc} ===\n{value}")
                except ValueError as e:
                    results.append(f"错误: {e}")

            if args.get_attribute:
                path, attr_name = args.get_attribute
                try:
                    value = get_attribute_value(f, path, attr_name)
                    results.append(f"=== 对象 '{path}' 的属性 '{attr_name}' 的值 ===\n{value}")
                except ValueError as e:
                    results.append(f"错误: {e}")

            if not results:
                print("未指定任何操作。使用 -h 查看帮助。", file=sys.stderr)
                sys.exit(1)

            # 确定输出模式
            output_mode = 'terminal'  # 默认
            output_file = None
            if args.both:
                output_mode = 'both'
                output_file = args.both
            elif args.to_file:
                output_mode = 'file'
                output_file = args.to_file

            # 输出结果
            final_result = "\n\n".join(results)
            output_result(final_result, output_mode, output_file)

    except Exception as e:
        print(f"处理文件时出错: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()