"""
dataset.py — 数据集加载与预处理

支持两种训练模式：
    digit  模式：加载 dataset/TibetanMNIST28x28/   （10 类数字，28×28 PNG）
    letter 模式：加载 dataset/TibetanLetter64x64/  （30 类字母，64×64 PNG）

目录结构约定：每个类别对应一个子文件夹，文件夹名即为类别标签（0, 1, 2, ...）。
"""

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# 默认数据集路径（相对于本文件的上两级目录自动推断）
_REPO_ROOT    = Path(__file__).resolve().parent.parent
_DATASET_ROOT = _REPO_ROOT / "dataset"

DIGIT_DIR  = _DATASET_ROOT / "TibetanMNIST28x28"
LETTER_DIR = _DATASET_ROOT / "TibetanLetter64x64"

# 类别索引 → 藏文字符的映射表
DIGIT_CHARS = {
    0: "༠", 1: "༡", 2: "༢", 3: "༣", 4: "༤",
    5: "༥", 6: "༦", 7: "༧", 8: "༨", 9: "༩",
}

LETTER_CHARS = {
    0: "ཀ",  1: "ཁ",  2: "ག",  3: "ང",  4: "ཅ",
    5: "ཆ",  6: "ཇ",  7: "ཉ",  8: "ཏ",  9: "ཐ",
    10: "ད", 11: "ན", 12: "པ", 13: "ཕ", 14: "བ",
    15: "མ", 16: "ཙ", 17: "ཚ", 18: "ཛ", 19: "ཝ",
    20: "ཞ", 21: "ཟ", 22: "འ", 23: "ཡ", 24: "ར",
    25: "ལ", 26: "ཤ", 27: "ས", 28: "ཧ", 29: "ཨ",
}

# 统一入口，供推理阶段查表使用
CHAR_MAPS = {"digit": DIGIT_CHARS, "letter": LETTER_CHARS}


def _build_transforms(mode: str, augment: bool = False):
    """
    构建图像预处理流水线。

    训练时启用数据增强（随机旋转 + 随机平移），验证和推理时只做标准化。
    digit 模式目标尺寸 28×28，letter 模式目标尺寸 64×64。
    """
    size = 28 if mode == "digit" else 64
    # 归一化到 [-1, 1]，与训练时保持一致
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))

    if augment:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转为单通道灰度图
            transforms.Resize((size, size)),
            transforms.RandomRotation(10),                # 随机旋转 ±10°
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 随机平移 5%
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders(
    mode: str,
    data_root: Optional[str] = None,
    batch_size: int = 64,
    val_split: float = 0.2,
    num_workers: int = 0,
) -> tuple:
    """
    构建训练集和验证集的 DataLoader。

    数据集按 80/20 比例随机划分（固定随机种子 42，保证可复现）。
    训练集启用数据增强，验证集仅做标准化。

    Args:
        mode:        训练模式，"digit" 或 "letter"
        data_root:   覆盖默认数据集目录（不传则自动推断）
        batch_size:  批大小
        val_split:   验证集比例
        num_workers: DataLoader 子进程数（Windows 建议设为 0）

    Returns:
        (train_loader, val_loader, num_classes)
    """
    if mode not in ("digit", "letter"):
        raise ValueError(f"mode 必须为 'digit' 或 'letter'，当前值：'{mode}'")

    root = Path(data_root) if data_root is not None else (
        DIGIT_DIR if mode == "digit" else LETTER_DIR
    )

    if not root.exists():
        raise FileNotFoundError(
            f"数据集目录不存在：{root}\n"
            "请将数据集放入 tibetan-hwr/dataset/ 目录。"
        )

    # 使用 ImageFolder，子文件夹名自动作为类别标签
    full_dataset = datasets.ImageFolder(
        root=str(root),
        transform=_build_transforms(mode, augment=True),
    )
    num_classes = len(full_dataset.classes)

    # 随机划分训练集和验证集，固定种子保证每次运行结果一致
    val_size   = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # 验证集单独构建无增强的 ImageFolder，确保评估结果不受数据增强干扰
    val_ds.dataset = datasets.ImageFolder(
        root=str(root),
        transform=_build_transforms(mode, augment=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # GPU 可用时启用内存钉页加速传输
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, num_classes


def get_inference_transform(mode: str) -> transforms.Compose:
    """返回推理阶段使用的确定性预处理流水线（无数据增强）。"""
    return _build_transforms(mode, augment=False)
