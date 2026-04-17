"""
model.py — 藏文手写识别 CNN 模型定义

DigitCNN  — 输入 1×28×28，输出 10 类（藏文数字 ༠–༩）
LetterCNN — 输入 1×64×64，输出 30 类（藏文字母 ཀ–ཨ）

使用工厂函数 get_model(mode) 获取对应模型实例。
"""

import torch
import torch.nn as nn
from typing import Optional


class DigitCNN(nn.Module):
    """
    用于藏文数字识别的轻量 CNN，输入 1×28×28 灰度图，输出 10 类。

    网络结构：
        卷积块 1：Conv(1→32, 3×3) → BN → ReLU → MaxPool(2)   28×28 → 14×14
        卷积块 2：Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)  14×14 → 7×7
        分类头：  Flatten → FC(3136→128) → Dropout(0.5) → FC(128→类别数)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 特征提取部分：两层卷积 + BatchNorm + 最大池化
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28×28 → 14×14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 14×14 → 7×7
        )
        # 分类头：全连接层 + Dropout 防止过拟合
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class LetterCNN(nn.Module):
    """
    用于藏文字母识别的深层 CNN，输入 1×64×64 灰度图，输出 30 类。

    相比 DigitCNN 增加了第三个卷积块和 BatchNorm，适应更大输入和更多类别。

    网络结构：
        卷积块 1：Conv(1→32,  3×3) → BN → ReLU → MaxPool(2)   64×64 → 32×32
        卷积块 2：Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)   32×32 → 16×16
        卷积块 3：Conv(64→128,3×3) → BN → ReLU → MaxPool(2)   16×16 → 8×8
        分类头：  Flatten → FC(8192→256) → Dropout(0.5) → FC(256→类别数)
    """

    def __init__(self, num_classes: int = 30):
        super().__init__()
        # 三层卷积块，逐层提升感受野；BatchNorm 加速收敛、稳定训练
        self.features = nn.Sequential(
            # 卷积块 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 64×64 → 32×32
            # 卷积块 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32×32 → 16×16
            # 卷积块 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16×16 → 8×8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def get_model(mode: str, num_classes: Optional[int] = None) -> nn.Module:
    """
    模型工厂函数，根据训练模式返回对应的模型实例。

    Args:
        mode:        "digit" 返回 DigitCNN，"letter" 返回 LetterCNN
        num_classes: 覆盖默认类别数（digit 默认 10，letter 默认 30）

    Returns:
        未加载权重的模型实例（需通过训练或加载 checkpoint 初始化权重）
    """
    if mode == "digit":
        return DigitCNN(num_classes=num_classes or 10)
    elif mode == "letter":
        return LetterCNN(num_classes=num_classes or 30)
    else:
        raise ValueError(f"mode 必须为 'digit' 或 'letter'，当前值：'{mode}'")
