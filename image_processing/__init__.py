"""
image_processing — 藏文手写图像预处理模块

从扫描的纸质手写表单中提取单个字符图像。

子模块：
    letter_processor  提取字母（红色格线方格纸，30 类，8×12 格）
    digit_processor   提取数字（深色背景，10 类，轮廓检测）
    utils             公共工具（重命名、去红色线、批量缩放）

快速上手：
    from image_processing.letter_processor import process_folder as process_letters
    from image_processing.digit_processor  import process_folder as process_digits
"""

from .letter_processor import process_folder as process_letter_folder
from .digit_processor  import process_folder as process_digit_folder

__all__ = ["process_letter_folder", "process_digit_folder"]
