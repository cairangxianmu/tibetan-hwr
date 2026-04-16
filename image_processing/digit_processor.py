"""
digit_processor.py — 藏文数字图像提取

处理深色背景手写数字图像，通过轮廓检测定位并保存每个独立数字。

处理流程：
    1. HSV 颜色提取，分离深色笔迹区域
    2. 中值滤波，去除椒盐噪声
    3. 二值化，增强前景对比度
    4. 轮廓检测，通过面积阈值过滤噪点和页面边框
    5. 按轮廓包围框裁剪单个数字并保存

CLI 用法：
    python digit_processor.py --input /原始图片目录 --output /输出目录
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── 图像处理函数 ──────────────────────────────────────────────────────────────

def extract_dark_ink(img: np.ndarray) -> np.ndarray:
    """HSV 颜色提取：提取深色笔迹（黑色/深色墨水）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 低亮度区域 → 深色墨水
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 140]))
    return mask


def preprocess(img: np.ndarray) -> np.ndarray:
    """颜色提取 → 中值滤波 → 二值化。"""
    mask   = extract_dark_ink(img)
    mask   = cv2.medianBlur(mask, 5)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def find_digit_contours(binary: np.ndarray,
                        min_area: int = 200,
                        max_area: int = 200_000) -> list:
    """
    轮廓检测，过滤噪点（面积太小）和页面边框（面积太大）。

    Returns:
        过滤后的轮廓列表
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    return [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]


def crop_digits(binary: np.ndarray, contours: list,
                output_dir: Path, stem: str, padding: int = 17) -> int:
    """
    按轮廓裁剪数字区域并保存。

    Returns:
        保存的图片数量
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        y1 = max(0, y - padding)
        y2 = min(binary.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(binary.shape[1], x + w + padding)
        cell = binary[y1:y2, x1:x2]
        if cell.size == 0:
            continue
        out_path = output_dir / f"{stem}_{i:03d}.jpg"
        cv2.imwrite(str(out_path), cell)
        count += 1
    return count


# ── 单文件处理 ────────────────────────────────────────────────────────────────

def process_image(img_path: Path, output_dir: Path,
                  min_area: int = 200, max_area: int = 200_000,
                  padding: int = 17) -> int:
    """
    处理单张图像，返回保存的数字图片数量。
    失败时记录日志并返回 0。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning("无法读取图像：%s", img_path)
        return 0

    try:
        binary   = preprocess(img)
        contours = find_digit_contours(binary, min_area=min_area, max_area=max_area)
        count    = crop_digits(binary, contours, output_dir, stem=img_path.stem,
                               padding=padding)
        logger.debug("  %s → %d 个数字", img_path.name, count)
        return count
    except Exception as exc:
        logger.warning("处理 %s 失败：%s", img_path.name, exc)
        return 0


# ── 批量处理 ──────────────────────────────────────────────────────────────────

def process_folder(input_dir, output_dir,
                   suffix: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
                   min_area: int = 200, max_area: int = 200_000,
                   padding: int = 17) -> dict:
    """
    批量处理文件夹下的所有数字图像。

    目录结构约定（平铺或按类别分子文件夹均支持）：
        input_dir/
            img_001.jpg
            img_002.jpg
            ...
    或
        input_dir/
            class_0/  ...
            class_1/  ...

    Returns:
        {'total_images': int, 'total_digits': int, 'failed': int}
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    stats = {"total_images": 0, "total_digits": 0, "failed": 0}

    def _process_dir(src: Path, dst: Path):
        for item in sorted(src.iterdir()):
            if item.is_dir():
                _process_dir(item, dst / item.name)
            elif item.suffix.lower() in suffix:
                stats["total_images"] += 1
                n = process_image(item, dst, min_area=min_area,
                                  max_area=max_area, padding=padding)
                if n == 0:
                    stats["failed"] += 1
                stats["total_digits"] += n

    logger.info("开始处理：%s", input_dir)
    _process_dir(input_dir, output_dir)
    logger.info(
        "完成：%d 张图像，提取 %d 个数字，失败 %d 张",
        stats["total_images"], stats["total_digits"], stats["failed"]
    )
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="从手写数字扫描图中提取藏文数字",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",    "-i", required=True, help="输入图像目录")
    p.add_argument("--output",   "-o", required=True, help="输出目录")
    p.add_argument("--min-area", type=int, default=200,     help="轮廓最小面积（过滤噪点）")
    p.add_argument("--max-area", type=int, default=200_000, help="轮廓最大面积（过滤边框）")
    p.add_argument("--padding",  type=int, default=17,      help="裁剪时四边留白像素数")
    p.add_argument("--verbose",  "-v", action="store_true", help="显示详细日志")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )
    process_folder(
        args.input, args.output,
        min_area=args.min_area,
        max_area=args.max_area,
        padding=args.padding,
    )
