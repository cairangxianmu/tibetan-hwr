"""
letter_processor.py — 藏文字母图像提取

处理红色格线方格纸（8 行 × 12 列）的扫描图像，提取每格中的单个字母并保存。

处理流程：
    1. HSV 颜色提取，分离红色格线
    2. 中值滤波，去除细线噪声
    3. 霍夫直线检测，补全断裂格线
    4. 轮廓检测，定位最外层方框并获取偏转角度
    5. 旋转校正，将图像调整至水平
    6. 裁剪外框区域
    7. 等分切割方格，逐格保存字母图像

CLI 用法：
    python letter_processor.py --input /扫描图片目录 --output /输出目录
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── 图像处理函数 ──────────────────────────────────────────────────────────────

def separate_color_red(img: np.ndarray) -> np.ndarray:
    """HSV 颜色提取：提取红色格线（两段色相范围取并集）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 红色在 HSV 中分布在 0–10° 和 170–180° 两段
    mask1 = cv2.inRange(hsv, np.array([0, 43, 46]),   np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 43, 46]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask1, mask2)


def complete_lines(mask: np.ndarray) -> np.ndarray:
    """霍夫直线检测：在格线掩膜上补全断线。"""
    edges = cv2.Canny(mask, 20, 250)
    result = mask.copy()
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120,
        minLineLength=50, maxLineGap=150
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(result, (x1, y1), (x2, y2), 255, 2)
    return result


def find_outer_rect(mask: np.ndarray, min_h: int = 3500, min_w: int = 2500,
                    max_h: int = 4680, max_w: int = 3310):
    """
    轮廓检测：找到最外层方格边框，返回 (box, angle)。

    box  : 旋转矩形四个顶点坐标 (4×2 int array)
    angle: 矩形相对水平面的偏转角度（度）
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    matched = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        box_pts = cv2.boxPoints(rect)
        h = abs(box_pts[3, 1] - box_pts[1, 1])
        w = abs(box_pts[3, 0] - box_pts[1, 0])
        if min_h <= h <= max_h and min_w <= w <= max_w:
            matched.append((rect, box_pts))

    if not matched:
        raise ValueError("未找到符合尺寸条件的外框轮廓，请检查图像或阈值参数。")

    # 取面积最大的候选
    rect, box_pts = max(matched, key=lambda x: x[0][1][0] * x[0][1][1])
    angle = rect[2]
    # 统一转换到 [-45, 45] 范围
    if abs(angle) > 45:
        angle = 90 - abs(angle)
    return np.int32(box_pts), angle


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """绕中心旋转图像。"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def crop_outer(img: np.ndarray, box: np.ndarray, margin: int = 10) -> np.ndarray:
    """按旋转矩形顶点坐标裁剪外框区域（轴对齐裁剪）。"""
    x1, y1 = box[1]
    x2, y2 = box[3]
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 < x1:
        x1, x2 = x2, x1
    return img[y1 + margin: y2 - margin, x1 + margin: x2 - margin]


def split_grid(img: np.ndarray, output_dir: Path, stem: str,
               rows: int = 8, cols: int = 12, margin: int = 5) -> int:
    """
    将裁剪后的方格图按 rows×cols 等分切割，保存各格图像。

    Returns:
        保存成功的图片数量
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    cell_h = h // rows
    cell_w = w // cols
    count = 0
    for r in range(rows):
        for c in range(cols):
            y1 = r * cell_h + margin
            y2 = (r + 1) * cell_h - margin
            x1 = c * cell_w + margin
            x2 = (c + 1) * cell_w - margin
            cell = img[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            out_path = output_dir / f"{stem}_{r * cols + c:03d}.jpg"
            cv2.imwrite(str(out_path), cell)
            count += 1
    return count


# ── 单文件处理 ────────────────────────────────────────────────────────────────

def process_image(img_path: Path, output_dir: Path) -> int:
    """
    处理单张扫描图像，返回保存的字母图片数量。
    失败时记录日志并返回 0（不中断批量处理）。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning("无法读取图像：%s", img_path)
        return 0

    try:
        mask = separate_color_red(img)
        mask = cv2.medianBlur(mask, 19)
        mask = complete_lines(mask)
        box, angle = find_outer_rect(mask)

        rotated_mask = rotate_image(mask, angle)
        box, _ = find_outer_rect(rotated_mask)

        rotated_img = rotate_image(img, angle)
        cropped = crop_outer(rotated_img, box)
        count = split_grid(cropped, output_dir, stem=img_path.stem)
        logger.debug("  %s → %d 个字母", img_path.name, count)
        return count
    except Exception as exc:
        logger.warning("处理 %s 失败：%s", img_path.name, exc)
        return 0


# ── 批量处理 ──────────────────────────────────────────────────────────────────

def process_folder(input_dir, output_dir,
                   suffix: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> dict:
    """
    批量处理文件夹下的所有扫描图像。

    目录结构约定：
        input_dir/
            class_0/   scan_01.jpg  scan_02.jpg ...
            class_1/   ...
        →
        output_dir/
            class_0/   scan_01_000.jpg  scan_01_001.jpg ...
            class_1/   ...

    Args:
        input_dir:  包含各类别子文件夹的扫描图像根目录
        output_dir: 输出根目录
        suffix:     允许的图片后缀（小写）

    Returns:
        {'total_images': int, 'total_cells': int, 'failed': int}
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    stats = {"total_images": 0, "total_cells": 0, "failed": 0}

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        out_class = output_dir / class_dir.name
        logger.info("处理类别：%s", class_dir.name)

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in suffix:
                continue
            stats["total_images"] += 1
            n = process_image(img_path, out_class)
            if n == 0:
                stats["failed"] += 1
            stats["total_cells"] += n

    logger.info(
        "完成：%d 张扫描图，提取 %d 个字母，失败 %d 张",
        stats["total_images"], stats["total_cells"], stats["failed"]
    )
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="从红色格线方格纸扫描图中提取藏文字母",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True, help="扫描图像根目录（含类别子文件夹）")
    p.add_argument("--output", "-o", required=True, help="输出目录")
    p.add_argument("--rows",   type=int, default=8,  help="每张表单的行数")
    p.add_argument("--cols",   type=int, default=12, help="每张表单的列数")
    p.add_argument("--margin", type=int, default=5,  help="格子边距（像素）")
    p.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )
    process_folder(args.input, args.output)
