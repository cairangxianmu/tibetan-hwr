"""
utils.py — 图像处理公共工具

提供三个独立的批处理工具：
    rename_images   批量重命名图像文件
    replace_red     将红色像素替换为白色（消除格线残留）
    resize_images   批量缩放图像到指定尺寸

均可作为 CLI 工具独立运行：
    python utils.py rename  --dir ./data
    python utils.py replace --dir ./data
    python utils.py resize  --dir ./data --size 64
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


# ── rename_images ─────────────────────────────────────────────────────────────

def rename_images(root_dir, pattern: str = "{folder}_{index:04d}{suffix}") -> int:
    """
    将目录下每个子文件夹的图像按顺序重命名。

    命名格式：{子文件夹名}_{序号}{后缀}，例如 ཀ_0001.jpg

    Args:
        root_dir: 包含类别子文件夹的根目录
        pattern:  Python format 字符串，支持 {folder}, {index}, {suffix}

    Returns:
        重命名的文件总数
    """
    root_dir = Path(root_dir)
    total = 0
    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir():
            continue
        images = sorted(
            f for f in folder.iterdir() if f.suffix.lower() in _IMAGE_SUFFIXES
        )
        for idx, img_path in enumerate(images, start=1):
            new_name = pattern.format(
                folder=folder.name, index=idx, suffix=img_path.suffix
            )
            new_path = folder / new_name
            if img_path != new_path:
                img_path.rename(new_path)
                logger.debug("重命名：%s → %s", img_path.name, new_name)
                total += 1
    logger.info("重命名完成：共 %d 个文件", total)
    return total


# ── replace_red ───────────────────────────────────────────────────────────────

def replace_red(root_dir, threshold: int = 220) -> int:
    """
    将图像中红色像素（R 通道 > threshold 且 G、B 较低）替换为白色。

    用于消除字母图像中残留的红色格线。

    Args:
        root_dir:  包含图像（可多级目录）的根目录，原地修改
        threshold: 红色 R 通道阈值

    Returns:
        处理的图像数量
    """
    root_dir = Path(root_dir)
    count = 0
    for img_path in sorted(root_dir.rglob("*")):
        if img_path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        try:
            img = Image.open(img_path).convert("RGBA")
            pixels = img.load()
            w, h = img.size
            modified = False
            for y in range(h):
                for x in range(w):
                    r, g, b, a = pixels[x, y]
                    if r > threshold and g < 100 and b < 100:
                        pixels[x, y] = (255, 255, 255, 255)
                        modified = True
            if modified:
                img.convert("RGB").save(img_path)
                logger.debug("去红：%s", img_path)
            count += 1
        except Exception as exc:
            logger.warning("处理 %s 失败：%s", img_path, exc)
    logger.info("去红完成：处理 %d 张图像", count)
    return count


# ── resize_images ─────────────────────────────────────────────────────────────

def resize_images(src_dir, dst_dir,
                  size: int = 64, keep_aspect: bool = False) -> int:
    """
    批量将图像缩放到 size×size（或保持宽高比后居中填充白色背景）。

    Args:
        src_dir:     源目录（保留子目录结构）
        dst_dir:     输出目录
        size:        目标边长（像素）
        keep_aspect: True 时保持宽高比，填充白色背景；False 时直接拉伸

    Returns:
        处理的图像数量
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    count = 0

    for src_path in sorted(src_dir.rglob("*")):
        if src_path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(src_path).convert("L")  # 灰度
            if keep_aspect:
                img.thumbnail((size, size), Image.LANCZOS)
                canvas = Image.new("L", (size, size), 255)
                offset = ((size - img.width) // 2, (size - img.height) // 2)
                canvas.paste(img, offset)
                canvas.save(dst_path)
            else:
                img.resize((size, size), Image.LANCZOS).save(dst_path)
            count += 1
            logger.debug("缩放：%s → %s", src_path, dst_path)
        except Exception as exc:
            logger.warning("处理 %s 失败：%s", src_path, exc)

    logger.info("缩放完成：处理 %d 张图像 → %s", count, dst_dir)
    return count


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="图像处理工具集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--verbose", "-v", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    # rename
    r = sub.add_parser("rename", help="批量重命名图像文件")
    r.add_argument("--dir", "-d", required=True, help="包含子文件夹的根目录")
    r.add_argument("--pattern", default="{folder}_{index:04d}{suffix}", help="命名格式")

    # replace
    rp = sub.add_parser("replace", help="将红色像素替换为白色")
    rp.add_argument("--dir", "-d", required=True, help="图像根目录（原地修改）")
    rp.add_argument("--threshold", type=int, default=220, help="R 通道阈值")

    # resize
    rs = sub.add_parser("resize", help="批量缩放图像")
    rs.add_argument("--src", required=True, help="源目录")
    rs.add_argument("--dst", required=True, help="输出目录")
    rs.add_argument("--size", type=int, default=64, help="目标边长（像素）")
    rs.add_argument("--keep-aspect", action="store_true", help="保持宽高比（白色填充）")

    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    if args.cmd == "rename":
        rename_images(args.dir, pattern=args.pattern)
    elif args.cmd == "replace":
        replace_red(args.dir, threshold=args.threshold)
    elif args.cmd == "resize":
        resize_images(args.src, args.dst, size=args.size, keep_aspect=args.keep_aspect)
