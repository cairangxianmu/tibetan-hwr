"""
eval_digit.py — 对数字模型做离线准确率评估

测试两条预处理路径：
  A. val_transform  ：Grayscale → Resize → GaussianBinarize → ToTensor → Normalize
                      （与训练验证集完全一致）
  B. web_transform  ：先 tight_crop，再走同样的 val_transform
                      （与 web/app.py 推理路径完全一致）

用法：
    python eval_digit.py
    python eval_digit.py --data-root ../dataset/TibetanMNIST28x28
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import DIGIT_DIR, get_inference_transform
from model import get_model

REPO_ROOT      = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoint"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# tight_crop（与 web/app.py 保持一致）
# --------------------------------------------------------------------------- #
def _tight_crop(img_gray: Image.Image, pad_ratio: float = 0.15) -> Image.Image:
    arr  = np.array(img_gray)
    mask = arr < 200
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img_gray
    rmin = int(np.where(rows)[0][0]);  rmax = int(np.where(rows)[0][-1])
    cmin = int(np.where(cols)[0][0]);  cmax = int(np.where(cols)[0][-1])
    char_h = rmax - rmin;  char_w = cmax - cmin
    pad    = max(4, int(max(char_h, char_w) * pad_ratio))
    H, W   = arr.shape
    rmin = max(0, rmin - pad);  rmax = min(H - 1, rmax + pad)
    cmin = max(0, cmin - pad);  cmax = min(W - 1, cmax + pad)
    cropped = img_gray.crop((cmin, rmin, cmax + 1, rmax + 1))
    cw, ch  = cropped.size
    side    = max(cw, ch)
    square  = Image.new("L", (side, side), 255)
    square.paste(cropped, ((side - cw) // 2, (side - ch) // 2))
    return square


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, data_root: Path, use_tight_crop: bool) -> tuple[float, int, int]:
    tfm   = get_inference_transform("digit")
    total = correct = 0

    for cls_dir in sorted(data_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        try:
            label = int(cls_dir.name)
        except ValueError:
            continue

        for img_path in cls_dir.glob("*.png"):
            img_gray = Image.open(img_path).convert("L")

            if use_tight_crop:
                img_gray = _tight_crop(img_gray)

            img    = img_gray.convert("RGB")
            tensor = tfm(img).unsqueeze(0).to(DEVICE)

            logits = model(tensor)
            pred   = int(logits.argmax(dim=1).item())

            correct += int(pred == label)
            total   += 1

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ckpt",      default=None,
                        help="checkpoint 路径，默认 checkpoint/digit_best.pth")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DIGIT_DIR
    ckpt_path = Path(args.ckpt) if args.ckpt else CHECKPOINT_DIR / "digit_best.pth"

    print(f"设备：{DEVICE}")
    print(f"数据集：{data_root}")
    print(f"模型：{ckpt_path}\n")

    ckpt        = torch.load(ckpt_path, map_location=DEVICE)
    num_classes = ckpt["num_classes"]
    model       = get_model("digit", num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"已加载模型（num_classes={num_classes}，训练 val_acc={ckpt.get('val_acc', '?'):.2f}%）\n")

    print("── A. 纯验证集预处理（无 tight_crop）──────────────────────────────")
    acc_a, cor_a, tot_a = evaluate(model, data_root, use_tight_crop=False)
    print(f"   准确率：{acc_a:.2f}%  ({cor_a}/{tot_a})\n")

    print("── B. Web 推理预处理（含 tight_crop）──────────────────────────────")
    acc_b, cor_b, tot_b = evaluate(model, data_root, use_tight_crop=True)
    print(f"   准确率：{acc_b:.2f}%  ({cor_b}/{tot_b})\n")

    delta = acc_a - acc_b
    if abs(delta) > 0.5:
        worse = "B（tight_crop）" if delta > 0 else "A（无 tight_crop）"
        print(f"差异：{abs(delta):.2f}%，{worse} 更差 — tight_crop 对数据集原图有负面影响")
    else:
        print(f"差异：{abs(delta):.2f}%，两条路径基本一致")


if __name__ == "__main__":
    main()
