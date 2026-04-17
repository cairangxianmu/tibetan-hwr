"""
app.py — 藏文手写识别 Web 后端（FastAPI）

接口：
    GET  /           返回前端页面 index.html
    GET  /health     服务健康检查
    POST /predict    接收 base64 图像 + 识别模式，返回识别结果和 Top-5 候选

启动：
    uvicorn app:app --reload --port 8000
"""

import base64
import io
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

# 将 recognition/ 加入模块搜索路径，以便直接导入 dataset 和 model
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "recognition"))

from dataset import CHAR_MAPS, get_inference_transform  # noqa: E402
from model import get_model                             # noqa: E402

# checkpoint 目录与前端静态文件目录
CHECKPOINT_DIR = REPO_ROOT / "checkpoint"
STATIC_DIR     = Path(__file__).resolve().parent / "static"

# ---------------------------------------------------------------------------
# FastAPI 应用初始化
# ---------------------------------------------------------------------------
app = FastAPI(title="藏文手写识别", version="1.0.0")
# 挂载静态文件目录，前端 JS/CSS/HTML 均从此处提供
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# 模型懒加载缓存
# 首次收到某模式的请求时才加载对应模型，之后复用缓存，避免重复磁盘 IO
# ---------------------------------------------------------------------------
_model_cache: dict = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(mode: str) -> torch.nn.Module:
    """
    按需加载指定模式的模型（digit 或 letter）并缓存。
    checkpoint 文件路径：checkpoint/{mode}_best.pth
    """
    if mode in _model_cache:
        return _model_cache[mode]

    ckpt_path = CHECKPOINT_DIR / f"{mode}_best.pth"
    # ckpt_path = CHECKPOINT_DIR / f"{mode}_epoch0030.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{ckpt_path}\n"
            f"请先执行训练：cd recognition && python train.py --mode {mode}"
        )

    checkpoint  = torch.load(ckpt_path, map_location=_device)
    num_classes = checkpoint["num_classes"]
    model = get_model(mode, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()  # 切换为推理模式，关闭 Dropout 和 BatchNorm 的训练行为

    _model_cache[mode] = model
    return model


# ---------------------------------------------------------------------------
# 请求 / 响应数据模型
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    image: str                       # base64 编码的图像（支持 data URL 前缀）
    mode: Literal["digit", "letter"] # 识别模式


class PredictResponse(BaseModel):
    label:      int         # 预测类别索引
    character:  str         # 对应的藏文字符
    confidence: float       # 置信度（0–100 %）
    top5:       list        # Top-5 候选：[{label, character, confidence}, ...]


# ---------------------------------------------------------------------------
# 推理辅助函数
# ---------------------------------------------------------------------------

def _tight_crop(img_gray: Image.Image, pad_ratio: float = 0.15) -> Image.Image:
    """
    将灰度图裁剪到字符的紧边界框，并等比填充为正方形白色背景。

    训练数据中每张图像的字符几乎铺满整帧；推理时画板是 400×400，
    字符仅占中间一小块，直接缩放会导致字符极小，与训练分布不匹配。
    此函数消除该差异，使推理输入与训练数据分布一致。

    Args:
        img_gray:  L 模式灰度图（白底黑字，背景 ~255，笔迹 ~0）
        pad_ratio: 在紧边界框外再扩展的比例（相对于字符高/宽）
    Returns:
        裁剪并填充为正方形的灰度图；若图像全白则原样返回。
    """
    arr = np.array(img_gray)
    # 找笔迹像素（暗像素，阈值 200 兼容轻笔压）
    mask = arr < 200
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():          # 画板为空，直接返回
        return img_gray

    rmin, rmax = int(np.where(rows)[0][[0, -1]].tolist()[0]), \
                 int(np.where(rows)[0][[0, -1]].tolist()[1])
    cmin, cmax = int(np.where(cols)[0][[0, -1]].tolist()[0]), \
                 int(np.where(cols)[0][[0, -1]].tolist()[1])

    char_h = rmax - rmin
    char_w = cmax - cmin
    pad    = max(4, int(max(char_h, char_w) * pad_ratio))

    H, W = arr.shape
    rmin = max(0, rmin - pad)
    rmax = min(H - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(W - 1, cmax + pad)

    cropped = img_gray.crop((cmin, rmin, cmax + 1, rmax + 1))

    # 等比填充为正方形（白色背景），避免后续 Resize 拉伸变形
    cw, ch  = cropped.size
    side    = max(cw, ch)
    square  = Image.new("L", (side, side), 255)
    square.paste(cropped, ((side - cw) // 2, (side - ch) // 2))
    return square


def _preprocess(image_data: bytes, mode: str) -> torch.Tensor:
    """
    将原始图像字节解码并转换为模型输入张量（形状：1×1×H×W）。

    处理步骤：
        1. 解码图像 → 转灰度
        2. 紧边界框裁剪（消除大量空白，对齐训练数据分布）
        3. 转 RGB（兼容 get_inference_transform 中的 Grayscale 步骤）
        4. Resize → GaussianBinarize（σ=1 模糊 + Otsu）→ ToTensor → Normalize
    与训练预处理完全一致，保证推理输入分布不偏移。
    """
    img_gray = Image.open(io.BytesIO(image_data)).convert("L")
    img_gray = _tight_crop(img_gray)
    img      = img_gray.convert("RGB")
    tfm      = get_inference_transform(mode)
    tensor   = tfm(img).unsqueeze(0)
    return tensor.to(_device)


def _predict(tensor: torch.Tensor, model: torch.nn.Module, mode: str) -> PredictResponse:
    """执行前向推理，计算 softmax 概率并返回 Top-5 结果。"""
    char_map = CHAR_MAPS[mode]
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]  # 转换为概率分布

    # 取概率最高的 5 个类别
    top5_probs, top5_idx = probs.topk(5)
    top5 = [
        {
            "label":      int(idx),
            "character":  char_map.get(int(idx), str(int(idx))),
            "confidence": round(float(prob) * 100, 2),
        }
        for idx, prob in zip(top5_idx.tolist(), top5_probs.tolist())
    ]

    best = top5[0]
    return PredictResponse(
        label=best["label"],
        character=best["character"],
        confidence=best["confidence"],
        top5=top5,
    )


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def index():
    """返回前端单页面。"""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    """服务健康检查，返回当前推理设备信息。"""
    return {"status": "ok", "device": str(_device)}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    接收手写图像和识别模式，返回识别结果。

    图像以 base64 字符串传入，支持 data URL 格式（data:image/png;base64,...）。
    """
    # 去掉 data URL 前缀，解码 base64 为字节
    try:
        raw = req.image
        if "," in raw:
            raw = raw.split(",", 1)[1]
        image_bytes = base64.b64decode(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"图像 base64 解码失败：{exc}")

    # 懒加载对应模式的模型
    try:
        model = _load_model(req.mode)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # 预处理 + 推理
    try:
        tensor = _preprocess(image_bytes, req.mode)
        return _predict(tensor, model, req.mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"推理失败：{exc}")
