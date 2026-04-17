---
title: TibetanHWR 系列三：CNN 训练与 Web 部署——手写识别在线演示实现
published: 2026-04-16
description: 藏文手写识别系列三：基于 TibetanCharacter 数据集，详解 DigitCNN / LetterCNN 两款模型的设计与训练方案，以及 FastAPI + Canvas 的 Web 演示应用搭建过程。
image: /assets/images/covers/python-opencv.svg
tags:
    [
        藏文手写识别,
        CNN,
        PyTorch,
        FastAPI,
        深度学习,
    ]
category: 项目
draft: false
---

> 本文是 TibetanCharacter 系列的第三篇，也是终章。前两篇分别解决了「数据从哪来」和「数据怎么清洗」的问题，本篇把数据喂进模型，再把模型搬进浏览器。
>
> - 系列一：[TibetanCharacter：藏文手写数字与字母数据集](/posts/tibetan-character/)
> - 系列二：[TibetanHWR 系列二：OpenCV 图像预处理——从红格纸扫描图到单字母图像](/posts/tibetan-hwr-preprocessing/)

## 一、整体架构

把「手写识别」拆成可以独立迭代的三个模块：数据、模型、服务。

```
dataset/                 ImageFolder 组织的 PNG 集合
   │
   ▼
recognition/             PyTorch 训练管线
   ├── dataset.py        DataLoader 工厂 + 字符映射表
   ├── model.py          DigitCNN / LetterCNN
   └── train.py          训练入口（日志 / 曲线 / 权重保存）
         │
         ▼
checkpoint/{mode}_best.pth
         │
         ▼
web/                     FastAPI + Canvas
   ├── app.py            懒加载模型 · /predict 接口
   └── static/           Canvas 画板 + 逐笔撤销
```

两个任务走**完全相同的管线**，只由 `--mode digit|letter` 这一个开关切换数据源、模型结构和类别数。这让代码可以大部分共用，同时各自有针对性的设计。

## 二、模型设计

### 2.1 为什么要两个网络

|              | 数字 | 字母 |
| :----------- | :--: | :--: |
| 输入尺寸     | 28×28 | 64×64 |
| 类别数       | 10    | 30    |
| 图像面积     | 1×    | 5.2×  |
| 字形复杂度   | 简单  | 较复杂 |

字母任务的**信息量**（面积 × 类别）是数字任务的十几倍，共用一套小网络会欠拟合，共用一套大网络对数字又是浪费。因此按任务复杂度分别设计。

### 2.2 DigitCNN（10 类）

LeNet 风格，两层卷积 + 两层全连接，参数量 **~420K**：

```
输入 1×28×28
  Conv(1→32, 3×3, pad=1) → ReLU → MaxPool(2)   →  32×14×14
  Conv(32→64, 3×3, pad=1) → ReLU → MaxPool(2)  →  64×7×7
  Flatten → FC(3136→128) → ReLU → Dropout(0.5)
  FC(128→10)
```

CPU 上单张推理 < 5 ms，训练 30 epoch 在笔记本上约 5 分钟。

### 2.3 LetterCNN（30 类）

在 DigitCNN 基础上增加第三个卷积块，并在每层卷积后加 **BatchNorm**，参数量 **~2.3M**：

```
输入 1×64×64
  Conv(1→32,  3×3) → BN → ReLU → MaxPool(2)   →  32×32×32
  Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)   →  64×16×16
  Conv(64→128,3×3) → BN → ReLU → MaxPool(2)   →  128×8×8
  Flatten → FC(8192→256) → ReLU → Dropout(0.5)
  FC(256→30)
```

BatchNorm 在这里做两件事：
- **加速收敛**——实测达到相同验证准确率所需 epoch 数减少约 30%；
- **稳定训练**——缓解内部协变量偏移，缩小训练 / 验证准确率差距。

两个模型都通过 `get_model(mode)` 工厂函数统一调用，训练、推理、权重加载全部复用同一套入口。

## 三、训练管线

### 3.1 数据加载

数据集按 `ImageFolder` 约定组织（子目录名即类别），通过 `get_dataloaders()` 一行获取 train/val DataLoader：

```python
train_loader, val_loader, num_classes = get_dataloaders(
    mode="letter",   # 或 "digit"
    batch_size=64,
    val_split=0.2,   # 固定随机种子 42，保证划分可复现
)
```

### 3.2 数据增强

藏文字符结构紧凑，增强幅度必须克制：

```python
transforms.RandomRotation(10)                            # ±10°
transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))  # 平移 5%
transforms.Normalize(mean=(0.5,), std=(0.5,))            # → [-1, 1]
```

**不做翻转**。这一点专门针对藏文：多个字母互为镜像（如 `ག / ད`），水平或垂直翻转会直接污染标签。这是「通用视觉增强」在专门领域需要让位于先验知识的典型案例。

### 3.3 优化与调度

```
优化器：Adam，lr=1e-3，weight_decay=1e-4
调度器：CosineAnnealingLR，T_max=epochs，eta_min=1e-6
损失：  CrossEntropyLoss
建议：  digit → 30 epoch，letter → 50 epoch（batch 128）
```

Cosine 退火的直觉：前期大 lr 快速下降到低损失盆地，后期小 lr 精细探索盆底，避免在最优解附近震荡。相比 StepLR，在同等 epoch 下通常能多挤出 0.5–1 个百分点。

### 3.4 日志与可视化

每次训练在 `runs/{mode}_{timestamp}/` 下自动生成四件套：

| 文件 | 用途 |
| :--- | :--- |
| `args.json` | 超参数快照 + 原始命令，复现实验 |
| `metrics.csv` | 逐 epoch 指标，便于后处理分析 |
| `events.out.*` | TensorBoard 事件文件 |
| `training_curves.png` | 训练结束时的总览图 |

终端输出按 epoch 滚动，行尾 `*` 标记该 epoch 刷新了最优验证准确率：

```
 Epoch    训练损失    训练准确率    验证损失    验证准确率    耗时
-----------------------------------------------------------------
     1      0.8231      73.42%     0.5617      82.10%     12s
     2      0.4102      86.78%     0.3914      87.93%     11s  *
    ...
    30      0.0521      98.62%     0.0489      98.31%     11s
```

### 3.5 权重保存

不止保存权重，还把元数据一起存进 checkpoint，让推理侧加载时无需再传参：

```python
torch.save({
    "epoch":            epoch,
    "mode":             args.mode,      # "digit" / "letter"
    "num_classes":      num_classes,
    "model_state_dict": model.state_dict(),
    "val_acc":          val_acc,
}, "checkpoint/{mode}_best.pth")
```

## 四、Web 在线演示

### 4.1 后端：FastAPI + 懒加载

三个路由就够了：

| 路由 | 方法 | 说明 |
| :--- | :--- | :--- |
| `/` | GET | 返回前端页面 |
| `/health` | GET | 健康检查，返回推理设备 |
| `/predict` | POST | 接收 base64 图像 + 模式，返回识别结果 |

**模型懒加载**是一个值得单独提的设计：服务启动不触碰权重文件，首次收到某模式请求时才读盘加载，之后缓存在内存中。好处是启动秒级响应、内存占用按需增长；如果只识别数字，字母模型永远不会被加载。

```python
_model_cache: dict = {}

def _load_model(mode: str) -> torch.nn.Module:
    if mode in _model_cache:
        return _model_cache[mode]
    checkpoint = torch.load(ckpt_path, map_location=_device)
    model = get_model(mode, num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _model_cache[mode] = model
    return model
```

### 4.2 前端：Canvas + 逐笔撤销

前端不依赖任何框架，核心是一个 400×400 的 Canvas。鼠标和触屏统一处理，真正想讲的是 **逐笔撤销**的实现——不存像素，而是存**每一笔落笔前的快照**：

```javascript
function startDraw(e) {
  // 落笔前拍快照
  currentStroke = ctx.getImageData(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function endDraw() {
  strokeHistory.push(currentStroke);   // 抬笔后入栈
}

undoBtn.addEventListener('click', () => {
  const prev = strokeHistory.pop();
  ctx.putImageData(prev, 0, 0);        // 还原到上一笔之前
});
```

这样任意笔画数都能无损撤销，且实现比维护笔画向量列表简单得多。代价是内存占用随撤销栈线性增长，对小画板无伤大雅。

其它交互细节：
- 图片上传（点击 / 拖放），自动居中缩放填充画板；
- `Ctrl+Z` 撤销、`Enter` 识别；
- 切换数字 / 字母模式时清空画板。

识别请求就是把 Canvas `toDataURL` 编码为 base64 发过去，响应体携带 Top-5 候选：

```json
{
  "label":      3,
  "character":  "༣",
  "confidence": 97.42,
  "top5": [
    {"label": 3, "character": "༣", "confidence": 97.42},
    {"label": 8, "character": "༨", "confidence":  1.83},
    ...
  ]
}
```

### 4.3 关键细节：训练—推理分布对齐

这是本项目调试中**收益最高的一步**，值得单独拎出来讲。

上线后首轮体验识别率很低。原因是：训练样本里字符几乎**铺满整帧**（28×28 或 64×64 都是紧贴字符边缘），但推理时用户在 400×400 画板上写一个字，字符只占中心一小块，周围大片白边。直接缩放，字符就被压成了一小坨，模型从未见过这种尺度。

解决办法：在推理预处理里**先做紧边界框裁剪**，再缩放。流程图：

```
Canvas 400×400（白底 + 一小块字符）
    │
    ▼  _tight_crop：找暗像素边界框 → 扩 15% padding → 裁剪 → 填充为正方形
字符铺满帧的图像
    │
    ▼  Resize → ToTensor → Normalize
模型输入（与训练分布一致）
```

核心代码：

```python
def _tight_crop(img_gray, pad_ratio=0.15):
    arr  = np.array(img_gray)
    mask = arr < 200                      # 暗像素掩码
    rows, cols = np.any(mask, 1), np.any(mask, 0)
    if not rows.any():                    # 画板为空
        return img_gray

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = max(4, int(max(rmax - rmin, cmax - cmin) * pad_ratio))
    # 外扩 padding 后裁剪，不越界
    ...
    cropped = img_gray.crop((cmin, rmin, cmax + 1, rmax + 1))
    # 填充为正方形白背景，避免后续 Resize 拉伸
    side   = max(cropped.size)
    square = Image.new("L", (side, side), 255)
    square.paste(cropped, ((side - cw) // 2, (side - ch) // 2))
    return square
```

加入这一步后，识别率显著提升。**经验**：训练和推理的数据分布不一致，即使模型本身没问题也会表现得「模型很差」——定位这类问题比调模型本身更重要。手写画板和上传图片走同一条预处理路径，保证两种输入的表现一致。

## 五、快速上手

```bash
# 1. 依赖
pip install -r requirements.txt

# 2. 数据集（见系列一下载链接），解压到：
#    dataset/TibetanMNIST28x28/{0..9}/*.png
#    dataset/TibetanLetter64x64/{0..29}/*.png

# 3. 训练（可选，仓库已附带权重）
cd recognition
python train.py --mode digit  --epochs 30                    # ~5 min CPU
python train.py --mode letter --epochs 50 --batch-size 128   # ~30 min CPU / 5 min GPU

# 4. 启动 Web 服务
cd ../web
uvicorn app:app --port 8000
# 浏览器打开 http://localhost:8000，书写或上传图片后点击「识别」
```

## 六、小结

本篇把数据变成了一个可交互的 Demo，沿途的关键选择：

1. **按任务复杂度差异化设计模型**：DigitCNN 轻量，LetterCNN 更深并加 BatchNorm，避免一刀切；
2. **增强策略服从先验**：藏文镜像字禁用翻转，这比盲目套用「常见视觉增强」更重要；
3. **懒加载服务**：启动快、内存按需增长，适合多模型共享后端的场景；
4. **训练—推理分布对齐**：Canvas 空白多、训练样本铺满帧，靠一个 `_tight_crop` 拉平差距，实测收益最高。

三篇合在一起，完整覆盖了这套系统**数据采集（系列一）→ 图像预处理（系列二）→ 模型训练与在线演示（本篇）** 的全链路。没有用到任何预训练模型，从零开始、轻量部署，可作为理解手写识别端到端链路的入门实践。
