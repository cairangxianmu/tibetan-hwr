# Tibetan HWR — 藏文手写识别系统

基于 PyTorch 的藏文手写数字与字母识别，含 Web 展示界面。

## 项目结构

```
tibetan-hwr/
├── dataset/                    # 软链接 → 原始数据集目录
│   ├── TibetanMNIST28x28/      # 10 类数字，28×28 PNG
│   └── TibetanLetter64x64/     # 30 类字母，64×64 PNG
├── image_processing/           # 扫描图像预处理模块
│   ├── __init__.py
│   ├── letter_processor.py     # 从红色格线方格纸提取字母
│   ├── digit_processor.py      # 从手写数字扫描图提取数字
│   └── utils.py                # 重命名 / 去红线 / 批量缩放
├── checkpoint/                 # 训练输出的模型权重文件
│   ├── digit_best.pth
│   └── letter_best.pth
├── recognition/
│   ├── dataset.py              # DataLoader 工厂（digit / letter 双模式）
│   ├── model.py                # CNN 模型定义（DigitCNN / LetterCNN）
│   └── train.py                # 训练入口（含可视化与日志）
├── runs/                       # 训练日志（每次运行生成带时间戳的子目录）
│   └── {mode}_{timestamp}/
│       ├── args.json           # 超参数与终端命令
│       ├── metrics.csv         # 逐 epoch 指标
│       ├── events.out.*        # TensorBoard 事件文件
│       └── training_curves.png # 损失与准确率曲线图
├── web/
│   ├── app.py                  # FastAPI 后端（从 checkpoint/ 加载模型）
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
├── requirements.txt
├── .gitignore
└── README.md
```

## 数据集

| 目录 | 尺寸 | 类别 | 样本数 |
|------|------|------|--------|
| `dataset/TibetanMNIST28x28/` | 28×28 | 10（数字 ༠–༩） | 17,768 |
| `dataset/TibetanLetter64x64/` | 64×64 | 30（字母 ཀ–ཨ） | 77,636 |

数据集下载：[百度网盘](https://pan.baidu.com/s/1TnM9Rxue9ae0bhPJ2EUP8g?pwd=4ata)　提取码：`4ata`

下载后将 `TibetanMNIST28x28` 与 `TibetanLetter64x64` 两个文件夹放入 `dataset/` 目录。

## 安装依赖

```bash
pip install -r requirements.txt
```

| 分组 | 包 | 用途 |
|------|-----|------|
| 训练 | `torch` `torchvision` | 模型训练与推理 |
| Web  | `fastapi` `uvicorn` `pillow` `python-multipart` | 后端服务 |
| 预处理 | `opencv-python` `numpy` | 图像处理流水线 |
| 可视化 | `matplotlib` `tensorboard` | 训练曲线图 / TensorBoard 日志 |

## 图像预处理（image_processing）

如果你有原始扫描手写表单，可通过预处理模块提取单字符图像。

**提取字母**（红色格线方格纸，8×12 格）：

```bash
python image_processing/letter_processor.py \
    --input  /path/to/scanned_sheets/ \
    --output /path/to/output/
```

**提取数字**（深色背景手写数字）：

```bash
python image_processing/digit_processor.py \
    --input  /path/to/digit_sheets/ \
    --output /path/to/output/
```

**工具函数**：

```bash
# 批量重命名
python image_processing/utils.py rename --dir ./data

# 去除残留红色格线
python image_processing/utils.py replace --dir ./data

# 批量缩放到 64×64
python image_processing/utils.py resize --src ./data_200 --dst ./data_64 --size 64
```

## 训练模型

```bash
cd recognition

# 训练数字模型（10 类，约 30 epoch）
python train.py --mode digit --epochs 30

# 训练字母模型（30 类，约 50 epoch）
python train.py --mode letter --epochs 50
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | 必填 | `digit` 或 `letter` |
| `--epochs` | 30 | 训练轮数 |
| `--lr` | 0.001 | 初始学习率 |
| `--batch-size` | 64 | 批大小 |
| `--val-split` | 0.2 | 验证集比例 |
| `--data-root` | 自动检测 | 覆盖数据集路径 |
| `--save-dir` | `../checkpoint/` | 模型保存目录 |
| `--log-dir` | `../runs/` | 日志根目录 |
| `--no-plot` | 否 | 禁用 matplotlib 曲线图输出 |

训练完成后，最优模型自动保存到 `checkpoint/{mode}_best.pth`。

## 训练日志与可视化

每次训练在 `runs/{mode}_{timestamp}/` 下自动生成以下文件：

| 文件 | 内容 |
|------|------|
| `args.json` | 所有超参数、完整终端命令、设备信息、时间戳 |
| `metrics.csv` | 逐 epoch 的训练损失、验证损失、准确率、学习率、耗时 |
| `events.out.*` | TensorBoard 事件文件（Loss / Accuracy / LR / 模型图） |
| `training_curves.png` | 损失与准确率双图，训练结束后自动保存 |

**查看 TensorBoard：**

```bash
tensorboard --logdir runs/
# 浏览器访问 http://localhost:6006
```

支持同时对比多次训练运行，面板包含 Loss、Accuracy、LR 折线图及模型计算图。

**不需要曲线图时跳过输出：**

```bash
python train.py --mode digit --epochs 30 --no-plot
```

**自定义日志目录：**

```bash
python train.py --mode letter --epochs 50 --log-dir /path/to/logs
```

## 启动 Web 服务

```bash
cd web
uvicorn app:app --reload --port 8000
```

浏览器访问 [http://localhost:8000](http://localhost:8000)

## Web 功能

- **模式切换**：顶部 Tab 切换数字 / 字母识别
- **手写输入**：Canvas 画板，支持鼠标和触屏；可调笔画粗细；支持逐笔撤销（Ctrl+Z）
- **图片上传**：点击上传区域或拖拽图片
- **识别结果**：显示识别字符、置信度进度条、Top-5 候选列表

## API

```
GET  /health
     → {"status": "ok", "device": "cpu"}

POST /predict
     Body:     {"image": "<base64>", "mode": "digit" | "letter"}
     Response: {"label": 0, "character": "༠", "confidence": 98.5,
                "top5": [{"label":0,"character":"༠","confidence":98.5}, ...]}
```

## 引用

> 周毛克, 才让先木, 龙从军, 等. 基于卷积神经网络的藏文手写数字和字母识别研究[J].
> 青海师范大学学报(自然科学版), 2019, 35(04): 34-39.
> DOI: 10.16229/j.cnki.issn1001-7542.2019.04.006
