"""
train.py — 藏文手写识别模型训练入口

用法：
    python train.py --mode digit  --epochs 30
    python train.py --mode letter --epochs 50 --lr 0.0005 --batch-size 128

每次运行在 runs/{mode}_{timestamp}/ 下生成：
    args.json           — 运行参数与终端命令
    metrics.csv         — 逐 epoch 训练指标
    events.out.*        — TensorBoard 事件文件（tensorboard --logdir runs/）
    training_curves.png — 损失与准确率曲线图

训练完成后，最优模型保存到 checkpoint/{mode}_best.pth。
每隔 --save-every 个 epoch 另外保存周期检查点 checkpoint/{mode}_epoch{N:04d}.pth。
"""

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 可选依赖：matplotlib 用于绘图，tensorboard 用于事件日志
try:
    import matplotlib
    matplotlib.use("Agg")   # 非交互后端，支持无 GUI / 服务器环境
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

# 支持从 recognition/ 目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_dataloaders
from model import get_model


# ---------------------------------------------------------------------------
# 训练 / 验证辅助函数
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    """执行一个 epoch 的前向传播、损失计算和反向传播，返回平均损失和准确率。"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在验证集上评估模型，返回平均损失和准确率（不更新梯度）。"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return running_loss / total, 100.0 * correct / total


def format_duration(seconds: float) -> str:
    """将秒数格式化为 "Xm Ys" 字符串，方便打印训练耗时。"""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ---------------------------------------------------------------------------
# 早停
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    监控验证准确率，连续 patience 个 epoch 无改善时触发早停。

    Args:
        patience: 容忍的连续不改善 epoch 数；0 表示禁用早停。
        delta:    最小改善量，小于此值视为无改善（防止噪声触发）。
    """

    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience = patience
        self.delta    = delta
        self.best_acc = 0.0
        self.counter  = 0
        self.best_state: dict | None = None

    def step(self, val_acc: float, model: nn.Module) -> bool:
        """
        用当前验证准确率更新状态。

        Returns:
            True 表示应触发早停，False 表示继续训练。
        """
        if val_acc > self.best_acc + self.delta:
            self.best_acc   = val_acc
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.patience > 0 and self.counter >= self.patience

    def restore(self, model: nn.Module):
        """将模型权重恢复到记录的最优状态。"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# 日志工具
# ---------------------------------------------------------------------------

def setup_run_dir(args, log_base: Path) -> Path:
    """
    为本次运行创建带时间戳的日志目录，并写入 args.json。

    目录命名格式：{mode}_{YYYYMMDD_HHMMSS}
    args.json 记录所有超参数、完整终端命令和设备信息。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_base / f"{args.mode}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "command": " ".join(sys.argv),
        "timestamp": timestamp,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "args": vars(args),
    }
    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return run_dir


class CSVLogger:
    """逐行写入 metrics.csv，每个 epoch 调用一次 log()。"""

    _FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "elapsed_s"]

    def __init__(self, path: Path):
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._f, fieldnames=self._FIELDS)
        self._writer.writeheader()

    def log(self, **kwargs):
        self._writer.writerow({k: kwargs.get(k, "") for k in self._FIELDS})
        self._f.flush()

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# 周期性检查点保存
# ---------------------------------------------------------------------------

def _save_periodic(model, epoch: int, args, num_classes: int, save_dir: Path):
    """保存当前 epoch 的周期检查点，并删除超出 keep_ckpts 数量限制的旧文件。"""
    path = save_dir / f"{args.mode}_epoch{epoch:04d}.pth"
    torch.save(
        {
            "epoch":            epoch,
            "mode":             args.mode,
            "num_classes":      num_classes,
            "model_state_dict": model.state_dict(),
            "is_periodic":      True,
        },
        path,
    )
    print(f"  周期检查点已保存：{path.name}")

    # 保留最近 keep_ckpts 个，删除多余旧文件
    if args.keep_ckpts > 0:
        ckpts = sorted(save_dir.glob(f"{args.mode}_epoch*.pth"))
        for old in ckpts[: -args.keep_ckpts]:
            old.unlink()


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_curves(history: dict, save_path: Path, mode: str):
    """
    绘制训练 / 验证的损失与准确率曲线，保存为 PNG。

    Args:
        history:   包含 train_loss / val_loss / train_acc / val_acc 列表的字典
        save_path: 图像保存路径
        mode:      训练模式（digit / letter），用于图标题
    """
    if not _HAS_MPL:
        print("警告：未安装 matplotlib，跳过曲线绘制（pip install matplotlib）")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training Curves — {mode}", fontsize=13)

    ax_loss.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
    ax_loss.plot(epochs, history["val_loss"],   label="Val Loss",   color="#F44336", linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="Train Acc", color="#2196F3")
    ax_acc.plot(epochs, history["val_acc"],   label="Val Acc",   color="#F44336", linestyle="--")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"训练曲线已保存至：{save_path}")


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="藏文手写识别模型训练")
    parser.add_argument(
        "--mode", choices=["digit", "letter"], required=True,
        help="训练模式：digit（10 类数字）或 letter（30 类字母）"
    )
    parser.add_argument("--epochs",          type=int,   default=30,   help="最大训练轮数")
    parser.add_argument("--lr",              type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--batch-size",      type=int,   default=64,   help="批大小")
    parser.add_argument("--val-split",       type=float, default=0.2,  help="验证集比例")
    parser.add_argument("--data-root",       type=str,   default=None, help="覆盖默认数据集路径")
    parser.add_argument("--save-dir",        type=str,   default=None,
                        help="模型保存目录（默认：../checkpoint/）")
    parser.add_argument("--num-workers",     type=int,   default=0,    help="DataLoader 子进程数")
    parser.add_argument("--log-dir",         type=str,   default=None,
                        help="日志根目录（默认：../runs/）；每次运行在其下建子目录")
    parser.add_argument("--no-plot",         action="store_true",
                        help="禁用 matplotlib 曲线图输出")
    # ── 泛化性 / 正则化 ──────────────────────────────────────────────────────
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="CrossEntropyLoss 标签平滑系数（0 禁用）")
    parser.add_argument("--weight-decay",    type=float, default=1e-4,
                        help="Adam weight_decay 参数")
    # ── 早停 ─────────────────────────────────────────────────────────────────
    parser.add_argument("--patience",        type=int,   default=10,
                        help="早停容忍 epoch 数（验证准确率连续无改善）；0 禁用早停")
    # ── 周期性保存 ────────────────────────────────────────────────────────────
    parser.add_argument("--save-every",      type=int,   default=5,
                        help="每 N 个 epoch 保存一次周期检查点；0 禁用")
    parser.add_argument("--keep-ckpts",      type=int,   default=0,
                        help="周期检查点最大保留数量；0 保留全部")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 自动检测并选择 GPU 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # ── 日志目录 ────────────────────────────────────────────────────────────
    log_base = Path(args.log_dir) if args.log_dir else (
        Path(__file__).resolve().parent.parent / "runs"
    )
    run_dir = setup_run_dir(args, log_base)
    print(f"日志目录：{run_dir}")

    csv_logger = CSVLogger(run_dir / "metrics.csv")

    writer = None
    if _HAS_TB:
        writer = SummaryWriter(log_dir=str(run_dir))
        print(f"TensorBoard：tensorboard --logdir {log_base}")
    else:
        print("提示：未安装 tensorboard，跳过事件日志（pip install tensorboard）")

    # ── 数据集 ──────────────────────────────────────────────────────────────
    print(f"加载 {args.mode} 数据集...")
    train_loader, val_loader, num_classes = get_dataloaders(
        mode=args.mode,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )
    print(
        f"  训练批次：{len(train_loader)} | "
        f"验证批次：{len(val_loader)} | "
        f"类别数：{num_classes}"
    )

    # ── 模型 ────────────────────────────────────────────────────────────────
    model = get_model(args.mode, num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型：{model.__class__.__name__}（{total_params:,} 个参数）")

    if writer:
        dummy = torch.zeros(1, 1, 28 if args.mode == "digit" else 64,
                            28 if args.mode == "digit" else 64).to(device)
        writer.add_graph(model, dummy)

    # ── 优化器 / 调度器 ─────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Checkpoint 路径 ──────────────────────────────────────────────────────
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(__file__).resolve().parent.parent / "checkpoint"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"{args.mode}_best.pth"

    # ── 早停初始化 ────────────────────────────────────────────────────────────
    early_stopper = EarlyStopping(patience=args.patience)
    if args.patience > 0:
        print(f"早停：patience={args.patience} epoch")

    # ── 训练主循环 ───────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    stopped_early = False

    print(f"\n{'Epoch':>6}  {'训练损失':>10}  {'训练准确率':>10}  {'验证损失':>8}  {'验证准确率':>9}  {'耗时':>6}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # 记录历史（用于绘图）
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # CSV 日志
        csv_logger.log(
            epoch=epoch,
            train_loss=round(train_loss, 6),
            train_acc=round(train_acc, 4),
            val_loss=round(val_loss, 6),
            val_acc=round(val_acc, 4),
            lr=f"{current_lr:.2e}",
            elapsed_s=round(elapsed, 2),
        )

        # TensorBoard 日志
        if writer:
            writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
            writer.add_scalar("LR", current_lr, epoch)

        # 终端输出
        marker = " *" if val_acc > best_val_acc else ""
        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.2f}%  "
            f"{val_loss:>8.4f}  {val_acc:>8.2f}%  {format_duration(elapsed):>6}{marker}"
        )

        # 保存最优权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":            epoch,
                    "mode":             args.mode,
                    "num_classes":      num_classes,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          val_acc,
                },
                best_path,
            )

        # 周期性保存
        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_periodic(model, epoch, args, num_classes, save_dir)

        # 早停检查
        if early_stopper.step(val_acc, model):
            print(
                f"\n早停触发：连续 {args.patience} 个 epoch 验证准确率无改善，"
                f"最优 {early_stopper.best_acc:.2f}%"
            )
            early_stopper.restore(model)
            stopped_early = True
            # 早停时保存当前（恢复后）的周期检查点
            if args.save_every > 0 and epoch % args.save_every != 0:
                _save_periodic(model, epoch, args, num_classes, save_dir)
            break

    # ── 末尾 epoch 周期保存（未被整除且未早停时） ──────────────────────────
    if not stopped_early and args.save_every > 0 and args.epochs % args.save_every != 0:
        _save_periodic(model, args.epochs, args, num_classes, save_dir)

    # ── 收尾 ─────────────────────────────────────────────────────────────────
    csv_logger.close()
    if writer:
        writer.close()

    print(f"\n训练完成，最优验证准确率：{best_val_acc:.2f}%")
    print(f"最优模型：{best_path}")

    if not args.no_plot:
        plot_curves(history, run_dir / "training_curves.png", mode=args.mode)


if __name__ == "__main__":
    main()
