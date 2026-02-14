"""
ISIC 2019 训练脚本：EfficientNet-B3，Balanced Accuracy 评估，输出中间结果与曲线图。
"""
import os
import argparse
import subprocess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# 使用非交互式后端，便于无图形界面环境保存图片
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from Mydataset import ISIC2019Dataset
from models import build_efficientnet_b3


def get_gpu_temperature_celsius():
    """
    读取当前 CUDA 显卡温度（°C）。依赖 nvidia-smi，非 NVIDIA 或不可用时返回 None。
    """
    if not torch.cuda.is_available():
        return None
    try:
        device_id = torch.cuda.current_device()
        out = subprocess.run(
            [
                'nvidia-smi', '-i', str(device_id),
                '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().split()[0])
    except (FileNotFoundError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return None


class FocalLoss(nn.Module):
    """
    Focal Loss：对易分样本降权、对难分样本聚焦，缓解类别不均衡。
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    logits: (N, C), target: (N,) 类别索引 long。
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # (C,) 或 None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        # log_pt, pt: (N,)
        log_pt = F.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_weight = (1 - pt).pow(self.gamma)
        loss = -focal_weight * log_pt
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def get_args():
    """解析命令行参数，返回训练所需的配置。"""
    parser = argparse.ArgumentParser(description='ISIC2019 EfficientNet-B3 训练')
    # 数据路径
    parser.add_argument('--img_dir', type=str,
                        default=r'D:\Documents\VSproject\graduation_project\SIFTNeXt\dataset_files\ISIC_2019_Training_Input',
                        help='训练图像目录')
    parser.add_argument('--csv_path', type=str,
                        default=r'D:\Documents\VSproject\graduation_project\SIFTNeXt\dataset_files\ISIC_2019_Training_GroundTruth.csv',
                        help='训练标签 CSV 路径')
    parser.add_argument('--test_img_dir', type=str,
                        default=r'D:\Documents\VSproject\graduation_project\SIFTNeXt\dataset_files\ISIC_2019_Test_Input',
                        help='测试集图像目录')
    parser.add_argument('--test_csv_path', type=str,
                        default=r'D:\Documents\VSproject\graduation_project\SIFTNeXt\dataset_files\ISIC_2019_Test_GroundTruth_without_unk.csv',
                        help='测试集标签 CSV 路径')
    # 模型与训练超参
    parser.add_argument('--num_classes', type=int, default=8, help='类别数（ISIC2019 为 8）')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=26, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader 子进程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型、日志与曲线图保存目录')
    parser.add_argument('--log_interval', type=int, default=50, help='每多少个 batch 打印一次当前训练 loss')
    parser.add_argument('--gpu_temp_threshold', type=int, default=85, help='GPU 温度阈值（°C），超过则暂停冷却；0 表示不监测')
    parser.add_argument('--gpu_temp_cooldown', type=int, default=60, help='过热时暂停冷却秒数')
    parser.add_argument('--no_pretrained', action='store_true', help='不使用 ImageNet 预训练；默认使用预训练（ISIC 数据规模下建议开启）')
    parser.add_argument('--img_size', type=int, default=300, help='输入图像边长（EfficientNet-B3 常用 300）')
    # 数据划分（竞赛规范：训练/验证从训练集分层划分，测试集仅最终评估一次）
    parser.add_argument('--val_ratio', type=float, default=0.2, help='从训练集中划分出的验证集比例（stratified）')
    parser.add_argument('--stratify_seed', type=int, default=1688, help='train/val 分层划分的随机种子')
    # Focal Loss（类别不均衡）
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss 的 gamma，越大越关注难分样本')
    # 可复现性
    parser.add_argument('--seed', type=int, default=1688, help='全局随机种子（torch/numpy/cuda）')
    parser.add_argument('--deterministic', action='store_true', help='开启后 cudnn 确定性模式，可完全复现但可能更慢')
    # 继续训练
    parser.add_argument('--resume', type=str, default='',
                        help='从指定 checkpoint 继续训练，如 checkpoints/last_model.pth；留空则从头训练')
    args = parser.parse_args()
    # 避免无效值导致除零或空循环
    args.log_interval = max(1, int(args.log_interval))
    args.epochs = max(1, int(args.epochs))
    args.batch_size = max(1, int(args.batch_size))
    return args


def get_transforms(img_size, is_train=True):
    """
    根据训练/验证阶段返回数据增强与归一化。
    训练：RandomResizedCrop 保持比例并做尺度变化，避免强制拉伸导致病灶形变。
    验证/测试：Resize 短边后 CenterCrop 成正方形，保持比例。
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    # 验证/测试：先按短边缩放到 img_size 再中心裁成正方形，避免拉伸
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def onehot_to_class(labels):
    """
    将 one-hot 标签转为类别索引，供 FocalLoss 与评估使用。
    输入形状 (B, C)，输出形状 (B,) 的 long 张量，供 FocalLoss / 评估使用。
    """
    return labels.argmax(dim=1).long()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval,
                    gpu_temp_threshold=0, gpu_temp_cooldown=60):
    """
    训练一个 epoch：前向、反向、更新参数，并计算该 epoch 平均 loss 与 Balanced Accuracy。
    每 log_interval 个 batch 检查一次 GPU 温度，超过 gpu_temp_threshold（°C）则暂停冷却。
    """
    if len(loader) == 0:
        return 0.0, 0.0
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels_onehot = labels.to(device)
        labels_idx = onehot_to_class(labels_onehot)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels_idx)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # 收集本 batch 的预测与真实标签，用于后续算 Balanced Accuracy
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels_idx.cpu().numpy())

        # 按间隔打印当前平均 loss，并检查 GPU 温度
        if (i + 1) % log_interval == 0:
            avg_loss = running_loss / (i + 1)
            print(f'  [Epoch {epoch}] Batch {i+1}/{len(loader)}  Loss: {avg_loss:.4f}')
            if gpu_temp_threshold > 0 and device.type == 'cuda':
                temp = get_gpu_temperature_celsius()
                if temp is not None:
                    print(f'  GPU 温度: {temp}°C')
                    if temp >= gpu_temp_threshold:
                        print(f'  [过热保护] 达到阈值 {gpu_temp_threshold}°C，暂停 {gpu_temp_cooldown}s 冷却...')
                        time.sleep(gpu_temp_cooldown)
                        temp_after = get_gpu_temperature_celsius()
                        if temp_after is not None:
                            print(f'  冷却后 GPU 温度: {temp_after}°C')

    avg_loss = running_loss / len(loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, bal_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    在验证集上评估：只前向，不更新参数，返回平均 loss 与 Balanced Accuracy。
    """
    if len(loader) == 0:
        return 0.0, 0.0
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_onehot = labels.to(device)
        labels_idx = onehot_to_class(labels_onehot)

        logits = model(images)
        loss = criterion(logits, labels_idx)
        running_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels_idx.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss = running_loss / len(loader)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, bal_acc


@torch.no_grad()
def get_predictions(model, loader, device):
    """在指定 DataLoader 上跑前向，返回真实标签与预测类别（numpy 一维数组）。"""
    if len(loader) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_idx = onehot_to_class(labels.to(device))
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels_idx.cpu().numpy())
    return np.concatenate(all_labels, axis=0), np.concatenate(all_preds, axis=0)


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path, class_names=None):
    """
    使用 matplotlib 绘制混淆矩阵并保存为图片。
    y_true, y_pred: 一维数组，类别索引。
    class_names: 可选，各类别显示名；缺省为 "0", "1", ...
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print('无预测结果，跳过混淆矩阵。')
        return
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.8), max(5, num_classes * 0.7)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'混淆矩阵已保存: {save_path}')


_HISTORY_KEYS = ('train_loss', 'val_loss', 'train_bal_acc', 'val_bal_acc')


def _normalize_history(history):
    """确保 history 包含四个等长列表，缺则用空列表补齐，长度取最小以避免错位。"""
    if not history or not isinstance(history, dict):
        return {k: [] for k in _HISTORY_KEYS}
    out = {}
    for k in _HISTORY_KEYS:
        v = history.get(k)
        out[k] = list(v) if isinstance(v, (list, np.ndarray)) else []
    n = min(len(out[k]) for k in _HISTORY_KEYS)
    if n == 0:
        return out
    return {k: out[k][:n] for k in _HISTORY_KEYS}


def _history_to_tensors(history):
    """将 history（dict of lists）转为 dict of 1D tensors，便于 weights_only=True 安全加载。"""
    out = {}
    for k in _HISTORY_KEYS:
        lst = history.get(k) or []
        out[k] = torch.tensor([float(x) for x in lst], dtype=torch.float64)
    return out


def _history_from_tensors(history_tensors):
    """从 checkpoint 中的 history 张量恢复为 dict of lists；缺或非张量则返回空 history。"""
    if not history_tensors or not isinstance(history_tensors, dict):
        return {k: [] for k in _HISTORY_KEYS}
    out = {}
    for k in _HISTORY_KEYS:
        v = history_tensors.get(k)
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu().tolist()
        else:
            out[k] = []
    return _normalize_history(out)


def _load_best_for_eval(model, save_dir, device):
    """若存在 best_model.pth 则加载其权重，用于最终测试集评估与混淆矩阵（避免用 last 过拟合）。"""
    best_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print('已加载 best_model.pth 用于最终测试集评估与混淆矩阵。')


def plot_curves(history, save_dir):
    """
    根据训练历史绘制 Loss 曲线与 Balanced Accuracy 曲线，并保存为 PNG。
    history 需包含：train_loss, val_loss, train_bal_acc, val_bal_acc（列表，按 epoch 顺序）。
    """
    history = _normalize_history(history)
    if not history.get('train_loss'):
        print('无有效训练历史，跳过曲线保存。')
        return
    n = len(history['train_loss'])
    epochs = np.arange(1, n + 1)

    # ---------- Loss 曲线 ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(save_dir, 'loss_curve.png')
    fig.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss 曲线已保存: {loss_path}')

    # ---------- Balanced Accuracy 曲线 ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history['train_bal_acc'], 'b-', label='Train Balanced Accuracy', linewidth=2)
    ax.plot(epochs, history['val_bal_acc'], 'r-', label='Val Balanced Accuracy', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Training & Validation Balanced Accuracy', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    acc_path = os.path.join(save_dir, 'balanced_accuracy_curve.png')
    fig.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Balanced Accuracy 曲线已保存: {acc_path}')


def print_hyperparameters(args, device):
    """在训练开始前打印当前使用的超参数与设备信息。"""
    print('\n' + '=' * 60)
    print('  训练超参数')
    print('=' * 60)
    print('  [设备]')
    print(f'    设备:              {device}')
    print('  [数据]')
    print(f'    图像目录:          {args.img_dir}')
    print(f'    标签 CSV:          {args.csv_path}')
    print(f'    测试集图像目录:    {args.test_img_dir}')
    print(f'    测试集标签 CSV:    {args.test_csv_path}')
    print(f'    类别数:            {args.num_classes}')
    print(f'    输入尺寸:          {args.img_size} x {args.img_size}')
    print('  [模型]')
    print(f'    是否使用预训练:    {"否" if args.no_pretrained else "是"}')
    print(f'    dropout:           0.3')
    print('  [训练]')
    print(f'    训练轮数:          {args.epochs}')
    print(f'    批大小:            {args.batch_size}')
    print(f'    学习率:            {args.lr}')
    print(f'    optimizer:         AdamW')
    print(f'    weight_decay:      1e-2')
    print(f'    scheduler:         CosineAnnealingLR (T_max=epochs)')
    print(f'    criterion:         FocalLoss(alpha=class_weight, gamma={getattr(args, "focal_gamma", 2.0)})')
    print('  [其他]')
    print(f'    val_ratio:         {getattr(args, "val_ratio", 0.2)}')
    print(f'    stratify_seed:     {getattr(args, "stratify_seed", 42)}')
    print(f'    focal_gamma:       {getattr(args, "focal_gamma", 2.0)}')
    print(f'    seed:              {getattr(args, "seed", 42)}')
    print(f'    保存目录:          {args.save_dir}')
    print(f'    log_interval:      {args.log_interval}')
    print(f'    gpu_temp_threshold: {getattr(args, "gpu_temp_threshold", 0)}°C (0=不监测)')
    print(f'    gpu_temp_cooldown:  {getattr(args, "gpu_temp_cooldown", 60)}s')
    print(f'    num_workers:       {args.num_workers}')
    if getattr(args, 'resume', '') and args.resume:
        print(f'    继续训练 (resume):  {args.resume}')
    print('=' * 60 + '\n')


def set_seed(seed, deterministic=False):
    """设置全局随机种子，便于复现。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """主流程：解析参数、构建数据与模型、训练、保存最佳/最新权重并绘制曲线。"""
    args = get_args()
    set_seed(getattr(args, 'seed', 42), getattr(args, 'deterministic', False))
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练开始前先打印使用的超参数
    print_hyperparameters(args, device)

    # 训练与验证使用不同 transform（训练带增强）
    train_transform = get_transforms(args.img_size, is_train=True)
    val_transform = get_transforms(args.img_size, is_train=False)

    # 竞赛规范：训练集分层划分 train/val（选 best 用 val），测试集仅最终评估一次，避免 data leakage
    df = pd.read_csv(args.csv_path)
    labels_full = df.iloc[:, 1:9].values
    class_indices = np.argmax(labels_full, axis=1)
    all_idx = np.arange(len(class_indices))
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=args.val_ratio,
        stratify=class_indices,
        random_state=args.stratify_seed,
    )
    full_train_ds = ISIC2019Dataset(args.img_dir, args.csv_path, transform=train_transform)
    full_val_ds = ISIC2019Dataset(args.img_dir, args.csv_path, transform=val_transform)
    train_dataset = Subset(full_train_ds, train_idx)
    val_dataset = Subset(full_val_ds, val_idx)
    test_dataset = ISIC2019Dataset(args.test_img_dir, args.test_csv_path, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if len(train_loader) == 0:
        raise ValueError(
            '训练集一个 batch 都没有（可能：数据为空、batch_size 大于样本数且 drop_last=False）。'
            '请检查 --img_dir、--csv_path 及 --batch_size。'
        )

    # Focal Loss + 类别权重：与 Balanced Accuracy 一致，缓解类别不均衡并聚焦难分样本
    # 若 train loss 震荡或 val recall 不稳，可尝试 alpha=1/sqrt(n)、focal_gamma=1.5
    train_class_indices = np.argmax(labels_full[train_idx], axis=1)
    class_counts = np.bincount(train_class_indices, minlength=args.num_classes)
    class_weights = 1.0 / (class_counts.astype(np.float64) + 1e-5)
    class_weights = class_weights / class_weights.sum() * args.num_classes
    focal_alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = build_efficientnet_b3(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        dropout=0.3,
    ).to(device)

    criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_path = os.path.join(args.save_dir, 'train_log.txt')
    # 默认：从头训练
    start_epoch = 1
    best_bal_acc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_bal_acc': [],
        'val_bal_acc': [],
    }

    # 若指定 --resume，则加载 checkpoint 并从中断处继续（checkpoint 仅含张量与安全类型，使用 weights_only=True）
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        if ckpt.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = max(1, int(ckpt.get('epoch', 0)) + 1)
        best_bal_acc = float(ckpt.get('best_bal_acc', ckpt.get('val_balanced_accuracy', 0.0)))
        history = _history_from_tensors(ckpt.get('history'))
        print(f'已从 checkpoint 恢复: {args.resume}')
        print(f'  上次训练到 Epoch {start_epoch - 1}，将从 Epoch {start_epoch} 继续至 {args.epochs}')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'\n[继续训练] {datetime.now().isoformat()} 从 {args.resume} 恢复，从 Epoch {start_epoch} 继续\n\n')
    else:
        if args.resume:
            print(f'警告: --resume 指定路径不存在或不是文件，将从头训练: {args.resume}')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f'Start: {datetime.now().isoformat()}\n')
            f.write(f'Args: {args}\n\n')

    # 恢复后若已超过目标轮数，则不再训练新 epoch，仅做收尾（写日志、画曲线、混淆矩阵）
    if start_epoch > args.epochs:
        print(f'当前已训练至 Epoch {start_epoch - 1}，不小于目标轮数 {args.epochs}，无需继续训练。')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'\nBest Val Balanced Accuracy: {best_bal_acc:.4f}\n')
            f.write(f'End: {datetime.now().isoformat()}\n')
        if history.get('train_loss'):
            plot_curves(history, args.save_dir)
            _load_best_for_eval(model, args.save_dir, device)
            y_true, y_pred = get_predictions(model, test_loader, device)
            plot_confusion_matrix(
                y_true, y_pred, args.num_classes,
                os.path.join(args.save_dir, 'confusion_matrix.png'),
            )
            if len(test_loader) > 0:
                test_loss, test_bal_acc = evaluate(model, test_loader, criterion, device)
                print(f'测试集  Loss: {test_loss:.4f}  BalancedAcc: {test_bal_acc:.4f}')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f'Test Loss: {test_loss:.4f}  Test BalancedAcc: {test_bal_acc:.4f}\n')
        return

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f'\n========== Epoch {epoch}/{args.epochs} ==========')
            train_loss, train_bal_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, args.log_interval,
                gpu_temp_threshold=getattr(args, 'gpu_temp_threshold', 0),
                gpu_temp_cooldown=getattr(args, 'gpu_temp_cooldown', 60),
            )
            val_loss, val_bal_acc = evaluate(model, val_loader, criterion, device)
            # 先取当前 epoch 使用的学习率再 step，这样打印的 LR 与本期训练一致
            lr_now = scheduler.get_last_lr()[0]
            scheduler.step()

            # 本轮结果写入控制台与日志文件
            line = (
                f'Epoch {epoch:3d}  '
                f'Train Loss: {train_loss:.4f}  Train BalancedAcc: {train_bal_acc:.4f}  '
                f'Val Loss: {val_loss:.4f}  Val BalancedAcc: {val_bal_acc:.4f}  '
                f'LR: {lr_now:.2e}\n'
            )
            print(line.strip())
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(line)

            # 使用 float() 保证内存中为 Python 标量，避免 numpy 进入 checkpoint
            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['train_bal_acc'].append(float(train_bal_acc))
            history['val_bal_acc'].append(float(val_bal_acc))

            # 若当前验证 Balanced Accuracy 为历史最高，保存为最佳模型（与 last 相同字段，便于 --resume 也可从 best 恢复）
            if val_bal_acc > best_bal_acc:
                best_bal_acc = val_bal_acc
                ckpt_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save({
                    'epoch': int(epoch),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_bal_acc': float(best_bal_acc),
                    'history': _history_to_tensors(history),
                    'val_balanced_accuracy': float(val_bal_acc),
                }, ckpt_path)
                print(f'  -> 保存最佳模型 (BalancedAcc={val_bal_acc:.4f}) -> {ckpt_path}')

            # 每轮都保存最新 checkpoint（history 存为张量，仅含 weights_only 安全类型）
            torch.save({
                'epoch': int(epoch),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_bal_acc': float(best_bal_acc),
                'history': _history_to_tensors(history),
            }, os.path.join(args.save_dir, 'last_model.pth'))

        # 正常结束：写入最佳指标与结束时间
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'\nBest Val Balanced Accuracy: {best_bal_acc:.4f}\n')
            f.write(f'End: {datetime.now().isoformat()}\n')
        print(f'\n训练结束。最佳 Val Balanced Accuracy: {best_bal_acc:.4f}')
        print(f'日志: {log_path}')

    except KeyboardInterrupt:
        # 用户按 Ctrl+C 中断：记录日志并尽量保存已完成的曲线
        print('\n\n训练被用户中断 (Ctrl+C)。')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'\n[用户中断] {datetime.now().isoformat()}\n')
            f.write(f'中断前最佳 Val Balanced Accuracy: {best_bal_acc:.4f}\n')
        if history['train_loss']:
            print('正在保存已完成的训练曲线...')
            plot_curves(history, args.save_dir)
            print('曲线已保存至:', args.save_dir)
            _load_best_for_eval(model, args.save_dir, device)
            y_true, y_pred = get_predictions(model, test_loader, device)
            plot_confusion_matrix(
                y_true, y_pred, args.num_classes,
                os.path.join(args.save_dir, 'confusion_matrix.png'),
            )
            if len(test_loader) > 0:
                test_loss, test_bal_acc = evaluate(model, test_loader, criterion, device)
                print(f'测试集  Loss: {test_loss:.4f}  BalancedAcc: {test_bal_acc:.4f}')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f'Test Loss: {test_loss:.4f}  Test BalancedAcc: {test_bal_acc:.4f}\n')
        else:
            print('尚无完整 epoch，跳过曲线保存。')
        print('已安全退出。')
        return

    # 根据 history 绘制 Loss 与 Balanced Accuracy 曲线并保存到 save_dir
    plot_curves(history, args.save_dir)
    # 用 best 模型在测试集上预测并绘制混淆矩阵，并输出测试集指标
    _load_best_for_eval(model, args.save_dir, device)
    y_true, y_pred = get_predictions(model, test_loader, device)
    plot_confusion_matrix(
        y_true, y_pred, args.num_classes,
        os.path.join(args.save_dir, 'confusion_matrix.png'),
    )
    if len(test_loader) > 0:
        test_loss, test_bal_acc = evaluate(model, test_loader, criterion, device)
        print(f'\n测试集  Loss: {test_loss:.4f}  BalancedAcc: {test_bal_acc:.4f}')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'Test Loss: {test_loss:.4f}  Test BalancedAcc: {test_bal_acc:.4f}\n')


if __name__ == '__main__':
    main()
