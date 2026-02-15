"""
ISIC 2019 训练脚本（竞赛规范版，科研向）

- 模型：EfficientNet-B3，可选 ImageNet 预训练
- 损失：默认 CrossEntropyLoss + class_weight + label_smoothing=0.1（推荐）；可选 --use_focal 用 Focal Loss
- 优化：AdamW(lr=3e-4, weight_decay=1e-4)，CosineAnnealingLR；可选 early stopping
- 评估：Balanced Accuracy；验证集选 best，测试集仅最终评估一次；输出每类 P/R/F1 与混淆矩阵 CSV
- 数据：Stratified train/val，无 test 泄露；增强含 RandomRotation/RandomAffine
- 可复现：种子、可选 deterministic、保存 config.json 与完整命令行到日志
"""
import os
import sys
import json
import argparse
import random
import subprocess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# 非交互式后端，便于无图形界面环境保存图片
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


def _log_ts():
    """当前时间戳，精确到秒，用于日志行前缀。格式如 [2026-02-15 12:41:30] """
    return datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ')


from Mydataset import ISIC2019Dataset
from models import build_efficientnet_b3

# ISIC 2019 官方 8 类名称（与常见 GroundTruth CSV 列顺序一致），用于混淆矩阵与报告
ISIC2019_CLASS_NAMES = [
    'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'SCC',
]


def get_gpu_temperature_celsius():
    """
    读取当前 CUDA 设备 GPU 温度（°C）。
    依赖系统 nvidia-smi；非 NVIDIA 或不可用时返回 None，不抛异常。
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
    公式：FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    输入：logits (N, C)，target (N,) 类别索引 long。
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # (C,) 各类别权重，None 表示不加权
        self.gamma = gamma  # 聚焦参数，越大越关注难分样本
        self.reduction = reduction

    def forward(self, logits, target):
        # 取目标类别的 log 概率与概率，用于计算 (1-p_t)^gamma * log(p_t)
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
    """
    解析命令行参数，返回训练所需配置（Namespace）。
    若指定 --config 或使用 --resume（则默认用 checkpoint 同目录 config.json），先从 config 加载默认值，再以命令行覆盖。
    """
    # 先解析 --config 与 --resume，确定要加载的 config 文件
    pre_parser = argparse.ArgumentParser()
    pre_parser.add_argument('--config', type=str, default=None, help='（由主解析器定义，此处仅用于先读）')
    pre_parser.add_argument('--resume', type=str, default='', help='（由主解析器定义，此处仅用于先读）')
    pre_args, _ = pre_parser.parse_known_args()
    _config_debug = lambda msg: print(f"[Config] {msg}", flush=True)
    _config_debug(f"预解析: --config={getattr(pre_args, 'config', None)!r}, --resume={getattr(pre_args, 'resume', None)!r}")

    config_path = None
    config_source = None
    if getattr(pre_args, 'config', None) and str(pre_args.config).strip():
        config_path = os.path.normpath(str(pre_args.config).strip())
        config_source = "命令行 --config"
    elif getattr(pre_args, 'resume', None) and str(pre_args.resume).strip():
        resume_path = os.path.normpath(str(pre_args.resume).strip())
        if os.path.isfile(resume_path):
            config_path = os.path.join(os.path.dirname(resume_path), 'config.json')
            config_source = f"resume 同目录 (--resume={resume_path!r})"
        else:
            _config_debug(f"跳过 resume 目录 config: --resume 文件不存在 {resume_path!r}")
    if config_path is not None:
        _config_debug(f"目标 config 路径: {config_path!r} (来源: {config_source})")

    config_dict = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        _config_debug(f"已加载 config 文件: 共 {len(config_dict)} 个键")
    elif config_path:
        _config_debug(f"未找到 config 文件，将使用脚本默认值: {config_path!r}")

    parser = argparse.ArgumentParser(description='ISIC2019 EfficientNet-B3 训练')
    # ---------- 数据路径 ----------
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
    # ---------- 模型与训练超参 ----------
    parser.add_argument('--num_classes', type=int, default=8, help='类别数（ISIC2019 为 8）')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=20, help='批大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率（推荐 3e-4，EfficientNet+AdamW）')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='AdamW 权重衰减（5e-4/1e-3 可减轻过拟合；过大影响 fine-tune）')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 子进程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型、日志与曲线图保存目录')
    parser.add_argument('--log_interval', type=int, default=100, help='每多少个 batch 打印一次当前训练 loss')
    parser.add_argument('--gpu_temp_threshold', type=int, default=85, help='GPU 温度阈值（°C），超过则暂停冷却；0 表示不监测')
    parser.add_argument('--gpu_temp_cooldown', type=int, default=30, help='过热时暂停冷却秒数')
    parser.add_argument('--no_pretrained', action='store_true', help='不使用 ImageNet 预训练；默认使用预训练（ISIC 数据规模下建议开启）')
    parser.add_argument('--img_size', type=int, default=300, help='输入图像边长（EfficientNet-B3 常用 300）')
    # ---------- 数据划分（竞赛规范：train/val 从训练集 stratified 划分，test 仅最终评估一次） ----------
    parser.add_argument('--val_ratio', type=float, default=0.2, help='从训练集中划分出的验证集比例（stratified）')
    parser.add_argument('--stratify_seed', type=int, default=1688, help='train/val 分层划分的随机种子')
    # ---------- 损失函数（推荐 CE+class_weight+label_smoothing，Focal 易与 class weight 叠加过度） ----------
    parser.add_argument('--use_focal', action='store_true', help='使用 Focal Loss；默认用 CrossEntropyLoss+class_weight+label_smoothing')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='CE 的 label smoothing（医学图像常 0.1）')
    parser.add_argument('--focal_gamma', type=float, default=3.0, help='Focal Loss 的 gamma（仅 --use_focal 时生效；极度不均衡可试 3.0～5.0）')
    # ---------- Early stopping ----------
    parser.add_argument('--early_stopping_patience', type=int, default=8, help='验证集无提升则提前停止的 epoch 数，0 表示不启用（推荐 8）')
    # ---------- 可复现性 ----------
    parser.add_argument('--seed', type=int, default=1688, help='训练/DataLoader/transform 等随机种子（torch/numpy/cuda）')
    parser.add_argument('--global_seed', type=int, default=None, help='若指定则同时作为 seed 与 stratify_seed，单参数统一全部随机源；不指定则使用上述 seed 与 stratify_seed')
    parser.add_argument('--deterministic', action='store_true', help='开启后 cudnn 确定性模式，可完全复现但可能更慢')
    # ---------- 继续训练 ----------
    parser.add_argument('--resume', type=str, default='',
                        help='从指定 checkpoint 继续训练，如 checkpoints/last_model.pth；留空则从头训练')
    parser.add_argument('--config', type=str, default=None,
                        help='指定 config.json 路径；未指定且使用 --resume 时默认使用 checkpoint 同目录下的 config.json；命令行参数可覆盖 config 中的值')
    # 用 config 中的值作为 parser 默认值，命令行传入的参数会覆盖这些默认值
    if config_dict:
        valid_dests = {a.dest for a in parser._actions if getattr(a, 'dest', None) and a.dest != 'help'}
        defaults_from_config = {k: v for k, v in config_dict.items() if k in valid_dests}
        skipped = set(config_dict) - set(defaults_from_config)
        if skipped:
            _config_debug(f"config 中已忽略的键（不在解析器中）: {sorted(skipped)}")
        if defaults_from_config:
            parser.set_defaults(**defaults_from_config)
            _config_debug(f"已将 config 作为默认值应用到 {len(defaults_from_config)} 个参数: {sorted(defaults_from_config.keys())}")
        else:
            _config_debug("config 中无与解析器匹配的键，未应用默认值")
    else:
        _config_debug("未加载 config，全部使用脚本默认值；命令行参数直接覆盖脚本默认值")
    args = parser.parse_args()
    # 若指定了 global_seed，则覆盖 seed 与 stratify_seed，保持与 config 兼容（config 会保存 global_seed 及生效后的 seed/stratify_seed）
    if getattr(args, 'global_seed', None) is not None:
        args.seed = args.global_seed
        args.stratify_seed = args.global_seed
    # 避免无效值导致除零或空循环
    args.log_interval = max(1, int(args.log_interval))
    args.epochs = max(1, int(args.epochs))
    args.batch_size = max(1, int(args.batch_size))
    # 打印最终生效的若干关键参数，便于核对 config/命令行 覆盖是否正确
    if config_dict:
        _config_debug(
            f"最终生效（config + 命令行覆盖）: save_dir={args.save_dir!r}, epochs={args.epochs}, "
            f"batch_size={args.batch_size}, lr={args.lr}, resume={getattr(args, 'resume', '')!r}"
        )
    return args


def _args_to_dict(args):
    """
    将 args 转为可 JSON 序列化的 dict（仅 int/float/str/bool/None），便于保存 config.json 复现实验。
    所有通过 argparse 定义的参数（含 use_focal、label_smoothing、weight_decay、early_stopping_patience 等）
    只要值为基本类型都会写入，无需单独维护字段列表。
    """
    return {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}


def _args_to_command(args):
    """
    根据当前 args 生成完整可执行命令（含全部参数），写入日志后可直接复制复现。
    规则：布尔 True → --key，False 不输出；空字符串 / None 不输出；其余 --key value；含空格的值加引号。
    与 config.json 一致：config 中有的键都可按此规则还原为命令行。
    """
    parts = ['python', 'train.py']
    for k, v in vars(args).items():
        if not isinstance(v, (int, float, str, bool, type(None))):
            continue
        if v is None or v == '':
            continue
        if isinstance(v, bool):
            if v:
                parts.append(f'--{k}')
        else:
            s = str(v)
            if ' ' in s or '\n' in s:
                s = f'"{s}"'
            parts.append(f'--{k} {s}')
    return ' '.join(parts)


def get_transforms(img_size, is_train=True):
    """
    根据阶段返回数据增强与 ImageNet 归一化。
    - 训练：强增强以缓解过拟合（ISIC 光照/颜色差异大）：更大裁剪范围、45° 旋转、shear、强 ColorJitter（含 hue）。
    - 验证/测试：先 Resize 再 CenterCrop，无随机性；先略放大再裁以保留中心区域。
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def onehot_to_class(labels):
    """
    将 one-hot 标签转为类别索引（单标签时取 argmax）。
    输入：(B, C) float；输出：(B,) long，供 FocalLoss 与评估使用。
    """
    return labels.argmax(dim=1).long()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval,
                    gpu_temp_threshold=0, gpu_temp_cooldown=60):
    """
    训练一个 epoch：前向、反向、更新参数；返回该 epoch 平均 loss 与 Balanced Accuracy。
    每 log_interval 个 batch 打印当前平均 loss，并在 GPU 温度超过阈值时暂停冷却。
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

        if i == 1:
            print('[提示] 数据加载完成，现在应该可以安全使用 Ctrl+C 中断。')

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels_idx)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # 收集本 batch 的预测与真实标签，用于该 epoch 结束后的 Balanced Accuracy
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels_idx.cpu().numpy())
        
        # 按间隔打印当前平均 loss，并可选地检查 GPU 温度与过热冷却
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


def _run_eval_loop(model, loader, criterion, device):
    """
    内部：在 loader 上跑一遍前向，不更新参数。
    返回 (avg_loss, bal_acc, all_labels, all_preds)；loader 为空时返回 (0, 0, [], [])。
    """
    if len(loader) == 0:
        return 0.0, 0.0, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels_idx = onehot_to_class(labels.to(device))
            logits = model(images)
            loss = criterion(logits, labels_idx)
            running_loss += loss.item()
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels_idx.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return running_loss / len(loader), balanced_accuracy_score(all_labels, all_preds), all_labels, all_preds


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在给定 DataLoader 上评估，仅前向。返回 (平均 loss, Balanced Accuracy)。"""
    avg_loss, bal_acc, _, _ = _run_eval_loop(model, loader, criterion, device)
    return avg_loss, bal_acc


@torch.no_grad()
def evaluate_and_get_predictions(model, loader, criterion, device):
    """
    在给定 DataLoader 上做一次前向，同时得到 loss、Balanced Accuracy 与 (y_true, y_pred)。
    测试集只遍历一次即可画混淆矩阵并输出 loss/bal_acc。
    """
    avg_loss, bal_acc, all_labels, all_preds = _run_eval_loop(model, loader, criterion, device)
    return avg_loss, bal_acc, all_labels, all_preds


def _get_class_names(num_classes):
    """返回类别显示名：ISIC 8 类用官方名，否则用 0..num_classes-1。"""
    if num_classes == len(ISIC2019_CLASS_NAMES):
        return ISIC2019_CLASS_NAMES.copy()
    return [str(i) for i in range(num_classes)]


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path, class_names=None, save_csv_path=None):
    """
    绘制混淆矩阵并保存为 PNG；可选同时保存 CSV（便于论文表格与后续分析）。
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print('无预测结果，跳过混淆矩阵。')
        return
    if class_names is None:
        class_names = _get_class_names(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    # PNG
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.8), max(5, num_classes * 0.7)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'混淆矩阵已保存: {save_path}')
    # CSV（科研：便于制表与复现）
    if save_csv_path:
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(save_csv_path, encoding='utf-8')
        print(f'混淆矩阵 CSV 已保存: {save_csv_path}')


# 训练历史字典的键，与 plot_curves / checkpoint 一致（含每轮测试集指标）
_HISTORY_KEYS = ('train_loss', 'val_loss', 'train_bal_acc', 'val_bal_acc', 'test_loss', 'test_bal_acc')


def _empty_history():
    """返回空训练历史（六键，空列表）。"""
    return {k: [] for k in _HISTORY_KEYS}


def _normalize_history(history):
    """
    规范化 history：保证包含 _HISTORY_KEYS 各键，值为列表；
    长度取「有数据的键」的最短长度，避免绘图错位；兼容旧 checkpoint 无 test_* 的情况。
    """
    if not history or not isinstance(history, dict):
        return _empty_history()
    out = {}
    for k in _HISTORY_KEYS:
        v = history.get(k)
        out[k] = list(v) if isinstance(v, (list, np.ndarray)) else []
    lengths = [len(out[k]) for k in _HISTORY_KEYS if len(out[k]) > 0]
    n = min(lengths) if lengths else 0
    if n == 0:
        return out
    return {k: out[k][:n] for k in _HISTORY_KEYS}


def _history_to_tensors(history):
    """
    将 history（dict of lists）转为 dict of 1D float64 张量，
    以便 checkpoint 仅含张量，可用 torch.load(..., weights_only=True) 安全加载。
    """
    out = {}
    for k in _HISTORY_KEYS:
        lst = history.get(k) or []
        out[k] = torch.tensor([float(x) for x in lst], dtype=torch.float64)
    return out


def _history_from_tensors(history_tensors):
    """
    从 checkpoint 中保存的 history 张量恢复为 dict of lists（Python 标量）；
    缺键或非张量则对应键用空列表，最后做 _normalize_history。
    """
    if not history_tensors or not isinstance(history_tensors, dict):
        return _empty_history()
    out = {}
    for k in _HISTORY_KEYS:
        v = history_tensors.get(k)
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu().tolist()
        else:
            out[k] = []
    return _normalize_history(out)


def _load_best_for_eval(model, save_dir, device):
    """
    若 save_dir 下存在 best_model.pth，则将其模型权重加载到 model。
    用于最终测试集评估与混淆矩阵，确保使用验证集上最佳模型而非 last。
    """
    best_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print('已加载 best_model.pth 用于最终测试集评估与混淆矩阵。')


def _run_final_test_eval(model, test_loader, criterion, device, save_dir, num_classes, log_path):
    """
    加载 best 权重 → 测试集评估 → 混淆矩阵（PNG+CSV）→ 每类 P/R/F1 与 BalancedAcc 写日志并保存 CSV。
    三处收尾（正常结束 / resume 已满轮 / Ctrl+C）共用。
    """
    _load_best_for_eval(model, save_dir, device)
    test_loss, test_bal_acc, y_true, y_pred = evaluate_and_get_predictions(
        model, test_loader, criterion, device)
    class_names = _get_class_names(num_classes)
    cm_png = os.path.join(save_dir, 'confusion_matrix.png')
    cm_csv = os.path.join(save_dir, 'confusion_matrix.csv')
    plot_confusion_matrix(
        y_true, y_pred, num_classes, cm_png,
        class_names=class_names, save_csv_path=cm_csv,
    )
    if len(test_loader) > 0 and len(y_true) > 0:
        # 每类 precision / recall / F1（科研报告常用）
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=np.arange(num_classes), zero_division=0,
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0,
        )
        print(f'\n测试集  Loss: {test_loss:.4f}  BalancedAcc: {test_bal_acc:.4f}')
        print(f'  Macro  Precision: {macro_p:.4f}  Recall: {macro_r:.4f}  F1: {macro_f1:.4f}')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(_log_ts() + f'Test Loss: {test_loss:.4f}  Test BalancedAcc: {test_bal_acc:.4f}\n')
            f.write(_log_ts() + f'Test Macro  P: {macro_p:.4f}  R: {macro_r:.4f}  F1: {macro_f1:.4f}\n')
        # 每类指标保存 CSV，便于论文表格
        metrics_path = os.path.join(save_dir, 'test_metrics_per_class.csv')
        metrics_df = pd.DataFrame({
            'class': class_names,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support.astype(int),
        })
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8')
        print(f'  每类指标已保存: {metrics_path}')


def plot_curves(history, save_dir):
    """
    根据训练历史绘制 Loss 与 Balanced Accuracy 曲线（含测试集），保存到 save_dir。
    history 需包含 train_loss, val_loss, train_bal_acc, val_bal_acc；若有 test_loss/test_bal_acc 则同图绘制。
    """
    history = _normalize_history(history)
    if not history.get('train_loss'):
        print('无有效训练历史，跳过曲线保存。')
        return
    n = len(history['train_loss'])
    epochs = np.arange(1, n + 1)
    has_test = len(history.get('test_loss') or []) > 0
    # Loss 曲线：train / val，若有则加 test
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    if has_test:
        ax.plot(epochs, history['test_loss'], 'g-', label='Test Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training, Validation & Test Loss', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Loss 曲线已保存: ' + os.path.join(save_dir, 'loss_curve.png'))
    # Balanced Accuracy 曲线：train / val，若有则加 test
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history['train_bal_acc'], 'b-', label='Train Balanced Accuracy', linewidth=2)
    ax.plot(epochs, history['val_bal_acc'], 'r-', label='Val Balanced Accuracy', linewidth=2)
    if has_test:
        ax.plot(epochs, history['test_bal_acc'], 'g-', label='Test Balanced Accuracy', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Training, Validation & Test Balanced Accuracy', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'balanced_accuracy_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Balanced Accuracy 曲线已保存: ' + os.path.join(save_dir, 'balanced_accuracy_curve.png'))


def print_hyperparameters(args, device):
    """在训练开始前将当前超参数与设备信息打印到控制台。"""
    sections = [
        ('设备', [('设备', device)]),
        ('数据', [
            ('图像目录', args.img_dir), ('标签 CSV', args.csv_path),
            ('测试集图像目录', args.test_img_dir), ('测试集标签 CSV', args.test_csv_path),
            ('类别数', args.num_classes), ('输入尺寸', f'{args.img_size} x {args.img_size}'),
        ]),
        ('模型', [
            ('是否使用预训练', '否' if args.no_pretrained else '是'), ('dropout', 0.3),
        ]),
        ('训练', [
            ('训练轮数', args.epochs), ('批大小', args.batch_size), ('学习率', args.lr),
            ('optimizer', 'AdamW'), ('weight_decay', getattr(args, 'weight_decay', 1e-4)),
            ('scheduler', 'CosineAnnealingLR (T_max=epochs)'),
            ('criterion', 'FocalLoss(alpha, gamma)' if getattr(args, 'use_focal', False) else f'CrossEntropyLoss(weight, label_smoothing={getattr(args, "label_smoothing", 0.1)})'),
            ('early_stopping_patience', getattr(args, 'early_stopping_patience', 0)),
        ]),
        ('其他', [
            ('val_ratio', args.val_ratio), ('stratify_seed', args.stratify_seed),
            ('focal_gamma', args.focal_gamma), ('seed', args.seed),
            ('deterministic', '是' if getattr(args, 'deterministic', False) else '否'),
            ('保存目录', args.save_dir), ('log_interval', args.log_interval),
            ('gpu_temp_threshold', f'{args.gpu_temp_threshold}°C (0=不监测)'),
            ('gpu_temp_cooldown', f'{args.gpu_temp_cooldown}s'), ('num_workers', args.num_workers),
        ]),
    ]
    if getattr(args, 'global_seed', None) is not None:
        sections[-1][1].append(('global_seed (已覆盖 seed/stratify_seed)', args.global_seed))
    if args.resume:
        sections[-1][1].append(('继续训练 (resume)', args.resume))
    print('\n' + '=' * 60 + '\n  训练超参数\n' + '=' * 60)
    for title, items in sections:
        print(f'  [{title}]')
        for label, val in items:
            print(f'    {label}: {val}')
    print('=' * 60 + '\n')


def set_seed(seed, deterministic=False):
    """
    设置全局随机种子（torch / numpy / cuda），便于复现。
    deterministic=True 时开启 cudnn 确定性并关闭 benchmark，完全可复现但可能更慢。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id, base_seed=0):
    """
    DataLoader 子进程内设置随机种子，使 transform（RandomCrop/Rotation/ColorJitter 等）可复现。
    torchvision 使用 Python random 与 torch；需同时设置 random、np、torch 的种子。
    """
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _SeedWorkerInit:
    """可 pickle 的 worker 初始化器（Windows spawn 下 DataLoader 需可序列化）。"""
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, worker_id):
        _worker_init_fn(worker_id, self.seed)


def main():
    """
    主流程：解析参数 → 设置种子与设备 → 构建数据与模型 → 训练循环 →
    保存 best/last checkpoint → 绘制曲线 → 用 best 在测试集上评估并输出指标与混淆矩阵。
    支持 --resume 从中断处继续；Ctrl+C 时也会保存曲线并用 best 做测试集评估。
    科研复现：同一 seed + --deterministic，且 config.json 与日志中的 Command 可精确复现实验。
    """
    args = get_args()
    set_seed(args.seed, args.deterministic)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 科研：保存完整配置便于复现与论文方法描述
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(_args_to_dict(args), f, indent=2, ensure_ascii=False)
    print(f'配置已保存: {config_path}')

    print_hyperparameters(args, device)

    # 训练用带增强的 transform，验证/测试用 Resize+CenterCrop
    train_transform = get_transforms(args.img_size, is_train=True)
    val_transform = get_transforms(args.img_size, is_train=False)

    # 竞赛规范：从训练 CSV 做 Stratified 划分 train/val；测试集单独路径，仅最终评估一次
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
    n_train, n_val = len(train_dataset), len(val_dataset)
    n_test = len(test_dataset)

    # 可复现：固定 shuffle 的随机源；子进程内固定 numpy 种子
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        generator=train_generator,
        worker_init_fn=_SeedWorkerInit(args.seed),
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

    # Focal Loss 的 alpha：按训练子集类别计数求逆频，归一化后作为类别权重，缓解不均衡
    train_class_indices = np.argmax(labels_full[train_idx], axis=1)
    class_counts = np.bincount(train_class_indices, minlength=args.num_classes)
    class_weights = 1.0 / (class_counts.astype(np.float64) + 1e-5)
    class_weights = class_weights / class_weights.sum() * args.num_classes  # 归一化后总质量为 num_classes
    focal_alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 模型、优化器、学习率调度器
    model = build_efficientnet_b3(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        dropout=0.3,
    ).to(device)

    # 推荐 CE + class_weight + label_smoothing；Focal+class_weight 易过度强化少数类、泛化变差
    if getattr(args, 'use_focal', False):
        criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(
            weight=focal_alpha,
            label_smoothing=getattr(args, 'label_smoothing', 0.1),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=getattr(args, 'weight_decay', 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_path = os.path.join(args.save_dir, 'train_log.txt')
    # 默认从头训练；若 --resume 则下面会覆盖 start_epoch / best_bal_acc / history
    start_epoch, best_bal_acc = 1, 0.0
    history = _empty_history()

    # 若指定 --resume 且文件存在，加载 checkpoint 并从中断处继续（weights_only=True 安全加载）
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
            f.write(_log_ts() + f'\n[继续训练] 从 {args.resume} 恢复，从 Epoch {start_epoch} 继续\n\n')
    else:
        if args.resume:
            print(f'警告: --resume 指定路径不存在或不是文件，将从头训练: {args.resume}')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(_log_ts() + f'Start: {datetime.now().isoformat()}\n')
            f.write(_log_ts() + f'Command: {_args_to_command(args)}\n')
            f.write(_log_ts() + f'Dataset sizes: train={n_train} val={n_val} test={n_test}\n')
            f.write(_log_ts() + f'Args: {args}\n\n')

    # 恢复后若已超过目标轮数：不再训练，仅写日志、画曲线、用 best 做测试集评估与混淆矩阵
    if start_epoch > args.epochs:
        print(f'当前已训练至 Epoch {start_epoch - 1}，不小于目标轮数 {args.epochs}，无需继续训练。')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(_log_ts() + f'\nBest Val Balanced Accuracy: {best_bal_acc:.4f}\n')
            f.write(_log_ts() + f'End: {datetime.now().isoformat()}\n')
        if history.get('train_loss'):
            plot_curves(history, args.save_dir)
            _run_final_test_eval(
                model, test_loader, criterion, device,
                args.save_dir, args.num_classes, log_path)
        return

    try:
        early_patience = getattr(args, 'early_stopping_patience', 0)
        epochs_without_improvement = 0
        print('\n[提示] 仅在训练批次进度（如 [Epoch 1] Batch 100/1014  Loss: ...）输出时可安全使用 Ctrl+C；'
              '每轮验证/测试及每轮开始时加载数据时请勿中断，否则可能出错。\n')
        # 训练循环：每轮 train → evaluate → 记录 history → 若 val 更优则存 best，每轮存 last
        for epoch in range(start_epoch, args.epochs + 1):
            print(f'\n========== Epoch {epoch}/{args.epochs} ==========')
            print('[提示] 加载数据集中，现在暂不可使用Ctrl+C中断，请等待数据加载完成。')
            train_loss, train_bal_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, args.log_interval,
                gpu_temp_threshold=args.gpu_temp_threshold,
                gpu_temp_cooldown=args.gpu_temp_cooldown,
            )
            print('[提示] 正在验证，请勿使用 Ctrl+C 中断。')
            val_loss, val_bal_acc = evaluate(model, val_loader, criterion, device)
            # 每轮在测试集上评估一次（仅记录曲线，不参与选 best；best 仍由 val 决定）
            print('[提示] 正在测试集评估，请勿使用 Ctrl+C 中断。')
            test_loss, test_bal_acc = evaluate(model, test_loader, criterion, device)
            lr_now = scheduler.get_last_lr()[0]  # 先取 LR 再 step，打印与本期训练一致
            scheduler.step()

            # 控制台与日志文件写入本轮指标（含测试集）
            line = (
                f'Epoch {epoch:3d}  '
                f'Train Loss: {train_loss:.4f}  Train BalAcc: {train_bal_acc:.4f}  '
                f'Val Loss: {val_loss:.4f}  Val BalAcc: {val_bal_acc:.4f}  '
                f'Test Loss: {test_loss:.4f}  Test BalAcc: {test_bal_acc:.4f}  '
                f'LR: {lr_now:.2e}\n'
            )
            print(line.strip())
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(_log_ts() + line)

            # 用 float() 转成 Python 标量，避免 numpy 等类型进入 checkpoint
            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['train_bal_acc'].append(float(train_bal_acc))
            history['val_bal_acc'].append(float(val_bal_acc))
            history['test_loss'].append(float(test_loss))
            history['test_bal_acc'].append(float(test_bal_acc))

            # 每轮保存 last；验证集 Balanced Accuracy 创新高则额外保存 best（结构一致，便于 --resume）
            ckpt_base = {
                'epoch': int(epoch),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_bal_acc': float(best_bal_acc),
                'history': _history_to_tensors(history),
            }
            if val_bal_acc > best_bal_acc:
                best_bal_acc = val_bal_acc
                epochs_without_improvement = 0
                torch.save({**ckpt_base, 'val_balanced_accuracy': float(val_bal_acc)},
                          os.path.join(args.save_dir, 'best_model.pth'))
                print(f'  -> 保存最佳模型 (BalancedAcc={val_bal_acc:.4f}) -> {os.path.join(args.save_dir, "best_model.pth")}')
            else:
                epochs_without_improvement += 1
            torch.save(ckpt_base, os.path.join(args.save_dir, 'last_model.pth'))

            if early_patience > 0 and epochs_without_improvement >= early_patience:
                print(f'\n[Early Stopping] 验证集 {early_patience} 轮无提升，停止训练。')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(_log_ts() + f'\n[Early Stopping] Epoch {epoch} 验证集 {early_patience} 轮无提升\n')
                break

        # 正常结束：写最佳指标与结束时间到日志
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(_log_ts() + f'\nBest Val Balanced Accuracy: {best_bal_acc:.4f}\n')
            f.write(_log_ts() + f'End: {datetime.now().isoformat()}\n')
        print(f'\n训练结束。最佳 Val Balanced Accuracy: {best_bal_acc:.4f}')
        print(f'日志: {log_path}')

    except KeyboardInterrupt:
        # Ctrl+C 中断：写日志、保存已有曲线、用 best 做测试集评估与混淆矩阵
        print('\n\n训练被用户中断 (Ctrl+C)。')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(_log_ts() + f'\n[用户中断] {datetime.now().isoformat()}\n')
            f.write(_log_ts() + f'中断前最佳 Val Balanced Accuracy: {best_bal_acc:.4f}\n')
        if history['train_loss']:
            print('正在保存已完成的训练曲线...')
            plot_curves(history, args.save_dir)
            print('曲线已保存至:', args.save_dir)
            _run_final_test_eval(
                model, test_loader, criterion, device,
                args.save_dir, args.num_classes, log_path)
        else:
            print('尚无完整 epoch，跳过曲线保存。')
        print('已安全退出。')
        return

    plot_curves(history, args.save_dir)
    _run_final_test_eval(
        model, test_loader, criterion, device,
        args.save_dir, args.num_classes, log_path)


if __name__ == '__main__':
    main()
