"""
ISIC 2019 训练脚本：EfficientNet-B3，Balanced Accuracy 评估，输出中间结果
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from Mydataset import ISIC2019Dataset
from models import build_efficientnet_b3


def get_args():
    parser = argparse.ArgumentParser(description='ISIC2019 EfficientNet-B3 训练')
    parser.add_argument('--img_dir', type=str,
                        default=r'D:\Documents\VSproject\graduation_project\SIFTNeXt\dataset_files\ISIC_2019_Training_Input',
                        help='训练图像目录')
    parser.add_argument('--csv_path', type=str,
                        default=r'D:\Documents\VSproject\graduation_project\SIFTNeXt\dataset_files\ISIC_2019_Training_GroundTruth.csv',
                        help='标签 CSV 路径')
    parser.add_argument('--num_classes', type=int, default=8, help='类别数')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 进程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型与日志保存目录')
    parser.add_argument('--log_interval', type=int, default=50, help='每多少 batch 打印一次训练 loss')
    parser.add_argument('--no_pretrained', action='store_true', help='不使用 ImageNet 预训练')
    parser.add_argument('--img_size', type=int, default=300, help='输入图像边长（EfficientNet-B3 常用 300）')
    return parser.parse_args()


def get_transforms(img_size, is_train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])


def onehot_to_class(labels):
    """(B, C) one-hot float -> (B,) long 类别索引"""
    return labels.argmax(dim=1).long()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval):
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
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels_idx.cpu().numpy())

        if (i + 1) % log_interval == 0:
            avg_loss = running_loss / (i + 1)
            print(f'  [Epoch {epoch}] Batch {i+1}/{len(loader)}  Loss: {avg_loss:.4f}')

    avg_loss = running_loss / len(loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, bal_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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


def plot_curves(history, save_dir):
    """绘制 Loss 曲线和 Balanced Accuracy 曲线并保存"""
    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Loss 曲线
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

    # Balanced Accuracy 曲线
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


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Args: {args}')

    train_transform = get_transforms(args.img_size, is_train=True)
    val_transform = get_transforms(args.img_size, is_train=False)

    # 若没有单独验证集，可用同一 CSV 做“验证”（仅做指标参考）
    train_dataset = ISIC2019Dataset(args.img_dir, args.csv_path, transform=train_transform)
    val_dataset = ISIC2019Dataset(args.img_dir, args.csv_path, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_efficientnet_b3(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_path = os.path.join(args.save_dir, 'train_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'Start: {datetime.now().isoformat()}\n')
        f.write(f'Args: {args}\n\n')

    best_bal_acc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_bal_acc': [],
        'val_bal_acc': [],
    }

    for epoch in range(1, args.epochs + 1):
        print(f'\n========== Epoch {epoch}/{args.epochs} ==========')
        train_loss, train_bal_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.log_interval
        )
        val_loss, val_bal_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        # 中间结果输出
        line = (
            f'Epoch {epoch:3d}  '
            f'Train Loss: {train_loss:.4f}  Train BalancedAcc: {train_bal_acc:.4f}  '
            f'Val Loss: {val_loss:.4f}  Val BalancedAcc: {val_bal_acc:.4f}  '
            f'LR: {lr_now:.2e}\n'
        )
        print(line.strip())
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_bal_acc'].append(train_bal_acc)
        history['val_bal_acc'].append(val_bal_acc)

        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            ckpt_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_balanced_accuracy': val_bal_acc,
                'args': args,
            }, ckpt_path)
            print(f'  -> 保存最佳模型 (BalancedAcc={val_bal_acc:.4f}) -> {ckpt_path}')

        # 每轮保存最新 checkpoint（可选）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(args.save_dir, 'last_model.pth'))

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f'\nBest Val Balanced Accuracy: {best_bal_acc:.4f}\n')
        f.write(f'End: {datetime.now().isoformat()}\n')
    print(f'\n训练结束。最佳 Val Balanced Accuracy: {best_bal_acc:.4f}')
    print(f'日志: {log_path}')

    # 绘制并保存 Loss 与 Balanced Accuracy 曲线
    plot_curves(history, args.save_dir)


if __name__ == '__main__':
    main()
