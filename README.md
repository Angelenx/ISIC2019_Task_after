# ISIC 2019 皮肤病变分类训练脚本

基于 **EfficientNet-B3** 的 ISIC 2019 八类皮肤病变分类训练脚本，面向竞赛规范与科研复现：Stratified 划分、Focal Loss、Balanced Accuracy、完整日志与可复现设置。

---

## 功能概览

| 项目 | 说明 |
|------|------|
| **模型** | EfficientNet-B3，默认 ImageNet 预训练，8 类分类头 |
| **损失** | Focal Loss + 按训练集类别统计的权重，缓解类别不均衡 |
| **评估** | Balanced Accuracy；验证集选 best，测试集仅最终评估一次 |
| **数据划分** | 训练集 Stratified 划分 train/val；测试集单独路径，无泄露 |
| **输出** | 曲线图、混淆矩阵（PNG+CSV）、每类 P/R/F1、config 与完整命令行日志 |

---

## 环境与依赖

- Python 3.8+
- PyTorch 1.9+（建议带 CUDA）
- 其他见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

主要依赖：`torch`, `torchvision`, `pandas`, `Pillow`, `scikit-learn`, `numpy`, `matplotlib`。

---

## 数据格式

### 目录与 CSV 约定

- **训练**：`--img_dir` 下放训练图像，`--csv_path` 为标签 CSV。
- **测试**：`--test_img_dir` 下放测试图像，`--test_csv_path` 为测试标签 CSV。

CSV 格式：

- **第 1 列**：图像文件名（不含 `.jpg`），脚本会自动补 `.jpg`。
- **第 2～9 列**：8 类 **one-hot** 标签，每行有且仅有一个 `1`，其余为 `0`。

类别顺序与脚本内 `ISIC2019_CLASS_NAMES` 一致：  
`MEL`, `NV`, `BCC`, `AKIEC`, `BKL`, `DF`, `VASC`, `SCC`。

示例（表头可有可无，列顺序固定）：

```
image_name, MEL, NV, BCC, AKIEC, BKL, DF, VASC, SCC
ISIC_001, 0, 1, 0, 0, 0, 0, 0, 0
...
```

数据集会在加载时检查：每行 one-hot 和为 1、且仅一个 1；CSV 至少 9 列，否则报错。

---

## 使用方法

### 1. 基本运行（使用默认路径与超参）

脚本内默认路径为 Windows 下的本地路径，若你的数据就在该位置，可直接：

```bash
python train.py
```

若数据在其他目录，请通过参数覆盖（见下）。

### 2. 指定数据路径与保存目录

```bash
python train.py --img_dir /path/to/train/images \
               --csv_path /path/to/train_ground_truth.csv \
               --test_img_dir /path/to/test/images \
               --test_csv_path /path/to/test_ground_truth.csv \
               --save_dir ./my_checkpoints
```

### 3. 常用参数示例

```bash
python train.py \
  --img_dir ./data/train_images \
  --csv_path ./data/train_gt.csv \
  --test_img_dir ./data/test_images \
  --test_csv_path ./data/test_gt.csv \
  --save_dir ./checkpoints \
  --epochs 80 \
  --batch_size 26 \
  --lr 1e-4 \
  --val_ratio 0.2 \
  --seed 1688
```

### 4. 继续训练（断点续训）

从上次保存的 `last_model.pth` 继续：

```bash
python train.py --resume ./checkpoints/last_model.pth --save_dir ./checkpoints
```

`--save_dir` 建议与首次训练一致，以便沿用同目录下的 best/last 与日志。

### 5. 科研复现（完全可复现）

使用固定种子并开启 cudnn 确定性（速度可能略慢）：

```bash
python train.py --seed 42 --deterministic --save_dir ./exp_repro
```

同一 `seed` + `--deterministic`，再配合 `save_dir` 下的 `config.json` 与 `train_log.txt` 中的完整 Command，可尽量复现同一次实验。

### 6. 其他常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_classes` | 8 | 类别数 |
| `--img_size` | 300 | 输入边长（EfficientNet-B3 常用 300） |
| `--no_pretrained` | 否 | 加上则不用 ImageNet 预训练 |
| `--focal_gamma` | 2.0 | Focal Loss 的 gamma |
| `--stratify_seed` | 1688 | train/val 分层划分的随机种子 |
| `--num_workers` | 2 | DataLoader 子进程数 |
| `--log_interval` | 50 | 每多少个 batch 打印一次训练 loss |
| `--gpu_temp_threshold` | 85 | GPU 温度阈值（°C），0 表示不监测 |
| `--gpu_temp_cooldown` | 60 | 过热时暂停冷却秒数 |

---

## 输出文件说明

所有输出默认在 `--save_dir`（如 `./checkpoints`）下。

| 文件 | 说明 |
|------|------|
| **config.json** | 本次运行的全部参数（可 JSON 反序列化），便于复现与写方法 |
| **train_log.txt** | 训练日志：完整命令行、数据量、每轮 train/val 指标、测试集指标、每类 P/R/F1 |
| **best_model.pth** | 验证集 Balanced Accuracy 最高的模型（用于最终测试评估与混淆矩阵） |
| **last_model.pth** | 最后一轮模型，可用于 `--resume` 继续训练 |
| **loss_curve.png** | 训练/验证 Loss 曲线 |
| **balanced_accuracy_curve.png** | 训练/验证 Balanced Accuracy 曲线 |
| **confusion_matrix.png** | 测试集混淆矩阵图 |
| **confusion_matrix.csv** | 测试集混淆矩阵表格（便于论文/报告） |
| **test_metrics_per_class.csv** | 测试集每类 precision、recall、f1、support |

训练结束后（或 Ctrl+C 后），会用 **best 模型**在测试集上算一次指标并生成混淆矩阵与上述 CSV。

---

## 流程简述

1. 解析参数 → 设置随机种子 → 创建 `save_dir`，写入 `config.json`。
2. 按 `--stratify_seed` 对训练 CSV 做 **Stratified** 划分得到 train/val，测试集单独从 `test_img_dir` + `test_csv_path` 读取。
3. 构建 EfficientNet-B3、Focal Loss（类别权重由训练子集统计）、AdamW、CosineAnnealingLR。
4. 若提供 `--resume` 且文件存在，则加载该 checkpoint 继续训练；否则从头训练。
5. 每轮：训练 → 验证 → 记录 history → 若验证 Balanced Accuracy 创新高则保存 best，每轮保存 last。
6. 正常结束或 Ctrl+C：绘制曲线 → 加载 best → 在测试集上评估 → 输出混淆矩阵（PNG+CSV）与每类 P/R/F1，并写入日志与 CSV。

---

## 注意事项

- **数据**：确保 CSV 列为「图像名 + 8 列 one-hot」，且每行恰好一个 1；图像文件名需与 CSV 第一列一致（脚本会自动加 `.jpg`）。
- **测试集**：测试集仅在全部训练流程结束后评估一次，用于报告最终结果，不参与选 best。
- **复现**：多进程时脚本已通过 `generator` 与 `worker_init_fn` 固定 shuffle；要尽量完全复现请加 `--deterministic`。
- **GPU**：若需温度保护，可设置 `--gpu_temp_threshold`（如 85）与 `--gpu_temp_cooldown`（如 60）；设为 0 则不监测。

---

## 项目结构（与训练相关）

```
ISIC2019_Task_after/
├── README.md           # 本说明
├── requirements.txt
├── train.py            # 训练入口
├── Mydataset/
│   └── __init__.py     # ISIC2019Dataset（CSV + one-hot 校验）
└── models/
    ├── __init__.py     # build_efficientnet_b3
    └── efficientnet_b3.py
```

运行前请保证数据路径与 CSV 格式符合上述约定，并按需修改 `--img_dir`、`--csv_path`、`--test_img_dir`、`--test_csv_path` 和 `--save_dir`。
