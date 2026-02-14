import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
import pandas as pd
from PIL import Image


class ISIC2019Dataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.transform = transform
        # 读取csv标签
        df = pd.read_csv(csv_path)
        # 假设第一列为图片名，后面9列为标签
        self.img_names = (df.iloc[:, 0].astype(str) + '.jpg').tolist()
        self.labels = df.iloc[:, 1:9].values
        
            
        self.name2idx = {name: i for i, name in enumerate(self.img_names)}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        label = self.labels[idx]
        # 单类别 one-hot：每行恰好一个 1，避免多标签/异常行导致 argmax 静默错误
        assert np.isclose(label.sum(), 1.0) and (np.isclose(label, 1.0).sum() == 1), (
            f'样本 {idx} 标签非单类别 one-hot（sum={label.sum()}, 非零数={(label > 0.5).sum()}）'
        )
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # 设置数据集路径和转换
    img_dir = 'D:\\Documents\\VSproject\\graduation_project\\SIFTNeXt\\dataset_files\\ISIC_2019_Training_Input'
    csv_path = 'D:\\Documents\\VSproject\\graduation_project\\SIFTNeXt\\dataset_files\\ISIC_2019_Training_GroundTruth.csv'
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ISIC2019Dataset(img_dir, csv_path, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    i = 1
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        print(images.shape, labels.shape)
        i+=1
        if i > 10:
            break

