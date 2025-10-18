# scripts/custom_dataset.py

import torch
from torch.utils.data import Dataset
import glob
import os
import numpy as np
from tqdm import tqdm

# 从我们刚写的文件中导入函数
from custom_data_utils import preprocess, get_random_patches

class LiTSDataset(Dataset):
    def __init__(self, data_dir, patch_size=(96, 96, 96), num_samples_per_image=4):
        self.patch_size = patch_size
        self.num_samples = num_samples_per_image

        # 1. 获取所有图像和标签的路径
        image_paths = sorted(glob.glob(os.path.join(data_dir, "volume-*.nii.gz")))
        label_paths = sorted(glob.glob(os.path.join(data_dir, "segmentation-*.nii.gz")))
        self.file_pairs = list(zip(image_paths, label_paths))
        
        # --- 关键策略：预处理并缓存所有 patch ---
        # 这种策略会占用大量内存（RAM），但会使训练时的 I/O 速度飞快
        # 因为我们是在 __init__ 中一次性加载所有数据
        self.all_patches = self._generate_all_patches()

    def _generate_all_patches(self):
        print("开始预处理和生成所有训练 patches...")
        all_patches = []
        
        # 遍历所有 131 个（或更少）训练图像
        for img_path, lbl_path in tqdm(self.file_pairs):
            # 2. 对每个图像执行完整的预处理
            image, label = preprocess(img_path, lbl_path)
            
            # 3. 从中采样 N 个 patches
            patches = get_random_patches(image, label, self.patch_size, self.num_samples)
            
            all_patches.extend(patches)
            
        print(f"总共生成了 {len(all_patches)} 个 patches.")
        return all_patches

    def __len__(self):
        # 数据集的总长度就是 patch 的总数
        return len(self.all_patches)

    def __getitem__(self, idx):
        # 从缓存中取出一个 patch
        image_patch, label_patch = self.all_patches[idx]
        
        # --- 在这里执行“在线”数据增强 ---
        # 例如，随机翻转
        if torch.rand(1) > 0.5:
            # 沿 Z 轴 (dim=1) 翻转
            image_patch = torch.flip(image_patch, dims=[1])
            label_patch = torch.flip(label_patch, dims=[0])
        if torch.rand(1) > 0.5:
            # 沿 Y 轴 (dim=2) 翻转
            image_patch = torch.flip(image_patch, dims=[2])
            label_patch = torch.flip(label_patch, dims=[1])
        if torch.rand(1) > 0.5:
            # 沿 X 轴 (dim=3) 翻转
            image_patch = torch.flip(image_patch, dims=[3])
            label_patch = torch.flip(label_patch, dims=[2])

        return {"image": image_patch, "label": label_patch}