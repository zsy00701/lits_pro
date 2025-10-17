# dataset.py
import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np

class LiTSDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.startswith('volume')])
        self.mask_files = sorted([f for f in os.listdir(data_dir) if f.startswith('segmentation')])
        
        self.slices = []
        for i in range(len(self.image_files)):
            # 读取一个病人的完整3D影像
            img_path = os.path.join(self.data_dir, self.image_files[i])
            img_itk = sitk.ReadImage(img_path)
            
            # 获取切片数量，为每个切片创建一个索引
            num_slices = img_itk.GetDepth()
            for s in range(num_slices):
                # 我们只训练包含肝脏的切片，以节省时间 (标签>0)
                mask_path = os.path.join(self.data_dir, self.mask_files[i])
                mask_itk = sitk.ReadImage(mask_path)
                mask_slice = sitk.GetArrayViewFromImage(mask_itk)[s, :, :]
                if np.sum(mask_slice) > 0: # 如果这个切片里有肝脏
                    self.slices.append((i, s)) # (病人索引, 切片索引)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        patient_idx, slice_idx = self.slices[idx]
        
        # --- 读取和处理图像 ---
        img_path = os.path.join(self.data_dir, self.image_files[patient_idx])
        img_itk = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img_itk) # (D, H, W)
        
        # 提取单个2D切片
        img_slice = img_array[slice_idx, :, :].astype(np.float32)
        
        # --- 窗位窗宽调整 (非常重要的医学图像预处理步骤) ---
        # 腹部CT的典型窗位窗宽
        window_center = 40
        window_width = 400
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        img_slice = np.clip(img_slice, min_val, max_val)
        
        # --- 归一化到 0-1 ---
        img_slice = (img_slice - min_val) / window_width
        
        # --- 读取和处理标签 ---
        mask_path = os.path.join(self.data_dir, self.mask_files[patient_idx])
        mask_itk = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask_itk)
        
        mask_slice = mask_array[slice_idx, :, :]
        # 我们只做二分类：背景(0) vs 肝脏(1)。肝脏和肿瘤都算作1。
        mask_slice[mask_slice > 0] = 1 
        mask_slice = mask_slice.astype(np.float32)

        # --- 转换成Torch Tensor ---
        # 添加一个通道维度 (C, H, W)
        img_tensor = torch.from_numpy(img_slice).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0)
        
        return img_tensor, mask_tensor