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
        # dataset.py __init__ 函数

        # ... (循环前的代码保持不变) ...
        
        # --- 这是我们升级后的、带详细监控的循环 ---
        for i in range(len(self.image_files)):
            try:
                # 打印将要处理的文件，这是我们的“心跳”
                print(f"--> [Patient {i+1}/{len(self.image_files)}] Reading mask file: {self.mask_files[i]}")
                
                mask_path = os.path.join(self.data_dir, self.mask_files[i])
                mask_itk = sitk.ReadImage(mask_path)
                mask_array = sitk.GetArrayViewFromImage(mask_itk)
                
                num_slices = mask_itk.GetDepth()
                
                # 打印找到了多少切片
                print(f"    - Found {num_slices} total slices. Now checking for valid ones...")

                processed_slices_for_this_patient = 0
                for s in range(num_slices):
                    if np.sum(mask_array[s, :, :]) > 0:
                        self.slices.append((i, s))
                        processed_slices_for_this_patient += 1
                
                print(f"    - Done. Found {processed_slices_for_this_patient} valid slices for this patient.")

            except Exception as e:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"!!!!!! ERROR processing patient {i} ({self.image_files[i]}) !!!!!!")
                print(f"!!!!!! Error message: {e}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # 即使一个病人出错，我们也继续处理下一个
                continue
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