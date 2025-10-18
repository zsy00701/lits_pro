# scripts/train.py

import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import glob
import numpy as np

# --- 导入我们自己编写的所有模块 ---
from model import CustomUNet3D
from custom_dataset import LiTSDataset
from custom_data_utils import DiceCELoss, preprocess, sliding_window_inference

def main():
    # --- 1. 设置超参数和路径 ---
    data_dir = '/data1/zhoushengyao/projects/media/nas/01_Datasets/CT/LITS/Training Batch 2'
    output_dir = '/data1/zhoushengyao/projects/media/nas/01_Datasets/CT/LITS/Training Batch 2'
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    PATCH_SIZE = (96, 96, 96)
    BATCH_SIZE = 2
    EPOCHS = 100
    LEARNING_RATE = 1e-4

    # --- 2. 准备数据 ---
    # 实例化我们的数据集。这将触发预处理和 patch 生成
    full_dataset = LiTSDataset(data_dir, patch_size=PATCH_SIZE, num_samples_per_image=4)
    
    # 划分训练集和验证集 (80% 训练, 20% 验证)
    # 注意：我们这里是按 patch 划分的
    train_size = int(0.8 * len(full_dataset))
    val_patch_size = len(full_dataset) - train_size
    train_dataset, val_patch_dataset = random_split(full_dataset, [train_size, val_patch_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # --- 验证集策略 ---
    # 我们不在 patch 上验证，而是在完整的 3D 图像上验证
    # 所以我们需要一个验证文件列表，而不是 patch Dataloader
    all_files = full_dataset.file_pairs
    _, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    print(f"训练 patches: {len(train_dataset)}, 验证图像: {len(val_files)}")

    # --- 3. 初始化模型、损失函数、优化器 ---
    model = CustomUNet3D(n_channels=1, n_classes=3).to(device)
    loss_function = DiceCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 4. 训练和验证循环 ---
    best_val_dice = -1.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        
        # --- 训练阶段 ---
        model.train() # 设置为训练模式
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            
            # 1. 前向传播
            outputs = model(images)
            # 2. 计算损失
            loss = loss_function(outputs, labels)
            # 3. 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 4. 更新权重
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} 训练损失: {avg_train_loss:.4f}")

        # --- 验证阶段 ---
        # 我们每 5 个 epoch 验证一次，因为验证很耗时
        if (epoch + 1) % 5 == 0:
            model.eval() # 设置为评估模式
            total_val_dice = 0
            
            with torch.no_grad(): # 关闭梯度计算
                for img_path, lbl_path in tqdm(val_files, desc="Validating"):
                    # 1. 加载并预处理单张完整的 3D 验证图像
                    image, label = preprocess(img_path, lbl_path)
                    
                    # 2. 转换为 Tensor 并增加 Batch 和 Channel 维度
                    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
                    
                    # 3. 执行滑动窗口推理
                    pred_logits = sliding_window_inference(model, image_tensor, patch_size=PATCH_SIZE)
                    
                    # 4. 获取最终分割图 (D, H, W)
                    pred_seg = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()
                    
                    # 5. 计算 Dice (这里我们用 NumPy 简单计算)
                    # 类别 1: 肝脏
                    liver_pred = (pred_seg == 1)
                    liver_true = (label == 1)
                    dice_liver = (2. * np.sum(liver_pred & liver_true)) / (np.sum(liver_pred) + np.sum(liver_true) + 1e-5)
                    
                    # 类别 2: 肿瘤
                    tumor_pred = (pred_seg == 2)
                    tumor_true = (label == 2)
                    dice_tumor = (2. * np.sum(tumor_pred & tumor_true)) / (np.sum(tumor_pred) + np.sum(tumor_true) + 1e-5)
                    
                    # 我们关心肝脏和肿瘤的平均 Dice
                    total_val_dice += (dice_liver + dice_tumor) / 2
            
            avg_val_dice = total_val_dice / len(val_files)
            print(f"Epoch {epoch + 1} 验证 Dice (肝脏+肿瘤): {avg_val_dice:.4f}")
            
            # --- 保存最佳模型 ---
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                save_path = os.path.join(output_dir, "best_model_custom.pth")
                torch.save(model.state_dict(), save_path)
                print(f"新最佳模型已保存至: {save_path}")

    print("训练完成！")

if __name__ == "__main__":
    main()