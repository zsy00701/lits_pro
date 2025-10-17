import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 从我们自己写的文件中导入
from dataset import LiTSDataset
from model import UNet

# --- 1. 配置参数 (这是你的“控制面板”) ---
DATA_DIR = "/data1/zhoushengyao/projects/media/nas/01_Datasets/CT/LITS/Training_Batch2" # <--- 必须修改成你的数据路径!
BATCH_SIZE = 8     
LEARNING_RATE = 1e-4
EPOCHS = 20       

def main():
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 3. 准备数据 ---
    print("Loading data...")
    dataset = LiTSDataset(data_dir=DATA_DIR)
    # DataLoader负责把数据分批 (batch)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print("Data loaded!")

    # --- 4. 初始化模型、损失函数、优化器 ---
    # U-Net的输入是1个通道(灰度图)，输出是1个通道(二分类的mask)
    model = UNet(n_channels=1, n_classes=1).to(device)
    
    # 损失函数：BCEWithLogitsLoss 在输出前会先做sigmoid，更稳定
    criterion = nn.BCEWithLogitsLoss()
    
    # 优化器：Adam是常用的选择
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. 训练循环 ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train() # 设置为训练模式
        epoch_loss = 0.0
        
        # 使用tqdm来显示进度条
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, masks in progress_bar:
            # 把数据移动到GPU
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, masks)
            
            # 反向传播和优化
            optimizer.zero_grad() # 梯度清零
            loss.backward()       # 计算梯度
            optimizer.step()      # 更新权重
            
            epoch_loss += loss.item()
            
            # 更新进度条上的损失显示
            progress_bar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {epoch_loss / len(dataloader)}")

    # --- 6. 保存模型 ---
    torch.save(model.state_dict(), 'unet_lits.pth')
    print("Training finished. Model saved to unet_lits.pth")

if __name__ == '__main__':
    main()