import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from model import DigitRecognizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch.backends.cudnn as cudnn

# 启用CUDA优化
cudnn.benchmark = True

class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.1307,), std=(0.3081,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.1307,), std=(0.3081,)),
            ToTensorV2(),
        ])

def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader)
    total_loss = 0
    correct = 0
    
    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 使用自动混合精度
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = F.nll_loss(output, target)
        
        # 使用scaler进行反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 使用scaler进行优化器步进
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += float(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        correct += int(pred.eq(target.view_as(pred)).sum().item())
        
        if batch_idx % 100 == 0:
            current_loss = float(loss.item())
            pbar.set_description(f'Train Epoch: {epoch} Loss: {current_loss:.6f}')

    accuracy = 100. * correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                output = model(data)
                test_loss += float(F.nll_loss(output, target, reduction='sum').item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += int(pred.eq(target.view_as(pred)).sum().item())
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    test_loss = float(test_loss) / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # 计算每个类别的准确率
    class_correct = [0] * 10
    class_total = [0] * 10
    for pred, target in zip(all_preds, all_targets):
        class_correct[int(target)] += int(pred == target)
        class_total[int(target)] += 1
    
    class_accuracies = [100. * float(correct) / float(total) if total > 0 else 0.0 
                       for correct, total in zip(class_correct, class_total)]
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    print('\nPer-class accuracy:')
    for i, acc in enumerate(class_accuracies):
        print(f'Class {i}: {acc:.2f}%')
    
    return accuracy, class_accuracies

def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查是否可以使用CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        # 设置CUDA设备
        torch.cuda.set_device(0)

    # 加载MNIST数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True)
    test_dataset = datasets.MNIST('./data', train=False)
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # 应用数据增强
    train_dataset = AlbumentationsDataset(train_dataset, transform=get_transforms('train'))
    val_dataset = AlbumentationsDataset(val_dataset, transform=get_transforms('val'))
    test_dataset = AlbumentationsDataset(test_dataset, transform=get_transforms('val'))

    # 创建数据加载器，增加num_workers和pin_memory
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, 
                          num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, 
                           num_workers=4, pin_memory=True)

    # 创建模型实例并移至GPU
    model = DigitRecognizer().to(device)
    
    # 如果有多个GPU，可以使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 使用OneCycleLR调度器
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, 
                          max_lr=0.01,
                          epochs=30,
                          steps_per_epoch=steps_per_epoch,
                          pct_start=0.3,
                          anneal_strategy='cos')

    # 训练模型
    best_accuracy = 0
    epochs = 30
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}')
        
        # 训练阶段
        train_loss, train_acc = train(model, device, train_loader, optimizer, scheduler, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证阶段
        val_acc, _ = test(model, device, val_loader)
        val_accuracies.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            # 如果使用DataParallel，需要保存model.module
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), 'mnist_model.pth')
            else:
                torch.save(model.state_dict(), 'mnist_model.pth')
            print(f"New best model saved with accuracy: {val_acc:.2f}%")

    # 最终测试
    print("\nFinal test on test set:")
    test_acc, class_accuracies = test(model, device, test_loader)
    
    # 绘制训练过程
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    # 绘制每个类别的准确率
    plt.figure(figsize=(10, 5))
    plt.bar(range(10), class_accuracies)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(10))
    plt.savefig('class_accuracies.png')
    plt.close()

if __name__ == '__main__':
    main() 