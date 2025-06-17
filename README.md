# 手写数字识别系统  

一个基于深度学习的智能手写数字识别系统，使用PyTorch框架和MNIST数据集构建。该系统能够准确识别0-9的手写数字，并提供Web界面进行实时预测。  

## 功能特点  

- **高精度识别**：使用优化的CNN架构，在MNIST测试集上达到99%+的准确率  
- **Web界面**：提供直观的Flask Web应用，支持实时手写数字识别  
- **数据增强**：使用Albumentations库进行高级数据增强，提高模型泛化能力  
- **混合精度训练**：支持自动混合精度训练，加速训练过程  
- **可视化分析**：提供训练曲线、类别准确率等详细的可视化分析  
- **GPU加速**：支持CUDA加速，大幅提升训练和推理速度  
- **模型优化**：使用OneCycleLR调度器和AdamW优化器，实现最佳训练效果  

## 环境要求  

- Python 3.8+  
- PyTorch 2.0.0+  
- CUDA 11.0+ (可选，用于GPU加速)  

## 安装指南  

### 1. 克隆项目  
```bash  
git clone <repository-url>  
cd 手写数字识别  
```  

### 2. 创建虚拟环境（推荐）  
```bash  
python -m venv venv  
# Windows  
venv\Scripts\activate  
# Linux/Mac  
source venv/bin/activate  
```  

### 3. 安装依赖  
```bash  
pip install -r requirements.txt  
```  

## 使用方法  

### 模型训练  

1. **开始训练**:  
```bash  
python train.py  
```  

2. **训练过程**:  
   - 自动下载MNIST数据集  
   - 应用数据增强策略  
   - 使用混合精度训练  
   - 生成训练曲线图  
   - 保存最佳模型  

3. **训练输出**:  
   - `mnist_model.pth`: 训练好的模型  
   - `training_curves.png`: 训练过程可视化  
   - `class_accuracies.png`: 各类别准确率分析  

### 命令行预测  

```bash  
python predict.py  
```  
**注意**: 修改 `predict.py` 中的 `image_path` 为您的图片路径。  

### Web应用  

1. **启动Web服务**:  
```bash  
python app.py  
```  

2. **访问应用**:  
   - 打开浏览器访问 `http://localhost:5000`  
   - 在画布上绘制数字  
   - 点击预测按钮获取结果  

## 项目结构  

```  
手写数字识别/  
├── app.py              # Flask Web应用  
├── model.py            # CNN模型定义  
├── train.py            # 模型训练脚本  
├── predict.py          # 命令行预测脚本  
├── requirements.txt    # 项目依赖  
├── README.md           # 项目说明  
├── mnist_model.pth     # 训练好的模型  
├── data/               # MNIST数据集  
└── templates/          # Web模板  
    └── index.html      # 主页面  
```  

## 模型架构  

### CNN网络结构  
- **输入层**: 28×28×1 灰度图像  
- **卷积层1**: 32个5×5卷积核，ReLU激活  
- **池化层1**: 2×2最大池化  
- **卷积层2**: 64个5×5卷积核，ReLU激活  
- **池化层2**: 2×2最大池化  
- **全连接层1**: 1024个神经元，ReLU激活，Dropout(0.5)  
- **全连接层2**: 10个神经元（对应0-9数字）  

### 训练策略  
- **优化器**: AdamW (lr=0.001, weight_decay=0.01)  
- **调度器**: OneCycleLR (max_lr=0.01, epochs=30)  
- **损失函数**: 负对数似然损失 (NLL Loss)  
- **数据增强**: 旋转、弹性变换、网格扭曲、亮度对比度调整  

## 性能指标  

- **测试集准确率**: 99.2%+  
- **训练时间**: ~5分钟 (GPU), ~30分钟 (CPU)  
- **推理速度**: <10ms (GPU), <50ms (CPU)  

## 高级功能  

### 数据增强策略  
```python  
# 训练时数据增强  
- 随机旋转 (±15度)  
- 弹性变换  
- 网格扭曲  
- 亮度对比度调整  
- 随机90度旋转  
```  

### 混合精度训练  
- 使用 `torch.cuda.amp` 进行自动混合精度训练  
- 减少显存使用，加速训练过程  
- 保持模型精度不变  

### 模型优化  
- 梯度裁剪防止梯度爆炸  
- 权重衰减防止过拟合  
- 学习率调度优化收敛  

## 故障排除  

### 常见问题  

1. **CUDA内存不足**:  
   ```bash  
   # 减少批次大小  
   # 在train.py中修改batch_size=128  
   ```  

2. **Flask应用无法启动**:  
   ```bash  
   # 检查端口占用  
   netstat -ano | findstr :5000  
   # 或修改端口  
   app.run(debug=True, port=5001)  
   ```  

3. **模型预测不准确**:  
   - 确保输入图像为灰度图  
   - 数字应该居中显示  
   - 图像大小会自动调整为28×28  

## 扩展功能  

- 支持更多数字数据集  
- 添加字母识别功能  
- 实现模型量化  
- 添加模型解释性分析  
- 支持批量预测  

## 贡献指南  

欢迎提交Issue和Pull Request来改进项目！  

## 许可证  

本项目采用MIT许可证。  

## 联系方式  

如有问题或建议，请通过以下方式联系：  
- 提交GitHub Issue  
- 发送邮件至3535474218@qq.com  

---  

**注意**: 首次运行时会自动下载MNIST数据集，请确保网络连接正常。