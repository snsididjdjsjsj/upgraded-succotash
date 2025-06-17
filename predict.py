import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from model import DigitRecognizer

def preprocess_image(image_path):
    # 打开图片并转换为灰度图
    image = Image.open(image_path).convert('L')
    
    # 调整图片大小为28x28
    image = image.resize((28, 28))
    
    # 转换为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 添加批次维度
    image = transform(image).unsqueeze(0)
    return image

def predict_digit(image_path, model_path='mnist_model.pth'):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitRecognizer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 预处理图片
    image = preprocess_image(image_path)
    image = image.to(device)

    # 进行预测
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)
        probabilities = F.softmax(output, dim=1)
        
    return pred.item(), probabilities[0][pred].item()

if __name__ == '__main__':
    # 示例使用
    image_path = '1.png'  # 替换为你的图片路径
    try:
        digit, confidence = predict_digit(image_path)
        print(f'预测的数字是: {digit}')
        print(f'置信度: {confidence:.2%}')
    except Exception as e:
        print(f'预测过程中出现错误: {str(e)}') 