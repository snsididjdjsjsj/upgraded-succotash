from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from model import DigitRecognizer
import numpy as np
import base64
import io
import cv2
from typing import Dict, Any

app = Flask(__name__)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitRecognizer().to(device)
model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
model.eval()

def preprocess_image(image):
    # 转换为numpy数组
    image = np.array(image)
    
    # 转换为灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 二值化
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 找到数字的边界框
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 获取最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 提取数字区域
        digit = image[y:y+h, x:x+w]
        
        # 计算填充量，使数字居中
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = digit
        
        # 调整大小为28x28
        image = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        # 如果没有找到轮廓，直接调整大小
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 转换为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 添加批次维度
    image = transform(image).unsqueeze(0)
    return image

def predict_digit(image_data):
    # 解码base64图像数据
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # 预处理图像
    image = preprocess_image(image)
    
    # 进行预测
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        pred = output.argmax(dim=1, keepdim=True)
        
        # 获取前三个最可能的预测结果
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
    digit = pred.item()
    confidence = probabilities[0][pred].item()
    
    # 返回预测结果和置信度
    return {
        'digit': digit,
        'confidence': f"{confidence:.2%}",
        'top3': [
            {'digit': idx.item(), 'confidence': f"{prob.item():.2%}"}
            for prob, idx in zip(top3_prob[0], top3_indices[0])
        ]
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.json['image']
        result = predict_digit(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 