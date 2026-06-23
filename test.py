from ultralytics import YOLO
import torch

print(f"YOLOv8 源码导入成功！")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")  # Intel核显应为 False
print(f"当前设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")

#本文件测试yolo是否成功导入，无实际作用