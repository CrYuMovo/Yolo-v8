# train_steel.py （你自己创建的文件）
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",  # 👈 指向你数据集根目录下的 data.yaml
    epochs=50,
    imgsz=416,
    batch=8,
    device="cpu",
    workers=1,
    project="runs_steel",
    name="steel_v1",
)
