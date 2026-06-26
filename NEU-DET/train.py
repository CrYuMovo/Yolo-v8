from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')

    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        amp=True,  # 保持你之前的设置
        workers=4
    )