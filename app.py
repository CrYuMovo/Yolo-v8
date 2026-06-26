from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import base64
import numpy as np
import os
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 确保必要目录存在
os.makedirs('static', exist_ok=True)
os.makedirs('temp', exist_ok=True)


@app.route('/')
def index():
    return send_file('static/index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    # 使用唯一ID避免多用户并发时临时文件互相覆盖
    task_id = str(uuid.uuid4())[:8]
    model_path = f'temp/model_{task_id}.pt'
    image_path = f'temp/input_{task_id}.jpg'

    try:
        # 1. 获取上传的文件与自定义参数
        model_file = request.files.get('model')
        image_file = request.files.get('image')

        if not model_file or not image_file:
            return jsonify({'error': '请同时上传模型和图片'}), 400

        # ✅ 接收前端传来的 conf 和 iou，若未传则使用默认值
        conf_threshold = float(request.form.get('conf', 0.25))
        iou_threshold = float(request.form.get('iou', 0.45))

        # 保存临时文件
        model_file.save(model_path)
        image_file.save(image_path)

        # 2. 加载模型并使用自定义参数推理
        model = YOLO(model_path)
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            save=False,
            verbose=False
        )
        result = results[0]

        # 3. 提取统计数据
        boxes = result.boxes
        total_count = len(boxes)

        stats = {
            "total": total_count,
            "defects": total_count,  # 若需区分缺陷/正常类别，可在此处按 cls 过滤
            "avg_conf": round(float(boxes.conf.mean()), 4) if total_count > 0 else 0,
            "infer_ms": round(result.speed.get('inference', 0), 1)
        }

        # 4. 将标注图转为 Base64
        annotated_bgr = result.plot()  # BGR 格式 numpy 数组
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', annotated_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # ✅ 返回 JSON（图片Base64 + 统计数据）
        return jsonify({
            "image_base64": img_base64,
            "stats": stats
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # 5. 无论成功失败都清理临时文件
        for path in [model_path, image_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)