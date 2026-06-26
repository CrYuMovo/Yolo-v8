"""
工业缺陷检测系统 — Flask 后端
功能：YOLO推理(V5/V8兼容)、模型管理、SQLite持久化、RBAC权限、数据集介绍、缺陷知识库、历史记录、报告导出、SSE进度推送
"""
from flask import Flask, request, jsonify, send_file, Response, session, stream_with_context
from ultralytics import YOLO
import cv2
import base64
import numpy as np
import os
import sys
import uuid
import sqlite3
import json
import csv
import io
import hashlib
import xml.etree.ElementTree as ET
import time
import threading
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB
app.secret_key = 'industrial-defect-detection-secret-2024'

for d in ['static', 'temp', 'models']:
    os.makedirs(d, exist_ok=True)

DB_PATH = 'detection.db'
SSE_PROGRESS = {}  # task_id → {"current": N, "total": M, "stage": "..."}

# ═══════════════ 数据集统计 ═══════════════
DATASET_INFO = {
    "name": "NEU-DET", "full_name": "Northeastern University Surface Defect Database",
    "description": "NEU-DET 是由东北大学发布的钢材表面缺陷数据集，广泛用于工业缺陷检测算法的训练与评估。"
                   "数据集包含热轧带钢表面的 6 种典型缺陷类型，共计 1,800 张灰度图像。",
    "classes": 6, "total_samples": 1800, "image_size": "200×200 像素",
    "image_type": "灰度图 (8-bit)", "format": "PNG + XML标注 (Pascal VOC格式)",
    "source": "东北大学 (Northeastern University, China)", "year": 2013,
    "application": "热轧带钢表面质量检测",
    "train_samples": 1440, "valid_samples": 360, "split_ratio": "8:2",
    "class_distribution": {"crazing": 300, "inclusion": 300, "patches": 300,
                           "pitted_surface": 300, "rolled-in_scale": 300, "scratches": 300}
}

# ═══════════════ 缺陷知识库 ═══════════════
DEFECT_KNOWLEDGE = [
    {"en": "crazing", "cn": "裂纹",
     "cause": "热应力或机械应力导致钢材表面产生网状裂纹，常见于热轧后冷却不均或外力冲击区域。",
     "appearance": "表面呈现不规则网状细纹，类似龟裂状纹理，在灰度图像中呈现为密集交错的深色细线。",
     "impact": "降低钢材的抗拉强度和疲劳寿命，裂纹可能在使用过程中持续扩展，最终导致构件断裂失效。",
     "severity": "高"},
    {"en": "inclusion", "cn": "夹杂物",
     "cause": "炼钢过程中脱氧、脱硫不充分，导致非金属杂质（如氧化铝、硫化锰等）残留在钢基体中。",
     "appearance": "点状或短条状的异色（通常偏暗）区域，边界清晰，单个体积较小但分布随机。",
     "impact": "破坏钢基体的连续性，在受力时成为应力集中点和裂纹萌生源。",
     "severity": "中高"},
    {"en": "patches", "cn": "斑块",
     "cause": "表面局部氧化、腐蚀或轧制前除鳞不彻底，造成区域性表面变色或质地变化。",
     "appearance": "不规则形状的深色或浅色斑块区域，面积较大，边界模糊。",
     "impact": "影响产品外观质量和后续涂层附着力，严重时可能导致局部腐蚀加速。",
     "severity": "中"},
    {"en": "pitted_surface", "cn": "麻面",
     "cause": "点蚀（局部电化学腐蚀）或热轧时氧化皮压入表面后脱落，形成密集分布的微小凹坑。",
     "appearance": "表面呈现密集分布的微小点状凹陷，在灰度图像中表现为均匀散布的暗色小斑点。",
     "impact": "降低表面光洁度和尺寸精度，凹坑底部可能成为应力集中点。",
     "severity": "中"},
    {"en": "rolled-in_scale", "cn": "氧化铁皮",
     "cause": "热轧过程中，加热炉内形成的一次氧化皮或轧制中产生的二次氧化皮被压入带钢表面。",
     "appearance": "片状或带状深色（近黑色）区域，沿轧制方向延伸，面积较大且边界不规则。",
     "impact": "严重影响后续冷轧、酸洗、镀层等工序质量；残留氧化皮会加速局部腐蚀。",
     "severity": "高"},
    {"en": "scratches", "cn": "划痕",
     "cause": "钢材在生产、搬运或加工过程中与硬物发生相对滑动摩擦，导致表面材料被机械去除。",
     "appearance": "沿某一方向的线性浅槽，通常呈连续的直线或弧线，宽度较窄。",
     "impact": "破坏表面完整性，划痕底部产生应力集中，可能成为疲劳裂纹的起始位置。",
     "severity": "中低"},
]

SAMPLE_IMAGES = {
    "crazing": "crazing_10.jpg", "inclusion": "inclusion_10.jpg",
    "patches": "patches_10.jpg", "pitted_surface": "pitted_surface_10.jpg",
    "rolled-in_scale": "rolled-in_scale_10.jpg", "scratches": "scratches_10.jpg",
}

# ═══════════════ 数据集对比 ═══════════════
DATASET_COMPARISON = [
    {"name": "NEU-DET", "classes": 6, "samples": 1800, "resolution": "200×200",
     "domain": "热轧带钢表面", "year": 2013, "format": "灰度图",
     "strength": "类别均衡、标注规范、学术界广泛使用",
     "weakness": "图像分辨率较低、场景单一"},
    {"name": "KSDD2", "classes": 1, "samples": 3335, "resolution": "可变（约500×1250）",
     "domain": "钢材表面（多种）", "year": 2019, "format": "灰度图",
     "strength": "样本量大、包含多种钢材类型",
     "weakness": "仅单一缺陷类别、负样本定义模糊"},
    {"name": "DAGM", "classes": 10, "samples": 12000, "resolution": "512×512",
     "domain": "多种工业表面", "year": 2007, "format": "灰度图",
     "strength": "类别丰富、样本量充足",
     "weakness": "合成图像占比大、与真实工业场景有偏差"},
    {"name": "Magnetic Tile", "classes": 5, "samples": 1344, "resolution": "可变",
     "domain": "磁砖表面", "year": 2018, "format": "彩色图",
     "strength": "真实生产线采集、包含不同光照条件",
     "weakness": "类别间样本不均衡、场景单一"},
]

# ═══════════════ 流程图数据（已修复特殊字符） ═══════════════
FLOWCHART_DATA = {
    "system_architecture": """
graph TB
    subgraph 用户层
        A[浏览器前端<br/>HTML/CSS/JS]
    end
    subgraph 应用层
        B[Flask Web Server<br/>app.py]
        C[YOLO推理引擎<br/>ultralytics/torch.hub]
    end
    subgraph 数据层
        D[SQLite数据库<br/>detection.db]
        E[模型文件<br/>models/*.pt]
        F[图像临时存储<br/>temp/]
    end
    A -->|HTTP请求| B
    B -->|加载模型| C
    B -->|读写| D
    B -->|读取| E
    B -->|临时存储| F
    C -->|推理结果| B
    B -->|JSON响应| A
""",
    "detection_flow": """
flowchart TD
    A[接收POST请求<br/>model和image文件] --> B{文件完整性校验}
    B -->|缺失| C[返回400错误]
    B -->|完整| D[保存临时文件<br/>UUID命名防冲突]
    D --> E[智能加载模型<br/>自动检测YOLOv5/v8]
    E --> F[执行推理<br/>model.predict<br/>conf/iou参数]
    F --> G[提取检测框<br/>xyxy坐标和置信度]
    G --> H[生成标注图<br/>OpenCV绘制边界框]
    H --> I[编码Base64图片]
    I --> J[统计检测数据<br/>总数/均值/耗时]
    J --> K[持久化到SQLite<br/>关联操作用户]
    K --> L[返回JSON响应<br/>图片和统计数据]
    D --> M[finally清理临时文件]
    C --> M
    L --> M
""",
    "data_flow": """
flowchart LR
    subgraph 输入
        A1[模型文件 .pt]
        A2[图片文件 .jpg/.png]
        A3[参数 conf/iou]
    end
    subgraph 处理
        B1[UUID生成任务ID]
        B2[临时文件存储]
        B3[YOLO模型推理]
        B4[OpenCV图像处理]
    end
    subgraph 输出
        C1[Base64标注图]
        C2[统计数据JSON]
        C3[检测详情数组]
        C4[SQLite持久化记录]
    end
    A1 --> B2
    A2 --> B2
    A3 --> B3
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> C1
    B3 --> C2
    B3 --> C3
    C2 --> C4
    C3 --> C4
""",
    "api_structure": """
graph LR
    subgraph 检测API
        A1[POST /upload 单图检测]
        A2[POST /upload/batch 批量检测]
    end
    subgraph 认证API
        AUTH1[POST /api/login]
        AUTH2[POST /api/logout]
        AUTH3[GET /api/me]
    end
    subgraph 数据API
        C1[GET /api/dataset/info]
        C2[GET /api/dataset/compare]
        C3[GET /api/defects]
    end
    subgraph 历史API
        D1[GET /api/history]
        D2[GET /api/history/stats]
    end
    subgraph 工具API
        E1[GET /api/export]
        E2[GET /api/flowchart]
    end
"""
}

# ═══════════════ 数据库 ═══════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detection_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL, model_name TEXT NOT NULL, image_name TEXT NOT NULL,
        conf_threshold REAL DEFAULT 0.25, iou_threshold REAL DEFAULT 0.45,
        total_count INTEGER DEFAULT 0, avg_confidence REAL DEFAULT 0,
        infer_time_ms REAL DEFAULT 0, defect_details TEXT DEFAULT '[]',
        username TEXT DEFAULT '', model_type TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now','localtime'))
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'operator',
        created_at TEXT DEFAULT (datetime('now','localtime'))
    )''')
    # 确保默认 admin 存在
    admin_exists = c.execute("SELECT 1 FROM users WHERE username='admin'").fetchone()
    if not admin_exists:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                  ('admin', hashlib.sha256('admin123'.encode()).hexdigest(), 'admin'))
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                  ('operator', hashlib.sha256('operator123'.encode()).hexdigest(), 'operator'))
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                  ('viewer', hashlib.sha256('viewer123'.encode()).hexdigest(), 'viewer'))
    # 尝试添加新列（兼容旧数据库）
    for col, col_def in [('username', "TEXT DEFAULT ''"), ('model_type', "TEXT DEFAULT ''")]:
        try:
            c.execute(f"ALTER TABLE detection_history ADD COLUMN {col} {col_def}")
        except sqlite3.OperationalError:
            pass  # 列已存在
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ═══════════════ 模型扫描 ═══════════════
def scan_models():
    models = []
    for f in sorted(os.listdir('models')) if os.path.exists('models') else []:
        if f.endswith('.pt'):
            path = os.path.join('models', f)
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
            nl = f.lower()
            if 'yolov3' in nl: mtype = 'YOLOv3'
            elif 'yolov5' in nl: mtype = 'YOLOv5'
            elif 'yolov6' in nl: mtype = 'YOLOv6'
            elif 'yolov7' in nl: mtype = 'YOLOv7'
            elif 'yolov8' in nl: mtype = 'YOLOv8'
            elif 'yolov9' in nl: mtype = 'YOLOv9'
            elif 'yolov10' in nl: mtype = 'YOLOv10'
            elif 'yolo11' in nl: mtype = 'YOLO11'
            elif 'yolo12' in nl: mtype = 'YOLO12'
            elif 'yolo26' in nl: mtype = 'YOLO26'
            elif 'rtdetr' in nl or 'rt-detr' in nl: mtype = 'RT-DETR'
            elif 'sam' in nl: mtype = 'SAM'
            else: mtype = '自定义'
            models.append({"name": f.replace('.pt', ''), "filename": f, "path": path,
                          "type": mtype, "size_mb": size_mb})
    return models

# ═══════════════ 权限装饰器 ═══════════════
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': '请先登录', 'need_login': True}), 401
        return f(*args, **kwargs)
    return decorated

def require_role(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'error': '请先登录', 'need_login': True}), 401
            if session.get('role') not in roles:
                return jsonify({'error': '权限不足'}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

# ═══════════════ YOLOv5/v8 兼容推理 ═══════════════

# 全局缓存 YOLOv5 模型，避免重复加载
_yolov5_model_cache = {}
_yolov5_legacy_imported = False

def _import_yolov5_legacy():
    """延迟导入 YOLOv5 本地模块（仅在需要时添加路径）"""
    global _yolov5_legacy_imported
    if not _yolov5_legacy_imported:
        legacy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5_legacy')
        if legacy_dir not in sys.path:
            sys.path.insert(0, legacy_dir)
        _yolov5_legacy_imported = True

def _infer_yolov5(model_path, image_path, conf, iou):
    """使用本地 YOLOv5 代码加载模型并推理（零网络依赖）"""
    import torch, importlib

    _import_yolov5_legacy()

    # 使用 importlib 避免 ultralytics 包污染 models/utils 命名空间
    def _v5_import(module_name):
        return importlib.import_module(module_name)

    # 加载模型（缓存以避免重复加载）
    model_key = f'{model_path}'
    if model_key not in _yolov5_model_cache:
        attempt_load = _v5_import('models.experimental').attempt_load
        select_device = _v5_import('utils.torch_utils').select_device
        device = select_device('cpu')
        model = attempt_load(model_path, map_location=device)
        if hasattr(model, 'module'):
            names = model.module.names
        else:
            names = model.names
        _yolov5_model_cache[model_key] = (model, names, device)
    model, names, device = _yolov5_model_cache[model_key]

    model.eval()
    img_size = 640

    # ── 预处理 ──
    letterbox = _v5_import('utils.datasets').letterbox
    im0 = cv2.imread(image_path)
    if im0 is None:
        raise ValueError(f'无法读取图片: {image_path}')
    img = im0[:, :, ::-1]
    img, ratio, pad = letterbox(img, new_shape=img_size, auto=True)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # ── 推理 ──
    t0 = time.time()
    with torch.no_grad():
        pred = model(img)[0]
    infer_time = round((time.time() - t0) * 1000, 1)

    # ── NMS ──
    general = _v5_import('utils.general')
    non_max_suppression = general.non_max_suppression
    scale_coords = general.scale_coords
    plot_one_box = general.plot_one_box
    pred = non_max_suppression(pred, conf_thres=conf, iou_thres=iou)

    # ── 提取结果 ──
    boxes_xyxy = []
    boxes_conf = []
    boxes_cls = []
    boxes_names = []
    annotated = im0.copy()

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, c, cls_id in reversed(det):
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                cls_id_int = int(cls_id)
                boxes_xyxy.append([float(x1), float(y1), float(x2), float(y2)])
                boxes_conf.append(float(c))
                boxes_cls.append(cls_id_int)
                name = names[cls_id_int] if cls_id_int < len(names) else str(cls_id_int)
                boxes_names.append(name)
                label_text = f'{name} {c:.2f}'
                plot_one_box([x1, y1, x2, y2], annotated, label=label_text,
                             color=(0, 255, 0), line_thickness=2)

    speed = {'inference': infer_time, 'preprocess': 0, 'postprocess': 0}
    return boxes_xyxy, boxes_conf, boxes_cls, boxes_names, annotated, speed


def _is_yolov5_checkpoint(model_path):
    """通过检查 checkpoint 元数据判断是否为 YOLOv5 模型"""
    import torch
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5_legacy'))
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        model_cls = type(ckpt.get('model', None))
        return model_cls.__module__ == 'models.yolo'
    except Exception:
        return False


def _load_and_infer(model_path, image_path, conf, iou):
    """智能加载模型并推理 — 兼容 YOLOv5 和 YOLOv8"""
    # 检测模型版本（必须在加载前判断，避免 ultralytics 污染 V5 的 fuse）
    if _is_yolov5_checkpoint(model_path):
        xyxy, confs, cls, names, annotated_bgr, speed = _infer_yolov5(
            model_path, image_path, conf, iou)
        return xyxy, confs, cls, names, annotated_bgr, speed, 'YOLOv5'

    # YOLOv8+
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, iou=iou, save=False, verbose=False)
    r = results[0]
    boxes = r.boxes
    boxes_xyxy = boxes.xyxy.tolist() if boxes is not None and len(boxes) > 0 else []
    boxes_conf = boxes.conf.tolist() if boxes is not None and len(boxes) > 0 else []
    boxes_cls = boxes.cls.tolist() if boxes is not None and len(boxes) > 0 else []
    names_dict = model.names if hasattr(model, 'names') else {}
    boxes_names = [names_dict.get(int(c), str(int(c))) for c in boxes_cls]
    annotated_bgr = r.plot()
    speed = r.speed
    return boxes_xyxy, boxes_conf, boxes_cls, boxes_names, annotated_bgr, speed, 'YOLOv8'


def _build_defect_details(boxes_xyxy, boxes_conf, boxes_cls, boxes_names):
    """根据推理结果构建 defect_details 列表"""
    details = []
    for i in range(len(boxes_xyxy)):
        details.append({
            "class_id": int(boxes_cls[i]),
            "class_name": boxes_names[i],
            "confidence": round(float(boxes_conf[i]), 4),
            "bbox": [round(float(x), 1) for x in boxes_xyxy[i]]
        })
    return details

# ═══════════════ 路由：页面 ═══════════════
@app.route('/')
def index():
    return send_file('static/index.html')

# ═══════════════ 路由：认证 ═══════════════
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or not password:
        return jsonify({'error': '请输入用户名和密码'}), 400
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    if not user:
        return jsonify({'error': '用户不存在'}), 401
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    if user['password_hash'] != pwd_hash:
        return jsonify({'error': '密码错误'}), 401
    session['user_id'] = user['id']
    session['username'] = user['username']
    session['role'] = user['role']
    return jsonify({'username': user['username'], 'role': user['role']})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'message': '已登出'})

@app.route('/api/me', methods=['GET'])
def api_me():
    if 'user_id' not in session:
        return jsonify({'logged_in': False})
    return jsonify({'logged_in': True, 'username': session.get('username'), 'role': session.get('role')})

@app.route('/api/users', methods=['GET'])
@require_role('admin')
def api_list_users():
    conn = get_db()
    users = conn.execute("SELECT id, username, role, created_at FROM users ORDER BY id").fetchall()
    conn.close()
    return jsonify({'users': [dict(u) for u in users]})

@app.route('/api/users', methods=['POST'])
@require_role('admin')
def api_create_user():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    role = data.get('role', 'operator')
    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400
    if role not in ('admin', 'operator', 'viewer'):
        return jsonify({'error': '无效的角色'}), 400
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = get_db()
    try:
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                     (username, pwd_hash, role))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': '用户名已存在'}), 409
    conn.close()
    return jsonify({'message': f'用户 {username} 创建成功'})

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@require_role('admin')
def api_delete_user(user_id):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return jsonify({'error': '用户不存在'}), 404
    if user['username'] == 'admin':
        conn.close()
        return jsonify({'error': '不能删除 admin 账号'}), 403
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': '已删除'})

# ═══════════════ 路由：模型列表 ═══════════════
@app.route('/api/models', methods=['GET'])
def api_models():
    return jsonify({"models": scan_models()})

# ═══════════════ 路由：缺陷知识库 ═══════════════
@app.route('/api/defects', methods=['GET'])
def api_defects():
    return jsonify({"defects": DEFECT_KNOWLEDGE})

@app.route('/api/defect/image/<class_name>', methods=['GET'])
def api_defect_sample_image(class_name):
    if class_name not in SAMPLE_IMAGES:
        return jsonify({'error': f'未找到类别 {class_name} 的样本图'}), 404
    img_filename = SAMPLE_IMAGES[class_name]
    img_path = os.path.join('NEU-DET', 'train', 'images', img_filename)
    xml_path = os.path.join('NEU-DET', 'ANNOTATIONS', img_filename.replace('.jpg', '.xml'))
    if not os.path.exists(img_path):
        return jsonify({'error': f'样本图片不存在'}), 404
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return jsonify({'error': '无法读取图片'}), 500
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_bgr = cv2.resize(img_bgr, (600, 600), interpolation=cv2.INTER_CUBIC)
    scale = 3.0
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        for obj in tree.getroot().iter('object'):
            name = obj.find('name').text
            bb = obj.find('bndbox')
            xmin = int(float(bb.find('xmin').text) * scale)
            ymin = int(float(bb.find('ymin').text) * scale)
            xmax = int(float(bb.find('xmax').text) * scale)
            ymax = int(float(bb.find('ymax').text) * scale)
            cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img_bgr, name, (xmin, max(ymin - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"class_name": class_name, "image_base64": img_base64})

# ═══════════════ 路由：模型性能对比 ═══════════════
@app.route('/api/model/compare', methods=['GET'])
@login_required
def api_model_compare():
    """使用一张测试图片对比所有可用模型的性能"""
    models = scan_models()
    if len(models) < 2:
        return jsonify({'error': '至少需要2个模型才能对比'}), 400
    # 找一张测试图
    test_img = None
    test_dir = os.path.join('NEU-DET', 'valid', 'images')
    if os.path.exists(test_dir):
        imgs = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        if imgs:
            test_img = os.path.join(test_dir, imgs[0])
    if not test_img:
        return jsonify({'error': '未找到测试图片'}), 404
    results_list = []
    for m in models:
        try:
            t0 = time.time()
            xyxy, confs, cls, names, _, _, mtype = _load_and_infer(m['path'], test_img, 0.25, 0.45)
            elapsed = round((time.time() - t0) * 1000, 1)
            results_list.append({
                "model_name": m['name'], "model_type": mtype, "size_mb": m['size_mb'],
                "detections": len(xyxy), "infer_time_ms": elapsed
            })
        except Exception as e:
            results_list.append({
                "model_name": m['name'], "model_type": m['type'],
                "size_mb": m['size_mb'], "detections": 0, "infer_time_ms": 0, "error": str(e)[:80]
            })
    return jsonify({"test_image": os.path.basename(test_img), "results": results_list})

# ═══════════════ 路由：数据集 ═══════════════
@app.route('/api/dataset/info', methods=['GET'])
def api_dataset_info():
    return jsonify(DATASET_INFO)

@app.route('/api/dataset/compare', methods=['GET'])
def api_dataset_compare():
    return jsonify({"datasets": DATASET_COMPARISON})

@app.route('/api/flowchart', methods=['GET'])
def api_flowchart():
    return jsonify(FLOWCHART_DATA)

# ═══════════════ 路由：检测 ═══════════════
def _resolve_model_path(request_obj, task_id):
    model_file = request_obj.files.get('model')
    preset_path = request_obj.form.get('model_path', '').strip()
    if model_file and model_file.filename:
        model_path = f'temp/model_{task_id}.pt'
        model_file.save(model_path)
        return model_path, model_file.filename
    if preset_path and os.path.exists(preset_path):
        return preset_path, os.path.basename(preset_path)
    return None, None

@app.route('/upload', methods=['POST'])
@require_role('admin', 'operator')
def upload_files():
    task_id = str(uuid.uuid4())[:8]
    image_path = f'temp/input_{task_id}.jpg'
    model_path = None
    try:
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({'error': '请上传待检测图片'}), 400
        model_path, model_name = _resolve_model_path(request, task_id)
        if not model_path:
            return jsonify({'error': '请上传模型文件或选择预设模型'}), 400
        conf_threshold = float(request.form.get('conf', 0.25))
        iou_threshold = float(request.form.get('iou', 0.45))
        image_file.save(image_path)

        xyxy, confs, cls, names, annotated_bgr, speed, model_type = _load_and_infer(
            model_path, image_path, conf_threshold, iou_threshold)
        defect_details = _build_defect_details(xyxy, confs, cls, names)
        total_count = len(defect_details)
        avg_conf = round(float(np.mean(confs)), 4) if confs else 0
        infer_ms = round(speed.get('inference', 0), 1) if speed else 0

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', annotated_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        conn = get_db()
        conn.execute('''INSERT INTO detection_history
            (task_id, model_name, image_name, conf_threshold, iou_threshold,
             total_count, avg_confidence, infer_time_ms, defect_details, username, model_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (task_id, model_name, image_file.filename, conf_threshold, iou_threshold,
             total_count, avg_conf, infer_ms, json.dumps(defect_details, ensure_ascii=False),
             session.get('username', ''), model_type))
        conn.commit(); conn.close()

        return jsonify({
            "task_id": task_id, "image_base64": img_base64,
            "stats": {"total": total_count, "defects": total_count, "avg_conf": avg_conf, "infer_ms": infer_ms},
            "details": defect_details, "model_type": model_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        for p in [f'temp/model_{task_id}.pt', image_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass

@app.route('/upload/batch', methods=['POST'])
@require_role('admin', 'operator')
def upload_batch():
    task_id = str(uuid.uuid4())[:8]
    model_path = None
    try:
        image_files = request.files.getlist('images')
        if not image_files:
            return jsonify({'error': '请上传至少一张图片'}), 400
        model_path, model_name = _resolve_model_path(request, task_id)
        if not model_path:
            return jsonify({'error': '请上传模型文件或选择预设模型'}), 400
        conf_threshold = float(request.form.get('conf', 0.25))
        iou_threshold = float(request.form.get('iou', 0.45))

        # 首次加载模型（之后复用）
        model = None
        use_v5 = False
        try:
            model = YOLO(model_path)
        except Exception as e:
            if 'YOLOv5' in str(e) or 'NOT forwards compatible' in str(e):
                use_v5 = True
                _import_yolov5_legacy()
                import torch, importlib as _v5lib
                def _v5i(m): return _v5lib.import_module(m)
                attempt_load = _v5i('models.experimental').attempt_load
                select_device = _v5i('utils.torch_utils').select_device
                device_obj = select_device('cpu')
                model = attempt_load(model_path, map_location=device_obj)
                if hasattr(model, 'module'):
                    v5_names = model.module.names
                else:
                    v5_names = model.names
                model.eval()
            else:
                raise

        total_images = len(image_files)
        SSE_PROGRESS[task_id] = {"current": 0, "total": total_images, "stage": "starting"}
        batch_results = []
        all_details = []

        for idx, img_file in enumerate(image_files):
            SSE_PROGRESS[task_id] = {"current": idx + 1, "total": total_images, "stage": "processing"}
            img_path = f'temp/batch_{task_id}_{img_file.filename}'
            img_file.save(img_path)

            if use_v5:
                import torch, importlib as _v5lib2
                def _v5i2(m): return _v5lib2.import_module(m)
                letterbox = _v5i2('utils.datasets').letterbox
                general = _v5i2('utils.general')
                non_max_suppression = general.non_max_suppression
                scale_coords = general.scale_coords
                plot_one_box = general.plot_one_box

                im0 = cv2.imread(img_path)
                img_v5 = im0[:, :, ::-1]
                img_v5, ratio, pad = letterbox(img_v5, new_shape=640, auto=True)
                img_v5 = img_v5.transpose((2, 0, 1))[::-1]
                img_v5 = np.ascontiguousarray(img_v5)
                img_v5 = torch.from_numpy(img_v5).to(device_obj).float() / 255.0
                if img_v5.ndimension() == 3:
                    img_v5 = img_v5.unsqueeze(0)

                with torch.no_grad():
                    pred = model(img_v5)[0]
                pred = non_max_suppression(pred, conf_thres=conf_threshold, iou_thres=iou_threshold)

                xyxy, confs, cls, names = [], [], [], []
                annotated_bgr = im0.copy()
                for det in pred:
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img_v5.shape[2:], det[:, :4], im0.shape).round()
                        for *xy, c, cls_id in reversed(det):
                            x1, y1, x2, y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
                            cls_i = int(cls_id)
                            xyxy.append([float(x1), float(y1), float(x2), float(y2)])
                            confs.append(float(c))
                            cls.append(cls_i)
                            nm = v5_names[cls_i] if cls_i < len(v5_names) else str(cls_i)
                            names.append(nm)
                            plot_one_box([x1, y1, x2, y2], annotated_bgr, label=f'{nm} {c:.2f}',
                                         color=(0, 255, 0), line_thickness=2)
                infer_ms = 0
                mtype = 'YOLOv5'
            else:
                results = model.predict(source=img_path, conf=conf_threshold, iou=iou_threshold, save=False, verbose=False)
                r = results[0]
                boxes = r.boxes
                xyxy = boxes.xyxy.tolist() if boxes is not None and len(boxes) > 0 else []
                confs = boxes.conf.tolist() if boxes is not None and len(boxes) > 0 else []
                cls = boxes.cls.tolist() if boxes is not None and len(boxes) > 0 else []
                names_dict = model.names if hasattr(model, 'names') else {}
                names = [names_dict.get(int(c), str(int(c))) for c in cls]
                annotated_bgr = r.plot()
                infer_ms = round(r.speed.get('inference', 0), 1)
                mtype = 'YOLOv8'

            defect_details = _build_defect_details(xyxy, confs, cls, names)
            all_details.extend(defect_details)
            total = len(defect_details)
            avg_c = round(float(np.mean(confs)), 4) if confs else 0

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', annotated_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            conn = get_db()
            conn.execute('''INSERT INTO detection_history
                (task_id, model_name, image_name, conf_threshold, iou_threshold,
                 total_count, avg_confidence, infer_time_ms, defect_details, username, model_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (task_id, model_name, img_file.filename, conf_threshold, iou_threshold,
                 total, avg_c, infer_ms, json.dumps(defect_details, ensure_ascii=False),
                 session.get('username', ''), mtype))
            conn.commit(); conn.close()

            batch_results.append({
                "image_name": img_file.filename, "image_base64": img_base64,
                "stats": {"total": total, "defects": total, "avg_conf": avg_c, "infer_ms": infer_ms},
                "details": defect_details, "model_type": mtype
            })
            if os.path.exists(img_path):
                os.remove(img_path)

        SSE_PROGRESS[task_id] = {"current": total_images, "total": total_images, "stage": "done"}
        return jsonify({"task_id": task_id, "model_name": model_name, "batch": batch_results, "all_details": all_details})
    except Exception as e:
        SSE_PROGRESS[task_id] = {"current": 0, "total": 0, "stage": "error", "error": str(e)}
        return jsonify({'error': str(e)}), 500
    finally:
        temp_model = f'temp/model_{task_id}.pt'
        if os.path.exists(temp_model):
            try: os.remove(temp_model)
            except OSError: pass

@app.route('/api/progress/<task_id>', methods=['GET'])
def api_progress(task_id):
    """SSE 实时进度推送"""
    def generate():
        last = None
        while True:
            info = SSE_PROGRESS.get(task_id, {"current": 0, "total": 0, "stage": "unknown"})
            current_data = json.dumps(info, ensure_ascii=False)
            if current_data != last:
                yield f"data: {current_data}\n\n"
                last = current_data
            if info.get('stage') in ('done', 'error'):
                break
            time.sleep(0.3)
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

# ═══════════════ 路由：历史记录 ═══════════════
@app.route('/api/history', methods=['GET'])
@login_required
def api_history():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    offset = (page - 1) * per_page
    conn = get_db()
    username = session.get('username', '')
    role = session.get('role', '')
    if role == 'admin':
        total = conn.execute('SELECT COUNT(*) FROM detection_history').fetchone()[0]
        rows = conn.execute('SELECT * FROM detection_history ORDER BY id DESC LIMIT ? OFFSET ?',
                           (per_page, offset)).fetchall()
    else:
        total = conn.execute('SELECT COUNT(*) FROM detection_history WHERE username=?',
                            (username,)).fetchone()[0]
        rows = conn.execute('SELECT * FROM detection_history WHERE username=? ORDER BY id DESC LIMIT ? OFFSET ?',
                           (username, per_page, offset)).fetchall()
    conn.close()
    records = [{"id": r['id'], "task_id": r['task_id'], "model_name": r['model_name'],
                "image_name": r['image_name'], "conf_threshold": r['conf_threshold'],
                "iou_threshold": r['iou_threshold'], "total_count": r['total_count'],
                "avg_confidence": r['avg_confidence'], "infer_time_ms": r['infer_time_ms'],
                "defect_details": json.loads(r['defect_details']) if r['defect_details'] else [],
                "username": r['username'] or '', "model_type": r['model_type'] or '',
                "created_at": r['created_at']} for r in rows]
    return jsonify({"total": total, "page": page, "per_page": per_page,
                    "total_pages": max(1, -(-total // per_page)), "records": records})

@app.route('/api/history/<int:record_id>', methods=['GET'])
@login_required
def api_history_detail(record_id):
    conn = get_db()
    r = conn.execute('SELECT * FROM detection_history WHERE id=?', (record_id,)).fetchone()
    conn.close()
    if not r:
        return jsonify({'error': '记录不存在'}), 404
    return jsonify(dict(r))

@app.route('/api/history/<int:record_id>', methods=['DELETE'])
@login_required
def api_history_delete(record_id):
    conn = get_db()
    r = conn.execute('SELECT * FROM detection_history WHERE id=?', (record_id,)).fetchone()
    if not r:
        conn.close(); return jsonify({'error': '记录不存在'}), 404
    role = session.get('role', '')
    username = session.get('username', '')
    if role != 'admin' and r['username'] != username:
        conn.close(); return jsonify({'error': '无权删除此记录'}), 403
    conn.execute('DELETE FROM detection_history WHERE id=?', (record_id,))
    conn.commit(); conn.close()
    return jsonify({"message": "已删除"})

@app.route('/api/history/stats', methods=['GET'])
@login_required
def api_history_stats():
    conn = get_db()
    username = session.get('username', '')
    role = session.get('role', '')
    if role == 'admin':
        total_detections = conn.execute('SELECT COUNT(*) FROM detection_history').fetchone()[0]
        avg_infer = conn.execute(
            "SELECT AVG(infer_time_ms) FROM detection_history WHERE infer_time_ms > 0").fetchone()[0] or 0
        all_details = conn.execute(
            "SELECT defect_details FROM detection_history WHERE defect_details != '[]'").fetchall()
    else:
        total_detections = conn.execute(
            'SELECT COUNT(*) FROM detection_history WHERE username=?', (username,)).fetchone()[0]
        avg_infer = conn.execute(
            "SELECT AVG(infer_time_ms) FROM detection_history WHERE infer_time_ms > 0 AND username=?",
            (username,)).fetchone()[0] or 0
        all_details = conn.execute(
            "SELECT defect_details FROM detection_history WHERE defect_details != '[]' AND username=?",
            (username,)).fetchall()
    conn.close()
    class_counter = {}
    for row in all_details:
        for d in json.loads(row['defect_details']):
            cn = d.get('class_name', 'unknown')
            class_counter[cn] = class_counter.get(cn, 0) + 1
    return jsonify({"total_detections": total_detections, "avg_infer_time_ms": round(avg_infer, 1),
                    "class_distribution": class_counter})

@app.route('/api/export/<int:record_id>', methods=['GET'])
@login_required
def api_export(record_id):
    fmt = request.args.get('format', 'json')
    conn = get_db()
    r = conn.execute('SELECT * FROM detection_history WHERE id=?', (record_id,)).fetchone()
    conn.close()
    if not r:
        return jsonify({'error': '记录不存在'}), 404
    details = json.loads(r['defect_details']) if r['defect_details'] else []
    if fmt == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['缺陷类别', '置信度', 'x1', 'y1', 'x2', 'y2'])
        for d in details:
            writer.writerow([d.get('class_name', ''), d.get('confidence', ''),
                            d['bbox'][0] if len(d.get('bbox', [])) > 0 else '',
                            d['bbox'][1] if len(d.get('bbox', [])) > 1 else '',
                            d['bbox'][2] if len(d.get('bbox', [])) > 2 else '',
                            d['bbox'][3] if len(d.get('bbox', [])) > 3 else ''])
        output.seek(0)
        return Response(output.getvalue().encode('utf-8-sig'), mimetype='text/csv',
                        headers={'Content-Disposition': f'attachment; filename=report_{record_id}.csv'})
    else:
        data = {"report_id": r['id'], "task_id": r['task_id'], "model_name": r['model_name'],
                "image_name": r['image_name'],
                "parameters": {"conf_threshold": r['conf_threshold'], "iou_threshold": r['iou_threshold']},
                "results": {"total_count": r['total_count'], "avg_confidence": r['avg_confidence'],
                           "infer_time_ms": r['infer_time_ms']},
                "details": details, "exported_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        return Response(json.dumps(data, ensure_ascii=False, indent=2), mimetype='application/json',
                        headers={'Content-Disposition': f'attachment; filename=report_{record_id}.json'})

# ═══════════════ 启动 ═══════════════
if __name__ == '__main__':
    init_db()
    print("=" * 50)
    print("🔍 工业缺陷检测系统 — 后端已启动")
    print(f"📦 可用模型: {len(scan_models())} 个")
    print(f"🗄️  数据库: {os.path.abspath(DB_PATH)}")
    print("👤 默认账号: admin/admin123 | operator/operator123 | viewer/viewer123")
    print("🌐 访问地址: http://localhost:8080")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=8080)
