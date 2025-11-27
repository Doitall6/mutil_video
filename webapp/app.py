#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频ReID Web界面 - 复用 video_reid.py 推理能力"""

import os
import sys
import time
import base64
import logging
import threading
from pathlib import Path
from queue import Queue, Empty
from collections import deque

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename

import pycuda.autoinit  # noqa: F401  初始化CUDA上下文

# ------------------------- 路径设置 -------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from video_reid import (
        YoloTRT,
        ONNXReIDModel,
        parse_detections,
        draw_detections,
        StreamConfig,
    )
except ImportError as exc:
    raise RuntimeError(f"无法导入 video_reid.py: {exc}")

try:
    from resize_image import resize_image
except ImportError:
    resize_image = None  # 备用：上传时可直接读取原图

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("webapp")

# ------------------------- Flask 应用 -------------------------
app = Flask(
    __name__,
    template_folder=str(CURRENT_DIR / "templates"),
    static_folder=str(CURRENT_DIR / "static"),
)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512MB
app.config['UPLOAD_FOLDER'] = CURRENT_DIR / "uploads"
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
(app.config['UPLOAD_FOLDER'] / 'videos').mkdir(parents=True, exist_ok=True)
(app.config['UPLOAD_FOLDER'] / 'queries').mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}
ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "bmp"}

# ------------------------- 全局状态 -------------------------
yolo_model = None
reid_model = None
models_loaded = False
models_loading = False

video_contexts = {}  # idx -> {reader, queue, event, trajectory, params, stream_config, fps,...}
video_sources = {}  # idx -> {'type': 'file'|'camera', 'source': path_or_id, 'label': str}
video_lock = threading.Lock()
model_lock = threading.Lock()

query_feats = None
query_ids = []
query_images_info = []
selected_query_indices = []
query_lock = threading.Lock()
DEFAULT_PARAMS = {
    'threshold': 0.6,
    'conf': 0.5,
    'iou': 0.4,
    'min_box_size': 40,
    'trajectory_length': 50,
    'show_all': True,
}
current_params = DEFAULT_PARAMS.copy()

# ------------------------- CUDA 上下文工具 -------------------------
def push_cuda_context():
    try:
        pycuda.autoinit.context.push()
        return True
    except Exception as err:
        logger.error("推入CUDA上下文失败: %s", err)
        return False


def pop_cuda_context():
    try:
        pycuda.autoinit.context.pop()
    except Exception as err:
        logger.warning("弹出CUDA上下文失败: %s", err)


class cuda_context_scope:
    def __enter__(self):
        self.pushed = push_cuda_context()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pushed:
            pop_cuda_context()


# ------------------------- 视频读取线程 -------------------------
class VideoReader(threading.Thread):
    def __init__(self, source, source_type: str, video_index: int, frame_queue: Queue, running_event: threading.Event):
        super().__init__(daemon=True)
        self.source = source
        self.source_type = source_type
        self.video_index = video_index
        self.frame_queue = frame_queue
        self.running_event = running_event
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error("VideoReader-%s 无法打开源: %s", self.video_index, self.source)
            return

        logger.info("VideoReader-%s 开始读取 %s (%s)", self.video_index, self.source, self.source_type)
        frame_counter = 0

        while self.running_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if self.source_type == 'file':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(0.05)
                continue

            frame_counter += 1
            if frame_counter % 300 == 0:
                logger.info("VideoReader-%s 已读取 %d 帧", self.video_index, frame_counter)

            try:
                self.frame_queue.put((self.video_index, frame), timeout=0.1)
            except Exception:
                pass  # 队列满则丢帧

        self.cap.release()
        logger.info("VideoReader-%s 结束", self.video_index)


# ------------------------- 模型初始化 -------------------------
def initialize_models():
    global yolo_model, reid_model, models_loaded, models_loading

    if models_loaded:
        return True, "模型已加载"
    if models_loading:
        return False, "模型正在加载中..."

    models_loading = True
    try:
        engine_path = PROJECT_ROOT / 'yolov5s.engine'
        plugin_path = PROJECT_ROOT / 'libmyplugins.so'
        reid_path = PROJECT_ROOT / 'osnet_x0_25.onnx'

        if not engine_path.exists():
            return False, f"YOLO 引擎不存在: {engine_path}"
        if not reid_path.exists():
            return False, f"ReID 模型不存在: {reid_path}"

        with cuda_context_scope():
            yolo_model = YoloTRT(str(engine_path), str(plugin_path), batch_size=1)
        reid_model = ONNXReIDModel(str(reid_path))

        models_loaded = True
        return True, "模型加载成功"
    except Exception as err:
        logger.exception("模型加载失败")
        models_loaded = False
        return False, f"模型加载失败: {err}"
    finally:
        models_loading = False


# ------------------------- 工具函数 -------------------------
def allowed_file(filename: str, allowed_ext):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext


def get_query_payload():
    with query_lock:
        return {
            'query_images': query_images_info,
            'selected_indices': selected_query_indices,
            'query_count': len(query_ids)
        }

def parse_detection_params(data: dict, *, update_defaults: bool = False):
    def _bool(val, default=True):
        if isinstance(val, bool):
            return val
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return bool(val)
        return str(val).lower() in {'1', 'true', 'yes', 'on'}

    base = current_params.copy()
    params = base.copy()
    params.update({
        'threshold': float(data.get('threshold', base['threshold'])),
        'conf': float(data.get('conf', base['conf'])),
        'iou': float(data.get('iou', base['iou'])),
        'min_box_size': int(data.get('min_box_size', base['min_box_size'])),
        'trajectory_length': max(5, int(data.get('trajectory_length', base['trajectory_length']))),
        'show_all': _bool(data.get('show_all', base['show_all']), base['show_all'])
    })
    if update_defaults:
        current_params.update(params)
    return params


def parse_camera_source(camera_id):
    if isinstance(camera_id, int):
        return camera_id
    if camera_id is None:
        return None
    try:
        return int(camera_id)
    except (TypeError, ValueError):
        if isinstance(camera_id, str):
            camera_id = camera_id.strip()
            if camera_id == '':
                return None
        return camera_id


def list_available_cameras(max_devices: int = 6):
    cameras = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cameras.append({'id': idx, 'name': f'摄像头 {idx}'})
        cap.release()
    return cameras


# ------------------------- Flask 路由 -------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/initialize_models', methods=['POST'])
def api_initialize_models():
    success, message = initialize_models()
    return jsonify({'success': success, 'message': message})


@app.route('/api/status')
def api_status():
    with video_lock:
        active_videos = set(video_contexts.keys())
        video_details = []
        for idx, info in video_sources.items():
            label = info.get('label') or os.path.basename(str(info.get('source', '')))
            video_details.append({
                'index': idx,
                'label': label,
                'filename': label,
                'type': info.get('type', 'file'),
                'source': info.get('source'),
                'active': idx in active_videos,
            })

    payload = get_query_payload()

    return jsonify({
        'models_loaded': models_loaded,
        'models_loading': models_loading,
        'video_details': video_details,
        'query_count': payload['query_count'],
        'query_images': payload['query_images'],
        'selected_indices': payload['selected_indices'],
    })


@app.route('/api/upload_query', methods=['POST'])
def api_upload_query():
    global query_feats, query_ids, query_images_info, selected_query_indices

    if not models_loaded or reid_model is None:
        return jsonify({'success': False, 'message': '请先初始化模型'})

    if 'query_image' not in request.files:
        return jsonify({'success': False, 'message': '没有上传文件'})

    file = request.files['query_image']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_IMAGE_EXT):
        return jsonify({'success': False, 'message': '文件格式不支持'})

    width = int(request.form.get('width', 128))
    height = int(request.form.get('height', 256))

    filename = secure_filename(file.filename)
    save_path = app.config['UPLOAD_FOLDER'] / 'queries' / f"{int(time.time()*1000)}_{filename}"
    file.save(save_path)

    if resize_image:
        resize_image(str(save_path), str(save_path), width=width, height=height)

    img = cv2.imread(str(save_path))
    if img is None:
        save_path.unlink(missing_ok=True)
        return jsonify({'success': False, 'message': '无法读取图片'})

    feat = reid_model.extract_features(img)
    if feat is None or len(feat) == 0:
        save_path.unlink(missing_ok=True)
        return jsonify({'success': False, 'message': '特征提取失败'})

    with query_lock:
        query_id = f"query_{len(query_ids)+1}"
        query_ids.append(query_id)
        query_images_info.append({
            'id': query_id,
            'filename': save_path.name,
            'thumbnail': image_to_base64(img, thumb_size=(100, 150)),
            'size': f"{width}x{height}"
        })

        if query_feats is None:
            query_feats = feat
        else:
            query_feats = np.vstack([query_feats, feat])

        selected_query_indices = list(range(len(query_ids)))

    payload = get_query_payload()
    return jsonify({'success': True, 'message': '上传成功', **payload})


def image_to_base64(img, thumb_size=(120, 160)):
    thumb = cv2.resize(img, thumb_size)
    _, buffer = cv2.imencode('.jpg', thumb)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


@app.route('/api/get_query_images')
def api_get_queries():
    return jsonify({'success': True, **get_query_payload()})


@app.route('/api/delete_query/<query_id>', methods=['DELETE'])
def api_delete_query(query_id):
    global query_feats, query_ids, query_images_info, selected_query_indices

    with query_lock:
        if query_id not in query_ids:
            return jsonify({'success': False, 'message': '查询图片不存在'})
        idx = query_ids.index(query_id)
        info = query_images_info[idx]
        file_path = app.config['UPLOAD_FOLDER'] / 'queries' / info['filename']
        if file_path.exists():
            file_path.unlink()

        query_ids.pop(idx)
        query_images_info.pop(idx)
        if query_feats is not None:
            if len(query_ids) == 0:
                query_feats = None
            else:
                query_feats = np.delete(query_feats, idx, axis=0)
        selected_query_indices = list(range(len(query_ids)))

    return jsonify({'success': True, **get_query_payload()})


@app.route('/api/load_default_queries')
def api_load_default_queries():
    global query_feats, query_ids, query_images_info, selected_query_indices

    if not models_loaded or reid_model is None:
        success, message = initialize_models()
        if not success:
            return jsonify({'success': False, 'message': message})

    default_dir = PROJECT_ROOT / 'query'
    if not default_dir.exists():
        return jsonify({'success': False, 'message': 'query 目录不存在'})

    imgs = []
    new_ids = []
    new_infos = []

    for img_path in sorted(default_dir.glob('*')):
        if img_path.suffix.lower().lstrip('.') not in ALLOWED_IMAGE_EXT:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        imgs.append(img)
        new_id = img_path.stem
        new_ids.append(new_id)
        new_infos.append({
            'id': new_id,
            'filename': img_path.name,
            'thumbnail': image_to_base64(img),
            'size': '原图'
        })

    if not imgs:
        return jsonify({'success': False, 'message': '默认目录中没有可用图片'})

    feats = reid_model.extract_features(imgs)
    with query_lock:
        query_feats = feats
        query_ids = new_ids
        query_images_info = new_infos
        selected_query_indices = list(range(len(query_ids)))

    return jsonify({'success': True, **get_query_payload()})


@app.route('/api/upload_video', methods=['POST'])
def api_upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': '没有上传文件'})
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_VIDEO_EXT):
        return jsonify({'success': False, 'message': '文件格式不支持'})

    video_index = int(request.form.get('video_index', 0))
    filename = secure_filename(file.filename)
    save_dir = app.config['UPLOAD_FOLDER'] / 'videos'
    save_path = save_dir / f"video_{video_index}_{int(time.time())}_{filename}"
    file.save(save_path)

    source_info = {
        'type': 'file',
        'source': str(save_path),
        'label': save_path.name,
    }
    with video_lock:
        stop_video_processing(video_index)
        video_sources[video_index] = source_info

    return jsonify({'success': True, 'video_index': video_index, 'label': save_path.name, 'type': 'file'})


@app.route('/api/list_cameras')
def api_list_cameras():
    try:
        max_devices = int(request.args.get('max', 6))
    except ValueError:
        max_devices = 6
    max_devices = max(1, min(max_devices, 16))
    cameras = list_available_cameras(max_devices=max_devices)
    return jsonify({'success': True, 'cameras': cameras})


@app.route('/api/assign_camera', methods=['POST'])
def api_assign_camera():
    data = request.get_json() or {}
    video_index = data.get('video_index')
    try:
        video_index = int(video_index)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'message': '缺少或非法的通道索引'})

    camera_id = parse_camera_source(data.get('camera_id'))
    if camera_id is None:
        return jsonify({'success': False, 'message': '请选择有效的摄像头'})

    label = data.get('label') or f'摄像头 {camera_id}'

    with video_lock:
        for idx, info in video_sources.items():
            if idx != video_index and info.get('type') == 'camera' and info.get('source') == camera_id:
                return jsonify({'success': False, 'message': '该摄像头已分配给其他通道'})

        stop_video_processing(video_index)
        video_sources[video_index] = {
            'type': 'camera',
            'source': camera_id,
            'label': label,
        }

    return jsonify({'success': True, 'video_index': video_index, 'label': label, 'type': 'camera'})


@app.route('/api/delete_video/<int:video_index>', methods=['DELETE'])
def api_delete_video(video_index):
    with video_lock:
        stop_video_processing(video_index)
        info = video_sources.pop(video_index, None)
    if info and info.get('type') == 'file':
        path = info.get('source')
        if path and os.path.exists(path):
            os.remove(path)
    return jsonify({'success': True})


def start_video_processing(video_index: int, params: dict):
    with video_lock:
        source_info = video_sources.get(video_index)
        if not source_info:
            return False, '请先上传视频或绑定摄像头'

        stop_video_processing(video_index)

        queue = Queue(maxsize=10)
        event = threading.Event()
        event.set()
        source_value = source_info.get('source')
        reader = VideoReader(source_value, source_info.get('type', 'file'), video_index, queue, event)
        reader.start()

        stream_cfg = StreamConfig(
            stream_id=video_index,
            source=str(source_value),
            name=source_info.get('label') or f"通道 {video_index + 1}"
        )
        video_contexts[video_index] = {
            'reader': reader,
            'queue': queue,
            'event': event,
            'trajectory': deque(maxlen=params.get('trajectory_length', DEFAULT_PARAMS['trajectory_length'])),
            'params': params,
            'stream_config': stream_cfg,
            'frame_count': 0,
            'last_fps_time': time.time(),
            'fps': 0.0,
        }
    return True, '检测已启动'


def stop_video_processing(video_index: int):
    ctx = video_contexts.pop(video_index, None)
    if not ctx:
        return
    ctx['event'].clear()
    ctx['reader'].join(timeout=1.0)


@app.route('/api/start_detection/<int:video_index>', methods=['POST'])
def api_start_detection(video_index):
    if not models_loaded:
        return jsonify({'success': False, 'message': '请先初始化模型'})

    data = request.get_json() or {}
    params = parse_detection_params(data, update_defaults=True)

    success, message = start_video_processing(video_index, params)
    return jsonify({'success': success, 'message': message})


@app.route('/api/stop_detection/<int:video_index>', methods=['POST'])
def api_stop_detection(video_index):
    stop_video_processing(video_index)
    return jsonify({'success': True, 'message': '已停止'})


def apply_params_to_active_contexts(new_params: dict):
    updated = 0
    with video_lock:
        for ctx in video_contexts.values():
            ctx['params'].update(new_params)
            if 'trajectory_length' in new_params:
                new_len = new_params['trajectory_length']
                if ctx['trajectory'].maxlen != new_len:
                    ctx['trajectory'] = deque(list(ctx['trajectory'])[-new_len:], maxlen=new_len)
            updated += 1
    return updated


@app.route('/api/update_params', methods=['POST'])
def api_update_params():
    data = request.get_json() or {}
    params = parse_detection_params(data, update_defaults=True)
    updated = apply_params_to_active_contexts(params)
    return jsonify({'success': True, 'updated': updated})


@app.route('/api/video_feed/<int:video_index>')
def api_video_feed(video_index):
    return Response(generate_video_stream(video_index), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_video_stream(video_index: int):
    logger.info("开始推流 video-%s", video_index)
    while True:
        with video_lock:
            ctx = video_contexts.get(video_index)
            if ctx is None:
                logger.info("视频%d上下文不存在，结束推流", video_index)
                break

        try:
            vid_idx, frame = ctx['queue'].get(timeout=0.2)
            if vid_idx != video_index:
                continue
        except Empty:
            time.sleep(0.01)
            continue

        with model_lock, cuda_context_scope():
            processed_img, _, _ = yolo_model.preprocess_img(frame)
            yolo_model.img_to_tensor(processed_img, batch_idx=0)
            outputs = yolo_model.do_inference(actual_batch_size=1)
            output = outputs[0]
            params = ctx['params']
            detections = parse_detections(output, params['conf'], params['iou'])

            with query_lock:
                if query_feats is not None and len(query_ids) > 0:
                    active_indices = selected_query_indices or list(range(len(query_ids)))
                    active_feats = query_feats[active_indices]
                    active_ids = [query_ids[i] for i in active_indices]
                else:
                    active_feats = None
                    active_ids = None

            img_draw, _, _, _ = draw_detections(
                processed_img,
                detections,
                params['conf'],
                params['min_box_size'],
                reid_model if active_feats is not None else None,
                active_feats,
                params['threshold'],
                params.get('show_all', True),
                ctx['stream_config'],
                active_ids,
                ctx['trajectory'],
                params.get('trajectory_length', ctx['trajectory'].maxlen or DEFAULT_PARAMS['trajectory_length']),
                False,
            )

        ctx['frame_count'] += 1
        now = time.time()
        if now - ctx['last_fps_time'] >= 1.0:
            ctx['fps'] = ctx['frame_count'] / (now - ctx['last_fps_time'])
            ctx['frame_count'] = 0
            ctx['last_fps_time'] = now

        frame_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_bgr, f"FPS: {ctx['fps']:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    logger.info("结束推流 video-%s", video_index)


# ------------------------- 入口 -------------------------
if __name__ == '__main__':
    logger.info("启动 视频ReID Web 服务")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
