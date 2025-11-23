#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三路视频流异步批处理推理 + ReID人物搜索
使用单 context + 异步 Batch 推理，充分利用 engine 的 batch_size=3 能力

核心特性:
- 单 context，避免 CUDA 资源冲突
- 异步视频读取，多线程解码
- 批处理推理（batch_size=3），最大化 GPU 利用率
- ReID 人物搜索功能
- 实时拼接显示三路视频

日期: 2025-11-09
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import ctypes
import argparse
import os
import threading
from pathlib import Path
from queue import Queue, Empty, Full
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

# ONNX Runtime for ReID
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("❌ onnxruntime未安装，ReID功能将不可用")
    print("请运行: pip install onnxruntime-gpu  # 或 onnxruntime")

# Constants
NMS_THRESH = 0.4
CONF_THRESH = 0.5
INPUT_H = 640
INPUT_W = 640
OUTPUT_SIZE = 1 + 1000 * 6  # num_det + max_boxes * (x,y,w,h,conf,class)

INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# 队列大小配置
VIDEO_QUEUE_SIZE = 10  # 每路视频的帧缓冲队列大小


@dataclass
class Frame:
    """单帧数据"""
    stream_id: int          # 视频流ID (0, 1, 2)
    frame_idx: int          # 帧序号
    timestamp: float        # 时间戳
    image: np.ndarray       # 原始图像 (BGR)
    processed: np.ndarray = None  # 预处理后的图像 (RGB, letterbox)
    detections: list = None # 检测结果
    scale_ratio: float = 1.0  # 缩放比例
    offset: Tuple[int, int] = (0, 0)  # 偏移量
    

@dataclass
class StreamConfig:
    """视频流配置"""
    stream_id: int
    source: str             # 视频源路径或摄像头ID
    name: str = ""          # 显示名称
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR颜色（用于边框）


class ONNXReIDModel:
    """ONNX ReID模型推理类"""
    
    def __init__(self, model_path, device='cuda', input_size=(256, 128)):
        """
        初始化ONNX ReID模型
        
        Args:
            model_path: ONNX模型路径
            device: 'cuda' 或 'cpu'
            input_size: 输入图片尺寸 (height, width)
        """
        self.device = device
        self.input_size = input_size
        
        print(f"✓ 加载ONNX ReID模型: {model_path}")
        print(f"  设备: {device}")
        print(f"  输入尺寸: {input_size}")
        
        # 配置ONNX Runtime - 优先使用TensorRT
        if device == 'cuda':
            providers = [
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']

        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✓ ONNX ReID模型加载成功")
        print(f"  提供者: {self.session.get_providers()}")
    
    def preprocess(self, img):
        """预处理图片"""
        from PIL import Image
        
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # Resize
        img = img.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)
        
        # 转换为numpy并归一化
        img = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        
        # 添加batch维度
        img = np.expand_dims(img, axis=0)
        
        return img.astype(np.float32)
    
    def extract_features(self, images):
        """提取特征"""
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = [images]
        
        features = []
        for img in images:
            input_data = self.preprocess(img)
            outputs = self.session.run([self.output_name], {self.input_name: input_data})
            feat = outputs[0]
            features.append(feat)
        
        features = np.concatenate(features, axis=0)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        return features


class VideoReader(threading.Thread):
    """异步视频读取线程"""
    
    def __init__(self, config: StreamConfig, frame_queue: Queue, stop_event: threading.Event, skip_frames: int = 1):
        super().__init__(daemon=True)
        self.config = config
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.skip_frames = skip_frames  # 跳帧间隔
        
        # 打开视频源
        if config.source.isdigit():
            self.cap = cv2.VideoCapture(int(config.source))
        else:
            self.cap = cv2.VideoCapture(config.source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {config.source}")
        
        self.frame_idx = 0
        self.read_count = 0  # 实际读取的帧数
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ 视频流{config.stream_id} ({config.name}): {self.width}x{self.height} @ {self.fps:.1f}fps")
    
    def run(self):
        """读取视频帧并放入队列"""
        while not self.stop_event.is_set():
            ret, img = self.cap.read()
            
            if not ret:
                print(f"视频流{self.config.stream_id} ({self.config.name}) 读取结束")
                break
            
            if img is None or img.size == 0:
                continue
            
            self.read_count += 1
            
            # 跳帧逻辑：只处理间隔帧
            if (self.read_count - 1) % self.skip_frames != 0:
                continue  # 跳过此帧
            
            # 创建Frame对象
            frame = Frame(
                stream_id=self.config.stream_id,
                frame_idx=self.frame_idx,
                timestamp=time.time(),
                image=img
            )
            
            # 放入队列（阻塞）
            try:
                self.frame_queue.put(frame, timeout=1.0)
                self.frame_idx += 1
            except Full:
                pass  # 队列满了，丢弃当前帧
        
        self.cap.release()
        print(f"视频流{self.config.stream_id} ({self.config.name}) 读取线程退出 (读取{self.read_count}帧, 处理{self.frame_idx}帧)")



class YoloTRT:
    """YOLOv5 TensorRT推理引擎（支持批处理）"""
    
    def __init__(self, engine_path: str, plugin_library: str = None, batch_size: int = 3):
        self.batch_size = batch_size
        
        # 加载插件库
        if plugin_library and os.path.exists(plugin_library):
            try:
                ctypes.CDLL(plugin_library)
                print(f"✓ 加载插件库: {plugin_library}")
            except Exception as e:
                print(f"⚠ 插件库加载失败: {e}")
        
        # 加载TensorRT引擎
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("引擎反序列化失败")
        
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("执行上下文创建失败")
        
        # 分配GPU内存
        self.input_shape = (batch_size, 3, INPUT_H, INPUT_W)
        self.output_shape = (batch_size, OUTPUT_SIZE)
        
        self.d_input = cuda.mem_alloc(batch_size * 3 * INPUT_H * INPUT_W * np.dtype(np.float32).itemsize)
        self.d_output = cuda.mem_alloc(batch_size * OUTPUT_SIZE * np.dtype(np.float32).itemsize)
        
        # 创建CUDA流
        self.stream = cuda.Stream()
        
        # 分配固定主机内存
        self.h_input = cuda.pagelocked_empty(int(np.prod(self.input_shape)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(int(np.prod(self.output_shape)), dtype=np.float32)
        
        # 获取绑定索引
        self.input_index = self.engine.get_binding_index(INPUT_BLOB_NAME)
        self.output_index = self.engine.get_binding_index(OUTPUT_BLOB_NAME)
        
        print(f"✓ TensorRT引擎加载成功")
        print(f"  批大小: {batch_size}")
        print(f"  输入尺寸: {INPUT_H}x{INPUT_W}")
    
    def preprocess_img(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox预处理"""
        h, w = img.shape[:2]
        
        # 计算缩放比例
        r = min(INPUT_W / w, INPUT_H / h)
        new_w, new_h = int(w * r), int(h * r)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建画布并粘贴
        canvas = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8)
        top = (INPUT_H - new_h) // 2
        left = (INPUT_W - new_w) // 2
        canvas[top:top+new_h, left:left+new_w] = resized
        
        # BGR转RGB
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        return canvas, r, (left, top)
    
    def img_to_tensor(self, img: np.ndarray, batch_idx: int = 0):
        """将图像转换为tensor并放入batch指定位置"""
        # 归一化到[0, 1]
        img = img.astype(np.float32) / 255.0
        
        # HWC转CHW
        img = np.transpose(img, (2, 0, 1))
        
        # 计算在batch中的偏移
        img_size = 3 * INPUT_H * INPUT_W
        offset = batch_idx * img_size
        
        # 复制到主机缓冲区
        img = np.ascontiguousarray(img)
        np.copyto(self.h_input[offset:offset+img_size], img.ravel())
    
    def do_inference(self, actual_batch_size: int = None):
        """执行推理"""
        if actual_batch_size is None:
            actual_batch_size = self.batch_size
        
        # 传输输入数据到GPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # 执行推理
        bindings = [int(self.d_input), int(self.d_output)]
        self.context.execute_async(
            batch_size=actual_batch_size,
            bindings=bindings,
            stream_handle=self.stream.handle
        )
        
        # 传输输出数据回主机
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        # 同步
        self.stream.synchronize()
        
        return self.h_output.reshape(self.output_shape)


def nms(detections, conf_threshold, nms_threshold):
    """非极大值抑制"""
    if not detections:
        return []
    
    boxes = []
    scores = []
    class_ids = []
    
    for det in detections:
        x, y, w, h, conf, class_id = det
        boxes.append([x - w/2, y - h/2, w, h])
        scores.append(conf)
        class_ids.append(int(class_id))
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        conf_threshold,
        nms_threshold
    )
    
    if len(indices) == 0:
        return []
    
    # 处理不同OpenCV版本的返回格式
    if isinstance(indices, tuple):
        indices = indices[0] if len(indices) > 0 else []
    
    if len(indices) > 0 and isinstance(indices[0], (list, np.ndarray)):
        indices = [i[0] if isinstance(i, (list, np.ndarray)) else i for i in indices]
    
    keep = indices.flatten() if hasattr(indices, 'flatten') else indices
    
    result = []
    for i in keep:
        result.append({
            'bbox': boxes[i],
            'conf': scores[i],
            'class_id': class_ids[i]
        })
    
    return result


def parse_detections(output, conf_thresh, nms_thresh, debug=False):
    """
    解析检测输出
    
    格式：output[0] = 检测数量
          output[1~6] = 第1个检测 [x, y, w, h, conf, class_id]
          output[7~12] = 第2个检测
          ...
    """
    num_det = int(output[0])
    
    if debug:
        print(f"  [DEBUG] output[0] (num_det) = {num_det}")
        print(f"  [DEBUG] output[1:8] = {output[1:8]}")
    
    detections = []
    
    # 只读取前num_det个有效检测
    det_size = 6
    for i in range(min(num_det, 1000)):
        idx = 1 + i * det_size
        if idx + det_size <= len(output):
            det = output[idx:idx + det_size]
            # det = [x, y, w, h, conf, class_id]
            x, y, w, h, conf, class_id = det[0], det[1], det[2], det[3], det[4], int(det[5])
            
            # 数据验证: 置信度必须在[0,1]范围内
            if conf < 0 or conf > 1:
                continue
            
            # 类别ID验证
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                continue
            
            # 边界框验证
            if w <= 0 or h <= 0:
                continue
            
            # 置信度过滤
            if conf >= conf_thresh:
                detections.append([x, y, w, h, conf, class_id])
                
                if debug and len(detections) <= 3:
                    print(f"  [DEBUG] 检测{len(detections)}: {CLASS_NAMES[class_id]}, "
                          f"x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}, conf={conf:.3f}")
    
    if debug:
        person_count = sum(1 for d in detections if int(d[5]) == 0)
        print(f"  [DEBUG] 总检测: {len(detections)}, person: {person_count}")
    
    # 只保留 person
    person_detections = [d for d in detections if int(d[5]) == 0]
    
    # 应用NMS
    if len(person_detections) > 0:
        return nms(person_detections, conf_thresh, nms_thresh)
    else:
        return []


def get_rect(bbox):
    """转换bbox为矩形坐标"""
    x_left, y_top, w, h = bbox
    
    left = int(max(0, x_left))
    top = int(max(0, y_top))
    width = int(w)
    height = int(h)
    
    return (left, top, width, height)


def draw_detections(img: np.ndarray, detections: list, person_min_conf: float,
                    min_box_size: int, reid_model, query_feats, dist_thres: float,
                    show_all: bool, stream_config: StreamConfig, query_ids: list = None, 
                    trajectory: deque = None, max_trajectory_length: int = 50,
                    debug: bool = False) -> Tuple[np.ndarray, int, int, Optional[Tuple[int, int]]]:
    """
    在图像上绘制检测框和ReID结果
    
    Args:
        query_ids: 查询图片的ID列表（如 ["001", "002"]）
        trajectory: 轨迹点队列（deque）
        max_trajectory_length: 最大轨迹长度
    
    Returns:
        img_draw: 绘制后的图像 (RGB)
        num_persons: 检测到的人数
        num_targets: 找到的目标数
        target_foot_pos: 目标人物脚底位置 (x, y)，如果没有目标则为 None
    """
    img_draw = img.copy()
    
    # 收集所有人物检测框
    person_boxes = []
    person_indices = []
    
    for i, det in enumerate(detections):
        if det['class_id'] == 0 and det['conf'] >= person_min_conf:  # person
            x, y, w, h = get_rect(det['bbox'])
            if w >= min_box_size and h >= min_box_size:
                person_boxes.append((x, y, w, h))
                person_indices.append(i)
    
    num_persons = len(person_boxes)
    num_targets = 0
    target_foot_pos = None  # 目标人物脚底位置
    
    # ReID匹配
    found_target = False
    target_index = -1
    target_query_id = ""  # 匹配的查询图片ID
    distmat = []
    
    if reid_model is not None and len(person_boxes) > 0:
        gallery_imgs = []
        for (x, y, w, h) in person_boxes:
            person_crop_rgb = img[y:y+h, x:x+w]
            person_crop_bgr = cv2.cvtColor(person_crop_rgb, cv2.COLOR_RGB2BGR)
            gallery_imgs.append(person_crop_bgr)
        
        # 提取gallery特征
        gallery_feats = reid_model.extract_features(gallery_imgs)
        
        # 计算与query的距离
        dist = np.linalg.norm(gallery_feats[:, np.newaxis] - query_feats[np.newaxis, :], axis=2)
        min_dists = dist.min(axis=1)
        distmat = min_dists.tolist()
        
        # 找到最佳匹配
        best_match_idx = np.argmin(min_dists)
        best_dist = min_dists[best_match_idx]
        
        # 找到匹配的查询图片索引
        best_query_idx = dist[best_match_idx].argmin()
        
        if best_dist < dist_thres:
            found_target = True
            target_index = best_match_idx
            num_targets = 1
            
            # 获取查询图片ID
            if query_ids is not None and best_query_idx < len(query_ids):
                target_query_id = query_ids[best_query_idx]
            
            if debug:
                print(f"  Stream {stream_config.stream_id}: 找到目标! ID={target_query_id}, 距离={best_dist:.3f}")
    
    # 绘制检测框
    for i, det in enumerate(detections):
        if det['class_id'] != 0:  # 只显示人物
            continue
        
        x, y, w, h = get_rect(det['bbox'])
        conf = det['conf']
        
        if conf < person_min_conf or w < min_box_size or h < min_box_size:
            continue
        
        # 检查是否是目标
        is_target = False
        distance = -1
        
        try:
            person_idx = [idx for idx, det_idx in enumerate(person_indices) if det_idx == i][0]
            distance = distmat[person_idx] if person_idx < len(distmat) else -1
            is_target = (person_idx == target_index and found_target)
        except:
            pass
        
        if is_target:
            # 目标人物: 红色粗框，显示查询图片ID
            color = (255, 0, 0)  # RGB: 红色
            thickness = 3
            if target_query_id:
                label = f"{target_query_id} d={distance:.2f}"
            else:
                label = f"TARGET! d={distance:.2f}"
            
            # 计算脚底位置（边界框的底部中心）
            foot_x = x + w // 2
            foot_y = y + h
            target_foot_pos = (foot_x, foot_y)
            
            # 在脚底画一个小圆点
            cv2.circle(img_draw, (foot_x, foot_y), 5, (255, 0, 0), -1)  # RGB红色实心圆
            
        elif show_all:
            # 其他人物: 绿色细框
            color = (0, 255, 0)  # RGB: 绿色
            thickness = 2
            label = f"person d={distance:.2f}" if distance >= 0 else f"person {conf:.2f}"
        else:
            # 不显示其他人
            continue
        
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(img_draw, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)
    
    # 更新并绘制轨迹
    if target_foot_pos is not None and trajectory is not None:
        # 添加新的轨迹点
        trajectory.append(target_foot_pos)
        
        # 限制轨迹长度
        while len(trajectory) > max_trajectory_length:
            trajectory.popleft()
        
        # 绘制轨迹线
        if len(trajectory) >= 2:
            points = np.array(list(trajectory), dtype=np.int32)
            
            # 绘制渐变的轨迹线（从透明到不透明）
            for i in range(1, len(points)):
                # 计算透明度：越新的点越不透明
                alpha = i / len(points)
                thickness = int(2 + alpha * 2)  # 线条从2像素渐变到4像素
                
                # 绘制线段（RGB红色）
                cv2.line(img_draw, tuple(points[i-1]), tuple(points[i]), 
                        (255, 0, 0), thickness, cv2.LINE_AA)
    
    return img_draw, num_persons, num_targets, target_foot_pos


def create_mosaic(frames: List[Frame], stream_configs: List[StreamConfig], 
                  stats: dict, fps: float) -> np.ndarray:
    """创建多视频流拼接显示"""
    rows, cols = 2, 2
    cell_h, cell_w = INPUT_H, INPUT_W
    
    # 创建空画布
    mosaic = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    for idx, frame in enumerate(frames):
        if idx >= 3:  # 最多显示3路
            break
        
        row = idx // cols
        col = idx % cols
        
        if frame.processed is not None:
            img = frame.processed
            
            # 转换为BGR用于显示
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 添加视频流信息
            stream_config = stream_configs[frame.stream_id]
            stream_name = f"{stream_config.name}: Frame {frame.frame_idx}"
            cv2.putText(img_bgr, stream_name, (10, 30),
                       cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
            
            # 添加检测统计
            if frame.stream_id in stats:
                stat = stats[frame.stream_id]
                det_info = f"Persons: {stat['persons']} | Targets: {stat['targets']}"
                cv2.putText(img_bgr, det_info, (10, 60),
                           cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
            
            # 放入mosaic
            y1 = row * cell_h
            y2 = y1 + cell_h
            x1 = col * cell_w
            x2 = x1 + cell_w
            mosaic[y1:y2, x1:x2] = img_bgr
    
    # 添加整体FPS信息
    fps_text = f"Batch FPS: {fps:.1f}"
    cv2.putText(mosaic, fps_text, (10, mosaic.shape[0] - 20),
               cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 2)
    
    return mosaic


def main():
    parser = argparse.ArgumentParser(description='三路视频流异步批处理推理 + ReID人物搜索')
    
    # 视频流配置
    parser.add_argument('--stream1', type=str, default='video2/campus4-c0_40s.avi',
                       help='视频流1: 摄像头ID或视频文件路径')
    parser.add_argument('--stream2', type=str, default='video2/campus4-c1_40s.avi',
                       help='视频流2: 摄像头ID或视频文件路径（可选）')
    parser.add_argument('--stream3', type=str, default='video2/campus4-c2_40s.avi',
                       help='视频流3: 摄像头ID或视频文件路径（可选）')
    
    # 模型配置
    parser.add_argument('--engine', type=str, default='yolov5s.engine',
                       help='TensorRT引擎文件路径')
    parser.add_argument('--plugin', type=str, default='libmyplugins.so',
                       help='插件库文件路径')
    
    # 检测配置
    parser.add_argument('--conf-thres', type=float, default=0.7,
                       help='置信度阈值（推荐0.7-0.8，越高误检越少）')
    parser.add_argument('--nms-thres', type=float, default=0.4,
                       help='NMS阈值')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='批处理大小（yolov5s.engine是batch=1，3路视频各自推理）')
    parser.add_argument('--min-confidence', type=float, default=0.75,
                       help='人物检测的最低置信度（额外过滤，推荐0.7-0.8）')
    parser.add_argument('--min-box-size', type=int, default=40,
                       help='最小检测框尺寸(像素，过滤远处的小框和误检)')
    
    # ReID配置
    parser.add_argument('--reid-model', type=str, 
                       default='osnet_x0_25.onnx',
                       help='ONNX ReID模型路径')
    parser.add_argument('--query', type=str, default='query',
                       help='查询图片目录路径')
    parser.add_argument('--dist-thres', type=float, default=0.75,
                       help='ReID距离阈值（越小越严格，推荐0.7-0.8）')
    parser.add_argument('--enable-reid', action='store_true',
                       help='启用ReID人物搜索功能')
    parser.add_argument('--show-all', action='store_true',
                       help='显示所有检测到的人(红色=目标,绿色=其他人)')
    
    # 显示配置
    parser.add_argument('--show', action='store_true',
                       help='显示检测结果窗口')
    parser.add_argument('--output', type=str, default='',
                       help='输出视频文件路径（可选）')
    parser.add_argument('--debug', action='store_true',
                       help='显示调试信息')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='跳帧间隔(1=每帧都处理, 2=每两帧处理一次, 3=每三帧处理一次...)')
    
    args = parser.parse_args()
    
    # 收集视频流配置
    stream_configs = []
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    names = ["Stream1", "Stream2", "Stream3"]
    
    for i, source in enumerate([args.stream1, args.stream2, args.stream3]):
        if source:
            config = StreamConfig(
                stream_id=i,
                source=source,
                name=names[i],
                color=colors[i]
            )
            stream_configs.append(config)
    
    if len(stream_configs) == 0:
        print("❌ 至少需要一个视频流")
        return -1
    
    print("=" * 60)
    print(f"三路视频流异步批处理推理 + ReID人物搜索")
    print(f"视频流数量: {len(stream_configs)}")
    print("=" * 60)
    
    # ReID初始化
    reid_model = None
    query_feats = None
    query_ids = []  # 存储查询图片的ID
    
    if args.enable_reid:
        if not ONNX_AVAILABLE:
            print("❌ 启用ReID需要安装onnxruntime")
            return -1
        
        if not os.path.exists(args.reid_model):
            print(f"❌ ReID模型文件不存在: {args.reid_model}")
            return -1
        
        if not os.path.exists(args.query):
            print(f"❌ 查询图片目录不存在: {args.query}")
            return -1
        
        print("\n" + "=" * 60)
        print("初始化ReID功能")
        print("=" * 60)
        reid_model = ONNXReIDModel(args.reid_model, device='cuda')
        
        # 加载查询图片
        query_dir = Path(args.query)
        query_files = list(query_dir.glob('*.jpg')) + list(query_dir.glob('*.png'))
        
        if len(query_files) == 0:
            print(f"❌ 查询目录中没有找到图片: {args.query}")
            return -1
        
        print(f"\n加载查询图片: {len(query_files)} 张")
        query_imgs = []
        for qf in query_files:
            img = cv2.imread(str(qf))
            if img is not None:
                query_imgs.append(img)
                
                # 从文件名提取ID（如 001_00.jpg -> 001）
                filename = qf.stem  # 不包含扩展名的文件名
                query_id = filename.split('_')[0]  # 取第一个下划线前的部分
                query_ids.append(query_id)
                
                print(f"  ✓ {qf.name}: {img.shape} -> ID: {query_id}")
        
        print("\n提取查询图片特征...")
        query_feats = reid_model.extract_features(query_imgs)
        print(f"✓ 查询特征提取完成: {query_feats.shape}")
        print("=" * 60 + "\n")
    
    # 初始化TensorRT引擎
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.engine):
        args.engine = os.path.join(script_dir, args.engine)
    if not os.path.isabs(args.plugin):
        args.plugin = os.path.join(script_dir, args.plugin)
    
    if not os.path.exists(args.engine):
        print(f"❌ 引擎文件不存在: {args.engine}")
        return -1
    
    print(f"✓ 加载引擎: {args.engine}")
    yolo_trt = YoloTRT(args.engine, args.plugin, batch_size=args.batch_size)
    
    # 创建视频读取线程
    stop_event = threading.Event()
    video_queues = []
    video_readers = []
    
    print(f"✓ 跳帧间隔: {args.skip_frames} (每{args.skip_frames}帧处理一次)")
    
    for config in stream_configs:
        queue = Queue(maxsize=VIDEO_QUEUE_SIZE)
        video_queues.append(queue)
        
        reader = VideoReader(config, queue, stop_event, skip_frames=args.skip_frames)
        reader.start()
        video_readers.append(reader)
    
    time.sleep(0.5)
    
    # 创建显示窗口
    window_name = "Triple Stream Detection + ReID"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
    
    # 输出视频
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, 30, (INPUT_W * 2, INPUT_H * 2))
        print(f"✓ 输出视频: {args.output}")
    
    # 统计信息
    total_frames = {i: 0 for i in range(len(stream_configs))}
    total_persons = {i: 0 for i in range(len(stream_configs))}
    total_targets = {i: 0 for i in range(len(stream_configs))}
    total_time = 0
    batch_count = 0
    
    # 为每个视频流创建轨迹队列
    trajectories = {i: deque(maxlen=50) for i in range(len(stream_configs))}
    
    person_min_conf = args.min_confidence if args.min_confidence > 0 else args.conf_thres
    
    print("\n" + "=" * 60)
    print("开始处理... (按 ESC 键退出)")
    print("=" * 60)
    
    try:
        while True:
            # 从每个视频流获取一帧
            batch_frames = []
            
            for i, queue in enumerate(video_queues):
                try:
                    frame = queue.get(timeout=0.1)
                    batch_frames.append(frame)
                except Empty:
                    pass
            
            if len(batch_frames) == 0:
                all_dead = all(not reader.is_alive() for reader in video_readers)
                if all_dead:
                    print("\n所有视频流处理完成")
                    break
                else:
                    time.sleep(0.01)
                    continue
            
            # 逐个推理（因为使用batch=1的引擎）
            start_time = time.time()
            
            # 解析每个流的检测结果并绘制
            current_stats = {}
            
            for i, frame in enumerate(batch_frames):
                # 预处理
                processed_img, scale_ratio, offset = yolo_trt.preprocess_img(frame.image)
                frame.processed = processed_img
                frame.scale_ratio = scale_ratio
                frame.offset = offset
                
                # 单帧推理
                yolo_trt.img_to_tensor(processed_img, batch_idx=0)
                outputs = yolo_trt.do_inference(actual_batch_size=1)
                output = outputs[0]
                
                # 调试：打印第一个batch的前20个值
                if args.debug and batch_count == 0 and i == 0:
                    print(f"\n[调试输出] Stream {frame.stream_id}, output.shape={output.shape}")
                    print(f"  前20个值: {output[:20]}")
                
                # 添加调试信息
                if args.debug and batch_count <= 3:
                    print(f"\n[调试] Batch {batch_count}, Stream {frame.stream_id}:")
                
                detections = parse_detections(output, args.conf_thres, args.nms_thres, debug=(args.debug and batch_count <= 3))
                frame.detections = detections
                
                # 获取当前流的轨迹队列
                stream_trajectory = trajectories[frame.stream_id]
                
                # 绘制检测框和ReID结果
                frame.processed, num_persons, num_targets, target_foot_pos = draw_detections(
                    frame.processed, detections, person_min_conf,
                    args.min_box_size, reid_model, query_feats, 
                    args.dist_thres, args.show_all,
                    stream_configs[frame.stream_id], query_ids, stream_trajectory, 50, args.debug
                )
                
                # 更新统计
                stream_id = frame.stream_id
                total_frames[stream_id] += 1
                total_persons[stream_id] += num_persons
                total_targets[stream_id] += num_targets
                
                current_stats[stream_id] = {
                    'persons': num_persons,
                    'targets': num_targets
                }
            
            # 更新时间统计
            inference_time = time.time() - start_time
            total_time += inference_time
            batch_count += 1
            
            # 创建拼接显示
            fps = batch_count / total_time if total_time > 0 else 0
            mosaic = create_mosaic(batch_frames, stream_configs, current_stats, fps)
            
            # 显示
            if args.show:
                cv2.imshow(window_name, mosaic)
                
                if cv2.waitKey(1) == 27:  # ESC
                    print("\nESC键按下，停止处理")
                    break
            
            # 保存
            if out is not None:
                out.write(mosaic)
            
            # 定期打印进度
            if batch_count % 30 == 0:
                print(f"[进度] Batch: {batch_count}, FPS: {fps:.2f}")
                for config in stream_configs:
                    sid = config.stream_id
                    avg_persons = total_persons[sid] / total_frames[sid] if total_frames[sid] > 0 else 0
                    avg_targets = total_targets[sid] / total_frames[sid] if total_frames[sid] > 0 else 0
                    print(f"  {config.name}: {total_frames[sid]}帧, "
                          f"平均{avg_persons:.1f}人/帧 (总{total_persons[sid]}人次), "
                          f"平均{avg_targets:.1f}目标/帧 (总{total_targets[sid]}次)")
    
    except KeyboardInterrupt:
        print("\n收到中断信号，停止处理")
    
    finally:
        # 清理
        stop_event.set()
        
        for reader in video_readers:
            reader.join(timeout=2.0)
        
        if out is not None:
            out.release()
        
        cv2.destroyAllWindows()
        
        # 打印统计信息
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        print(f"总批次数: {batch_count}")
        if total_time > 0:
            print(f"总时间: {total_time:.2f}s")
            print(f"平均批处理FPS: {batch_count/total_time:.2f}")
        
        for config in stream_configs:
            sid = config.stream_id
            avg_persons = total_persons[sid] / total_frames[sid] if total_frames[sid] > 0 else 0
            avg_targets = total_targets[sid] / total_frames[sid] if total_frames[sid] > 0 else 0
            target_rate = 100 * total_targets[sid] / total_frames[sid] if total_frames[sid] > 0 else 0
            
            print(f"\n{config.name}:")
            print(f"  总帧数: {total_frames[sid]}")
            print(f"  检测: 总{total_persons[sid]}人次, 平均{avg_persons:.1f}人/帧")
            print(f"  目标: 总{total_targets[sid]}次, 平均{avg_targets:.1f}次/帧")
            print(f"  目标检出率: {target_rate:.1f}%")
        print("=" * 60)
    
    return 0


if __name__ == "__main__":
    main()
