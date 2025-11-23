# 视频ReID Web界面计划

本目录将承载新的 Flask + Tailwind Web 前端，用于以更直观的方式运行 `video_reid.py` 推理流程。下面是设计概要。

## 功能需求

- **模型管理**：初始化 YOLOv5 TensorRT 与 ReID ONNX，展示加载状态。
- **查询图片管理**：上传、列出、删除人物图片；可一键载入 `query/` 默认示例；支持自定义尺寸。
- **视频源管理**：最多 4 路视频，可上传文件或引用 Jetson 本地视频；支持删除与状态显示。
- **检测控制**：设置 `conf / iou / dist` 阈值，启动/停止检测任务。
- **实时展示**：以 MJPEG 流显示每路视频的检测结果，标注 ReID 命中。
- **状态轮询**：周期性刷新模型、查询、视频状态，使界面在刷新后也能恢复。

## 后端 API 草案

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 渲染前端页面 |
| GET | `/api/status` | 返回模型、查询、视频、活动流状态 |
| POST | `/api/initialize_models` | 初始化 YOLO + ReID |
| POST | `/api/upload_query` | 上传并提取查询特征 |
| GET | `/api/get_query_images` | 获取已上传查询图列表 |
| DELETE | `/api/delete_query/<id>` | 删除指定查询图 |
| GET | `/api/load_default_queries` | 从 `query/` 目录批量载入 |
| POST | `/api/upload_video` | 上传视频到 `uploads/videos` 并注册索引 |
| DELETE | `/api/delete_video/<idx>` | 删除指定视频并停止处理 |
| POST | `/api/start_detection/<idx>` | 根据当前参数启动检测 |
| POST | `/api/stop_detection/<idx>` | 停止检测 |
| GET | `/api/video_feed/<idx>` | 返回 MJPEG 视频流 |

## 架构要点

- 复用 `video_reid.py` 中的 `YoloTRT`, `ONNXReIDModel`, `parse_detections`, `draw_detections`。
- 每路视频使用 `VideoReader` 线程和 `Queue` 缓冲，主推理线程串行处理并确保 CUDA 上下文安全。
- 通过 `pycuda.autoinit` 管理 CUDA，上下文使用自定义 `cuda_context_scope`。
- 所有上传文件位于 `webapp/uploads`（自动创建子目录 videos / queries）。
- 前端采用 Tailwind + 原生 JS，布局为左侧控制面板、右侧 2x2 视频墙。

## UI 草图

- 左侧：
  - 模型状态 + 初始化按钮
  - 参数滑块 (Conf / IoU / ReID dist)
  - 查询图片卡片（上传按钮、默认加载、缩略图网格）
  - 视频源卡片（上传按钮、当前列表、开始/停止按钮）
- 右侧：4 个视频面板，显示“检测中 / 等待中”，提供删除按钮。

接下来将按此方案实现后端与前端。
