# 视频 ReID 多路检测项目

利用 Jetson 平台上的 TensorRT YOLOv5 引擎与 ONNX ReID 特征模型，实现多路视频中的人物检索。仓库同时包含命令行脚本与现代化的 Flask Web 界面，方便调试、可视化与现场演示。

## ✨ 功能亮点
- **多路视频并行**：独立 `VideoReader` 线程+队列缓冲，主推理线程串行使用单一 CUDA 上下文，避免资源冲突。
- **YOLOv5 + ReID 级联**：先做行人检测，再使用 OSNet ReID 向量与查询图库进行相似度匹配。
- **可视化界面**：`webapp/` 提供 Tailwind + 原生 JS 的控制台，可上传人物/视频、调整参数、查看 4 路实时 MJPEG 流。
- **参数热更新**：置信度、IoU、ReID 阈值、最小框尺寸、轨迹长度、显示模式皆可在界面即时生效。
- **工具链**：提供图片尺寸预处理脚本、示例查询图 (`query/`) 与示例视频 (`video2/`) 便于开箱体验。

## 📁 目录结构
```
project/
├── video_reid.py          # CLI 版本：三路视频批处理推理 + ReID 搜索
├── web/                   # 旧版 Flask Web（保留参考）
├── webapp/                # 当前使用的 Web 前端与 API 服务
│   ├── app.py             # 主 Flask 应用，复用 video_reid 的模型/推理逻辑
│   ├── templates/         # Tailwind UI（index.html）
│   ├── static/            # app.js、样式、前端逻辑
│   └── uploads/           # 运行时生成：videos/ & queries/
├── query/                 # 默认人物查询图片
├── video2/                # 示例监控视频片段
├── resize_image.py        # 图片等比例缩放/填充工具
├── yolov5s.engine         # TensorRT 引擎（batch=1）
├── libmyplugins.so        # YOLO 依赖的自定义 plugin
└── osnet_x0_25.onnx       # ReID 模型
```

## 🧩 运行前提
- **硬件**：NVIDIA Jetson (已安装 JetPack/TensorRT) 或具备 CUDA 的 x86 设备。
- **系统**：Ubuntu 18.04+/Jetson Linux。
- **Python**：3.6 及以上。
- **依赖库**（pip / apt）：
  - `flask`, `werkzeug`, `requests`（web 界面）
  - `opencv-python`, `numpy`, `Pillow`
  - `pycuda`, `tensorrt` (JetPack 自带)
  - `onnxruntime` 或 `onnxruntime-gpu`
  - `tqdm`、`dataclasses`（Py3.6 环境已包含 backport）

> 若缺少 `onnxruntime`，ReID 功能会禁用；请运行 `pip3 install onnxruntime`.

## 📦 模型与素材
- `yolov5s.engine`：TensorRT 推理文件（需与 `libmyplugins.so` 配套）。
- `osnet_x0_25.onnx`：ONNX ReID 模型，默认走 TensorRT + CUDA EP。
- `query/`：示例人物图片，`webapp` 可一键导入。
- `video2/`：示例多路视频。

## 🚀 快速体验
### 1. 命令行批处理 (`video_reid.py`)
```bash
cd /home/ye/桌面/mount/Desktop/project
python3 video_reid.py \
  --stream1 video2/campus4-c0_40s.avi \
  --stream2 video2/campus4-c1_40s.avi \
  --stream3 video2/campus4-c2_40s.avi \
  --enable-reid --show --skip-frames 2
```
常用参数：
- `--conf-thres / --nms-thres`：检测阈值。
- `--dist-thres`：ReID 相似度阈值（越小越严格，默认 0.75）。
- `--min-confidence / --min-box-size`：过滤小框/低置信度误检。
- `--streamX`：可替换为 RTSP/摄像头 ID。

### 2. Web 可视化 (`webapp/`)
```bash
cd /home/ye/桌面/mount/Desktop/project/webapp
python3 app.py
```
浏览器访问 `http://<Jetson-IP>:5000`，典型流程：
1. **初始化模型**：点击“初始化模型”，等待状态变为“运行中”。
2. **管理查询图片**：上传人物照或“加载默认查询”，自带尺寸重排。
3. **上传视频**：可同时添加 4 路，本地文件会保存到 `webapp/uploads/videos/`。
4. **调整参数**：左侧滑块/输入框可修改 `Conf / IoU / ReID 阈值`、最小框、轨迹长度、显示模式。
5. **开始检测**：选择通道点击“开始检测”，右侧视频墙会显示检测框与 ReID 命中；可随时“停止检测”或“清除视频”。
6. **状态恢复**：界面轮询 `/api/status`，即使刷新页面也能恢复已上传的数据。

### 3. 图片批量缩放
```bash
python3 resize_image.py --dir query ./query_resized
```
或单图：
```bash
python3 resize_image.py path/to/img.jpg  # 输出到 img_256x128.jpg
```

## 🛠 常见问题
| 问题 | 可能原因 & 处理 |
| --- | --- |
| `parse_detection_params` 未定义 | 确保运行的是 `webapp/app.py` 最新版本（函数为全局定义），必要时重启服务。 |
| 模型加载失败 | 检查 `yolov5s.engine`, `libmyplugins.so`, `osnet_x0_25.onnx` 是否位于项目根目录，路径区分大小写。 |
| ReID 无命中 | 确认已上传/加载查询图，调低 `ReID 阈值`（例如 0.6）。 |
| 视频无法打开 | 上传路径需为 Jetson 可访问的绝对路径，或者在 webapp 中重新上传。 |
| Web 页面无画面 | 检查浏览器控制台/Flask 日志；确认已点击“开始检测”并允许 MJPEG (某些浏览器需禁用拦截)。 |

## 🗂 其他说明
- `web/` 目录保留了旧版本 Web UI，如无需要可忽略。
- `uploads/` 在运行过程中自动生成，可定期清理历史视频/图片以节省空间。
- 如需自定义更多参数（如轨迹颜色、批大小），可直接修改 `video_reid.py` / `webapp/app.py` 内的默认常量。

欢迎根据部署环境继续扩展，例如接入 RTSP 摄像头、将结果发布到消息总线、或添加 PyQt5 桌面界面。祝使用愉快！
