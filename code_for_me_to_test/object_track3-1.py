from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8n.pt')

# 打开视频文件
video_path = "target.mp4"
cap = cv2.VideoCapture(video_path)

# 存储追踪历史
track_history = defaultdict(lambda: [])

# 循环遍历视频帧
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 运行YOLOv8追踪
    results = model.track(frame, persist=True)

    # 初始化boxes和track_ids
    boxes = []
    track_ids = []

    if results[0].boxes is not None:
        # 获取检测框
        boxes = results[0].boxes.xywh.cpu()
        # 安全获取追踪ID
        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        except AttributeError:
            track_ids = []

    # 绘制检测结果
    annotated_frame = results[0].plot()

    # 绘制追踪路径
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > 30:
            track.pop(0)

        # 绘制轨迹线
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    cv2.imshow("YOLOv8 Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()