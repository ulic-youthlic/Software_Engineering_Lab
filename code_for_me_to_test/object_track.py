from ultralytics import YOLO
import lap
#model=YOLO('yolov8n.pt')#检测模型
model=YOLO('yolov8n-seg.pt')#分割模型
#model=YOLO('yolov8n-pose.pt')#姿态模型
#用模型进行追踪
#results=model.track(source="./target.mp4",show=True)
results=model.track(source="./target.mp4",show=True,tracker="bytetrack.yaml",save=True)
#results=model.track(source="./target.mp4",save=True)
