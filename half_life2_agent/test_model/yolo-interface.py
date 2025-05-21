from ultralytics import YOLO
import os
#print(os.environ.get('CUDA_VISIBLE_DEVICES'))  # 应为 None 或有效 GPU 索引
import torch
import onnx
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"检测到的 GPU 数量: {torch.cuda.device_count()}")

def main():
    model = YOLO('yolov8n-seg.pt')
    results = model.train(
        data='D:\\pycharm\\PythonProject1\\half_life2_agent\\test_model\\data_2.yaml',
        batch=16,
        device='cpu',
        project='yolov8_train1-20epochs',
        epochs=20,
        save=True,
        workers=0,  # 降低 worker 数量
        task='segment',
        exist_ok=False
    )
    model.export(format='onnx')  # 导出为 ONNX

if __name__ == '__main__':
    main()
