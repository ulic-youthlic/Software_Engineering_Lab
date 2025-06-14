import traceback
import torch.nn as nn
import cv2
from ultralytics import YOLO
import pyautogui
import numpy as np
import time


class EnhancedVisionEngine:
    def __init__(self, model_path='best.pt'):
        self.screen_height = 1080
        self.screen_width = 1920
        self.model = YOLO(model_path)
        self.orb = cv2.ORB_create(1000)
        self.max_pic_num = 40
        self.pic_idx = 0
        self.last_frame = None
        self.frame_rate = 30  # 目标帧率
        self.last_capture_time = time.time()


    def capture(self, force=False):
        """智能截图，控制帧率避免过度消耗资源"""
        current_time = time.time()
        elapsed = current_time - self.last_capture_time

        # 控制帧率，避免过度消耗资源
        if not force and elapsed < 1.0 / self.frame_rate:
            return self.last_frame

        screen = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        self.last_frame = frame
        self.last_capture_time = current_time
        return frame

    def detect(self, frame):
        """增强的目标检测，添加置信度阈值和类别过滤"""
        if frame is None:
            return {'enemies': [], 'items': [], 'features': np.zeros(256)}

        try:
            results = self.model(frame, verbose=False)  # 禁用详细日志
            results = results[0] if isinstance(results, list) else results

            # 可视化并保存检测结果
            if hasattr(results, 'boxes') and results.boxes is not None:
                self.pic_idx = (self.pic_idx + 1) % self.max_pic_num
                annotated_frame = results.plot()
                cv2.imwrite(f'debug/detection_{self.pic_idx:02d}.jpg', annotated_frame)

            enemies = []
            valid_classes = ['enemy', 'hostile', 'target']  # 可识别的敌对目标类别

            if hasattr(results, 'boxes') and results.boxes is not None:
                for box in results.boxes:
                    # 置信度过滤
                    if box.conf.item() < 0.5:
                        continue

                    # 类别过滤
                    cls_id = int(box.cls.item())
                    cls_name = self.model.names[cls_id]
                    if cls_name not in valid_classes:
                        continue

                    # 计算目标中心坐标
                    xyxy = box.xyxy[0].cpu().numpy()
                    center = (
                        (xyxy[0] + xyxy[2]) / 2 / self.screen_width,
                        (xyxy[1] + xyxy[3]) / 2 / self.screen_height
                    )
                    enemies.append(center)


            # 特征提取
            features = self._extract_deep_features(results)
            print(f"EnhancedVisionEngine,detect:feature.shape{features.shape},enemies:{enemies}")
            # 确保最终输出为260维
            if len(features) != 260:
                features = features[:260] if len(features) > 260 else np.pad(features, (0, 260 - len(features)))

            return {'enemies': enemies, 'items': [], 'features': features}

        except Exception as e:
            print(f"检测失败: {str(e)}")
            return {'enemies': [], 'items': [], 'features': np.zeros(256)}

    def _extract_deep_features(self, results):
        """更健壮的特征提取方法"""
        try:
            # 尝试不同的特征提取方法
            if hasattr(results, 'features') and results.features is not None:
                return results.features.cpu().mean(axis=(1, 2)).numpy()
            if hasattr(results, 'probs') and results.probs is not None:
                return results.probs.cpu().numpy()
            if hasattr(results, 'boxes') and results.boxes is not None:
                # 使用边界框信息作为特征
                features = []
                for box in results.boxes:
                    #features.extend(box.xyxy[0].cpu().numpy() / [self.screen_width, self.screen_height] * 2)
                    # 正确归一化边界框坐标,上面哪个是4/2显然不对
                    xyxy = box.xyxy[0].cpu().numpy()
                    normalized = np.array([
                        xyxy[0] / self.screen_width,
                        xyxy[1] / self.screen_height,
                        xyxy[2] / self.screen_width,
                        xyxy[3] / self.screen_height
                    ])
                    features.extend(normalized * 2)  # 缩放特征值

                return np.array(features[:256])  # 截断为256维
        except Exception as e:
            print("_extract_deep_features failure:")
            print(f"视觉系统详细错误: {str(e)}\n{traceback.format_exc()}")

        return np.random.randn(256)  # 默认返回随机向量

    def build_map(self, frame):
        """增强的环境地图构建，添加障碍物检测"""
        if frame is None:
            return np.zeros((15, 15))

        try:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, _ = self.orb.detectAndCompute(gray, None)
            return self._create_grid(kp)

        except Exception as e:
            print(f"地图构建失败: {str(e)}")
            return np.zeros((15, 15))

    def _create_grid(self, kp):
        """创建导航网格地图，添加障碍物标记"""
        grid = np.zeros((15, 15))
        if not kp:
            return grid

        for pt in kp:
            x = min(int(pt.pt[0] // (self.screen_width / 15)), 14)
            y = min(int(pt.pt[1] // (self.screen_height / 15)), 14)
            grid[y, x] = 1  # 标记为可通行区域

        # 添加边界障碍
        grid[0, :] = 0  # 上边界
        grid[-1, :] = 0  # 下边界
        grid[:, 0] = 0  # 左边界
        grid[:, -1] = 0  # 右边界

        return grid