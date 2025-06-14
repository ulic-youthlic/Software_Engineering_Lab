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
        """统一输出260维特征向量"""
        if frame is None:
            return {'enemies': [], 'items': [], 'features': np.zeros(260)}

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
            #features = self._extract_deep_features(results)
            features = self._build_feature_vector(results, enemies)
            print(f"EnhancedVisionEngine,detect:feature.shape{features.shape},enemies:{enemies}")
            return {'enemies': enemies, 'items': [], 'features': features}

        except Exception as e:
            print(f"检测失败: {str(e)}")
            return {'enemies': [], 'items': [], 'features': np.zeros(260)}

    def _extract_deep_features(self, results):
        """智能特征压缩方法，保留关键信息的同时确保256维输出"""
        try:
            # 1. 基础特征：敌人数量归一化
            num_enemies = len(results.boxes) / 10.0 if results.boxes is not None else 0.0

            # 2. 基础特征：物品数量归一化（当前无物品检测）
            num_items = 0.0

            # 3. 目标位置特征（最多4个目标）
            position_features = []
            if results.boxes is not None:
                for i, box in enumerate(results.boxes[:4]):  # 最多取4个目标
                    xyxy = box.xyxy[0].cpu().numpy()
                    position_features.extend([
                        xyxy[0] / self.screen_width,  # 左上X
                        xyxy[1] / self.screen_height,  # 左上Y
                        xyxy[2] / self.screen_width,  # 右下X
                        xyxy[3] / self.screen_height  # 右下Y
                    ])

            # 4. 填充位置特征到16维（4个目标*4维）
            position_features = position_features[:16]  # 最多16维
            if len(position_features) < 16:
                position_features += [0.0] * (16 - len(position_features))

            # 5. 组合所有特征（2基础 + 16位置 = 18维）
            all_features = [num_enemies, num_items] + position_features

            # 6. 扩展到256维
            if len(all_features) < 256:
                # 使用零填充
                all_features += [0.0] * (256 - len(all_features))
            else:
                # 截断到256维
                all_features = all_features[:256]

            return np.array(all_features, dtype=np.float32)

        except Exception as e:
            print(f"特征提取失败: {str(e)}")
            return np.zeros(256, dtype=np.float32)

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

    def _compress_features(self, features, target_dim=260):
        """智能特征压缩算法"""
        # 如果特征少于目标维度，使用PCA+插值扩展
        if len(features) < target_dim:
            return self._expand_features(features, target_dim)

        # 如果特征多于目标维度，使用PCA降维
        if len(features) > target_dim:
            return self._reduce_features(features, target_dim)

        return np.array(features)

    def _reduce_features(self, features, target_dim):
        """PCA降维保留信息量最大的特征"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # 标准化
        scaler = StandardScaler()
        scaled = scaler.fit_transform(np.array(features).reshape(1, -1))

        # 计算保留95%信息的组件数
        pca = PCA(n_components=0.95)
        reduced = pca.fit_transform(scaled)

        # 如果降维后仍超过目标维度，取前n个主成分
        if reduced.shape[1] > target_dim:
            return reduced[0, :target_dim]

        # 填充剩余维度（使用最后几个特征，避免零填充）
        remaining = target_dim - reduced.shape[1]
        filler = features[-remaining:] if remaining > 0 else []
        return np.concatenate([reduced[0], np.array(filler)])

    def _expand_features(self, features, target_dim):
        """特征扩展算法"""
        from sklearn.decomposition import PCA
        from scipy import interpolate

        # 使用PCA找到特征扩展方向
        pca = PCA(n_components=min(len(features), target_dim))
        transformed = pca.fit_transform(np.array(features).reshape(1, -1))

        # 插值扩展
        x = np.arange(len(features))
        x_new = np.linspace(0, len(features) - 1, target_dim)

        # 创建插值函数
        f = interpolate.interp1d(x, features, kind='cubic', fill_value="extrapolate")
        return f(x_new)

    def _build_feature_vector(self, results, enemies):
        """构建260维特征向量：1(敌人数)+1(物品数)+2(最近敌坐标)+256(视觉特征)"""
        # 1. 基础特征
        num_enemies = len(enemies) / 10.0
        num_items = 0.0  # 暂时没有物品检测

        # 2. 最近敌人坐标
        nearest_enemy = [0.0, 0.0]
        if enemies:
            nearest_enemy = list(enemies[0])

        # 3. 视觉特征（保持256维）
        visual_features = self._extract_visual_features(results)

        # 4. 合并所有特征
        combined = np.concatenate([
            [num_enemies, num_items],
            nearest_enemy,
            visual_features
        ])

        # 5. 确保维度正确
        if combined.size < 260:
            return np.pad(combined, (0, 260 - combined.size))
        return combined[:260]

    def _extract_visual_features(self, results):
        """提取256维视觉特征"""
        try:
            # 基础特征：敌人数量和位置
            position_features = []
            if results.boxes is not None:
                for i, box in enumerate(results.boxes[:4]):
                    xyxy = box.xyxy[0].cpu().numpy()
                    position_features.extend([
                        xyxy[0] / self.screen_width,
                        xyxy[1] / self.screen_height,
                        xyxy[2] / self.screen_width,
                        xyxy[3] / self.screen_height
                    ])

            # 填充到256维
            features = position_features[:256]
            if len(features) < 256:
                features += [0.0] * (256 - len(features))
            return np.array(features, dtype=np.float32)

        except Exception as e:
            print(f"视觉特征提取失败: {str(e)}")
            return np.zeros(256, dtype=np.float32)