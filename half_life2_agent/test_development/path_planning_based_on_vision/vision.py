# 依赖库：opencv-python, ultralytics, numpy
import cv2
from ultralytics import YOLO
import pyautogui
import numpy as np

class VisionEngine:
    def __init__(self):
        self.screen_height = 1080
        self.screen_width = 1920
        self.model = YOLO('best.pt').cuda()  # GPU加速[6](@ref)
        self.orb = cv2.ORB_create(1000)  # 特征提取[3](@ref)

    def capture(self):
        '''
        def capture() -> frame:
    ├─ 调用pyautogui截图（1920x1080分辨率）
    └─ 返回BGR格式的numpy数组
        '''
        screen = pyautogui.screenshot()  # 使用网页5方法[5](@ref)
        return cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

    def detect(self, frame):
        '''
        def detect(frame) -> objects:
    ├─ 输入：frame（cv2图像）
    ├─ 通过YOLOv8检测敌人（best.pt模型）
    ├─ 解析包围盒坐标（_get_center方法）
    ├─ 提取深层特征（_extract_deep_features）
    └─ 返回字典：{'enemies': [(x1,y1),...], 'features': array}
        :param frame:
        :return:
        '''
        results = self.model(frame)[0]

        # 调试：输出原始检测结果
        print(f"检测到{len(results.boxes)}个目标")

        enemies = []
        for box in results.boxes:
            cls_id = int(box.cls)
            cls_name = results.names[cls_id]  # 确保正确获取类别名
            if cls_name == 'enemy':  # 根据实际类别名调整
                enemies.append(self._get_center(box))
            # 调试：输出每个目标的类别
            print(f"目标类别: {cls_name}")

        return {
            'enemies': enemies,
            'items': [],  # 暂留空，后续补充
            'features': self._extract_deep_features(results)
        }

    def _extract_deep_features(self, results):
        """改为提取模型深层特征"""
        if hasattr(results, 'features'):  # 如果YOLO模型有特征输出
            return results.features.mean(axis=(1,2))  # 示例调整
        else:
            return np.random.randn(256)  # 模拟特征向量

    def build_map(self, frame):
        '''
        ├─ 输入：frame（cv2图像）
    ├─ 提取ORB特征点（1000个关键点）
    └─ 返回15x15栅格地图（_create_grid方法）
        :param frame:
        :return:
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)  # ORB特征[3](@ref)
        return self._create_grid(kp)  # 转换为15x15栅格

    def _get_center(self, box):
        """计算目标包围盒中心坐标（屏幕相对坐标）"""
        xyxy = box.xyxy[0].cpu().numpy()
        return ((xyxy[0] + xyxy[2]) / 2 / self.screen_width,  # 归一化坐标
                (xyxy[1] + xyxy[3]) / 2 / self.screen_height)


    def _create_grid(self, kp):
        """将特征点转换为可导航栅格地图"""
        grid = np.zeros((15, 15))
        for pt in kp:
            x = int(pt.pt[0] // (self.screen_width / 15))
            y = int(pt.pt[1] // (self.screen_height / 15))
            grid[y, x] = 1  # 1表示可通行区域
        return grid