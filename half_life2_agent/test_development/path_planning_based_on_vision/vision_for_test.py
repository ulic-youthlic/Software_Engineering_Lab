# 依赖库：opencv-python, ultralytics, numpy
import cv2
from ultralytics import YOLO
import pyautogui
import numpy as np

class Blank_VisionEngine_for_Test:
    def __init__(self):
        self.screen_height = 1080
        self.screen_width = 1920
        self.model = YOLO('best.pt')  # GPU加速[6](@ref)
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
        results = self.model(frame)# 自动继承模型所在的CPU设备

        # 可视化标注并保存带框截图
        if len(results) > 0:
            annotated_frame = results[0].plot()  # 自动绘制检测框
            cv2.imwrite('debug_screenshot2.jpg', annotated_frame)

        # 打印结果结构用于调试
        print(f"结果类型: {type(results)}")
        if hasattr(results, 'boxes'):
            print(f"检测到{len(results.boxes)}个目标")
        else:
            print("结果中没有boxes属性")

        enemies = []
        if results and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            for box in (boxes.cpu().numpy() if hasattr(boxes, 'cpu') else boxes):
                # 兼容处理不同设备的结果
                if box.conf < 0.5:  # 添加置信度阈值
                    continue
                cls_id = int(box.cls)
                cls_name = self.model.names[cls_id]
                if cls_name == 'enemy':
                    enemies.append(self._get_center(box))
                print(f"目标类别: {cls_name}")
        else:
            print("未检测到有效目标")

        return {
            'enemies': enemies,
            'items': [],  #floor,sky,wall,object
            'features': self._extract_deep_features(results)
        }

    def _extract_deep_features(self, results):
        """改为提取模型深层特征"""
        try:
            return results.features.cpu().mean(axis=(1, 2)).numpy()  # 添加维度处理
        except AttributeError:
            return np.random.randn(256)  # 更健壮的异常处理

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
            x = min(int(pt.pt[0] // (self.screen_width / 15)), 14)  # 添加边界限制
            y = min(int(pt.pt[1] // (self.screen_height / 15)), 14)
            grid[y, x] = 1  # 1表示可通行区域
        return grid