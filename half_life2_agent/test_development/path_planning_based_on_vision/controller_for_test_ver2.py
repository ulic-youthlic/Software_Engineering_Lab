# 依赖库：pydirectinput, pyautogui
import pydirectinput
import pyautogui
import time
import numpy as np
import random

class Blank_InputSimulator_for_test:
    def __init__(self):
        self.pos_history = []

    def execute(self, path):
        '''
        def execute(path):
    ├─ 输入：path（坐标序列）
    ├─ 贝塞尔曲线模拟移动（_bezier_curve）
    │├─ 生成20个中间点
    │└─ 添加随机扰动（±50像素）
    └─ 执行点击操作（pydirectinput.click）
        :param path:
        :return:
        '''
        if not path:
            return

            # 只取前3个路径点作为演示
        for target in path[:3]:
            self._move_mouse(target[0], target[1])
            pydirectinput.click()
            time.sleep(0.3)

    def _move_mouse(self, x, y):
        """贝塞尔曲线模拟人类移动"""
        AIM_MODE=0
        screen_width, screen_height = pyautogui.size()

        if AIM_MODE:
            """带随机扰动的移动"""
            # 获取当前鼠标位置（起点）
            start_x, start_y = pyautogui.position()
            start = (start_x / screen_width, start_y / screen_height)
            end = (x, y)

            # 生成贝塞尔曲线路径
            path = self._bezier_curve(start, end)

            # 沿路径移动鼠标
            for point in path:
                # 转换回屏幕坐标
                target_x = int(point[0] * screen_width)
                target_y = int(point[1] * screen_height)

                # 边界检查
                target_x = max(0, min(screen_width - 1, target_x))
                target_y = max(0, min(screen_height - 1, target_y))

                # 移动到路径点
                pydirectinput.moveTo(target_x, target_y)
                # time.sleep(0.01)  # 每个点之间的延迟

        else:
            x = max(0, min(screen_width - 1, int(x * screen_width) + random.randint(-50, 50)))
            y = max(0, min(screen_height - 1, int(y * screen_height) + random.randint(-50, 50)))
            # x = int(x * 1920) + random.randint(-50, 50)
            # y = int(y * 1080) + random.randint(-50, 50)
            pydirectinput.moveTo(x, y, duration=0.3)


    def _bezier_curve(self, start, end, n=20):
        """生成平滑的贝塞尔曲线轨迹"""
        # 计算起点和终点的方向向量
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5

        # 如果距离太短，直接返回直线
        if distance < 0.01:  # 归一化距离阈值
            return [start, end]

        # 生成两个控制点（在路径两侧偏移）
        # 控制点偏移量（基于距离的随机比例）
        offset_factor = random.uniform(0.2, 0.4) * distance

        # 计算垂直方向（与移动方向垂直）
        perp_x = -dy / max(distance, 0.0001)  # 避免除以零
        perp_y = dx / max(distance, 0.0001)

        '''
        # 为两个控制点添加随机偏移
        offset1 = offset_factor * random.uniform(0.5, 1.0)
        offset2 = offset_factor * random.uniform(0.5, 1.0)
        '''
        offset1=0   #先不加试试
        offset2=0

        # 计算控制点位置（在路径两侧）
        ctrl1 = (
            start[0] + dx * 0.3 + perp_x * offset1,
            start[1] + dy * 0.3 + perp_y * offset1
        )

        ctrl2 = (
            start[0] + dx * 0.7 + perp_x * offset2 * (1 if random.random() > 0.5 else -1),
            start[1] + dy * 0.7 + perp_y * offset2 * (1 if random.random() > 0.5 else -1)
        )

        # 生成曲线上的点
        points = []
        for i in range(n):
            t = i / (n - 1)
            # 三次贝塞尔曲线公式
            x = (1 - t) ** 3 * start[0] + 3 * (1 - t) ** 2 * t * ctrl1[0] + 3 * (1 - t) * t ** 2 * ctrl2[0] + t ** 3 * \
                end[0]
            y = (1 - t) ** 3 * start[1] + 3 * (1 - t) ** 2 * t * ctrl1[1] + 3 * (1 - t) * t ** 2 * ctrl2[1] + t ** 3 * \
                end[1]
            points.append((x, y))

        return points

    def get_position(self):
        """模拟返回鼠标当前位置"""
        x, y = pyautogui.position()
        return (x/1920.0, y/1080.0)