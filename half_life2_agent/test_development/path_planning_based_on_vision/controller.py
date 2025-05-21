# 依赖库：pydirectinput, pyautogui
import pydirectinput
import pyautogui
import time
import numpy as np
import random

class InputSimulator:
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
            print("警告：空路径，不执行移动")
            return
        for (x, y) in path:
            self._move_mouse(x, y)
            pydirectinput.click()  # 精确点击[5](@ref)
            time.sleep(0.05)

    def _move_mouse(self, x, y):
        """贝塞尔曲线模拟人类移动"""
        points = self._bezier_curve(
            pyautogui.position(), (x, y), n=20
        )
        for p in points:
            pydirectinput.moveTo(*p)

    def _bezier_curve(self, start, end, n=20):
        """生成平滑移动轨迹"""
        """三阶贝塞尔曲线生成"""
        control1 = (start[0] + random.randint(-50, 50),
                    start[1] + random.randint(-50, 50))
        control2 = (end[0] + random.randint(-50, 50),
                    end[1] + random.randint(-50, 50))
        points = []
        for t in np.linspace(0, 1, n):
            x = (1 - t) ** 3 * start[0] + 3 * (1 - t) ** 2 * t * control1[0] + 3 * (1 - t) * t ** 2 * control2[
                0] + t ** 3 * end[0]
            y = (1 - t) ** 3 * start[1] + 3 * (1 - t) ** 2 * t * control1[1] + 3 * (1 - t) * t ** 2 * control2[
                1] + t ** 3 * end[1]
            points.append((int(x), int(y)))
        return points

    def get_position(self):
        """获取当前鼠标坐标（归一化）"""
        x, y = pyautogui.position()
        return (x / 1920, y / 1080)  # 根据实际分辨率调整