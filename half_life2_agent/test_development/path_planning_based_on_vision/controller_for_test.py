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

        #先拿这个错测试
        """带随机扰动的移动"""
        x = int(x * 1920) + random.randint(-50, 50)
        y = int(y * 1080) + random.randint(-50, 50)
        pydirectinput.moveTo(x, y, duration=0.3)

    def _bezier_curve(self, start, end, n=20):
        """生成平滑移动轨迹"""
        pass

    def get_position(self):
        """模拟返回鼠标当前位置"""
        x, y = pyautogui.position()
        return (x/1920.0, y/1080.0)