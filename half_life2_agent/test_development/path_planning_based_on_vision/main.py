# 主程序架构 main.py
import cv2
import torch
from memory import LSTM_Memory
from planner import PathPlanner
import vision
from vision import VisionEngine
from controller import InputSimulator
import numpy as np

class GameBot:
    def __init__(self,test_mode=False):
        # 模块初始化
        self.vision = VisionEngine()  # 视觉处理
        self.memory = LSTM_Memory()  # 记忆建模
        self.planner = PathPlanner()  # 路径规划
        self.controller = InputSimulator()  # 控制执行

    def run_test(self):
        """不依赖LSTM的测试流程"""
        self.memory.simulation_mode = True
        # 其他测试逻辑...

    def run(self):
        while True:
            # 处理流程
            frame = self.vision.capture()  # 图像采集[5](@ref)
            objects = self.vision.detect(frame)  # 目标检测[6](@ref)
            env_map = self.vision.build_map(frame)  # 场景建模[3](@ref)

            memory_data = self.memory.update(
                objects, env_map  # 记忆更新[8,4](@ref)
            )

            path = self.planner.plan(
                current_pos=self.controller.get_position(),
                targets=objects['enemies'],
                memory=memory_data  # 路径生成[2](@ref)
            )

            self.controller.execute(path)  # 动作执行[5](@ref)


if __name__ == "__main__":
    bot = GameBot(test_mode=True)

    # 单独测试Vision模块
    frame = bot.vision.capture()
    cv2.imwrite('debug_screenshot.jpg', frame)  # 保存截图供检查

    objects = bot.vision.detect(frame)
    print(f"检测到敌人坐标: {objects['enemies']}")

    env_map = bot.vision.build_map(frame)
    print(f"环境地图特征点数量: {np.sum(env_map)}")

    # 测试完整流程
    bot.run()
    '''
VisionEngine.capture()
   ↓（frame）
VisionEngine.detect() → objects → LSTM_Memory.update()
   ↓（frame）             ↑
VisionEngine.build_map() → env_map → LSTM_Memory.update()
                                         ↓（memory_data）
                             PathPlanner.plan()
                                   ↓（path）
                         InputSimulator.execute()
    '''