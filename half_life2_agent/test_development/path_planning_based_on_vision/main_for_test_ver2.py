# 主程序架构 main.py
import cv2
import torch
from memory_for_test_ver2 import Blank_LSTM_Memory_For_Test
from planner_for_test_ver2 import Blank_PathPlanner_For_Test
import vision
from vision import VisionEngine
from vision_for_test_ver2 import Blank_VisionEngine_for_Test
from controller_for_test_ver2 import Blank_InputSimulator_for_test
import numpy as np
import time
import random
import keyboard

class GameBot:
    def __init__(self):
        # 模块初始化
        self.vision =Blank_VisionEngine_for_Test()   # 视觉处理
        self.memory = Blank_LSTM_Memory_For_Test()  # 记忆建模
        self.planner = Blank_PathPlanner_For_Test()  # 路径规划
        self.controller = Blank_InputSimulator_for_test()  # 控制执行

    def run_test(self):
        """不依赖LSTM的测试流程"""
        self.memory.simulation_mode = True
        # 其他测试逻辑...

    def run(self):
        while True:
            try:
                # 检查ESC键是否被按下
                if keyboard.is_pressed('esc'):
                    print("检测到ESC按键，程序退出")
                    self.running = False
                    exit(0)
                # 处理流程
                frame = self.vision.capture()  # 图像采集[5](@ref)
                objects = self.vision.detect(frame)  # 目标检测[6](@ref)
                env_map = self.vision.build_map(frame)  # 场景建模[3](@ref)

                # 调试输出
                print(f"检测到敌人数量: {len(objects['enemies'])}")
                print(f"环境地图密度: {np.sum(env_map)}/{env_map.size}")

                memory_data = self.memory.update(
                    objects, env_map  # 记忆更新[8,4](@ref)
                )
                print(f"记忆向量维度: {memory_data.shape}")

                current_pos = self.controller.get_position()

                # 路径规划
                path = self.planner.plan(
                    current_pos=current_pos,
                    targets=objects['enemies'],
                    memory=memory_data  # 路径生成[2](@ref)
                )

                if path:  # 添加空路径保护
                    self.controller.execute(path)

                else:
                    print("未找到有效路径")
                    print("random_move未启用")
                    #self._random_move()


                time.sleep(0.1)  # 降低循环频率



            except Exception as e:
                print(f"运行异常: {str(e)}")
                import traceback
                traceback.print_exc()
                '''
                # 重置关键状态
                self.memory.hidden = None
                time.sleep(1)  # 防止错误循环
                '''
                break

    def _random_move(self):
        """无目标时随机移动"""
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        self.controller._move_mouse(x, y)

    def input(key):
        if key=="escape":
            exit(0)

    def stop(self):
        self.running=False
if __name__ == "__main__":
    bot = GameBot()

    # 单独测试Vision模块
    frame = bot.vision.capture()
    cv2.imwrite('debug_screenshot.jpg', frame)  # 保存截图供检查

    objects = bot.vision.detect(frame)
    print(f"检测到敌人坐标: {objects['enemies']}")

    env_map = bot.vision.build_map(frame)
    print(f"环境地图特征点数量: {np.sum(env_map)}")

    # 测试完整流程
    bot.run()

    #############

    # 初始化记忆模块
    memory = Blank_LSTM_Memory_For_Test()

    # 测试数据
    test_objects = {
        'enemies': [(0.5, 0.3)],  # 归一化坐标
        'items': ['ammo', 'health']
    }
    test_map = np.ones((15, 15))  # 模拟环境地图

    # 正常模式测试
    output = memory.update(test_objects, test_map)
    print(f"记忆输出形状: {output.shape}")

    # 模拟模式测试
    memory.simulation_mode = True
    sim_output = memory.update({}, np.zeros((15, 15)))
    print(f"模拟记忆输出: {sim_output}")
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