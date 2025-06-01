# 主程序架构 main.py
import cv2
import torch
#from memory_for_test import Blank_LSTM_Memory_For_Test
from .planner_for_test_without_memory import Blank_PathPlanner_For_Test
from .vision_for_test_without_memory import Blank_VisionEngine_for_Test
from .controller_for_test_without_memory import Blank_InputSimulator_for_test
import numpy as np
import time

'''
这个版本是不通过视觉特征建模地图而仅使用硬编码小地图测试除记忆模块外其余部分的
'''

class GameBot:
    def __init__(self):
        # 模块初始化
        self.vision =Blank_VisionEngine_for_Test()   # 视觉处理
        #self.memory = Blank_LSTM_Memory_For_Test()  # 记忆建模        先不做了
        self.planner = Blank_PathPlanner_For_Test()  # 路径规划
        self.controller = Blank_InputSimulator_for_test()  # 控制执行

    def run_test(self):
        """不依赖LSTM的测试流程"""


    def run(self):
        while True:
            try:
                # 处理流程
                frame = self.vision.capture()  # 图像采集[5](@ref)
                objects = self.vision.detect(frame)  # 目标检测[6](@ref)
                env_map = self.vision.build_map(frame)  # 场景建模[3](@ref)

                # 调试输出
                print(f"检测到敌人数量: {len(objects['enemies'])}")
                print(f"环境地图密度: {np.sum(env_map)}/{env_map.size}")


                current_pos = self.controller.get_position()

                path = self.planner.plan(
                    current_pos=self.controller.get_position(),
                    targets=objects['enemies'],
                    #memory=memory_data  # 路径生成[2](@ref)
                )

                if path:  # 添加空路径保护
                    self.controller.execute(path)
                else:
                    print("未找到有效路径")

                time.sleep(0.1)  # 降低循环频率
            except Exception as e:
                print(f"运行异常: {str(e)}")
                break

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



    # 测试数据
    test_objects = {
        'enemies': [(0.5, 0.3)],  # 归一化坐标
        'items': ['ammo', 'health']
    }
    test_map = np.ones((15, 15))  # 模拟环境地图

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
