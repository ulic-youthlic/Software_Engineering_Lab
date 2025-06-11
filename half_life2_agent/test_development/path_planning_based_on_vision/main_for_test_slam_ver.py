import cv2
import torch
import numpy as np
import time
import keyboard
import os
from vision_for_test_slam_ver import EnhancedVisionEngine
from controller_for_test_slam_ver import EnhancedInputSimulator
from memory_for_test_slam_ver import Blank_LSTM_Memory_For_Test
from planner_for_test_slam_ver import Blank_PathPlanner_For_Test


class GameBot:
    def __init__(self):
        # 创建调试目录
        os.makedirs('debug', exist_ok=True)

        # 初始化模块
        self.vision = EnhancedVisionEngine('best.pt')
        self.memory = Blank_LSTM_Memory_For_Test()
        self.planner = Blank_PathPlanner_For_Test()
        self.controller = EnhancedInputSimulator()

        # 状态变量
        self.running = True
        self.debug_mode = True
        self.cycle_count = 0
        self.start_time = time.time()

        # 性能统计
        self.processing_times = []
        self.detection_counts = []

        print("game bot initialized")

    def run(self):
        """主循环，增强错误处理和性能监控"""
        print("activating main loop...")
        self.start_time = time.time()

        try:
            while self.running:
                cycle_start = time.time()

                # 检查退出按键
                if keyboard.is_pressed('esc'):
                    print("ESC detected,exit")
                    self.running = False
                    break

                # 处理流程
                try:
                    # 1. 采集图像
                    frame = self.vision.capture()

                    # 2. 目标检测
                    objects = self.vision.detect(frame)

                    # 3. 环境建模
                    env_map = self.vision.build_map(frame)

                    # 4. 记忆更新
                    memory_data = self.memory.update(objects, env_map)

                    # 5. 获取当前位置
                    current_pos = self.controller.get_position()

                    # 6. 路径规划
                    path = self.planner.plan(current_pos, objects['enemies'], memory_data)

                    # 7. 执行动作
                    prev_pos = self.controller.get_position()
                    prev_dir = self.controller.get_direction()

                    if path:
                        self.controller.execute(path)
                    else:
                        self.controller.random_move()

                    # 获取执行后的状态
                    new_pos = self.controller.get_position()
                    new_dir = self.controller.get_direction()

                    # 计算移动向量
                    dx = new_pos[0] - prev_pos[0]
                    dy = new_pos[1] - prev_pos[1]
                    moved_distance = (dx ** 2 + dy ** 2) ** 0.5

                    # 判断是否成功移动（移动距离>阈值）A
                    movement_success = moved_distance > 0.05

                    # 更新记忆中的位置和地图
                    self.memory.agent_direction = new_dir
                    self.memory.update_position([dx, dy], movement_success)

                except Exception as e:
                    print(f"main loop error: {str(e)}")
                    # 重置关键状态
                    self.memory.hidden = None
                    # 保存错误截图
                    if frame is not None:
                        cv2.imwrite('debug/error_frame.jpg', frame)

                # 性能监控
                cycle_time = time.time() - cycle_start
                self.processing_times.append(cycle_time)
                self.detection_counts.append(len(objects.get('enemies', [])))
                self.cycle_count += 1

                # 调试输出
                if self.debug_mode and self.cycle_count % 10 == 0:
                    self.print_status()

                # 控制循环频率
                time.sleep(max(0, 0.1 - cycle_time))

        finally:
            self.shutdown()

    def print_status(self):
        """打印系统状态和性能统计"""
        avg_time = np.mean(self.processing_times[-10:]) if self.processing_times else 0
        fps = 1 / avg_time if avg_time > 0 else 0
        avg_detections = np.mean(self.detection_counts[-10:]) if self.detection_counts else 0

        print("\n===== 系统状态 =====")
        print(f"运行时长: {time.time() - self.start_time:.1f}秒")
        print(f"循环次数: {self.cycle_count}")
        print(f"平均处理时间: {avg_time * 1000:.1f}ms")
        print(f"估计帧率: {fps:.1f}FPS")
        print(f"平均检测目标: {avg_detections:.1f}")
        print("===================")

    def shutdown(self):
        """清理资源"""
        print("shuting down bot...")
        # 保存性能数据
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            print(f"平均循环时间: {avg_time * 1000:.1f}ms")
        # 保存截图
        frame = self.vision.capture(force=True)
        if frame is not None:
            cv2.imwrite('debug/final_screenshot.jpg', frame)

        print("game bot offline")


if __name__ == "__main__":
    bot = GameBot()

    # 模块测试
    print("测试视觉模块...")
    test_frame = bot.vision.capture(force=True)
    cv2.imwrite('debug/test_screenshot.jpg', test_frame)

    print("测试目标检测...")
    test_objects = bot.vision.detect(test_frame)
    print(f"检测到 {len(test_objects['enemies'])} 个目标")

    print("测试环境建模...")
    test_map = bot.vision.build_map(test_frame)
    print(f"地图密度: {np.sum(test_map)}/{test_map.size}")

    print("测试控制模块...")
    bot.controller.move_to(0.5, 0.5)

    # 启动主循环
    print("启动主循环")
    bot.run()