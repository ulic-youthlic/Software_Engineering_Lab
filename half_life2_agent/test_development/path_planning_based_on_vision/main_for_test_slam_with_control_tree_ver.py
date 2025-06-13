import math
import traceback
import cv2
import torch
import numpy as np
import time
import keyboard
import os
from vision_for_test_slam_with_control_tree_ver import EnhancedVisionEngine
from controller_for_test_slam_with_control_tree_ver import EnhancedInputSimulator
from memory_for_test_slam_with_control_tree_ver import Blank_LSTM_Memory_For_Test
from planner_for_test_slam_with_control_tree_ver import Blank_PathPlanner_For_Test
from decision_tree_for_test_slam_with_control_tree_ver import EnhancedDecisionTree

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

        # 决策树
        self.decision_tree = EnhancedDecisionTree(self.memory)
        self.last_decision = None
        self.last_position = (0.5, 0.5)

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

                    # 5. 获取当前位置(统一用归一化坐标)
                    current_pos = self.controller.get_position()
                    '''
                    # 6. 路径规划
                    path = self.planner.plan(current_pos, objects['enemies'], memory_data)

                    # 7. 执行动作
                    prev_pos = self.controller.get_position()
                    prev_dir = self.controller.get_direction()
                
                    if path:
                        self.controller.execute(path)
                    else:
                        self.controller.random_move()
                    '''

                    # 6. 决策制定
                    decision = self.decision_tree.decide(current_pos)
                    self.last_decision = decision  # 记录决策

                    # 7. 执行动作
                    prev_pos = self.controller.get_position()
                    prev_dir = self.controller.get_direction()

                    print("\n===== 决策执行 =====")
                    print(f"当前坐标: ({prev_pos[0]:.4f}, {prev_pos[1]:.4f})")
                    print(f"当前方向: {prev_dir}°")

                    # 根据决策类型执行不同动作
                    action_type = decision[0]

                    if action_type == 'move':
                        target_pos, turn_angle = decision[1], decision[2]
                        print(f"决策行为: 移动到({target_pos:.4f}, {target_pos[1]:.4f})")
                        print(f"需要转向: {turn_angle}°")

                        # 移动决策 (来自决策树或神经网络)    移动决策返回 (action, distance, turn_angle, pitch)
                        #_, direction, distance, angle, pitch = decision
                        distance, turn_angle, pitch = decision[1], decision[2], decision[3]
                        # 执行转向（俯仰角固定为0）
                        self.controller.precise_turn(turn_angle, 0)
                        #self.controller.move_to(target_pos[0], target_pos[1])
                        # 执行移动
                        self.controller.move_direction('forward', distance)

                        # 获取新位置
                        new_pos = self.controller.get_position()



                        # 计算实际移动向量
                        actual_dx = new_pos[0] - prev_pos[0]
                        actual_dy = new_pos[1] - prev_pos[1]
                        moved_distance = math.sqrt(actual_dx ** 2 + actual_dy ** 2)

                        # 评估移动是否成功
                        movement_success = moved_distance > max(0.05, distance * 0.3)
                        success_msg = "成功" if movement_success else "失败"
                        print(f"移动结果: {success_msg} (预期:{distance:.4f}, 实际:{moved_distance:.4f})")

                        # 更新记忆（只使用位移向量）
                        dx = new_pos[0] - prev_pos[0]
                        dy = new_pos[1] - prev_pos[1]
                        self.memory.update_position(dx, dy, movement_success)

                        # 更新记忆中的方向
                        new_dir = self.controller.get_direction()
                        self.memory.agent_direction = new_dir  # 直接赋值单个角度值

                        # 评估决策效果    成功（移动距离>阈值的50%）
                        decision_reward = self.decision_tree.evaluate_decision(
                            prev_pos, new_pos, movement_success
                        )

                        print(f"决策奖励: {decision_reward:.2f}")
                        print("====================\n")

                        # 记录决策用于训练
                        state_vector = self.decision_tree._build_state_vector(prev_pos)
                        self.decision_tree.record_decision(state_vector, decision, decision_reward)

                        # 更新记忆中的位置和地图
                        self.memory.agent_direction = new_dir[0]  # 只储存方向角度，不储存元组（还有个俯仰角不需要）
                        self.memory.update_position(actual_dx, actual_dy, movement_success)



                        '''
                        # 计算移动距离（基于目标距离）
                        dx = target_pos[0] - prev_pos[0]
                        dy = target_pos[1] - prev_pos[1]
                        distance = (dx ** 2 + dy ** 2) ** 0.5
                        
                        
                        # 执行移动
                        self.controller.move_direction('forward', distance)
                        
                        # 执行决策
                        self._execute_decision(decision)


                        # 执行移动并获取位移向量
                        displacement = self.controller.move_direction('forward')

                        # 获取执行后的状态
                        new_pos = self.controller.get_position()
                        print(f"新坐标: ({new_pos[0]:.4f}, {new_pos[1]:.4f})")
                        new_dir = self.controller.get_direction()

                        # 计算实际移动距离
                        moved_distance = ((new_pos[0] - prev_pos[0]) ** 2 +
                                          (new_pos[1] - prev_pos[1]) ** 2) ** 0.5
                        # 计算预期移动距离
                        expected_distance = math.sqrt(
                            (target_pos[0] - prev_pos[0]) ** 2 +
                            (target_pos[1] - prev_pos[1]) ** 2
                        )
                        # 判断是否成功移动（移动距离>阈值的75%）A
                        movement_success = moved_distance > expected_distance * 0.5
                        success_msg = "成功" if movement_success else "失败"
                        print(f"移动结果: {success_msg} (预期:{distance:.4f}, 实际:{moved_distance:.4f})")

                        # 评估决策效果    成功（移动距离>阈值的50%）
                        decision_reward = self.decision_tree.evaluate_decision(
                            prev_pos, new_pos, movement_success
                        )


                        print(f"决策奖励: {decision_reward:.2f}")
                        print("====================\n")

                        # 更新记忆中的位置和地图
                        self.memory.agent_direction = new_dir[0]#只储存方向角度，不储存元组（还有个俯仰角不需要）
                        #self.memory.update_position([dx, dy], movement_success)
                        self.memory.update_position(displacement, movement_success)
                        '''
                    elif action_type == 'turn':
                        # 转向决策 (主要来自神经网络)
                        _, angle, pitch = decision

                        # 执行转向
                        self.controller.precise_turn(angle, pitch)

                        # 获取新方向
                        new_dir = self.controller.get_direction()

                        # 评估转向是否成功（角度误差小于10度）
                        angle_diff = abs((new_dir - prev_dir + 180) % 360 - 180)
                        turn_success = angle_diff < 10

                        # 转向不影响位置
                        new_pos = prev_pos
                        movement_success = turn_success  # 复用变量

                        self.memory.agent_direction = new_dir

                        # 评估决策效果
                        decision_reward = self.decision_tree.evaluate_decision(
                            prev_pos, new_pos, movement_success
                        )

                        print(f"决策奖励: {decision_reward:.2f}")
                        print("====================\n")

                        # 记录决策用于训练
                        state_vector = self.decision_tree._build_state_vector(prev_pos)
                        self.decision_tree.record_decision(state_vector, decision, decision_reward)


                    elif action_type == 'wait':
                        # 等待决策 (主要来自神经网络)
                        _, duration = decision
                        time.sleep(duration)

                        # 等待不影响位置和方向
                        new_pos = prev_pos
                        new_dir = prev_dir
                        movement_success = True  # 等待总是成功
                        self.memory.agent_direction = new_dir

                        # 评估决策效果
                        decision_reward = self.decision_tree.evaluate_decision(
                            prev_pos, new_pos, movement_success
                        )

                        print(f"决策奖励: {decision_reward:.2f}")
                        print("====================\n")

                        # 记录决策用于训练
                        state_vector = self.decision_tree._build_state_vector(prev_pos)
                        self.decision_tree.record_decision(state_vector, decision, decision_reward)

                except Exception as e:
                    print(f"main loop error: {str(e)}\n{traceback.format_exc()}")
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

    def _execute_decision(self, decision):
        """执行神经网络生成的决策"""
        action_type = decision[0]

        if action_type == 'move':
            _, distance, angle, pitch = decision
            self.controller.precise_turn(angle, pitch)
            self.controller.move_direction('forward', distance)

        elif action_type == 'turn':
            _, angle, pitch = decision
            self.controller.precise_turn(angle, pitch)

        else:  # wait
            _, duration = decision
            time.sleep(duration)

    def _calculate_reward(self, prev_pos, new_pos, success):
        """计算决策奖励"""
        base_reward = 0.1  # 基础奖励

        # 成功移动奖励
        if success:
            base_reward += 0.5

        # 探索新区域奖励
        grid_x, grid_y = self.memory._pos_to_grid(new_pos)
        if self.memory.env_map[grid_y, grid_x] < 0.3:
            base_reward += 0.4

        # 碰撞惩罚
        if not success:
            base_reward -= 0.7

        return base_reward

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