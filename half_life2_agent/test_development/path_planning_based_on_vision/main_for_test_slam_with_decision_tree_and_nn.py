import traceback
from collections import deque
import cv2
import torch
import numpy as np
import time
import keyboard
import os
#from vision_for_test_slam_with_decision_tree_and_nn import EnhancedVisionEngine
from vision_for_test_slam_with_decision_tree_and_nn_test import EnhancedVisionEngine
from controller_for_test_slam_with_decision_tree_and_nn import EnhancedInputSimulator
from memory_for_test_slam_with_decision_tree_and_nn import Blank_LSTM_Memory_For_Test
from planner_for_test_slam_with_decision_tree_and_nn import Blank_PathPlanner_For_Test
from decision_tree_and_nn import EnhancedDecisionNetwork
from decision_tree_and_nn import ExplorationPlanner
from decision_tree_and_nn import DecisionTreeRegressor
from decision_tree_and_nn import HybridDecisionSystem
from log_system import EnhancedLogger

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

        # 添加决策系统
        self.decision_system = HybridDecisionSystem()

        # 记录近期奖励
        self.recent_rewards = deque(maxlen=100)

        # 训练控制
        self.train_interval = 10  # 每10步训练一次
        self.step_count = 0

        # 地图缓存
        self.prev_map = None

        self.decision_system = HybridDecisionSystem()

        #日志系统
        self.logger = EnhancedLogger()

        #自瞄开启按钮
        self.is_auto_aim=False

        #战斗持续
        self.is_in_combat=False
        self.combat_duration=5
        self.combat_time=0

        print(f"[系统] 决策系统初始化: {type(self.decision_system).__name__}")
        print(f"记忆模块模拟模式状态: {self.memory.simulation_mode}")
        print(f"视觉特征维度: {self.vision.detect(np.zeros((1600, 2560, 3), dtype=np.uint8))['features'].shape}")
        print(f"内存输入维度: {self.memory.feature_encoder[0].in_features}")
        print(f"决策网络输入维度: {self.decision_system.decision_net.encoder[0].in_features}")

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

                    # 2. 目标检测WW
                    objects = self.vision.detect(frame)

                    # 2.1 战斗分支——中断当前行为进入战斗
                    if objects.get('enemies') and len(objects['enemies']) > 0 and self.combat_time<=0:
                        #self._handle_combat(objects['enemies'])
                        self.is_in_combat=True
                        self.combat_time=self.combat_duration
                    if self.is_in_combat:
                        if self.combat_time>=0:
                            self._handle_combat(objects['enemies'])
                            self.combat_time -= 1
                            print(f"combat end in {self.combat_time}")
                            continue

                    self.is_in_combat=False

                    # 3. 环境建模
                    env_map = self.vision.build_map(frame)

                    # 4. 记忆更新
                    #memory_data = self.memory.update(objects, env_map)
                    try:
                        memory_data = self.memory.update(objects, env_map)
                        # 增加维度检查
                        if 'features' in memory_data:
                            print(f"记忆特征维度: {memory_data['features'].shape}")
                        if 'map' in memory_data:
                            print(f"地图维度: {memory_data['map'].shape}")

                        # 强制转换为字典
                        if not isinstance(memory_data, dict):
                            print(f"记忆数据不是字典，实际类型: {type(memory_data)}")
                            memory_data = {'features': np.zeros(128), 'map': np.zeros(225)}
                    except Exception as e:
                        print(f"记忆更新异常: {str(e)}")
                        memory_data = {'features': np.zeros(128), 'map': np.zeros(225)}

                    # 5. 获取当前位置
                    current_pos = self.controller.get_position()

                    # 6. 路径规划
                    path = self.planner.plan(current_pos, objects['enemies'], memory_data)

                    # 7. 混合决策
                    try:
                        # 增加特征维度检查
                        vision_features = objects['features']
                        print(f"视觉特征维度: {vision_features.shape}")

                        net_input = self.decision_system.prepare_input(
                            self.memory, vision_features, self.controller
                        )
                        print(f"决策网络输入实际维度: {net_input.shape[1]}")

                        action = self.decision_system.decide_action(
                            # self.memory.env_map.detach().cpu().numpy(),
                            memory_data,  # 改为传入memory data (字典，后面对应处理的地方也要求用字典)
                            objects['features'],
                            self.controller,
                            self.memory
                        )
                        print(f"[决策系统] 返回动作: {action}")  # 添加这行
                    except Exception as e:
                        print(f"决策系统错误: {str(e)}")
                        # 默认探索行为
                        action = ('explore', 0)
                    # 8. 执行动作
                    prev_pos = self.controller.get_position()
                    prev_map = self.memory.env_map.clone().detach()
                    # 获取执行前角色位置
                    prev_char_pos = self.controller.get_character_position()

                    # 在决策系统调用后
                    print(f"[主循环] 准备执行动作: {action}")
                    # 执行动作
                    if action[0] == 'move':
                        #self.controller.move_direction('forward', action[1])
                        self.controller.move_direction('forward')
                        print(f"[控制器] 执行移动动作")
                    elif action[0] == 'turn':
                        self.controller.precise_turn(action[1], 0)  # 无俯仰角
                        print(f"[控制器] 执行转向动作: {action[1]}度")
                    else:  # 等待
                        time.sleep(0.5)
                        print(f"[控制器] 等待")
                    # 获取执行后状态
                    new_pos = self.controller.get_position()#这个是鼠标位置
                    new_dir = self.controller.get_direction()
                    # 执行动作后角色位置
                    new_char_pos = self.controller.get_character_position()

                    # 计算角色移动向量
                    dx = new_char_pos[0] - prev_char_pos[0]
                    dy = new_char_pos[1] - prev_char_pos[1]
                    moved_distance = (dx ** 2 + dy ** 2) ** 0.5

                    # 判断是否成功移动（移动距离>阈值）
                    movement_success = moved_distance > 0.05

                    # 更新记忆中的位置和地图
                    #self.memory.agent_direction = new_dir[0]#只储存方向角度，不储存元组（还有个俯仰角不需要）
                    self.memory.agent_direction = new_dir  # 只储存方向角度，不储存元组（还有个俯仰角不需要）
                    self.memory.update_position([dx, dy], movement_success)


                    # 9.计算奖励
                    success = self._is_movement_successful(prev_pos, new_pos)
                    reward = self.decision_system.calculate_reward(
                        self.memory.env_map.detach().cpu().numpy(),
                        prev_map.cpu().numpy(),
                        action,
                        success,
                        new_pos  # 添加当前的位置
                    )
                    self.recent_rewards.append(reward)
                    print(f"[行为奖励]:{reward}")

                    # 在执行行为后，收集日志数据
                    cycle_data = {
                        'cycle_count': self.cycle_count,
                        'action': action,  # (action_type, param)
                        'prev_map': prev_map,  # 行为前的地图
                        'current_map': self.memory.env_map.detach().cpu().numpy().copy(),
                        'character_pos': self.controller.get_character_position(),
                        'direction': self.controller.get_direction(),
                        'success': success,
                        'reward': reward
                    }

                    # 记录日志
                    self.logger.log_cycle(cycle_data)



                    # 10. 记录经验
                    current_state = self.decision_system.prepare_input(
                        self.memory, objects['features'], self.controller
                    )
                    self.decision_system.record_experience(
                        current_state,
                        action,
                        reward,
                        self.decision_system.prepare_input(self.memory, objects['features'], self.controller),
                        False  # 假设不会结束
                    )

                    # 11. 定期训练
                    self.step_count += 1
                    if self.step_count % self.train_interval == 0:
                        loss = self.decision_system.update_network()
                        print(f"训练神经网络，损失值: {loss:.4f}")

                        # 调整神经网络影响力
                        self.decision_system.adjust_influence_weight(self.recent_rewards)
                        print(f"更新神经网络影响力权重: {self.decision_system.nn_influence:.2f}")


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

            self.logger.finalize_logs()
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

    def _is_movement_successful(self, prev_pos, new_pos):
        """判断移动是否成功"""
        dx = new_pos[0] - prev_pos[0]
        dy = new_pos[1] - prev_pos[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        return distance > 0.03  # 移动超过阈值视为成功

    def _handle_combat(self, enemies):
        """执行战斗行为：瞄准并攻击敌人"""
        print(f"战斗状态！发现{len(enemies)}个敌人")

        if(len(enemies)>0):
            # 选择最近的敌人（相对于屏幕中心）
            screen_center = (0.5, 0.5)
            closest_enemy = min(
                enemies,
                key=lambda e: (e[0] - screen_center[0]) ** 2 + (e[1] - screen_center[1]) ** 2
            )

            print(closest_enemy[0],closest_enemy[1])
            # 移动准心到敌人位置
            self.controller._direct_move(closest_enemy[0], closest_enemy[1])

            # 执行攻击
            self.controller.click()

            # 保存战斗截图
            frame = self.vision.capture()
            if frame is not None:
                cv2.imwrite(f'debug/combat_{self.cycle_count:04d}.jpg', frame)
        else:
            print("no visual of the enemy now,do nothing")


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