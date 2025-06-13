import random
import math
import torch.nn as nn
import torch
from EnhancedDecisionNetwork import EnhancedDecisionNetwork
import numpy as np

class EnhancedDecisionTree:
    def __init__(self, memory):
        self.memory = memory
        # 决策评估参数
        self.exploration_reward = 1.0
        self.collision_penalty = -0.5
        self.repeat_penalty = -0.3
        self.max_angle = 90  # 最大转向角度
        self.nn = EnhancedDecisionNetwork()
        self.buffer = []  # 存储训练数据
        self.nn_weight = 0.7  # 神经网络决策权重

    def decide(self, current_pos):
        """基于当前状态做出决策"""

        # 获取当前状态信息
        map_data = self.memory.env_map
        direction = self.memory.agent_direction
        grid_x, grid_y = self.memory._pos_to_grid(current_pos)#current_pos 是归一化坐标

        '''
        # 决策树核心逻辑
        if self._should_explore_new_area(map_data, grid_x, grid_y):
            return self._explore_new_area(map_data, grid_x, grid_y)
        elif self._should_revisit_area(map_data, grid_x, grid_y):
            return self._revisit_area(map_data, grid_x, grid_y)
        else:
            return self._safe_random_move()
        '''

        # 构建神经网络输入
        state_vector = self._build_state_vector(current_pos)

        # 获取神经网络预测
        with torch.no_grad():
            action_probs, action_params = self.nn(state_vector)
        # 决策融合 (神经网络预测+启发式规则)
        if random.random() < self.nn_weight:
            # 神经网络主导决策
            action_idx = torch.argmax(action_probs).item()
            params = action_params.squeeze().tolist()
            return self._nn_decision(action_idx, params)
        else:
            # 启发式规则备用
            return self._heuristic_decision()




    def evaluate_decision(self, prev_pos, new_pos, movement_success):
        """评估决策效果并更新决策树参数"""
        # 计算移动距离
        dx = new_pos[0] - prev_pos[0]
        dy = new_pos[1] - prev_pos[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5

        # 计算探索收益
        exploration_gain = 0
        if movement_success:
            # 成功移动到新区域
            new_grid_x, new_grid_y = self.memory._pos_to_grid(new_pos)
            if self.memory.env_map[new_grid_y, new_grid_x] < 0.3:  # 未探索区域
                exploration_gain = self.exploration_reward

        # 计算惩罚
        penalty = 0
        if not movement_success:
            penalty = self.collision_penalty
        elif distance < 0.05:  # 移动距离过小
            penalty = self.repeat_penalty

        # 更新决策参数（简单示例）
        total_reward = exploration_gain + penalty
        if total_reward > 0:
            # 奖励探索行为
            self.exploration_reward = min(1.5, self.exploration_reward * 1.05)
        elif total_reward < 0:
            # 惩罚无效行为
            self.collision_penalty = max(-1.0, self.collision_penalty * 0.95)

        return total_reward

    def evaluate_position(self, grid_pos):
        """评估位置的探索价值"""
        # 将地图数据转换为张量
        map_tensor = torch.tensor(self.memory.env_map.flatten(), dtype=torch.float32)
        with torch.no_grad():
            score = self.nn(map_tensor).item()
        return score

    def _should_explore_new_area(self, map_data, x, y):
        """检查周围是否有未探索区域"""
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 15 and 0 <= ny < 15:
                    if map_data[ny, nx] < 0.3:  # 未探索区域
                        return True
        return False

    def _explore_new_area(self, map_data, x, y):
        """向未探索区域移动"""
        # 找到最近的未探索区域
        best_score = float('-inf')
        best_target = None

        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 15 and 0 <= ny < 15:
                    if map_data[ny, nx] < 0.3:  # 未探索区域
                        # 计算距离得分（偏好较近区域）
                        distance_score = 1.0 / (abs(dx) + abs(dy) + 0.1)
                        # 计算方向得分（偏好前方区域）
                        angle = self._calculate_angle_to_target(dx, dy)
                        direction_score = 1.0 - abs(angle) / 180
                        # 综合得分
                        score = distance_score * 0.7 + direction_score * 0.3

                        if score > best_score:
                            best_score = score
                            best_target = (nx, ny)

        if best_target:
            current_pos = self.memory.agent_position
            target_pos = self._grid_to_pos(best_target)
            turn_angle = self._calculate_turn_angle(best_target, x, y)
            # 计算移动距离（归一化）
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            distance = math.sqrt(dx ** 2 + dy ** 2)

            return ('move', distance, turn_angle, 0)  # 俯仰角设为0
        return self._safe_random_move()

    def _should_revisit_area(self, map_data, x, y):
        """检查是否有需要重新探索的区域"""
        # 简单实现：有一定概率重新探索
        return random.random() < 0.2

    def _revisit_area(self, map_data, x, y):
        """重新探索低置信度区域"""
        # 找到置信度最低的区域
        min_confidence = float('inf')
        best_target = None

        for dx in range(-5, 6):
            for dy in range(-5, 6):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 15 and 0 <= ny < 15:
                    confidence = map_data[ny, nx]
                    if confidence < min_confidence and confidence > 0.1:  # 避免完全未探索区域
                        min_confidence = confidence
                        best_target = (nx, ny)

        if best_target:
            target_pos = self._grid_to_pos(best_target)
            turn_angle = self._calculate_turn_angle(best_target, x, y)
            return ('move', target_pos, turn_angle)

        return self._safe_random_move()

    def _safe_random_move(self):
        """安全的随机移动"""
        # 产生一个随机的移动方向和角度
        angle = random.uniform(-self.max_angle, self.max_angle)
        # 短距离移动
        distance = random.uniform(0.1, 0.3)
        # 计算目标位置
        rad = math.radians(self.memory.agent_direction + angle)
        dx = distance * math.cos(rad)
        dy = distance * math.sin(rad)
        new_x = max(0, min(1, self.memory.agent_position[0] + dx))
        new_y = max(0, min(1, self.memory.agent_position[1] + dy))
        print("当前决策来源于随机移动")
        return ('move', distance, angle, 0)  # 俯仰角设为0

    def _calculate_angle_to_target(self, dx, dy):
        """计算到目标的相对角度"""
        # 计算目标方向向量
        target_dir = math.degrees(math.atan2(dy, dx))
        # 计算与当前方向的夹角
        angle_diff = (target_dir - self.memory.agent_direction) % 360
        if angle_diff > 180:
            angle_diff -= 360
        return angle_diff

    def _calculate_turn_angle(self, target_grid, current_x, current_y):
        """计算需要转向的角度"""
        # 计算目标方向
        dx = target_grid[0] - current_x
        dy = target_grid[1] - current_y
        target_angle = math.degrees(math.atan2(dy, dx))

        # 计算转向角度
        angle_diff = (target_angle - self.memory.agent_direction) % 360
        if angle_diff > 180:
            angle_diff -= 360

        # 限制最大转向角度
        return max(-self.max_angle, min(self.max_angle, angle_diff))

    def _grid_to_pos(self, grid):
        """将栅格坐标转换为屏幕位置"""
        return (grid[0] / 14.0, grid[1] / 14.0)

    def _build_state_vector(self, current_pos):
        """构建神经网络输入向量"""
        # 获取地图数据
        map_flat = self.memory.env_map.flatten()

        # 获取当前栅格坐标并归一化
        grid_x, grid_y = self.memory._pos_to_grid(current_pos)
        norm_pos = [grid_x / 14.0, grid_y / 14.0]

        # 当前归一化坐标
        norm_pos = [current_pos[0], current_pos[1]]

        # 获取当前方向 (归一化到0-1)
        direction_norm = self.memory.agent_direction / 360.0

        # 随机选择目标位置 (实际应用中由路径规划提供)
        target_x = random.uniform(0, 1)
        target_y = random.uniform(0, 1)

        # 合并所有特征
        state_vector = np.concatenate([
            map_flat,
            norm_pos,# 直接使用归一化坐标
            [target_x, target_y],
            [direction_norm]
        ])
        return torch.tensor(state_vector).float().unsqueeze(0)

    def _nn_decision(self, action_idx, params):
        """将神经网络输出转换为决策"""
        actions = ['move', 'turn', 'wait']
        base_action = actions[action_idx]

        if base_action == 'move':
            # 使用预测参数: 距离、角度、俯仰
            distance = max(0.1, min(1.0, params[0]))
            angle = params[1] * 180  # 转换为角度 (-180~180)
            pitch = max(-90, min(90, params[2] * 90))
            return (base_action, distance, angle, pitch)

        elif base_action == 'turn':
            # 使用预测的转向参数
            angle = params[0] * 360 - 180  # 转换为角度范围
            pitch = max(-90, min(90, params[1] * 90))
            return (base_action, angle, pitch)

        else:  # wait
            return (base_action, random.uniform(0.5, 2.0))  # 等待时间

    def record_decision(self, state, action, reward):
        """记录决策数据用于训练"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward
        })

        # 当积累足够数据时训练
        if len(self.buffer) >= 100:
            self.train_network()

    def train_network(self):
        """使用积累的数据训练网络"""
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        for _ in range(10):  # 10个epoch
            for data in self.buffer:
                state = data['state']
                reward = data['reward']

                # 前向传播
                action_probs, action_params = self.nn(state)

                # 计算损失 (奖励最大化)
                loss = -reward * torch.log(action_probs).mean()

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 清空缓冲区
        self.buffer = []
        print("神经网络训练完成，更新决策模型")

    def _heuristic_decision(self):
        """启发式决策：安全随机移动"""
        '''
                # 随机选择移动方向
                direction = random.choice(['forward', 'backward', 'left', 'right'])
                distance = random.uniform(0.1, 0.3)
                angle = random.uniform(-45, 45)
                return ('move', direction, distance, angle, 0)  # 俯仰角设为0
                '''
        # 获取当前状态
        current_pos = self.memory.agent_position
        grid_x, grid_y = self.memory._pos_to_grid(current_pos)
        map_data = self.memory.env_map

        # 决策树核心逻辑
        if self._should_explore_new_area(map_data, grid_x, grid_y):
            return self._explore_new_area(map_data, grid_x, grid_y)
        elif self._should_revisit_area(map_data, grid_x, grid_y):
            return self._revisit_area(map_data, grid_x, grid_y)
        else:
            return self._safe_random_move()

