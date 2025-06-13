#decision_tree_and_nn here
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import heapq
from sklearn.tree import DecisionTreeRegressor
import random

class EnhancedDecisionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: 地图(225) + 当前位置(2) + 视觉特征(256) + 方向(1) = 484维
        self.encoder = nn.Sequential(
            nn.Linear(484, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 行为预测分支 (分类)
        self.action_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3种行为: 移动、转向、等待
            nn.Softmax(dim=1)
        )

        # 参数预测分支 (回归)
        self.param_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3个参数: 移动距离、转向角度、俯仰角
        )

        # 价值评估分支
        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 状态价值
        )

    def forward(self, x):
        encoded = self.encoder(x)
        action_probs = self.action_head(encoded)
        action_params = self.param_head(encoded)
        state_value = self.value_head(encoded)
        return action_probs, action_params, state_value


class ExplorationPlanner:
    def __init__(self, grid_size=(15, 15)):
        self.grid_size = grid_size
        self.uncertainty_threshold = 0.3  # 置信度低于此值视为不确定区域
        self.exploration_weight = 0.7  # 探索权重（vs 利用）

    def find_nearest_uncertain(self, map_data, current_pos):
        """决策树寻找最近的低置信度位置"""
        # 转换为栅格坐标
        grid_x = int(current_pos[0] * (self.grid_size[0] - 1))
        grid_y = int(current_pos[1] * (self.grid_size[1] - 1))

        uncertain_points = []

        # 扫描整个地图寻找低置信度区域
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                confidence = map_data[y, x]
                # 低置信度且不是当前位置
                if confidence < self.uncertainty_threshold and not (x == grid_x and y == grid_y):
                    # 计算曼哈顿距离
                    distance = abs(x - grid_x) + abs(y - grid_y)
                    # 使用最小堆维护最近的点
                    heapq.heappush(uncertain_points, (distance, (x, y)))

        # 返回最近的低置信度点
        if uncertain_points:
            _, target = heapq.heappop(uncertain_points)
            return target
        return None


class HybridDecisionSystem:
    def __init__(self):
        # 初始化决策网络
        self.decision_net = EnhancedDecisionNetwork()
        self.optimizer = optim.Adam(self.decision_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        # 初始化探索规划器
        self.exploration_planner = ExplorationPlanner()

        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=10000)

        # 神经网络影响权重
        self.nn_influence = 0.5  # 初始权重

        # 训练状态
        self.train_mode = True

    def prepare_input(self, memory, vision_features, controller):
        """准备神经网络输入数据"""
        # 获取当前地图（从memory）
        map_data = memory.env_map.detach().cpu().numpy().flatten()

        # 获取当前位置和方向
        current_pos = controller.get_position()
        direction = controller.get_direction()

        # 合并所有输入特征
        input_features = np.concatenate([
            map_data,
            [current_pos[0], current_pos[1]],
            vision_features,
            [direction]
        ])

        return torch.FloatTensor(input_features).unsqueeze(0)

    def decide_action(self, map_data, vision_features, controller, memory):
        """混合决策过程"""
        # 1. 准备神经网络输入
        net_input = self.prepare_input(memory, vision_features, controller)

        # 2. 神经网络推理
        with torch.no_grad():
            action_probs, action_params, state_value = self.decision_net(net_input)

        # 3. 决策树寻找探索目标
        current_pos = controller.get_position()
        uncertain_target = self.exploration_planner.find_nearest_uncertain(
            map_data, current_pos
        )

        # 4. 混合决策
        if uncertain_target:
            # 转换目标为归一化坐标
            target_pos = (
                uncertain_target[0] / (self.exploration_planner.grid_size[0] - 1),
                uncertain_target[1] / (self.exploration_planner.grid_size[1] - 1)
            )

            # 计算神经网络建议的移动方向
            net_direction = action_params[0, 0].item() * 360  # 转换为角度

            # 计算实际方向：结合神经网络建议和决策树目标
            actual_direction = self._blend_directions(
                current_pos, target_pos, net_direction
            )

            # 执行转向
            return ('turn', actual_direction)

        # 没有探索目标时执行神经网络建议
        action_idx = torch.argmax(action_probs).item()
        action_type = ['move', 'turn', 'wait'][action_idx]

        if action_type == 'move':
            distance = action_params[0, 0].item()
            return (action_type, distance)
        elif action_type == 'turn':
            angle = action_params[0, 1].item() * 360  # 转换为角度
            return (action_type, angle)
        else:
            return (action_type, 0)  # 等待

    def _blend_directions(self, current_pos, target_pos, net_direction):
        """混合神经网络建议方向和决策树目标方向"""
        # 计算到目标的方向
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        target_direction = np.degrees(np.arctan2(dy, dx)) % 360

        # 混合两个方向（基于神经网络影响权重）
        blended_direction = (self.nn_influence * net_direction +
                             (1 - self.nn_influence) * target_direction)

        return blended_direction % 360

    def record_experience(self, state, action, reward, next_state, done):
        """记录经验到回放缓冲区"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def calculate_reward(self, map_data, prev_map, action, success):
        """计算奖励函数"""
        reward = 0.0

        # 1. 成功探索奖励
        if success and action[0] == 'move':
            # 计算信息增益（地图变化）
            info_gain = np.sum(np.abs(map_data - prev_map))
            reward += 0.1 * info_gain

            # 探索未知区域奖励
            if np.mean(map_data) < 0.3:  # 低平均置信度
                reward += 0.5

        # 2. 验证不确定区域奖励
        if action[0] == 'move':
            # 检查是否移动到低置信度区域
            x, y = self._pos_to_grid(action[1])
            if map_data[y, x] < 0.3:
                reward += 1.0

        # 3. 碰撞惩罚
        if not success and action[0] == 'move':
            reward -= 0.5

        # 4. 探索效率奖励（减少在已知区域徘徊）
        if action[0] == 'move' and np.mean(map_data) > 0.7:
            reward -= 0.2

        return reward

    def _pos_to_grid(self, pos):
        """将归一化位置转换为网格坐标"""
        x = int(pos[0] * (self.exploration_planner.grid_size[0] - 1))
        y = int(pos[1] * (self.exploration_planner.grid_size[1] - 1))
        return x, y

    def update_network(self, batch_size=32, gamma=0.99):
        """使用经验回放更新网络"""
        if len(self.replay_buffer) < batch_size:
            return 0.0  # 经验不足

        # 随机采样一批经验
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # 计算目标Q值
        with torch.no_grad():
            _, _, next_values = self.decision_net(next_states)
            target_values = rewards + gamma * next_values * (1 - dones)

        # 计算当前Q值
        _, _, current_values = self.decision_net(states)

        # 计算价值损失
        value_loss = self.loss_fn(current_values, target_values.unsqueeze(1))

        # 策略梯度更新
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        return value_loss.item()

    def adjust_influence_weight(self, recent_rewards):
        """根据近期奖励调整神经网络影响力权重"""
        avg_reward = np.mean(recent_rewards)

        if avg_reward > 0.5:  # 表现良好
            self.nn_influence = min(0.8, self.nn_influence + 0.05)
        elif avg_reward < 0:  # 表现不佳
            self.nn_influence = max(0.2, self.nn_influence - 0.05)