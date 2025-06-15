#decision_tree_and_nn here
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import heapq
from sklearn.tree import DecisionTreeRegressor
import random

class EnhancedDecisionNetwork(nn.Module):
    def __init__(self,input_dim=484):
        super().__init__()
        # 输入: 地图(225) + 当前位置(2) + 视觉特征(256) + 方向(1) = 484维
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
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
        self.uncertainty_threshold = 0.5  # 置信度低于此值视为不确定区域
        self.exploration_weight = 0.7  # 探索权重（vs 利用）

    def find_nearest_uncertain(self, map_data, current_pos):
        """决策树寻找最近的低置信度位置"""
        '''
        # 确保map_data是NumPy数组
        if not isinstance(map_data, np.ndarray):
            map_data = np.array(map_data)
            print("find_nearest_uncertain:map_data is not a np.array now it's transferred into it")
        '''
        # 确保map_data是二维数组
        # 确保map_data是NumPy数组
        if not isinstance(map_data, np.ndarray):
            print("警告：map_data不是NumPy数组，重置为零矩阵")
            map_data = np.zeros(self.grid_size)

        if map_data.ndim == 0:  # 处理0维数组情况
            print("警告：map_data是0维数组，重置为15x15零矩阵")
            map_data = np.zeros(self.grid_size)
        elif map_data.ndim == 1:  # 处理1维数组情况
            print("警告：map_data是1维数组，尝试重塑为二维")
            if map_data.size == 225:  # 15x15=225
                map_data = map_data.reshape(self.grid_size)
            else:
                print(f"无法重塑大小{map_data.size}的数组，重置为15x15零矩阵")
                map_data = np.zeros(self.grid_size)

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
        self.decision_net = EnhancedDecisionNetwork(input_dim=484)

    def prepare_input(self, memory, vision_features, controller):
        """准备神经网络输入数据"""
        """准备484维输入向量：225(地图)+2(位置)+256(视觉)+1(方向)"""
        try:
            # 获取当前地图（从memory）
            map_data = memory.env_map.detach().cpu().numpy().flatten()
            # 检查所有输入维度
            if map_data.size != 225:
                print(f"prepare_input warning!:map_data.size={map_data.size}!=225,transferred to np.zeros(225)")
                map_data = map_data[:225]  # 截取前225维
                if map_data.size < 225:
                    map_data = np.pad(map_data, (0, 225 - map_data.size))

            if vision_features.size != 256:
                print(f"prepare_input warning!:vision_features.size={vision_features.size}!=225,transferred to np.zeros(225)")
                vision_features = vision_features[:256]
                if vision_features.size < 256:
                    print(f"prepare_input warning!: vision_features.size={vision_features.size}<256, padding with zeros")
                    padded = np.zeros(256)
                    padded[:vision_features.size] = vision_features
                    vision_features = padded
            # 获取当前位置和方向
            current_pos = controller.get_position()
            direction = controller.get_direction()
            pos_features = [current_pos[0], current_pos[1]]

            # 合并所有输入特征
            input_features = np.concatenate([
                map_data,
                pos_features,
                vision_features,
                [direction]
            ])

            # 检查最终维度
            if input_features.size != 484:
                # 自动修正维度
                if input_features.size < 484:
                    input_features = np.pad(input_features, (0, 484 - input_features.size))
                else:
                    input_features = input_features[:484]

            # 合并特征前打印维度
            print(f"输入维度: 地图={map_data.size}, 位置=2, 视觉={vision_features.size}, 方向=1")
            return torch.FloatTensor(input_features).unsqueeze(0)
        except Exception as e:
            print(f"准备输入失败: {str(e)}")
            # 返回安全的默认输入
            return torch.zeros(1, 484)

    def decide_action(self, memory_data, vision_features, controller, memory):
        """混合决策过程"""
        '''
        # 确保 memory_data 是字典
        if not isinstance(memory_data, dict):
            print(f"[决策系统] 警告: memory_data 是 {type(memory_data)} 类型，不是字典")
            map_grid = np.zeros((15, 15))
        else:
            # 正确获取地图数据
            map_grid = memory_data.get('map', np.zeros(225)).reshape(15, 15)


        # 正确解析记忆数据，获取地图数据
        map_grid = memory_data['map'].reshape(15, 15)  # 重塑为网格

        # 使用连续置信度地图寻找目标
        uncertain_target = self.exploration_planner.find_nearest_uncertain(
            map_grid, controller.get_position()
        )
        '''
        try:
            # 确保特征维度正确
            if vision_features is None or vision_features.ndim == 0 or len(vision_features) != 256:
                print(f"修正视觉特征维度: {vision_features.shape if vision_features is not None else 'None'} -> 256")
                vision_features = np.zeros(256)

            '''
            # 确保地图数据正确
            if 'map' not in memory_data or memory_data['map'].size != 225:
                print("重置地图数据")
                memory_data['map'] = np.zeros(225)
            '''
            # 确保地图数据是二维数组
            if 'map' not in memory_data or memory_data['map'].ndim != 2:
                print("重置地图数据为15x15零矩阵")
                memory_data['map'] = np.zeros((15, 15))

            # 获取地图数据
            map_data = memory_data['map']
            print(f"地图维度: {map_data.ndim}维, 形状: {map_data.shape if hasattr(map_data, 'shape') else '无形状'}")

            # 计算输入特征前检查维度
            input_vector = self.prepare_input(memory, vision_features, controller)
            if input_vector.nelement() != 484:  # 检查元素总数
                print(f"输入维度错误: 实际元素数={input_vector.nelement()} (期望484)")
                print(f"输入维度错误: {input_vector.size} != 484, 使用随机探索")
                return ('explore', 0)

            # 处理可能的输入类型
            if isinstance(memory_data, dict):
                # 从字典获取地图数据并重塑为15x15
                map_data = memory_data.get('map', np.zeros(225)).reshape(15, 15)
            elif hasattr(memory, 'env_map'):
                # 直接从memory对象获取地图
                map_data = memory.env_map.detach().cpu().numpy().reshape(15, 15)
            else:
                # 未知类型：使用零数组
                print(f"[决策系统] 错误: 无法解析 memory_data 类型 {type(memory_data)}")
                # 默认地图
                map_data = np.zeros((15, 15))

            # 确保地图数据正确重塑
            try:
                #map_grid = map_data.reshape(15, 15)
                map_grid = memory_data['map']
            except Exception as e:
                print(f"地图重塑失败: {str(e)}，使用默认地图")
                map_grid = np.zeros((15, 15))

            # 调试输出地图信息
            print(f"地图形状: {map_data.shape}, 数据类型: {type(map_data)}")

            # 后续决策逻辑...
            current_pos = controller.get_position()
            uncertain_target = self.exploration_planner.find_nearest_uncertain(
                map_grid, current_pos
            )

            # 打印调试信息
            print(f"地图置信度统计: min={map_grid.min():.2f} max={map_grid.max():.2f} mean={map_grid.mean():.2f}")
            if uncertain_target:
                print(
                    f"发现探索目标: {uncertain_target} (置信度={map_grid[uncertain_target[1], uncertain_target[0]]:.2f})")

            # 1. 准备神经网络输入
            net_input = self.prepare_input(memory, vision_features, controller)
            print(f"神经网络输入形状: {net_input.shape}")
            # 2. 神经网络推理
            with torch.no_grad():
                action_probs, action_params, state_value = self.decision_net(net_input)

            # 3. 决策树寻找探索目标
            current_pos = controller.get_position()
            '''
            uncertain_target = self.exploration_planner.find_nearest_uncertain(
                memory_data, current_pos
            )
            '''
            if 'map' in memory_data:
                map_grid = memory_data['map']
                if map_grid.size == 225:  # 如果是展平的地图，重塑为15x15
                    map_grid = map_grid.reshape(15, 15)
                uncertain_target = self.exploration_planner.find_nearest_uncertain(map_grid, current_pos)
            else:
                uncertain_target = None

            # 在 HybridDecisionSystem.decide_action() 中添加
            print(f"探索目标: {'存在' if uncertain_target else '不存在'}")
            print(f"神经网络建议: action_probs={action_probs}, action_params={action_params}")

            # 4. 混合决策
            if uncertain_target:
                print("有搜索目标，执行uncertain_target策略")
                # 转换目标为归一化坐标
                target_pos = (
                    uncertain_target[0] / (self.exploration_planner.grid_size[0] - 1),
                    uncertain_target[1] / (self.exploration_planner.grid_size[1] - 1)
                )

                # 计算当前位置到目标的距离和方向
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
                distance = (dx ** 2 + dy ** 2) ** 0.5
                target_direction = np.degrees(np.arctan2(dy, dx)) % 360

                # 计算神经网络建议的移动方向
                net_direction = action_params[0, 0].item() * 360  # 转换为角度

                # 计算实际方向：结合神经网络建议和决策树目标
                actual_direction = self._blend_directions(
                    current_pos, target_pos, net_direction
                )

                current_direction = controller.get_direction()  # 这是关键修改

                # 计算当前方向与目标方向的夹角
                angle_diff = abs((current_direction - target_direction + 180) % 360 - 180)

                # 决策：当方向偏差较大时转向，较小时移动
                if angle_diff > 15:  # 角度偏差大于15度
                    # 计算需要转向的角度（最小转向）
                    turn_angle = (target_direction - current_direction + 180) % 360 - 180
                    if abs(turn_angle) > 180:
                        turn_angle = 360 - abs(turn_angle) * np.sign(turn_angle)

                    print(f"方向偏差较大({angle_diff:.1f}度)，转向{abs(turn_angle):.1f}度")
                    return ('turn', turn_angle)
                else:
                    # 根据距离决定移动距离（最大不超过1.0）
                    move_distance = min(1.0, distance * 2)
                    print(f"方向已对准({angle_diff:.1f}度)，移动{move_distance:.2f}单位")
                    return ('move', move_distance)
            else:
                print("无搜索目标，执行神经网络建议")
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
        except Exception as e:
            print(f"决策系统详细错误: {str(e)}\n{traceback.format_exc()}")
            # 返回默认移动动作而非explore
            return ('move', 1.0)

    def _blend_directions(self, current_pos, target_pos, net_direction):
        """混合神经网络建议方向和决策树目标方向"""
        # 确保位置是有效的元组
        if not isinstance(current_pos, tuple) or len(current_pos) < 2:
            print(f"无效当前位置: {current_pos}, 使用默认位置(0.5,0.5)")
            current_pos = (0.5, 0.5)

        if not isinstance(target_pos, tuple) or len(target_pos) < 2:
            print(f"无效目标位置: {target_pos}, 使用默认位置(0.6,0.6)")
            target_pos = (0.6, 0.6)

        # 计算到目标的方向
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        # 避免除以零错误
        if abs(dx) < 0.0001 and abs(dy) < 0.0001:
            return net_direction

        target_direction = np.degrees(np.arctan2(dy, dx)) % 360

        # 混合两个方向（基于神经网络影响权重）
        blended_direction = (self.nn_influence * net_direction +
                             (1 - self.nn_influence) * target_direction)

        return blended_direction % 360

    def record_experience(self, state, action, reward, next_state, done):
        """记录经验到回放缓冲区"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def calculate_reward(self, map_data, prev_map, action, success,current_pos=None):
        """计算奖励函数"""
        reward = 0.0

        # 使用当前的位置而不是action[1]
        if current_pos is not None:
            x, y = self._pos_to_grid(current_pos)
            # 检查是否移动到低置信度区域
            if map_data[y, x] < 0.7:
                reward += 4.0

        # 处理探索动作
        if action[0] == 'explore':
            print("WARNING! current action is 'explore' ")
            # 探索未知区域奖励
            if np.mean(map_data) < 0.7:
                reward += 0.7
            # 探索动作惩罚
            else:
                reward -= 0.5


        # 1. 成功探索奖励
        if success and action[0] == 'move':
            #成功移动基础奖励
            reward+=1.0
            # 计算信息增益（地图变化）
            info_gain = np.sum(np.abs(map_data - prev_map))
            reward += 0.1 * info_gain

            # 探索未知区域奖励
            if np.mean(map_data) < 0.7:  # 低平均置信度
                reward += 4.0

        # 2. 验证不确定区域奖励
        if action[0] == 'move':
            # 检查是否移动到低置信度区域
            #x, y = self._pos_to_grid(action[1])    因为action改了，action[1]现在是移动距离
            x, y = self._pos_to_grid(current_pos)
            if map_data[y, x] < 0.3:
                reward += 8.0           #鼓励探索

        # 3. 碰撞惩罚
        if not success and action[0] == 'move':
            reward -= 0.5

        # 4. 探索效率奖励（减少在已知区域徘徊）
        if action[0] == 'move' and np.mean(map_data) > 0.7:
            reward -= 0.2               #降低试错成本

        # 5. 等待惩罚
        if action[0]=='wait':
            reward-=4.0                 #不鼓励等待

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