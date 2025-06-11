# 依赖库：torch, numpy, neo4j
import torch.nn as nn
from neo4j import GraphDatabase
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

class Blank_LSTM_Memory_For_Test(nn.Module):
    def __init__(self):
        super().__init__()
        # 增强的特征编码层
        '''
        nn.Linear(485, 64),  # 输入维度调整,需要按照实际特征维度修改
        '''
        self.feature_encoder = nn.Sequential(
            nn.Linear(260, 64),# 输入维度调整为：敌人数量(1) + 物品数量(1) + 最近敌人坐标(2) + YOLO特征(256) = 260
            nn.ReLU()
        )
        # LSTM层（双层层结构）
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # 环境地图建模（15x15栅格）
        self.env_map = torch.zeros((15, 15), dtype=torch.float32)  # 初始全0矩阵
        self.map_update_layer = nn.Sequential(  # 地图更新网络
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 225),  # 15x15=225
            nn.Sigmoid()  # 转换为0-1的概率图
        )


        # 调试和统计相关
        debug_interval=20
        self.debug_interval = debug_interval
        self.update_count = 0
        self.total_accuracy = 0.0
        self.total_recall = 0.0
        self.start_time = time.time()

        # 根据提供的数据初始化真实地图
        self.ground_truth_map = self._create_ground_truth_map()

        # 模拟数据库连接
        #self.db = GraphDatabase.driver("bolt://localhost:7687")
        self.db_initialized = False
        self.hidden = None
        # 模拟模式开关（用于测试其他模块）
        self.simulation_mode = False
        self.test_hidden = torch.randn(2, 1, 128)

    def _create_ground_truth_map(self):
        """根据提供的迷宫布局创建真实地图"""
        # 14x14迷宫矩阵（1为墙体，0为通道）
        maze_layout = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
        ]

        # 创建15x15地图（扩展边界）
        gt_map = torch.zeros((15, 15), dtype=torch.float32)

        # 复制迷宫布局到地图中心
        for i in range(14):
            for j in range(14):
                gt_map[i + 0, j + 0] = maze_layout[i][j]

        # 设置边界为墙体
        gt_map[0, :] = 1.0  # 上边界
        gt_map[-1, :] = 1.0  # 下边界
        gt_map[:, 0] = 1.0  # 左边界
        gt_map[:, -1] = 1.0  # 右边界

        return gt_map

    def _encode_features(self, objects):
        """根据vision模块特征输出获取特征编码（CPU版本）"""
        # 敌人数量归一化
        num_enemies = len(objects.get('enemies', [])) / 10.0

        # 物品数量归一化（模拟值）
        num_items = len(objects.get('items', [])) / 5.0

        '''
        # 环境栅格展开（确保15x15=225维）
        grid_features = np.array(env_map).flatten()#一维数组
        if len(grid_features) != 225:
            grid_features = np.zeros(225)  # 容错处理
        '''

        # 最近敌人坐标
        if objects.get('enemies'):
            last_enemy = np.array(objects['enemies'][0])
        else:
            last_enemy = np.zeros(2)

        # 添加YOLO特征
        #yolo_features = objects.get('features', np.zeros(256))
        yolo_features = objects.get('features')
        if yolo_features is None or not hasattr(yolo_features, 'ndim'):
            yolo_features = np.zeros(256)  # 默认零向量
        if yolo_features.ndim == 0:  # 处理标量情况
            yolo_features = np.array([yolo_features])

        # 合并所有特征
        combined = np.concatenate([
            [num_enemies, num_items],
            #grid_features,     #现在不需要它了，这个特征在其它部分处理
            last_enemy,
            yolo_features
        ])

        # 添加维度检查
        print(f"特征维度检查: "
              f"num_enemies: {type(num_enemies)} "
              #f"grid_features: {grid_features.shape} "
              f"last_enemy: {last_enemy.shape} "
              f"yolo_features: {yolo_features.shape}")

        # 手动构建特征向量（229维）
        #manual_features = [num_enemies, num_items] + grid_features + list(last_enemy)
        '''
        # 添加YOLO深层特征
        yolo_features = objects['features'].tolist()  # 假设是numpy数组

        #这个是把yolo深层特征和手动构建的特征结合，以后再修改，现在先用手动构建的特征
        # 合并特征
        combined = manual_features + yolo_features
        return torch.FloatTensor(combined).cpu()
        '''

        print(f"[DEBUG] 手动特征统计: "
              f"敌人数={num_enemies:.2f} 物品数={num_items:.2f} "
              f"最近敌人坐标={last_enemy}") #网格非零数={np.sum(env_map)}


        # 转换为CPU tensor
        #return torch.FloatTensor(manual_features).cpu()
        return torch.FloatTensor(combined).cpu()

    def _calculate_metrics(self, predicted_map):
        """计算预测地图与真实地图的准确率和召回率"""
        # 转换为numpy数组
        pred_np = predicted_map.numpy()
        gt_np = self.ground_truth_map.numpy()

        # 计算混淆矩阵
        true_pos = np.logical_and(pred_np == 1, gt_np == 1).sum()
        true_neg = np.logical_and(pred_np == 0, gt_np == 0).sum()
        false_pos = np.logical_and(pred_np == 1, gt_np == 0).sum()
        false_neg = np.logical_and(pred_np == 0, gt_np == 1).sum()

        # 计算指标
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-8)
        recall = true_pos / (true_pos + false_neg + 1e-8)

        return accuracy, recall

    def _visualize_map(self, predicted_map, accuracy, recall):
        """可视化地图并保存为图片"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # 真实地图
        ax1.imshow(self.ground_truth_map, cmap='gray', vmin=0, vmax=1)
        ax1.set_title('Ground Truth Map')
        ax1.set_xticks([])
        ax1.set_yticks([])

        # 预测地图
        ax2.imshow(predicted_map, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f'Predicted Map (Acc: {accuracy:.2f}, Rec: {recall:.2f})')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # 保存图片
        plt.tight_layout()
        plt.savefig(f'map_debug_{self.update_count}.png')
        plt.close()

        print(f"地图已保存为 map_debug_{self.update_count}.png")

    def update(self, objects, env_map):
        """更新记忆并动态构建环境地图"""
        '''
        def update(objects, env_map) -> memory_data:
    ├─ 输入：
    │   ├─ objects（视觉检测结果）
    │   └─ env_map（15x15栅格）
    ├─ 特征整合（_encode_features）：
    │   ├─ 敌人数量归一化
    │   ├─ 物品数量归一化
    │   ├─ 栅格展开为225维
    │   └─ 拼接为229维特征向量
    ├─ LSTM处理（输入64→输出128）
    ├─ 长期记忆存储（_save_landmarks）
    └─ 返回记忆向量（numpy数组）
        :param objects:
        :param env_map:
        :return:
        '''
        """测试版记忆更新（全CPU操作）"""
        if self.simulation_mode:
            return self._simulate_memory()

         # 更新计数器
        self.update_count += 1

        # 将视觉输入的环境地图转换为张量
        visual_map = torch.tensor(env_map, dtype=torch.float32)

        # 特征编码
        with torch.no_grad():
            # 提取当前帧的特征
            raw_features = self._encode_features(objects)
            encoded = self.feature_encoder(raw_features.unsqueeze(0))  # 添加batch维度

            # LSTM处理获取短期记忆
            lstm_out, self.hidden = self.lstm(
                encoded.view(1, 1, -1),  # 调整形状为 (batch, seq, feature)
                self.hidden
            )

            #  更新环境地图（核心功能）
            #  将LSTM输出转换为地图更新量
            map_update = self.map_update_layer(lstm_out.squeeze(0))
            map_update = map_update.view(15, 15)

            # 动态更新地图：融合当前视觉信息和历史记忆
            self.env_map = 0.3 * self.env_map + 0.7 * visual_map

            # 应用LSTM生成的更新量（突出重要区域）
            self.env_map = torch.maximum(self.env_map, map_update)

            # 二值化处理（阈值0.5）
            predicted_map = (self.env_map > 0.5).float()

            # 打印地图更新信息
            obstacle_count = torch.sum(predicted_map > 0).item()
            print(f"[记忆模块] 地图更新: 新增障碍{obstacle_count}个")
#
        # 定期调试输出
        if self.update_count % self.debug_interval == 0:
            # 计算准确率和召回率
            accuracy, recall = self._calculate_metrics(predicted_map)

            # 更新统计
            self.total_accuracy += accuracy
            self.total_recall += recall

            # 计算平均指标
            avg_accuracy = self.total_accuracy / (self.update_count // self.debug_interval)
            avg_recall = self.total_recall / (self.update_count // self.debug_interval)

            # 计算运行时间
            elapsed_time = time.time() - self.start_time
            fps = self.update_count / elapsed_time

            # 打印统计信息
            print("\n===== 地图统计 =====")
            print(f"更新次数: {self.update_count}")
            print(f"当前准确率: {accuracy:.4f}")
            print(f"当前召回率: {recall:.4f}")
            print(f"平均准确率: {avg_accuracy:.4f}")
            print(f"平均召回率: {avg_recall:.4f}")
            print(f"运行时间: {elapsed_time:.2f}秒")
            print(f"处理速度: {fps:.2f} FPS")
            print("==================\n")

            # 可视化地图
            self._visualize_map(predicted_map, accuracy, recall)

        # 返回包含地图信息的记忆向量
        memory_vector = torch.cat([
            lstm_out.squeeze(),
            predicted_map.flatten()  # 将地图信息加入记忆向量
        ])

        return memory_vector.numpy()
        '''
        # 转换为numpy输出
        return lstm_out.squeeze().numpy()
        '''

    def _save_landmarks(self, objects):
        """存储关键位置到知识图谱"""
        if not self.db_initialized:
            print("[模拟] 初始化知识图谱连接")
            self.db_initialized = True

        print(f"[模拟] 存储地标数据: {objects.get('enemies', [])}")

    def _simulate_memory(self):
        """为其他模块提供测试用的模拟数据"""
        """生成测试用记忆数据"""
        self.test_hidden = 0.9 * self.test_hidden + 0.1 * torch.randn(2, 1, 128)
        return self.test_hidden.mean().item()

