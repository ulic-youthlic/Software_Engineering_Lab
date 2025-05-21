# 依赖库：torch, numpy, neo4j
import torch.nn as nn
from neo4j import GraphDatabase
import torch

class LSTM_Memory(nn.Module):
    def __init__(self):
        super().__init__()
        # 增强的特征编码层
        self.feature_encoder = nn.Sequential(
            nn.Linear(229, 64),  # 输入维度调整
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=64,  # 视觉特征维度
            hidden_size=128,
            num_layers=2  # 网页8的双层结构[8](@ref)
        )
        self.db = GraphDatabase.driver("bolt://localhost:7687")
        self.hidden = None  # 维护隐藏状态

        # 模拟模式开关（用于测试其他模块）
        self.simulation_mode = False
        self.test_hidden = torch.randn(2, 1, 128)

    def _encode_features(self, objects, env_map):
        """整合多源数据为特征向量"""
        # 处理目标检测结果
        enemy_coords = objects.get('enemies', [])
        num_enemies = len(enemy_coords)

        # 处理环境地图特征
        grid_features = env_map.flatten()

        # 构建特征向量
        features = [
            num_enemies / 10,  # 归一化
            len(objects.get('items', [])) / 5,
            *grid_features
        ]

        # 添加最近敌人的坐标
        if enemy_coords:
            features.extend(enemy_coords[0])
        else:
            features.extend([0.0, 0.0])

        return torch.FloatTensor(features)


    def update(self, objects, env_map):
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
        if self.simulation_mode:
            return self._simulate_memory()

        # 1. 特征编码
        raw_features = self._encode_features(objects, env_map)
        encoded = self.feature_encoder(raw_features.unsqueeze(0))

        # 2. LSTM处理
        with torch.no_grad():
            lstm_out, self.hidden = self.lstm(
                encoded.unsqueeze(0),  # 添加batch维度
                self.hidden
            )

        # 3. 长期记忆存储
        self._save_landmarks(objects)

        return lstm_out.squeeze().numpy()

    def _save_landmarks(self, objects):
        """存储关键位置到知识图谱"""
        with self.db.session() as session:
            for enemy in objects.get('enemies', []):
                session.run(
                    "MERGE (a:Landmark {pos: $pos}) "
                        "SET a.last_seen = timestamp()",
                    pos=enemy
                )

    def _simulate_memory(self):
        """为其他模块提供测试用的模拟数据"""
        self.test_hidden = 0.9 * self.test_hidden + 0.1 * torch.randn(2, 1, 128)
        return self.test_hidden.mean().item()

