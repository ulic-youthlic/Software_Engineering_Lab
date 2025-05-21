# 依赖库：torch, numpy, neo4j
import torch.nn as nn
from neo4j import GraphDatabase
import torch
import numpy as np

class Blank_LSTM_Memory_For_Test(nn.Module):
    def __init__(self):
        super().__init__()
        # 增强的特征编码层
        self.feature_encoder = nn.Sequential(
            nn.Linear(229, 64),  # 输入维度调整
            nn.ReLU()
        )
        # LSTM层（双层层结构）
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        # 模拟数据库连接
        #self.db = GraphDatabase.driver("bolt://localhost:7687")
        self.db_initialized = False
        self.hidden = None

        # 模拟模式开关（用于测试其他模块）
        self.simulation_mode = False
        self.test_hidden = torch.randn(2, 1, 128)

    def _encode_features(self, objects, env_map):
        """模拟特征编码（CPU版本）"""
        # 敌人数量归一化
        num_enemies = len(objects.get('enemies', [])) / 10.0

        # 物品数量归一化（模拟值）
        num_items = len(objects.get('items', [])) / 5.0

        # 环境栅格展开（确保15x15=225维）
        grid_features = np.array(env_map).flatten().tolist()
        if len(grid_features) != 225:
            grid_features = [0.0] * 225  # 容错处理

        # 最近敌人坐标（模拟数据）
        if objects['enemies']:
            last_enemy = objects['enemies'][0]
        else:
            last_enemy = (0.0, 0.0)

        # 构建特征向量（229维）
        features = [num_enemies, num_items] + grid_features + list(last_enemy)

        # 转换为CPU tensor
        return torch.FloatTensor(features).cpu()


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
        """测试版记忆更新（全CPU操作）"""
        if self.simulation_mode:
            return self._simulate_memory()

        # 特征编码
        with torch.no_grad():
            raw_features = self._encode_features(objects, env_map)
            encoded = self.feature_encoder(raw_features.unsqueeze(0))  # 添加batch维度

            # LSTM处理
            lstm_out, self.hidden = self.lstm(
                encoded.view(1, 1, -1),  # 调整形状为 (batch, seq, feature)
                self.hidden
            )

        # 转换为numpy输出
        return lstm_out.squeeze().numpy()

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

