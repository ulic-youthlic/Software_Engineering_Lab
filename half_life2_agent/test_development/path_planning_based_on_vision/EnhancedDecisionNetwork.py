import torch
import torch.nn as nn

class EnhancedDecisionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: 地图(225) + 当前位置(2) + 目标位置(2) + 方向(1) = 230维
        self.encoder = nn.Sequential(
            nn.Linear(230, 128),
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

    def forward(self, x):
        encoded = self.encoder(x)
        action_probs = self.action_head(encoded)
        action_params = self.param_head(encoded)
        return action_probs, action_params