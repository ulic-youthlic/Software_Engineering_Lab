import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import torch


class EnhancedLogger:
    def __init__(self, log_dir="logs", map_size=(15, 15)):
        """
        高级日志记录器，记录游戏循环的详细数据
        :param log_dir: 日志存储目录
        :param map_size: 地图尺寸 (rows, cols)
        """
        self.log_dir = log_dir
        self.map_size = map_size
        self.cycle_count = 0
        self.log_data = []
        os.makedirs(log_dir, exist_ok=True)

        # 创建自定义颜色映射 (红色表示障碍物，绿色表示可通行区域)
        self.cmap = LinearSegmentedColormap.from_list('custom', ['#ff0000', '#ffff00', '#00ff00'], N=256)

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"game_log_{timestamp}.json")
        self.image_dir = os.path.join(log_dir, "map_images")
        os.makedirs(self.image_dir, exist_ok=True)

        print(f"日志系统初始化: 日志文件 -> {self.log_file}, 地图图像 -> {self.image_dir}")

    def log_cycle(self, cycle_data):
        """
        记录单次循环数据并生成可视化地图
        :param cycle_data: 包含以下键的字典:
            - 'cycle_count': 循环计数
            - 'action': 执行的行为 (元组: (行为类型, 参数))
            - 'prev_map': 行为前的地图置信度矩阵 (15x15)
            - 'current_map': 行为后的地图置信度矩阵 (15x15)
            - 'character_pos': 角色位置 (归一化坐标, (x, y))
            - 'direction': 角色方向 (角度)
            - 'success': 行为是否成功 (bool)
            - 'reward': 获得的奖励值
        """
        self.cycle_count = cycle_data['cycle_count']

        # 确保所有地图数据都转换为正确的格式
        cycle_data = self._normalize_data(cycle_data)

        log_entry = {
            'cycle': self.cycle_count,
            'timestamp': datetime.now().isoformat(),
            'action': {
                'type': cycle_data['action'][0],
                'param': cycle_data['action'][1]
            },
            'character': {
                'x': cycle_data['character_pos'][0],
                'y': cycle_data['character_pos'][1],
                'direction': cycle_data['direction']
            },
            'success': cycle_data['success'],
            'reward': cycle_data['reward'],
            'map_image': f"map_{self.cycle_count:04d}.png"
        }

        self.log_data.append(log_entry)
        self._save_log_entry(log_entry)
        self._generate_map_visualization(cycle_data)

        print(f"日志记录: 周期 {self.cycle_count} | 行为: {cycle_data['action'][0]}({cycle_data['action'][1]}) "
              f"| 成功: {cycle_data['success']} | 奖励: {cycle_data['reward']:.2f}")

    def _normalize_data(self, cycle_data):
        """确保所有数据格式正确（在Logger内部处理）"""
        # 转换地图数据为numpy数组
        for key in ['prev_map', 'current_map']:
            if key in cycle_data:
                cycle_data[key] = self._convert_to_numpy(cycle_data[key])

        # 确保角色位置是元组格式
        if 'character_pos' in cycle_data:
            if isinstance(cycle_data['character_pos'], list):
                cycle_data['character_pos'] = tuple(cycle_data['character_pos'])
            elif torch.is_tensor(cycle_data['character_pos']):
                cycle_data['character_pos'] = tuple(cycle_data['character_pos'].tolist())

        # 确保方向是浮点数
        if 'direction' in cycle_data and isinstance(cycle_data['direction'], torch.Tensor):
            cycle_data['direction'] = cycle_data['direction'].item()

        return cycle_data

    def _convert_to_numpy(self, data):
        """将各种格式的地图数据转换为numpy数组"""
        if isinstance(data, np.ndarray):
            return data
        elif torch.is_tensor(data):
            return data.detach().cpu().numpy()
        elif hasattr(data, 'numpy'):
            return data.numpy()
        elif isinstance(data, list):
            return np.array(data)
        else:
            print(f"警告: 无法识别的数据类型 {type(data)}，使用零数组替代")
            return np.zeros(self.map_size)

    def _save_log_entry(self, entry):
        """保存日志条目到JSON文件"""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def _generate_map_visualization(self, cycle_data):
        """生成地图可视化图像，标注变化和角色位置"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle(f"Cycle {self.cycle_count}: {cycle_data['action'][0]}({cycle_data['action'][1]}) - "
                     f"Success: {cycle_data['success']}, Reward: {cycle_data['reward']:.2f}",
                     fontsize=14, fontweight='bold')

        # 绘制行为前地图
        self._plot_single_map(ax1, cycle_data['prev_map'], cycle_data['character_pos'],
                              cycle_data['direction'], "Before Action")

        # 绘制行为后地图
        self._plot_single_map(ax2, cycle_data['current_map'], cycle_data['character_pos'],
                              cycle_data['direction'], "After Action",
                              prev_map=cycle_data['prev_map'])

        # 保存图像
        image_path = os.path.join(self.image_dir, f"map_{self.cycle_count:04d}.png")
        plt.tight_layout()
        plt.savefig(image_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"地图图像已保存: {image_path}")

    def _plot_single_map(self, ax, map_data, char_pos, direction, title, prev_map=None):
        """绘制单个地图并标注角色位置和变化"""
        # 确保地图数据格式正确
        map_data = self._convert_to_numpy(map_data)
        if prev_map is not None:
            prev_map = self._convert_to_numpy(prev_map)

        # 确保角色位置格式正确
        if isinstance(char_pos, torch.Tensor):
            char_pos = char_pos.tolist()
        elif not isinstance(char_pos, (tuple, list)):
            char_pos = (0.5, 0.5)  # 默认位置

        # 确保方向是数值类型
        if isinstance(direction, torch.Tensor):
            direction = direction.item()

        # 转换角色位置到网格坐标
        grid_x = int(char_pos[0] * (self.map_size[1] - 1))
        grid_y = int(char_pos[1] * (self.map_size[0] - 1))

        # 显示地图置信度热力图
        im = ax.imshow(map_data, cmap=self.cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Confidence')

        # 添加网格线
        ax.set_xticks(np.arange(-0.5, self.map_size[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.map_size[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        # 标注置信度值
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                ax.text(j, i, f"{map_data[i, j]:.1f}",
                        ha="center", va="center", color="black", fontsize=8)

        # 标记角色位置和方向
        ax.plot(grid_x, grid_y, 'ro', markersize=10)  # 角色位置
        ax.text(grid_x + 0.5, grid_y - 0.5, f"({char_pos[0]:.2f},{char_pos[1]:.2f})\nDir: {direction}°",
                color='red', fontsize=9, ha='center')

        # 绘制方向箭头
        arrow_length = 1.0
        angle_rad = np.radians(direction)
        dx = arrow_length * np.cos(angle_rad)
        dy = arrow_length * np.sin(angle_rad)
        ax.arrow(grid_x, grid_y, dx, dy, head_width=0.5, head_length=0.7, fc='red', ec='red', linewidth=2)

        # 标记变化区域（如果有前一状态的地图）
        if prev_map is not None:
            # 计算变化区域（置信度变化超过0.1）
            changes = np.abs(map_data - prev_map) > 0.1
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    if changes[i, j]:
                        # 在变化的网格周围添加红色边框
                        rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2,
                                         edgecolor='purple', facecolor='none')
                        ax.add_patch(rect)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")

    def finalize_logs(self):
        """完成日志记录，保存总结信息"""
        summary = {
            "total_cycles": self.cycle_count,
            "start_time": self.log_data[0]['timestamp'] if self.log_data else "",
            "end_time": datetime.now().isoformat(),
            "average_reward": np.mean([entry['reward'] for entry in self.log_data]) if self.log_data else 0,
            "success_rate": np.mean([int(entry['success']) for entry in self.log_data]) if self.log_data else 0,
            "action_counts": self._count_actions()
        }

        with open(os.path.join(self.log_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"日志完成: 共记录 {self.cycle_count} 次循环")

        # 自动导出为Markdown格式
        self.export_to_markdown()

        return summary

    def _count_actions(self):
        """统计各类行为的执行次数"""
        action_count = {}
        for entry in self.log_data:
            action_type = entry['action']['type']
            action_count[action_type] = action_count.get(action_type, 0) + 1
        return action_count

    def export_to_markdown(self, output_file="game_report.md"):
        """将日志数据导出为Markdown格式报告"""
        markdown_content = "# 游戏智能体行为报告\n\n"
        markdown_content += f"## 实验概览\n"
        markdown_content += f"- **实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        markdown_content += f"- **总循环次数**: {self.cycle_count}\n"

        # 添加行为统计表格
        action_counts = self._count_actions()
        markdown_content += "\n## 行为统计\n"
        markdown_content += "| 行为类型 | 执行次数 | 执行比例 |\n"
        markdown_content += "|----------|----------|----------|\n"

        for action, count in action_counts.items():
            percentage = count / self.cycle_count * 100
            markdown_content += f"| {action} | {count} | {percentage:.1f}% |\n"

        # 添加地图演变过程
        markdown_content += "\n## 地图演变过程\n"
        for entry in self.log_data:
            cycle = entry['cycle']
            action = f"{entry['action']['type']}({entry['action']['param']})"
            success = "✓" if entry['success'] else "✗"
            reward = entry['reward']

            # 图片路径（相对于Markdown文件）
            img_path = f"map_images/map_{cycle:04d}.png"

            # 使用Markdown图片语法嵌入图片
            markdown_content += f"\n### 循环 {cycle}: {action} (成功: {success}, 奖励: {reward:.2f})\n"
            markdown_content += f"\n"
            markdown_content += f"**行为详情**:\n"
            markdown_content += f"- 行为类型: {entry['action']['type']}\n"
            markdown_content += f"- 行为参数: {entry['action']['param']}\n"
            markdown_content += f"- 是否成功: {'是' if entry['success'] else '否'}\n"
            markdown_content += f"- 获得奖励: {entry['reward']:.2f}\n"
            markdown_content += f"- 角色位置: ({entry['character']['x']:.2f}, {entry['character']['y']:.2f})\n"
            markdown_content += f"- 角色方向: {entry['character']['direction']}°\n"
            markdown_content += f"![map_{cycle:04d}.png]({img_path})"
            markdown_content += f"\n\n"

        # 添加报告生成时间
        markdown_content += f"\n**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # 保存Markdown文件
        full_path = os.path.join(self.log_dir, output_file)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"Markdown报告已保存至: {full_path}")
        print(f"图片目录: {os.path.abspath(self.image_dir)}")
        return full_path