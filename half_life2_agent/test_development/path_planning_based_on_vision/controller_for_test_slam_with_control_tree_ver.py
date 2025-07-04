import pydirectinput
import pyautogui
import time
import random
import numpy as np
from mpmath.libmp import to_float
from numba.np.random.old_distributions import random_f
import math

class EnhancedInputSimulator:
    def __init__(self):
        # 添加方向属性（0-360度，0=正右方）
        self.direction = 0  # 初始方向设为0度（正右方）
        self.pitch = 0  # 俯仰角（-90到90度）
        self.pos_history = []
        self.screen_width, self.screen_height = pyautogui.size()
        self.click_delay = 0.3
        self.bezier_points = 20
        self.jitter_range = 20  # 随机抖动范围

        # 鼠标灵敏度设置
        self.horizontal_sensitivity = 1.0  # 水平转向灵敏度
        self.vertical_sensitivity = 0.5  # 垂直转向灵敏度
        self.smoothness = 5  # 转动平滑度（步数）,太高了转的慢


        # 移动配置
        self.move_config = {
            'min_duration': 0.2,
            'max_duration': 0.5,
            'human_like': True
        }

        self.move_distance = 0.3  # 默认移动距离

        # 移动按键及时间配置
        self.move_keys = {
            'forward': 'w',
            'backward': 's',
            'left': 'a',
            'right': 'd'
        }
        self.move_duration = 0.4  # 移动持续时间
        self.turn_duration = 0.1  # 转向持续时间

        # 使用栅格坐标系统一位置表示
        self.agent_position = [0.5,0.5]  # 初始位置设为地图中心(实际上当然不是)，使用归一化坐标
        self.agent_direction = 0
        # 角色位置属性（归一化坐标）                         还是用栅格坐标吧
        #self.agent_position = [0.5, 0.5]  # 角色初始位置

    def get_position(self):
        """获取角色位置（不是鼠标位置,在地图里的位置）"""
            #return tuple(self.agent_position)#原本获取的
        return self.agent_position#栅格坐标
    def precise_turn(self, angle_degrees, pitch_degrees=0):
        """精确转向指定角度和俯仰角（使用鼠标移动）"""
        # 计算基于游戏引擎的视角移动量（经验值）
        MOUSE_SCALE_FACTOR = 10  # 需要根据游戏调整

        # 计算水平移动（考虑方向）
        horizontal_move = angle_degrees * MOUSE_SCALE_FACTOR

        # 计算垂直移动（考虑俯仰）
        vertical_move = pitch_degrees * MOUSE_SCALE_FACTOR

        '''临时措施'''
        # 强制垂直移动为0，保持俯仰角不变
        vertical_move = 0
        '''临时措施'''

        # 实际移动鼠标（使用pyautogui确保精确移动）
        pyautogui.move(horizontal_move, vertical_move, duration=0.2)

        # 更新内部方向状态
        self.direction = (self.direction + angle_degrees) % 360
        self.pitch = max(-90, min(90, self.pitch + pitch_degrees))

        '''临时措施'''
        self.pitch=0
        '''临时措施'''

        print(f"转向完成: 角度={self.direction}°, 俯仰={self.pitch}°")
        '''
        """精确转向指定角度和俯仰角（使用鼠标移动）"""
        print(f"\n执行动作：转向{angle_degrees}度，俯仰{pitch_degrees}度\n")

        # 计算水平移动距离（像素）
        horizontal_move = angle_degrees * self.horizontal_sensitivity
        # 计算垂直移动距离（像素）
        vertical_move = pitch_degrees * self.vertical_sensitivity

        # 更新内部方向
        self.direction = (self.direction + angle_degrees) % 360
        self.pitch = max(-90, min(90, self.pitch + pitch_degrees))

        # 分步平滑移动鼠标
        steps = max(1, int(self.smoothness * abs(angle_degrees) / 45))
        step_x = horizontal_move / steps
        step_y = vertical_move / steps

        print(
            f"转向: 角度={angle_degrees}°, 俯仰={pitch_degrees}°, 步数={steps}, 每步移动: X={step_x:.2f}, Y={step_y:.2f}")

        for _ in range(steps):
            pydirectinput.moveRel(int(step_x), int(step_y), relative=True)
            time.sleep(0.01)  # 短延迟确保平滑
        print(f"\n执行动作：转向{angle_degrees}度，俯仰{pitch_degrees}度\n")
        '''
        '''
        """精确转向指定角度"""
        # 计算需要按下的时间（假设180度需要0.5秒）
        turn_duration = abs(angle_degrees) / 360 * 1.0

        if angle_degrees > 0:
            pydirectinput.keyDown(self.move_keys['right'])
            time.sleep(turn_duration)
            pydirectinput.keyUp(self.move_keys['right'])
        else:
            pydirectinput.keyDown(self.move_keys['left'])
            time.sleep(turn_duration)
            pydirectinput.keyUp(self.move_keys['left'])

        # 更新内部方向
        self.direction = (self.direction + angle_degrees) % 360
        print(f"\n执行动作：转向\n")
        '''

    def move_direction(self, direction,distance=None):
        """按指定方向移动，添加距离控制，更新位置"""
        distance = distance or self.move_distance

        if direction in self.move_keys:
            # 计算位移向量
            rad = math.radians(self.direction)
            dx = distance * math.cos(rad)
            dy = distance * math.sin(rad)
            '''
            # 更新归一化坐标
            rad = math.radians(self.direction)
            dx = distance * math.cos(rad) / self.screen_width * 2.0  # 转换为归一化位移
            dy = distance * math.sin(rad) / self.screen_height * 2.0
            '''
            # 更新角色位置
            self.agent_position[0] = max(0, min(1, self.agent_position[0] + dx))
            self.agent_position[1] = max(0, min(1, self.agent_position[1] + dy))

            # 执行按键
            key = self.move_keys[direction]
            # 根据距离计算按键持续时间
            duration = distance * self.move_duration
            pydirectinput.keyDown(key)
            time.sleep(duration)
            pydirectinput.keyUp(key)
            time.sleep(0.1)  # 按键释放后短暂延迟

        print(f"\n执行动作：向{direction}移动{distance:.2f}距离\n")

    def turn(self, angle, pitch=0):
        """转向指定角度和俯仰角并更新方向"""
        print(f"\n执行动作：转向\n")
        self.precise_turn(angle, pitch)

        '''
        """转向指定角度并更新方向"""
        # 更新内部方向记录
        self.direction = (self.direction + angle) % 360

        # 实际转向操作
        if angle > 0:
            self.move_direction('right')
        elif angle < 0:
            self.move_direction('left')
        '''

    def get_direction(self):
        """获取当前方向（0-360度）"""
        #return self.direction,self.pitch
        return self.direction   #返回元组会出问题，主循环也改过了反正不用pitch
    def explore_behavior(self):                 #用不到了
        """增强的探索行为：包含精确转向"""
        # 随机选择动作类型：移动或转向
        action_type = random.choice(['turn','move','move','turn','move'])

        if action_type == 'move':
            # 随机选择移动方向
            direction = random.choice(['forward', 'backward', 'left', 'right'])
            self.move_direction(direction)
        if action_type == 'turn':
            print("\n    这里没问题     \n")
            # 随机转向角度（-180到180度）
            #turn_angle = random.uniform(-180, 180)
            turn_angle = random.choice([-90,90,180])#要么-90，要么+90,要么180
            print(f"\nturn_angle{turn_angle}\n")
            # 随机俯仰角度（-5到5度）
            #pitch_angle = (random.uniform(-5, 5))
            pitch_angle = random.choice([-5,0,0,0,0,0,0,0])
            print(f"\npitch_angle{pitch_angle}\n")
            #self.precise_turn(turn_angle)
            self.precise_turn(turn_angle, pitch_angle)
        # 添加随机延迟使动作更自然
        time.sleep(random.uniform(0.1, 0.3))

    def _find_low_confidence_areas(self, map_data, current_x, current_y):
        """找到附近的低置信度区域"""
        low_confidence = []
        radius = 3  # 搜索半径

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < 15 and 0 <= ny < 15:
                    confidence = map_data[ny, nx]
                    # 优先选择置信度低于0.4的区域
                    if confidence < 0.4:
                        distance = abs(dx) + abs(dy)
                        # 加入位置和距离信息
                        low_confidence.append(((nx, ny), distance))

        # 按距离排序（优先近距离区域）
        return sorted(low_confidence, key=lambda x: x[1])


    def execute(self, path):                #这个函数用不到了，但是我留着不删，可以参考一下
        """执行路径，添加空路径检查和随机移动"""
        if not path:
            # 没有有效路径时随机移动
            self.random_move()
            return

        # 探索模式只关注移动
        for action in path:
            self.explore_behavior()
        '''
        # 只取前3个目标点
        for target in path[:3]:
            try:
                self.move_to(target[0], target[1])
                self.click()
                time.sleep(self.click_delay)
            except Exception as e:
                print(f"执行失败: {str(e)}")
                self.random_move()
        '''
    def move_to(self, x, y):
        """更智能的移动方法，支持直线和曲线移动"""
        if self.move_config['human_like']:
            self._bezier_move(x, y)
        else:
            self._direct_move(x, y)

    def _direct_move(self, x, y):
        """直线移动，添加随机抖动"""
        target_x = int(x * self.screen_width) + random.randint(-self.jitter_range, self.jitter_range)
        target_y = int(y * self.screen_height) + random.randint(-self.jitter_range, self.jitter_range)

        # 边界检查
        target_x = max(0, min(self.screen_width - 1, target_x))
        target_y = max(0, min(self.screen_height - 1, target_y))

        # 随机移动时间
        duration = random.uniform(self.move_config['min_duration'], self.move_config['max_duration'])
        pydirectinput.moveTo(target_x, target_y, duration=duration)

    def _bezier_move(self, x, y):
        """贝塞尔曲线移动，更自然的鼠标轨迹"""
        start_x, start_y = pyautogui.position()
        start = (start_x / self.screen_width, start_y / self.screen_height)
        end = (x, y)

        # 生成路径点
        path = self._bezier_curve(start, end, self.bezier_points)

        for point in path:
            target_x = int(point[0] * self.screen_width)
            target_y = int(point[1] * self.screen_height)

            # 边界检查
            target_x = max(0, min(self.screen_width - 1, target_x))
            target_y = max(0, min(self.screen_height - 1, target_y))

            pydirectinput.moveTo(target_x, target_y)
            time.sleep(0.01)

    def _bezier_curve(self, start, end, n=20):
        """优化贝塞尔曲线生成算法"""
        # 计算起点和终点的方向向量
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5

        if distance < 0.01:
            return [start, end]

        # 计算控制点偏移量
        offset_factor = random.uniform(0.1, 0.3) * distance

        # 计算垂直方向
        perp_x = -dy / max(distance, 0.0001)
        perp_y = dx / max(distance, 0.0001)

        # 随机偏移方向
        direction = 1 if random.random() > 0.5 else -1

        # 控制点位置
        ctrl1 = (
            start[0] + dx * 0.3 + perp_x * offset_factor * direction,
            start[1] + dy * 0.3 + perp_y * offset_factor * direction
        )

        ctrl2 = (
            start[0] + dx * 0.7 + perp_x * offset_factor * -direction,
            start[1] + dy * 0.7 + perp_y * offset_factor * -direction
        )

        # 生成曲线点
        points = []
        for i in range(n):
            t = i / (n - 1)
            # 三次贝塞尔曲线公式
            x = (1 - t) ** 3 * start[0] + 3 * (1 - t) ** 2 * t * ctrl1[0] + 3 * (1 - t) * t ** 2 * ctrl2[0] + t ** 3 * \
                end[0]
            y = (1 - t) ** 3 * start[1] + 3 * (1 - t) ** 2 * t * ctrl1[1] + 3 * (1 - t) * t ** 2 * ctrl2[1] + t ** 3 * \
                end[1]
            points.append((x, y))

        return points

    def click(self):
        """模拟点击，添加随机延迟"""
        pydirectinput.click()
        time.sleep(random.uniform(0.1, 0.3))

    def get_mouse_position(self):
        """获取当前鼠标位置"""
        x, y = pyautogui.position()
        return (x / self.screen_width, y / self.screen_height)

    def random_move(self):
        """随机移动鼠标，避免卡住"""
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        self.move_to(x, y)
        print("执行随机移动")

    def execute_decision(self, decision):
        """执行决策树做出的决策"""
        if decision[0] == 'move':
            target_pos, turn_angle = decision[1], decision[2]
            self.precise_turn(turn_angle, 0)
            self.move_to(target_pos[0], target_pos[1])