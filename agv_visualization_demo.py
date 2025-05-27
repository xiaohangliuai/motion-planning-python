"""
AGV导航过程详细可视化演示
展示AGV从开始运行到结束节点的完整过程，包含完善的障碍物规避

Author: Xiaohang Liu
Date: 2025年5月
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import math
from typing import List, Tuple
import heapq

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimpleAGVEnvironment:
    """简化的AGV环境"""
    
    def __init__(self, width=20, height=15):
        self.width = width
        self.height = height
        self.obstacles = []
        self.dynamic_obstacles = []
        self._setup_warehouse()
        self._setup_dynamic_obstacles()
    
    def _setup_warehouse(self):
        """设置仓库环境"""
        # 货架位置 (x, y, width, height) - 调整布局增大通道
        shelves = [
            (3, 3, 2, 8),   # 左侧货架1
            (7, 3, 2, 8),   # 左侧货架2
            (11, 3, 2, 8),  # 右侧货架1
            (15, 3, 2, 7),  # 右侧货架2
            (3, 12, 2, 2),  # 上方小货架1
            (7, 12, 2, 2),  # 上方小货架2
            (11, 12, 2, 2), # 上方小货架3
            (15, 12, 2, 2), # 上方小货架4
        ]
        self.obstacles = shelves
    
    def _setup_dynamic_obstacles(self):
        """设置动态障碍物"""
        self.dynamic_obstacles = [
            {'x': 5.0, 'y': 8.0, 'vx': 0.3, 'vy': 0.0, 'radius': 0.4, 'type': 'Other AGV', 'id': 1},
            {'x': 12.0, 'y': 6.0, 'vx': 0.0, 'vy': 0.2, 'radius': 0.3, 'type': 'Person', 'id': 2},
            {'x': 8.0, 'y': 10.0, 'vx': -0.2, 'vy': 0.1, 'radius': 0.35, 'type': 'Other AGV', 'id': 3},
        ]
    
    def update_dynamic_obstacles(self, dt):
        """更新动态障碍物位置"""
        for obs in self.dynamic_obstacles:
            obs['x'] += obs['vx'] * dt
            obs['y'] += obs['vy'] * dt
            
            # 边界反弹
            if obs['x'] <= obs['radius'] or obs['x'] >= self.width - obs['radius']:
                obs['vx'] *= -1
            if obs['y'] <= obs['radius'] or obs['y'] >= self.height - obs['radius']:
                obs['vy'] *= -1
            
            # 保持在边界内
            obs['x'] = max(obs['radius'], min(self.width - obs['radius'], obs['x']))
            obs['y'] = max(obs['radius'], min(self.height - obs['radius'], obs['y']))
    
    def is_collision(self, x, y, agv_size=0.8):
        """检查位置是否与障碍物碰撞"""
        # 检查静态障碍物
        for ox, oy, ow, oh in self.obstacles:
            if (x - agv_size/2 < ox + ow and x + agv_size/2 > ox and
                y - agv_size/2 < oy + oh and y + agv_size/2 > oy):
                return True
        
        # 检查动态障碍物
        for obs in self.dynamic_obstacles:
            dist = math.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
            if dist < obs['radius'] + agv_size/2 + 0.5:  # 增加安全边距
                return True
        
        # 检查边界
        if x < agv_size/2 or x > self.width - agv_size/2 or y < agv_size/2 or y > self.height - agv_size/2:
            return True
        
        return False
    
    def get_obstacle_info(self, x, y, agv_size=0.8):
        """获取障碍物信息用于调试"""
        obstacles_nearby = []
        
        # 检查动态障碍物
        for obs in self.dynamic_obstacles:
            dist = math.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
            if dist < obs['radius'] + agv_size/2 + 1.0:  # 检测范围
                obstacles_nearby.append({
                    'type': obs['type'],
                    'distance': dist,
                    'position': (obs['x'], obs['y']),
                    'id': obs['id']
                })
        
        return obstacles_nearby

class AdvancedAStarPlanner:
    """增强的A*路径规划器 - 网格化移动"""
    
    def __init__(self, environment):
        self.env = environment
        self.resolution = 0.5  # 增大网格分辨率，使路径更规整
        self.last_plan_time = 0
        self.replan_interval = 1.0  # 重规划间隔
    
    def heuristic(self, a, b):
        """启发式函数：曼哈顿距离（适合网格移动）"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, node):
        """获取邻居节点 - 只允许4个方向移动"""
        neighbors = []
        # 只允许上、下、左、右四个方向移动
        directions = [
            (0, 1),   # 上
            (0, -1),  # 下
            (-1, 0),  # 左
            (1, 0)    # 右
        ]
        
        for dx, dy in directions:
            new_x = node[0] + dx * self.resolution
            new_y = node[1] + dy * self.resolution
            
            if not self.env.is_collision(new_x, new_y):
                cost = self.resolution  # 所有移动成本相同
                neighbors.append(((new_x, new_y), cost))
        
        return neighbors
    
    def _align_to_grid(self, point):
        """将点对齐到网格"""
        x, y = point
        aligned_x = round(x / self.resolution) * self.resolution
        aligned_y = round(y / self.resolution) * self.resolution
        return (aligned_x, aligned_y)
    
    def plan_path(self, start, goal):
        """A*路径规划 - 网格化版本"""
        # 将起点和终点对齐到网格
        start = self._align_to_grid(start)
        goal = self._align_to_grid(goal)
        
        print(f"开始网格化路径规划: 从 ({start[0]:.1f}, {start[1]:.1f}) 到 ({goal[0]:.1f}, {goal[1]:.1f})")
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        explored_nodes = []
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            explored_nodes.append(current)
            
            if self.heuristic(current, goal) < self.resolution:
                print(f"网格化路径规划完成! 探索了 {len(explored_nodes)} 个节点")
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return self._optimize_grid_path(path), explored_nodes
            
            for neighbor, move_cost in self.get_neighbors(current):
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print("未找到路径!")
        return None, explored_nodes
    
    def _optimize_grid_path(self, path):
        """优化网格路径 - 合并同方向的连续移动"""
        if len(path) < 3:
            return path
        
        optimized = [path[0]]
        
        i = 0
        while i < len(path) - 1:
            current = path[i]
            
            # 找到同一方向上的最远点
            direction = None
            j = i + 1
            
            while j < len(path):
                next_point = path[j]
                
                # 计算当前方向
                dx = next_point[0] - current[0]
                dy = next_point[1] - current[1]
                
                # 标准化方向
                if abs(dx) > abs(dy):
                    current_dir = (1 if dx > 0 else -1, 0)
                else:
                    current_dir = (0, 1 if dy > 0 else -1)
                
                if direction is None:
                    direction = current_dir
                elif direction != current_dir:
                    break
                
                # 检查直线路径是否安全
                if not self._is_grid_line_clear(current, next_point):
                    break
                
                j += 1
            
            # 添加最远的安全点
            if j > i + 1:
                optimized.append(path[j - 1])
                i = j - 1
            else:
                optimized.append(path[i + 1])
                i += 1
        
        return optimized
    
    def _is_grid_line_clear(self, start, end):
        """检查网格线路径是否无障碍"""
        # 确保是水平或垂直线
        if start[0] != end[0] and start[1] != end[1]:
            return False
        
        # 检查路径上的每个网格点
        if start[0] == end[0]:  # 垂直线
            y_start, y_end = sorted([start[1], end[1]])
            y = y_start
            while y <= y_end:
                if self.env.is_collision(start[0], y):
                    return False
                y += self.resolution
        else:  # 水平线
            x_start, x_end = sorted([start[0], end[0]])
            x = x_start
            while x <= x_end:
                if self.env.is_collision(x, start[1]):
                    return False
                x += self.resolution
        
        return True

class AGVNavigationVisualizer:
    """AGV导航过程可视化器"""
    
    def __init__(self, environment, planner):
        self.env = environment
        self.planner = planner
        self.agv_position = None
        self.goal_position = None
        self.path = None
        self.explored_nodes = None
        self.path_index = 0
        self.agv_trail = []
        self.step_count = 0
        self.total_distance = 0
        self.replan_count = 0
        self.collision_count = 0
        self.last_replan_time = 0
        
        # AGV状态
        self.agv_speed = 0
        self.agv_direction = 0
        self.navigation_status = "初始化"
        
        # 可视化设置
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.ax.set_xlim(0, environment.width)
        self.ax.set_ylim(0, environment.height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('AGV Navigation Process Demo', fontsize=16, fontweight='bold')
        
        # 状态文本
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                       verticalalignment='top', fontsize=10,
                                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def draw_environment(self):
        """绘制环境"""
        self.ax.clear()
        self.ax.set_xlim(0, self.env.width)
        self.ax.set_ylim(0, self.env.height)
        self.ax.set_aspect('equal')
        
        # 绘制网格线
        grid_resolution = self.planner.resolution
        for x in np.arange(0, self.env.width + grid_resolution, grid_resolution):
            self.ax.axvline(x, color='lightgray', alpha=0.3, linewidth=0.5)
        for y in np.arange(0, self.env.height + grid_resolution, grid_resolution):
            self.ax.axhline(y, color='lightgray', alpha=0.3, linewidth=0.5)
        
        # 绘制静态障碍物（货架）
        for ox, oy, ow, oh in self.env.obstacles:
            rect = patches.Rectangle((ox, oy), ow, oh, 
                                   linewidth=2, edgecolor='black', 
                                   facecolor='gray', alpha=0.7)
            self.ax.add_patch(rect)
            # 添加货架标签
            self.ax.text(ox + ow/2, oy + oh/2, 'Shelf', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')
        
        # 绘制动态障碍物
        for obs in self.env.dynamic_obstacles:
            color = 'orange' if obs['type'] == 'Other AGV' else 'red'
            circle = patches.Circle((obs['x'], obs['y']), obs['radius'],
                                  facecolor=color, alpha=0.6, edgecolor='black')
            self.ax.add_patch(circle)
            # 添加标签和ID
            label = f"{obs['type']}\nID:{obs['id']}"
            self.ax.text(obs['x'], obs['y'], label, ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white')
            
            # 绘制运动方向箭头
            if abs(obs['vx']) > 0.01 or abs(obs['vy']) > 0.01:
                arrow_end_x = obs['x'] + obs['vx'] * 2
                arrow_end_y = obs['y'] + obs['vy'] * 2
                self.ax.arrow(obs['x'], obs['y'], obs['vx'] * 2, obs['vy'] * 2,
                            head_width=0.2, head_length=0.2, fc=color, ec=color, alpha=0.8)
    
    def draw_planning_process(self):
        """绘制规划过程"""
        if self.explored_nodes:
            # 绘制探索的节点
            explored_x = [node[0] for node in self.explored_nodes]
            explored_y = [node[1] for node in self.explored_nodes]
            self.ax.scatter(explored_x, explored_y, c='lightcoral', s=8, alpha=0.4, label='Explored Nodes')
        
        if self.path:
            # 绘制网格化路径 - 使用直角线段
            for i in range(len(self.path) - 1):
                start = self.path[i]
                end = self.path[i + 1]
                
                # 绘制路径段
                self.ax.plot([start[0], end[0]], [start[1], end[1]], 
                           'g-', linewidth=4, alpha=0.8, solid_capstyle='round')
                
                # 在路径段上添加方向箭头
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    # 标准化方向
                    length = math.sqrt(dx**2 + dy**2)
                    dx_norm = dx / length * 0.3
                    dy_norm = dy / length * 0.3
                    
                    self.ax.arrow(mid_x - dx_norm/2, mid_y - dy_norm/2, 
                                dx_norm, dy_norm,
                                head_width=0.15, head_length=0.15, 
                                fc='darkgreen', ec='darkgreen', alpha=0.8)
            
            # 绘制路径点
            path_x = [point[0] for point in self.path]
            path_y = [point[1] for point in self.path]
            self.ax.scatter(path_x, path_y, c='green', s=40, alpha=0.9, 
                          zorder=5, edgecolors='darkgreen', linewidth=1, label='Planned Path')
            
            # 标记当前目标点
            if self.path_index < len(self.path):
                target = self.path[self.path_index]
                self.ax.plot(target[0], target[1], 'yo', markersize=15, 
                           markeredgecolor='orange', markeredgewidth=3, label='Current Target')
    
    def draw_agv(self):
        """绘制AGV"""
        if self.agv_position:
            x, y = self.agv_position
            
            # AGV主体 - 方形设计更符合网格移动
            agv_rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8,
                                       facecolor='blue', edgecolor='darkblue',
                                       linewidth=3, alpha=0.9)
            self.ax.add_patch(agv_rect)
            
            # AGV方向指示器 - 显示当前移动方向
            if self.path and self.path_index < len(self.path) - 1:
                target = self.path[self.path_index + 1] if self.path_index + 1 < len(self.path) else self.path[-1]
                dx = target[0] - x
                dy = target[1] - y
                
                # 只显示主要方向的箭头
                if abs(dx) > abs(dy):
                    # 水平移动
                    direction = 1 if dx > 0 else -1
                    arrow_end_x = x + direction * 0.5
                    arrow_end_y = y
                else:
                    # 垂直移动
                    direction = 1 if dy > 0 else -1
                    arrow_end_x = x
                    arrow_end_y = y + direction * 0.5
                
                arrow = patches.FancyArrowPatch((x, y), (arrow_end_x, arrow_end_y),
                                              arrowstyle='->', mutation_scale=25,
                                              color='white', linewidth=3)
                self.ax.add_patch(arrow)
            
            # 传感器范围
            sensor_circle = patches.Circle((x, y), 2.5, fill=False, 
                                         edgecolor='cyan', alpha=0.4, 
                                         linestyle='--', linewidth=2)
            self.ax.add_patch(sensor_circle)
            
            # AGV轨迹 - 显示网格化的移动轨迹
            if len(self.agv_trail) > 1:
                for i in range(len(self.agv_trail) - 1):
                    start = self.agv_trail[i]
                    end = self.agv_trail[i + 1]
                    self.ax.plot([start[0], end[0]], [start[1], end[1]], 
                               'b--', alpha=0.6, linewidth=2)
                
                # 标记轨迹点
                trail_x = [pos[0] for pos in self.agv_trail[::3]]  # 每3个点显示一个
                trail_y = [pos[1] for pos in self.agv_trail[::3]]
                self.ax.scatter(trail_x, trail_y, c='lightblue', s=15, alpha=0.7, label='AGV Trail')
            
            # AGV标签
            self.ax.text(x, y-0.8, 'Main AGV', ha='center', va='top',
                        fontsize=9, fontweight='bold', color='blue',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    def draw_start_goal(self):
        """绘制起点和终点"""
        if hasattr(self, 'start_pos'):
            self.ax.plot(self.start_pos[0], self.start_pos[1], 'go', 
                        markersize=15, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
            self.ax.text(self.start_pos[0], self.start_pos[1]+0.5, 'START', 
                        ha='center', va='bottom', fontweight='bold', color='green')
        
        if self.goal_position:
            self.ax.plot(self.goal_position[0], self.goal_position[1], 'ro', 
                        markersize=15, label='Goal', markeredgecolor='darkred', markeredgewidth=2)
            self.ax.text(self.goal_position[0], self.goal_position[1]+0.5, 'GOAL', 
                        ha='center', va='bottom', fontweight='bold', color='red')
    
    def update_status(self):
        """更新状态信息"""
        if self.path and self.agv_position:
            remaining_distance = 0
            if self.path_index < len(self.path):
                for i in range(self.path_index, len(self.path)-1):
                    p1, p2 = self.path[i], self.path[i+1]
                    remaining_distance += abs(p2[0]-p1[0]) + abs(p2[1]-p1[1])  # 曼哈顿距离
            
            total_path_distance = sum(abs(self.path[i+1][0]-self.path[i][0]) + 
                                    abs(self.path[i+1][1]-self.path[i][1])
                                    for i in range(len(self.path)-1))
            
            progress = ((total_path_distance - remaining_distance) / total_path_distance * 100) if total_path_distance > 0 else 0
            
            # 检查附近障碍物
            nearby_obstacles = self.env.get_obstacle_info(self.agv_position[0], self.agv_position[1])
            obstacle_info = ""
            if nearby_obstacles:
                obstacle_info = f"\nNearby Obstacles: {len(nearby_obstacles)}"
                for obs in nearby_obstacles[:2]:  # 只显示最近的2个
                    obstacle_info += f"\n  {obs['type']} (ID:{obs['id']}) - {obs['distance']:.1f}m"
            
            # 确定当前移动方向
            current_direction = "Stationary"
            if self.path_index < len(self.path) - 1:
                current_pos = self.agv_position
                target_pos = self.path[self.path_index + 1]
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
                
                if abs(dx) > abs(dy):
                    current_direction = "→ East" if dx > 0 else "← West"
                else:
                    current_direction = "↑ North" if dy > 0 else "↓ South"
            
            status = f"""AGV Grid Navigation Status:
Step: {self.step_count}
Status: {self.navigation_status}
Current Position: ({self.agv_position[0]:.1f}, {self.agv_position[1]:.1f})
Goal Position: ({self.goal_position[0]:.1f}, {self.goal_position[1]:.1f})
Current Direction: {current_direction}
Progress: {progress:.1f}%
Remaining Distance: {remaining_distance:.1f}m (Manhattan)
Grid Waypoints: {self.path_index}/{len(self.path)}
Total Distance Traveled: {self.total_distance:.1f}m
Replanning Count: {self.replan_count}
Collision Avoidance: {self.collision_count}
Explored Nodes: {len(self.explored_nodes) if self.explored_nodes else 0}
Grid Resolution: {self.planner.resolution}m{obstacle_info}"""
            
            self.status_text.set_text(status)
    
    def run_scenario(self, start, goal, scenario_name):
        """运行单个场景"""
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_name}")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"{'='*80}")
        
        self.start_pos = start
        self.agv_position = start
        self.goal_position = goal
        self.path_index = 0
        self.agv_trail = [start]
        self.step_count = 0
        self.total_distance = 0
        self.replan_count = 0
        self.collision_count = 0
        self.navigation_status = "Path Planning"
        
        # 路径规划
        print("Step 1: Starting path planning...")
        self.path, self.explored_nodes = self.planner.plan_path(start, goal)
        
        if not self.path:
            print("Path planning failed!")
            self.navigation_status = "Planning Failed"
            return
        
        print(f"Step 2: Path planning successful! Path contains {len(self.path)} points")
        self.navigation_status = "Navigating"
        
        # 创建新图形
        plt.figure(figsize=(16, 12))
        self.fig = plt.gcf()
        self.ax = plt.gca()
        
        # 绘制初始状态
        self.draw_environment()
        self.draw_planning_process()
        self.draw_start_goal()
        self.draw_agv()
        self.update_status()
        
        self.ax.legend(loc='upper right')
        self.ax.set_title(f'{scenario_name} - AGV Navigation Demo', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2)  # 显示初始状态2秒
        
        # 开始导航
        print("Step 3: Starting AGV navigation...")
        dt = 0.15  # 时间步长
        last_position = self.agv_position
        
        while self.path_index < len(self.path) - 1:
            self.step_count += 1
            
            # 更新动态障碍物
            self.env.update_dynamic_obstacles(dt)
            
            # 检查是否需要重规划
            current_time = time.time()
            if (current_time - self.last_replan_time > 2.0 and  # 每2秒检查一次
                self.path_index < len(self.path) - 1):
                
                # 检查当前路径是否仍然安全
                next_few_points = self.path[self.path_index:min(self.path_index + 5, len(self.path))]
                need_replan = False
                
                for point in next_few_points:
                    if self.env.is_collision(point[0], point[1]):
                        need_replan = True
                        break
                
                if need_replan:
                    print(f"  Obstacle detected on path! Replanning...")
                    self.navigation_status = "Replanning"
                    new_path, new_explored = self.planner.plan_path(self.agv_position, self.goal_position)
                    if new_path:
                        self.path = new_path
                        self.explored_nodes.extend(new_explored)
                        self.path_index = 0
                        self.replan_count += 1
                        print(f"  Replanning successful! New path has {len(new_path)} points")
                        self.navigation_status = "Navigating"
                    else:
                        print(f"  Replanning failed! Continuing with current path")
                        self.navigation_status = "Navigation Warning"
                
                self.last_replan_time = current_time
            
            # 移动AGV - 网格化移动逻辑
            if self.path_index < len(self.path) - 1:
                target = self.path[self.path_index + 1]
                current = self.agv_position
                
                # 计算移动方向 - 只允许水平或垂直移动
                dx = target[0] - current[0]
                dy = target[1] - current[1]
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < 0.1:  # 接近路径点
                    # 精确对齐到目标点
                    self.agv_position = target
                    self.agv_trail.append(self.agv_position)
                    self.path_index += 1
                    print(f"  Reached grid waypoint {self.path_index}/{len(self.path)} at ({target[0]:.1f}, {target[1]:.1f})")
                    if self.path_index >= len(self.path) - 1:
                        self.navigation_status = "Goal Reached"
                else:
                    # 网格化移动 - 优先完成一个方向的移动
                    speed = 1.0  # m/s
                    move_distance = speed * dt
                    
                    # 确定主要移动方向
                    if abs(dx) > abs(dy):
                        # 水平移动优先
                        direction = 1 if dx > 0 else -1
                        move_x = min(abs(dx), move_distance) * direction
                        new_x = current[0] + move_x
                        new_y = current[1]
                    else:
                        # 垂直移动优先
                        direction = 1 if dy > 0 else -1
                        move_y = min(abs(dy), move_distance) * direction
                        new_x = current[0]
                        new_y = current[1] + move_y
                    
                    # 检查碰撞
                    if not self.env.is_collision(new_x, new_y):
                        self.agv_position = (new_x, new_y)
                        
                        # 只在位置有明显变化时添加到轨迹
                        if (len(self.agv_trail) == 0 or 
                            abs(new_x - self.agv_trail[-1][0]) > 0.05 or 
                            abs(new_y - self.agv_trail[-1][1]) > 0.05):
                            self.agv_trail.append(self.agv_position)
                        
                        # 计算移动距离
                        move_dist = math.sqrt((new_x - last_position[0])**2 + (new_y - last_position[1])**2)
                        self.total_distance += move_dist
                        last_position = self.agv_position
                    else:
                        print(f"  Collision detected! AGV stopped at ({current[0]:.1f}, {current[1]:.1f})")
                        self.collision_count += 1
                        self.navigation_status = "Collision Avoidance"
                        
                        # 尝试紧急重规划
                        emergency_path, _ = self.planner.plan_path(self.agv_position, self.goal_position)
                        if emergency_path:
                            self.path = emergency_path
                            self.path_index = 0
                            self.replan_count += 1
                            print(f"  Emergency replanning successful!")
                            self.navigation_status = "Navigating"
            
            # 更新可视化
            self.draw_environment()
            self.draw_planning_process()
            self.draw_start_goal()
            self.draw_agv()
            self.update_status()
            
            self.ax.legend(loc='upper right')
            self.ax.set_title(f'{scenario_name} - Step {self.step_count} - {self.navigation_status}', 
                            fontsize=16, fontweight='bold')
            
            plt.draw()
            plt.pause(0.08)  # 控制动画速度
        
        print(f"Step 4: AGV successfully reached the goal!")
        print(f"Total steps: {self.step_count}")
        print(f"Total distance traveled: {self.total_distance:.1f}m")
        print(f"Number of replanning events: {self.replan_count}")
        print(f"Number of collision avoidance events: {self.collision_count}")
        
        # 最终状态显示
        self.navigation_status = "Mission Complete"
        self.ax.set_title(f'{scenario_name} - Navigation Complete!', 
                         fontsize=16, fontweight='bold', color='green')
        self.update_status()
        plt.draw()
        
        input("Press Enter to continue to next scenario...")

def main():
    """主函数"""
    print("AGV Navigation Process Detailed Demo")
    print("=" * 80)
    
    # 创建环境和规划器
    env = SimpleAGVEnvironment()
    planner = AdvancedAStarPlanner(env)
    visualizer = AGVNavigationVisualizer(env, planner)
    
    # 定义场景
    scenarios = [
        {
            "name": "Scenario 1: Dynamic Obstacle Avoidance",
            "start": (1.0, 1.0),
            "goal": (18.0, 13.0),
            "description": "AGV navigates while avoiding moving obstacles"
        },
        {
            "name": "Scenario 2: Narrow Passage Navigation",
            "start": (1.0, 7.0),
            "goal": (18.0, 7.0),
            "description": "AGV navigates through narrow passages between shelves"
        },
        {
            "name": "Scenario 3: Complex Multi-Obstacle Environment",
            "start": (2.0, 13.0),
            "goal": (16.0, 2.0),
            "description": "AGV performs long-distance navigation in complex environment"
        }
    ]
    
    # 运行所有场景
    for scenario in scenarios:
        visualizer.run_scenario(
            scenario["start"], 
            scenario["goal"], 
            scenario["name"]
        )
    
    print("\n" + "=" * 80)
    print("All scenario demonstrations completed!")
    print("Demo Features:")
    print("✓ Real-time path planning visualization")
    print("✓ Dynamic obstacle detection and avoidance")
    print("✓ AGV movement trajectory tracking")
    print("✓ Detailed navigation status information")
    print("✓ Step-by-step process demonstration")
    print("✓ Automatic replanning when obstacles detected")
    print("✓ Collision avoidance with emergency replanning")
    print("✓ Performance metrics tracking")
    print("=" * 80)
    
    plt.show()

if __name__ == "__main__":
    main() 