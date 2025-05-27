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
    """增强的A*路径规划器"""
    
    def __init__(self, environment):
        self.env = environment
        self.resolution = 0.3  # 减小网格分辨率以获得更精确的路径
        self.last_plan_time = 0
        self.replan_interval = 1.0  # 重规划间隔
    
    def heuristic(self, a, b):
        """启发式函数：欧几里得距离"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, node):
        """获取邻居节点"""
        neighbors = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in directions:
            new_x = node[0] + dx * self.resolution
            new_y = node[1] + dy * self.resolution
            
            if not self.env.is_collision(new_x, new_y):
                cost = self.resolution if dx == 0 or dy == 0 else self.resolution * math.sqrt(2)
                neighbors.append(((new_x, new_y), cost))
        
        return neighbors
    
    def plan_path(self, start, goal):
        """A*路径规划"""
        print(f"开始路径规划: 从 ({start[0]:.1f}, {start[1]:.1f}) 到 ({goal[0]:.1f}, {goal[1]:.1f})")
        
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
                print(f"路径规划完成! 探索了 {len(explored_nodes)} 个节点")
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return self._smooth_path(path), explored_nodes
            
            for neighbor, move_cost in self.get_neighbors(current):
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print("未找到路径!")
        return None, explored_nodes
    
    def _smooth_path(self, path):
        """路径平滑处理"""
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # 尝试直线连接到更远的点
            for j in range(len(path) - 1, i + 1, -1):
                if self._is_line_clear(path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    break
            else:
                i += 1
                if i < len(path):
                    smoothed.append(path[i])
        
        return smoothed
    
    def _is_line_clear(self, start, end):
        """检查两点间直线是否无障碍"""
        steps = int(self.heuristic(start, end) / (self.resolution / 2))
        if steps == 0:
            return True
        
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            if self.env.is_collision(x, y):
                return False
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
        self.ax.grid(True, alpha=0.3)
        
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
            # 绘制规划路径
            path_x = [point[0] for point in self.path]
            path_y = [point[1] for point in self.path]
            self.ax.plot(path_x, path_y, 'g-', linewidth=3, alpha=0.8, label='Planned Path')
            
            # 绘制路径点
            self.ax.scatter(path_x, path_y, c='green', s=25, alpha=0.8, zorder=5)
            
            # 标记当前目标点
            if self.path_index < len(self.path):
                target = self.path[self.path_index]
                self.ax.plot(target[0], target[1], 'yo', markersize=12, 
                           markeredgecolor='orange', markeredgewidth=2, label='Current Target')
    
    def draw_agv(self):
        """绘制AGV"""
        if self.agv_position:
            x, y = self.agv_position
            
            # AGV主体
            agv_rect = patches.Rectangle((x-0.4, y-0.6), 0.8, 1.2,
                                       facecolor='blue', edgecolor='darkblue',
                                       linewidth=2, alpha=0.8)
            self.ax.add_patch(agv_rect)
            
            # AGV方向指示器
            if self.path and self.path_index < len(self.path) - 1:
                target = self.path[self.path_index + 1] if self.path_index + 1 < len(self.path) else self.path[-1]
                dx = target[0] - x
                dy = target[1] - y
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    length = math.sqrt(dx**2 + dy**2)
                    arrow_end_x = x + (dx / length) * 0.6
                    arrow_end_y = y + (dy / length) * 0.6
                    arrow = patches.FancyArrowPatch((x, y), (arrow_end_x, arrow_end_y),
                                                  arrowstyle='->', mutation_scale=20,
                                                  color='white', linewidth=2)
                    self.ax.add_patch(arrow)
            
            # 传感器范围
            sensor_circle = patches.Circle((x, y), 3.0, fill=False, 
                                         edgecolor='cyan', alpha=0.3, 
                                         linestyle='--', linewidth=2)
            self.ax.add_patch(sensor_circle)
            
            # AGV轨迹
            if len(self.agv_trail) > 1:
                trail_x = [pos[0] for pos in self.agv_trail]
                trail_y = [pos[1] for pos in self.agv_trail]
                self.ax.plot(trail_x, trail_y, 'b--', alpha=0.6, linewidth=2, label='AGV Trail')
            
            # AGV标签
            self.ax.text(x, y-1.0, 'Main AGV', ha='center', va='top',
                        fontsize=9, fontweight='bold', color='blue',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
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
                    remaining_distance += math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            
            total_path_distance = sum(math.sqrt((self.path[i+1][0]-self.path[i][0])**2 + 
                                              (self.path[i+1][1]-self.path[i][1])**2)
                                    for i in range(len(self.path)-1))
            
            progress = ((total_path_distance - remaining_distance) / total_path_distance * 100) if total_path_distance > 0 else 0
            
            # 检查附近障碍物
            nearby_obstacles = self.env.get_obstacle_info(self.agv_position[0], self.agv_position[1])
            obstacle_info = ""
            if nearby_obstacles:
                obstacle_info = f"\nNearby Obstacles: {len(nearby_obstacles)}"
                for obs in nearby_obstacles[:2]:  # 只显示最近的2个
                    obstacle_info += f"\n  {obs['type']} (ID:{obs['id']}) - {obs['distance']:.1f}m"
            
            status = f"""AGV Navigation Status:
Step: {self.step_count}
Status: {self.navigation_status}
Current Position: ({self.agv_position[0]:.1f}, {self.agv_position[1]:.1f})
Goal Position: ({self.goal_position[0]:.1f}, {self.goal_position[1]:.1f})
Progress: {progress:.1f}%
Remaining Distance: {remaining_distance:.1f}m
Path Points: {self.path_index}/{len(self.path)}
Total Distance Traveled: {self.total_distance:.1f}m
Replanning Count: {self.replan_count}
Collision Avoidance: {self.collision_count}
Explored Nodes: {len(self.explored_nodes) if self.explored_nodes else 0}{obstacle_info}"""
            
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
            
            # 移动AGV
            if self.path_index < len(self.path) - 1:
                target = self.path[self.path_index + 1]
                current = self.agv_position
                
                # 计算移动方向
                dx = target[0] - current[0]
                dy = target[1] - current[1]
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < 0.2:  # 接近路径点
                    self.path_index += 1
                    print(f"  Reached waypoint {self.path_index}/{len(self.path)}")
                    if self.path_index >= len(self.path) - 1:
                        self.navigation_status = "Goal Reached"
                else:
                    # 移动向目标
                    speed = 1.2  # m/s
                    move_distance = speed * dt
                    if move_distance > distance:
                        move_distance = distance
                    
                    new_x = current[0] + (dx / distance) * move_distance
                    new_y = current[1] + (dy / distance) * move_distance
                    
                    # 检查碰撞
                    if not self.env.is_collision(new_x, new_y):
                        self.agv_position = (new_x, new_y)
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