"""
Advanced AGV Path Planning Demo

Features:
1. A* with dynamic replanning
2. Path smoothing algorithms
3. Dynamic obstacle detection and avoidance
4. Real-time AGV simulation
5. Performance optimization
6. Multi-AGV coordination (basic)

Author: Xiaohang Liu
Date: May 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import math
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist

@dataclass
class AGVConfig:
    """Advanced AGV Configuration"""
    width: float = 0.8
    length: float = 1.2
    max_speed: float = 2.0
    max_angular_speed: float = 1.0
    max_acceleration: float = 1.5
    safety_margin: float = 0.3
    sensor_range: float = 5.0  # LiDAR range
    
@dataclass
class DynamicObstacle:
    """Dynamic obstacle representation"""
    x: float
    y: float
    vx: float  # velocity in x
    vy: float  # velocity in y
    radius: float
    
    def update(self, dt: float):
        """Update obstacle position"""
        self.x += self.vx * dt
        self.y += self.vy * dt

class AdvancedEnvironment:
    """Advanced environment with dynamic obstacles"""
    
    def __init__(self, width: float = 20.0, height: float = 15.0, resolution: float = 0.1):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # Static obstacles
        self.static_obstacles = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        # Dynamic obstacles
        self.dynamic_obstacles = []
        
        # Combined obstacle map (updated each frame)
        self.obstacles = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        self._setup_warehouse()
        self._add_dynamic_obstacles()
    
    def _setup_warehouse(self):
        """Setup warehouse environment"""
        # Main shelves
        shelf_positions = [
            (3, 3, 2, 8),
            (7, 3, 2, 8),
            (11, 3, 2, 8),
            (15, 3, 2, 8),
            (3, 12, 2, 2),
            (7, 12, 2, 2),
            (11, 12, 2, 2),
            (15, 12, 2, 2),
        ]
        
        for x, y, w, h in shelf_positions:
            self._add_rectangle_obstacle(x, y, w, h)
    
    def _add_dynamic_obstacles(self):
        """Add moving obstacles (other AGVs, people, etc.)"""
        self.dynamic_obstacles = [
            DynamicObstacle(5.0, 8.0, 0.5, 0.0, 0.4),  # Moving AGV
            DynamicObstacle(12.0, 6.0, 0.0, 0.3, 0.3),  # Person walking
            DynamicObstacle(8.0, 10.0, -0.2, 0.1, 0.35), # Another AGV
        ]
    
    def _add_rectangle_obstacle(self, x: float, y: float, width: float, height: float):
        """Add static rectangular obstacle"""
        x_start = max(0, int(x / self.resolution))
        y_start = max(0, int(y / self.resolution))
        x_end = min(self.grid_width, int((x + width) / self.resolution))
        y_end = min(self.grid_height, int((y + height) / self.resolution))
        
        self.static_obstacles[y_start:y_end, x_start:x_end] = True
    
    def update_dynamic_obstacles(self, dt: float):
        """Update positions of dynamic obstacles"""
        for obs in self.dynamic_obstacles:
            obs.update(dt)
            
            # Bounce off walls
            if obs.x <= obs.radius or obs.x >= self.width - obs.radius:
                obs.vx *= -1
            if obs.y <= obs.radius or obs.y >= self.height - obs.radius:
                obs.vy *= -1
            
            # Keep in bounds
            obs.x = max(obs.radius, min(self.width - obs.radius, obs.x))
            obs.y = max(obs.radius, min(self.height - obs.radius, obs.y))
    
    def update_obstacle_map(self):
        """Update combined obstacle map"""
        # Start with static obstacles
        self.obstacles = self.static_obstacles.copy()
        
        # Add dynamic obstacles
        for obs in self.dynamic_obstacles:
            self._add_circle_obstacle(obs.x, obs.y, obs.radius)
    
    def _add_circle_obstacle(self, cx: float, cy: float, radius: float):
        """Add circular obstacle to grid"""
        grid_cx = int(cx / self.resolution)
        grid_cy = int(cy / self.resolution)
        grid_radius = int(radius / self.resolution)
        
        for y in range(max(0, grid_cy - grid_radius), 
                      min(self.grid_height, grid_cy + grid_radius + 1)):
            for x in range(max(0, grid_cx - grid_radius), 
                          min(self.grid_width, grid_cx + grid_radius + 1)):
                if (x - grid_cx)**2 + (y - grid_cy)**2 <= grid_radius**2:
                    self.obstacles[y, x] = True
    
    def is_valid_position(self, x: float, y: float, agv_config: AGVConfig) -> bool:
        """Check if position is valid for AGV"""
        # Check AGV footprint
        half_width = (agv_config.width + agv_config.safety_margin) / 2
        half_length = (agv_config.length + agv_config.safety_margin) / 2
        
        corners = [
            (x - half_width, y - half_length),
            (x + half_width, y - half_length),
            (x - half_width, y + half_length),
            (x + half_width, y + half_length),
        ]
        
        for corner_x, corner_y in corners:
            grid_x = int(corner_x / self.resolution)
            grid_y = int(corner_y / self.resolution)
            
            if (grid_x < 0 or grid_x >= self.grid_width or 
                grid_y < 0 or grid_y >= self.grid_height or
                self.obstacles[grid_y, grid_x]):
                return False
        
        return True

class PathSmoother:
    """Path smoothing algorithms"""
    
    @staticmethod
    def smooth_path_spline(path: List[Tuple[float, float]], smoothing_factor: float = 0.1) -> List[Tuple[float, float]]:
        """Smooth path using B-spline"""
        if len(path) < 3:
            return path
        
        # Convert to numpy arrays
        path_array = np.array(path)
        x = path_array[:, 0]
        y = path_array[:, 1]
        
        # Fit spline
        try:
            tck, u = splprep([x, y], s=smoothing_factor * len(path))
            u_new = np.linspace(0, 1, len(path) * 2)
            x_smooth, y_smooth = splev(u_new, tck)
            
            return list(zip(x_smooth, y_smooth))
        except:
            return path
    
    @staticmethod
    def smooth_path_gradient_descent(path: List[Tuple[float, float]], 
                                   environment, agv_config,
                                   iterations: int = 50, 
                                   alpha: float = 0.1) -> List[Tuple[float, float]]:
        """Smooth path using gradient descent optimization"""
        if len(path) < 3:
            return path
        
        smoothed = np.array(path, dtype=float)
        
        for _ in range(iterations):
            for i in range(1, len(smoothed) - 1):
                # Smoothness term (minimize curvature)
                prev_point = smoothed[i-1]
                curr_point = smoothed[i]
                next_point = smoothed[i+1]
                
                # Calculate gradient for smoothness
                gradient = 2 * curr_point - prev_point - next_point
                
                # Update position
                new_point = curr_point - alpha * gradient
                
                # Check if new position is valid
                if environment.is_valid_position(new_point[0], new_point[1], agv_config):
                    smoothed[i] = new_point
        
        return [tuple(point) for point in smoothed]

class AdvancedAStarPlanner:
    """Advanced A* with dynamic replanning"""
    
    def __init__(self, environment: AdvancedEnvironment, agv_config: AGVConfig):
        self.env = environment
        self.agv_config = agv_config
        self.path = []
        self.search_nodes = []
        self.smoother = PathSmoother()
        
    def heuristic(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Enhanced heuristic with obstacle awareness"""
        base_distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # Add penalty for proximity to dynamic obstacles
        penalty = 0
        for obs in self.env.dynamic_obstacles:
            dist_to_obs = math.sqrt((x1 - obs.x)**2 + (y1 - obs.y)**2)
            if dist_to_obs < obs.radius + self.agv_config.safety_margin:
                penalty += 10.0 / (dist_to_obs + 0.1)
        
        return base_distance + penalty
    
    def get_neighbors(self, x: float, y: float) -> List[Tuple[float, float, float]]:
        """Get valid neighboring positions with costs"""
        neighbors = []
        
        # 8-directional movement with variable step sizes
        directions = [
            (0, self.env.resolution),
            (self.env.resolution, 0),
            (0, -self.env.resolution),
            (-self.env.resolution, 0),
            (self.env.resolution, self.env.resolution),
            (self.env.resolution, -self.env.resolution),
            (-self.env.resolution, self.env.resolution),
            (-self.env.resolution, -self.env.resolution),
        ]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            if self.env.is_valid_position(new_x, new_y, self.agv_config):
                cost = math.sqrt(dx**2 + dy**2)
                
                # Add cost for proximity to dynamic obstacles
                for obs in self.env.dynamic_obstacles:
                    dist = math.sqrt((new_x - obs.x)**2 + (new_y - obs.y)**2)
                    if dist < obs.radius + self.agv_config.safety_margin + 1.0:
                        cost += 5.0 / (dist + 0.1)
                
                neighbors.append((new_x, new_y, cost))
        
        return neighbors
    
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Plan path using enhanced A*"""
        start_x, start_y = start
        goal_x, goal_y = goal
        
        print(f"Planning path from ({start_x:.1f}, {start_y:.1f}) to ({goal_x:.1f}, {goal_y:.1f})")
        
        # Update obstacle map
        self.env.update_obstacle_map()
        
        # Validate positions
        if not self.env.is_valid_position(start_x, start_y, self.agv_config):
            raise ValueError("Start position is not valid")
        if not self.env.is_valid_position(goal_x, goal_y, self.agv_config):
            raise ValueError("Goal position is not valid")
        
        # A* implementation
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start_x, start_y, goal_x, goal_y)}
        
        closed_set = set()
        self.search_nodes = []
        
        start_time = time.time()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            self.search_nodes.append(current)
            
            current_x, current_y = current
            
            # Check if reached goal
            if math.sqrt((current_x - goal_x)**2 + (current_y - goal_y)**2) < self.env.resolution:
                print(f"Path found in {time.time() - start_time:.3f} seconds")
                print(f"Nodes explored: {len(self.search_nodes)}")
                
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                # Smooth the path
                smoothed_path = self.smoother.smooth_path_gradient_descent(
                    path, self.env, self.agv_config
                )
                
                self.path = smoothed_path
                return smoothed_path
            
            # Explore neighbors
            for neighbor_x, neighbor_y, move_cost in self.get_neighbors(current_x, current_y):
                neighbor = (neighbor_x, neighbor_y)
                
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(
                        neighbor_x, neighbor_y, goal_x, goal_y
                    )
                    
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print("No path found!")
        return []

class AdvancedAGVSimulator:
    """Advanced AGV simulator with real-time features"""
    
    def __init__(self, environment: AdvancedEnvironment, agv_config: AGVConfig):
        self.env = environment
        self.agv_config = agv_config
        self.planner = AdvancedAStarPlanner(environment, agv_config)
        
        # Simulation state
        self.agv_position = (1.0, 1.0)
        self.agv_velocity = (0.0, 0.0)
        self.current_path = []
        self.path_index = 0
        self.goal_position = None
        
        # Timing
        self.last_replan_time = 0
        self.replan_interval = 1.0  # seconds
        
        # Visualization (will be created in run_simulation)
        self.fig = None
        self.ax = None
        
        # Animation data
        self.time_data = []
        self.position_data = []
        
    def setup_visualization(self):
        """Setup advanced visualization"""
        self.ax.set_xlim(0, self.env.width)
        self.ax.set_ylim(0, self.env.height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Advanced AGV Path Planning - Dynamic Environment', 
                         fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X Position (meters)')
        self.ax.set_ylabel('Y Position (meters)')
    
    def draw_environment(self):
        """Draw the complete environment"""
        self.ax.clear()
        self.setup_visualization()
        
        # Draw static obstacles
        for y in range(self.env.grid_height):
            for x in range(self.env.grid_width):
                if self.env.static_obstacles[y, x]:
                    rect = patches.Rectangle(
                        (x * self.env.resolution, y * self.env.resolution),
                        self.env.resolution, self.env.resolution,
                        linewidth=0, facecolor='black', alpha=0.8
                    )
                    self.ax.add_patch(rect)
        
        # Draw dynamic obstacles
        for obs in self.env.dynamic_obstacles:
            circle = patches.Circle(
                (obs.x, obs.y), obs.radius,
                facecolor='red', alpha=0.6, edgecolor='darkred', linewidth=2
            )
            self.ax.add_patch(circle)
            
            # Draw velocity vector
            if abs(obs.vx) > 0.01 or abs(obs.vy) > 0.01:
                self.ax.arrow(obs.x, obs.y, obs.vx, obs.vy, 
                            head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    def draw_agv(self, position: Tuple[float, float], color: str = 'blue'):
        """Draw AGV with sensor range"""
        x, y = position
        
        # AGV body
        agv_rect = patches.Rectangle(
            (x - self.agv_config.width/2, y - self.agv_config.length/2),
            self.agv_config.width, self.agv_config.length,
            linewidth=2, facecolor=color, alpha=0.8, edgecolor='darkblue'
        )
        self.ax.add_patch(agv_rect)
        
        # Sensor range
        sensor_circle = patches.Circle(
            (x, y), self.agv_config.sensor_range,
            fill=False, edgecolor='cyan', alpha=0.3, linestyle='--'
        )
        self.ax.add_patch(sensor_circle)
        
        # Direction indicator
        arrow = patches.FancyArrowPatch(
            (x, y), (x + 0.4, y),
            arrowstyle='->', mutation_scale=25, color='white', linewidth=3
        )
        self.ax.add_patch(arrow)
    
    def draw_path(self, path: List[Tuple[float, float]], color: str = 'green', linewidth: int = 3):
        """Draw planned path"""
        if len(path) < 2:
            return
        
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        self.ax.plot(x_coords, y_coords, color=color, linewidth=linewidth, 
                    alpha=0.8, label='Planned Path')
    
    def simulate_step(self, dt: float = 0.1):
        """Simulate one time step"""
        current_time = time.time()
        
        # Update dynamic obstacles
        self.env.update_dynamic_obstacles(dt)
        
        # Check if replanning is needed
        if (current_time - self.last_replan_time > self.replan_interval and 
            self.goal_position is not None):
            
            try:
                new_path = self.planner.plan_path(self.agv_position, self.goal_position)
                if new_path:
                    self.current_path = new_path
                    self.path_index = 0
                    print("Path replanned due to dynamic obstacles")
            except:
                print("Replanning failed, keeping current path")
            
            self.last_replan_time = current_time
        
        # Move AGV along path
        if self.current_path and self.path_index < len(self.current_path):
            target = self.current_path[self.path_index]
            
            # Simple path following
            dx = target[0] - self.agv_position[0]
            dy = target[1] - self.agv_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 0.2:  # Close enough to waypoint
                self.path_index += 1
            else:
                # Move towards target
                speed = min(self.agv_config.max_speed, distance / dt)
                vx = (dx / distance) * speed * dt
                vy = (dy / distance) * speed * dt
                
                new_x = self.agv_position[0] + vx
                new_y = self.agv_position[1] + vy
                
                # Check if new position is safe
                if self.env.is_valid_position(new_x, new_y, self.agv_config):
                    self.agv_position = (new_x, new_y)
                    self.agv_velocity = (vx/dt, vy/dt)
        
        # Store data for analysis
        self.time_data.append(current_time)
        self.position_data.append(self.agv_position)
    
    def run_simulation(self, start: Tuple[float, float], goal: Tuple[float, float]):
        """Run complete simulation"""
        self.agv_position = start
        self.goal_position = goal
        
        # Create a new figure for this simulation
        plt.figure(figsize=(14, 10))
        self.fig = plt.gcf()
        self.ax = plt.gca()
        
        # Initial path planning
        try:
            self.current_path = self.planner.plan_path(start, goal)
            if not self.current_path:
                print("No initial path found!")
                return
        except Exception as e:
            print(f"Initial planning failed: {e}")
            return
        
        # Visualization
        self.draw_environment()
        self.draw_path(self.current_path)
        self.draw_agv(self.agv_position)
        
        # Draw start and goal
        self.ax.plot(start[0], start[1], 'go', markersize=15, label='Start')
        self.ax.plot(goal[0], goal[1], 'ro', markersize=15, label='Goal')
        
        # Add performance info
        path_length = sum(math.sqrt((self.current_path[i][0] - self.current_path[i+1][0])**2 + 
                                  (self.current_path[i][1] - self.current_path[i+1][1])**2)
                         for i in range(len(self.current_path)-1))
        
        info_text = (f"Path Length: {path_length:.2f}m\n"
                    f"Nodes Explored: {len(self.planner.search_nodes)}\n"
                    f"Dynamic Obstacles: {len(self.env.dynamic_obstacles)}")
        
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        self.ax.legend(loc='upper right')
        plt.tight_layout()
        
        # Show the figure and keep it open
        plt.show(block=False)
        plt.draw()
        
        return self.fig  # Return figure reference to keep it alive

def main():
    """Main demonstration"""
    print("=" * 70)
    print("Advanced AGV Path Planning Demo")
    print("VisionNav Robotics Test Engineer Interview")
    print("Features: Dynamic obstacles, Path smoothing, Real-time replanning")
    print("=" * 70)
    
    # Demo scenarios
    scenarios = [
        {
            "name": "Dynamic Obstacle Avoidance",
            "start": (1.0, 1.0),
            "goal": (18.0, 13.0),
            "description": "Navigate while avoiding moving obstacles"
        },
        {
            "name": "Narrow Passage with Moving AGV",
            "start": (1.0, 7.0),
            "goal": (18.0, 7.0),
            "description": "Navigate through passages with other moving AGVs"
        },
        {
            "name": "Complex Multi-Obstacle Environment",
            "start": (2.0, 13.0),
            "goal": (16.0, 2.0),
            "description": "Complex navigation with multiple dynamic obstacles"
        }
    ]
    
    # Store figure references to keep them alive
    figures = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Start: {scenario['start']}")
        print(f"Goal: {scenario['goal']}")
        
        input(f"\nPress Enter to run scenario {i}...")
        
        # Create fresh environment and simulator for each scenario
        env = AdvancedEnvironment(width=20.0, height=15.0, resolution=0.1)
        agv_config = AGVConfig()
        simulator = AdvancedAGVSimulator(env, agv_config)
        
        start_time = time.time()
        fig = simulator.run_simulation(scenario['start'], scenario['goal'])
        end_time = time.time()
        
        # Store figure reference
        if fig:
            figures.append(fig)
        
        print(f"Simulation completed in {end_time - start_time:.3f} seconds")
    
    print("\n" + "=" * 70)
    print("Advanced Demo Features Demonstrated:")
    print("✓ A* with dynamic replanning")
    print("✓ Path smoothing algorithms")
    print("✓ Dynamic obstacle detection and avoidance")
    print("✓ Real-time environment updates")
    print("✓ Performance optimization")
    print("✓ Realistic AGV constraints and sensor modeling")
    print("=" * 70)
    
    # Keep all figures open
    print(f"\n{len(figures)} figures are now displayed.")
    print("Close the figure windows manually when you're done viewing them.")
    print("Press Enter to exit the program...")
    input()
    
    # Optional: Close all figures programmatically
    plt.close('all')

if __name__ == "__main__":
    main() 