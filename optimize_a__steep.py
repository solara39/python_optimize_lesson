#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'nbagg')
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import heapq
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

grid_width = 10
grid_height = 10
grid = np.zeros((grid_width, grid_height))
obstacle_positions = [(0, 4), (0, 5), (0, 6), (0, 7), (1, 2), (1, 5), (1, 6), (1, 7), (1, 8), (2, 1), (2, 2), (2, 6), (2, 7), (2, 8), (3, 1), (3, 2), (4, 1), (4, 2), (4, 6), (4, 7), (5, 1), (5, 2), (5, 4), (5, 5), (5, 6), (5, 7), (6, 1), (6, 2), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (7, 1), (7, 2), (7, 6), (7, 7), (7, 8), (8, 1), (8, 2), (8, 6), (8, 7), (8, 8), (9, 1), (9, 2)]
for x, y in obstacle_positions:
    grid[x][y] = 1

# スタート位置とゴール位置を設定
start_position = (3, 0)
goal_position = (2, 9)

# スタート位置にエージェントを配置
grid[start_position[0]][start_position[1]] = 2  # 2をエージェントを表す値として使用

# ゴール位置を別の値で表す（通常は3など別の整数値を使用）
grid[goal_position[0]][goal_position[1]] = 3

#移動コストが5のセルを配置
for _ in range(5, 9):
    grid[3][_] = 4

# Create the figure and axis for the plot
fig, ax = plt.subplots()
ax.set_xlim(0, grid_width)
ax.set_ylim(0, grid_height)
ax.set_aspect('equal')

for i in range(10):
      for j in range(10):
        if grid[i][j] == 0:
          ax.add_patch(patches.Rectangle((j, grid_height - i - 1), 1, 1, color='w'))
        elif grid[i][j] == 1:
          ax.add_patch(patches.Rectangle((j, grid_height - i - 1), 1, 1, color='k'))
        elif grid[i][j] == 3:
          ax.add_patch(patches.Rectangle((j, grid_height - i - 1), 1, 1, color='g'))
        elif grid[i][j] == 4:
          ax.add_patch(patches.Rectangle((j, grid_height - i - 1), 1, 1, color = 'grey'))
        else:
          ax.add_patch(patches.Rectangle((j, grid_height - i - 1), 1, 1, color='r'))

def generate_h_grid(grid,goal):
  grid_width, grid_height = grid.shape
  h_grid = np.zeros((grid_width, grid_height))
  for x in range(grid_width):
    for y in range(grid_height):
      h_grid[x][y] = heuristic((x, y),goal)
  
  return h_grid
        
# Custom move_cost function to handle specific positions
def move_cost(current_position, next_position):
    x1, y1 = current_position
    x2, y2 = next_position
    
    if (x1, y1) in [(3, 5), (3, 6), (3, 7), (3, 8)] or (x2, y2) in [(3, 5), (3, 6), (3, 7), (3, 8)]:
        return 5
    else:
        return 1

# マンハッタン距離をヒューリスティック関数として使用
def heuristic(position, goal):
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

# A*アルゴリズム
def astar(grid, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {position: float('inf') for position in np.ndindex(grid.shape)}
    g_score[start] = 0

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_score[current]  # 経路とステップ数を返す

        for neighbor in get_neighbors(current, grid):
            tentative_g_score = g_score[current] + move_cost(current, neighbor)
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))

    return None, float('inf')  # ゴールに到達できない場合

# 迷路内での隣接セルを取得（縦横のみ許可）
def get_neighbors(position, grid):
    neighbors = []

    # 縦横の移動のみを許可
    possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in possible_moves:
        x, y = position[0] + dx, position[1] + dy
        if 0 <= x < grid_width and 0 <= y < grid_height and grid[x][y] != 1:
            neighbors.append((x, y))

    return neighbors


# 経路の再構築
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    return path

# Initialize the agent's position
x, y = start_position
agent_patch = patches.Circle((y + 0.5, grid_height - x - 1 + 0.5), radius=0.4, color='b')
ax.add_patch(agent_patch)

# Function to update the agent's position in the animation frame
def update(frame):
    x, y = paths[frame]
    agent_patch.set_center((y + 0.5, grid_height - x - 1 + 0.5))
    return agent_patch

# A*を実行
h_grid = generate_h_grid(grid, goal_position)
paths, steps = astar(grid, start_position, goal_position)
print(paths)

# Generate the animation frames
frames = len(paths)

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, interval=100, repeat=False)

# Show the plot
plt.title('Agent Path Animation')
plt.show()

