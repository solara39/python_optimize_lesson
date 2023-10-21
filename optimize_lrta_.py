#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().run_line_magic('matplotlib', 'nbagg')
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# グリッドの初期化と障害物の設定
grid_width = 10
grid_height = 10
grid = np.zeros((grid_width, grid_height))
obstacle_positions = [(0, 4), (0, 5), (0, 6), (0, 7), (1, 2), (1, 5), (1, 6), (1, 7), (1, 8), (2, 1), (2, 2), (2, 6), (2, 7), (2, 8), (3, 1), (3, 2), (4, 1), (4, 2), (4, 6), (4, 7), (5, 1), (5, 2), (5, 4), (5, 5), (5, 6), (5, 7), (6, 1), (6, 2), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (7, 1), (7, 2), (7, 6), (7, 7), (7, 8), (8, 1), (8, 2), (8, 6), (8, 7), (8, 8), (9, 1), (9, 2)]
for x, y in obstacle_positions:
    grid[x][y] = 1

def generate_h_grid(grid,goal):
  grid_width, grid_height = grid.shape
  h_grid = np.zeros((grid_width, grid_height))
  for x in range(grid_width):
    for y in range(grid_height):
      h_grid[x][y] = heuristic((x, y),goal)
  
  return h_grid

# スタート位置とゴール位置を設定
start_position = (3, 0)
goal_position = (2, 9)

# スタート位置にエージェントを配置
grid[start_position[0]][start_position[1]] = 2

# ゴール位置を別の値で表す（通常は3など別の整数値を使用）
grid[goal_position[0]][goal_position[1]] = 3

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
        else:
          ax.add_patch(patches.Rectangle((j, grid_height - i - 1), 1, 1, color='r'))
        
def heuristic(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def _init_(start, goal, path):
    state = start
    path = [start]
    return state, path

def search_move(grid, start, state, h_grid):
    grid_width, grid_height = grid.shape
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    next_fs = []

    for action in actions:
      new_x, new_y = state[0] + action[0], state[1] + action[1]

      if 0 <= new_x < grid_width and 0 <= new_y < grid_height and grid[new_x][new_y] != 1:
        next_f = 1 + h_grid[new_x][new_y]
        next_fs.append(next_f)
      else:
        next_fs.append(float('inf'))
    return next_fs

def update_h(state, next_fs, h_grid):
  index = next_fs.index(min(next_fs))
  h_grid[state[0], state[1]] = min(next_fs)
  return h_grid, index

def LRTA_Star(grid, start, goal, h_grid):
    grid_width, grid_height = grid.shape
    path = [start]
    paths = []
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    next_fs = []
    state = (start[0], start[1])
    counter = 0
    
    while True: #not(np.array_equal(h_keep, h_grid) and state == goal):
      state, path = _init_(start, goal, path)
      h_keep = copy.deepcopy(h_grid)
      while state != goal:
        next_fs = search_move(grid, start, state, h_grid)
        h_grid, index = update_h(state, next_fs, h_grid)
        state = state[0] + actions[index][0], state[1] + actions[index][1]
        if 0 > state[0] or state[0] >= grid_width or 0 > state[1] or state[1] >= grid_height:
            paths.append(path)
            break;
        elif np.array_equal(h_keep, h_grid) and state == goal:
            path
            break;
        else:
            path.append(state)
      counter += 1
      paths.append(path)
      if np.array_equal(h_keep, h_grid):
        break;
    return paths

# Initialize the agent's position
x, y = start_position
agent_patch = patches.Circle((y + 0.5, grid_height - x - 1 + 0.5), radius=0.4, color='b')
ax.add_patch(agent_patch)

# Function to update the agent's position in the animation frame
def update(frame):
    x, y = paths[frame]
    agent_patch.set_center((y + 0.5, grid_height - x - 1 + 0.5))
    return agent_patch

# LRTA*を実行
h_grid = generate_h_grid(grid, goal_position)
paths = LRTA_Star(grid, start_position, goal_position, h_grid)

# Generate the animation frames
print(paths)
paths = sum(paths,[])
frames = len(paths)


# Create the animation
ani = FuncAnimation(fig, update, frames=frames, interval=100, repeat=False)

# Show the plot
plt.title('Agent Path Animation')
plt.show()

