import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

MAZE_SIZE = 10
start_position = [3, 0]
agent_position = [3, 0]
goal_position = [2, 9]
loop_count = 0
total_cost = 0
cost_list = []

land_map=[[1,1,1,1,0,0,0,0,1,1],
          [1,1,0,1,1,0,0,0,0,1],
          [1,0,0,1,1,1,0,0,0,1],
          [1,0,0,1,1,5,5,5,5,1],
          [1,0,0,1,1,1,0,0,1,1],
          [1,0,0,1,0,0,0,0,1,1],
          [1,0,0,1,0,0,0,0,0,1],
          [1,0,0,1,1,1,0,0,0,1],
          [1,0,0,1,1,1,0,0,0,1],
          [1,0,0,1,1,1,1,1,1,1]]

f_map=[[0 for _ in range(10)] for _ in range(10)]
wall_positions=[]

for i in range(10):
    for j in range(10):
        if land_map[i][j] == 0:
            wall_positions.append((i, j))
        else:
            f_map[i][j] = abs(i - 2) + abs(j - 9)

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_xlim(0, MAZE_SIZE)
ax.set_ylim(0, MAZE_SIZE)
for i in range(1, MAZE_SIZE):
    ax.plot([i, i], [0, MAZE_SIZE], color='black', linewidth=2)
    ax.plot([0, MAZE_SIZE], [i, i], color='black', linewidth=2)

for wall_position in wall_positions:
    ax.add_patch(patches.Rectangle((wall_position[1], MAZE_SIZE - wall_position[0] - 1), 1, 1, color='black'))

ax.add_patch(patches.Rectangle((start_position[1], MAZE_SIZE - start_position[0] - 1), 1, 1, color='red'))
ax.add_patch(patches.Rectangle((goal_position[1], MAZE_SIZE - goal_position[0] - 1), 1, 1, color='green'))
agent_patch = patches.Circle((agent_position[1] + 0.5, MAZE_SIZE - agent_position[0] - 0.5), 0.4, color='blue')
ax.add_patch(agent_patch)
text_object = ax.text(0.5, 0, "0", transform=ax.transAxes, fontsize=16, ha='center')




def move_agent(position, f_map, land_map):
    global total_cost
    x, y = position
    next_list = []
    if x > 0 and (x - 1, y) not in wall_positions:
        next_list.append([(x - 1, y), f_map[x - 1][y] + land_map[x - 1][y]])

    if x < MAZE_SIZE - 1 and (x + 1, y) not in wall_positions:
        next_list.append([(x + 1, y), f_map[x + 1][y] + land_map[x + 1][y]])

    if y > 0 and (x, y - 1) not in wall_positions:
        next_list.append([(x, y - 1), f_map[x][y - 1] + land_map[x][y - 1]])

    if y < MAZE_SIZE - 1 and (x, y + 1) not in wall_positions:
        next_list.append([(x, y + 1), f_map[x][y + 1] + land_map[x][y + 1]])

    next_position, cost = min(next_list, key=lambda x: x[1])
    f_map[x][y] = cost
    total_cost += land_map[next_position[0]][next_position[1]]
    return next_position

def update(frame):
    global agent_position, step_count, loop_count, total_cost, cost_list
    agent_position = move_agent(agent_position, f_map, land_map)
    agent_patch.set_center((agent_position[1] + 0.5, MAZE_SIZE - agent_position[0] - 0.5))
    
    text_object.set_text(f'Cost: {total_cost}')
    
    
    
    # ゴールに到達したかを確認
    if agent_position == goal_position or (abs(agent_position[0] - goal_position[0]) < 0.01 and abs(agent_position[1] - goal_position[1]) < 0.01):
        print("Goal reached! Stopping animation.")
        # ani.event_source.stop()  # アニメーションを停止
        step_count = 0
        agent_position = start_position
        loop_count +=1
        print(total_cost)
        cost_list.append(total_cost)
        total_cost = 0
        

    if loop_count == 20:
        ani.event_source.stop()

    
    return agent_patch, text_object,

ani = FuncAnimation(fig, update, frames=200, interval=100, blit=True)
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure()
plt.plot(cost_list, marker='o', linestyle='-', color='b')
plt.xlabel('Loop Count')
plt.ylabel('Total Cost')
plt.title('Total Cost per Loop')
plt.grid(True)
plt.show()