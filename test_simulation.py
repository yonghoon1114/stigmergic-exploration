import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import animation
from agents import Agent
from map_generator import generate_map, get_valid_position, set_obstacles_on_potential

WIDTH = 50
HEIGHT = 50
NUM_AGENTS = 10
STEPS = 400
DECAY = 1
OBSTACLE_RATIO = 0.2

# 파라미터 설정
ALPHA = 20
BETA = 5

# 맵 생성
map_grid = generate_map(WIDTH, HEIGHT, OBSTACLE_RATIO, seed=42)
target_pos = get_valid_position(map_grid, seed=43)
target_y, target_x = target_pos

print(f"Map created with {np.sum(map_grid)} obstacles")
print(f"Target position: ({target_y}, {target_x})")

potential = np.zeros((HEIGHT, WIDTH)) + 1e-6

# 장애물 설정
potential = set_obstacles_on_potential(potential, map_grid)

agents = [
    Agent(
        y=random.randint(0, 5),      # 맵 위쪽 (y: 0~5)
        x=random.randint(0, 5),      # 맵 왼쪽 (x: 0~5)
        height=HEIGHT,
        width=WIDTH,
        alpha=ALPHA,
        beta=BETA
    )
    for _ in range(NUM_AGENTS)
]

fig, ax = plt.subplots(figsize=(8, 8))

def update(step):
    ax.clear()

    # Decay 적용
    global potential
    potential *= DECAY
    
    # 장애물 유지
    potential = set_obstacles_on_potential(potential, map_grid)

    for agent in agents:
        agent.step(potential, agents)

    # Potential 시각화
    ax.imshow(potential, cmap="hot", origin="lower", vmin=0, vmax=2)

    # 타겟 표시 (빨간 별)
    ax.scatter([target_x], [target_y], c="red", s=300, marker='*', edgecolors='white', linewidth=2, label='Target')

    # 에이전트 표시 (사이안 점)
    ys = [a.y for a in agents]
    xs = [a.x for a in agents]
    ax.scatter(xs, ys, c="cyan", s=50, edgecolors='white', linewidth=1.5, label='Agents')

    ax.set_title(f"Step {step} (Alpha={ALPHA}, Beta={BETA})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right')

ani = animation.FuncAnimation(
    fig,
    update,
    frames=STEPS,
    interval=50
)

output_filename = f"test_simulation_alpha{ALPHA}_beta{BETA}.gif"
ani.save(output_filename, writer="pillow", dpi=150)

plt.close()

print(f"Simulation saved: {output_filename}")
print(f"Parameters: Alpha={ALPHA}, Beta={BETA}, Agents={NUM_AGENTS}, Steps={STEPS}")
print(f"Obstacle ratio: {np.sum(map_grid) / (WIDTH * HEIGHT):.1%}")
print(f"Parameters: Alpha={ALPHA}, Beta={BETA}, Agents={NUM_AGENTS}, Steps={STEPS}")
