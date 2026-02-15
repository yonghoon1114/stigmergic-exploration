import numpy as np
import matplotlib.pyplot as plt
import random
from agents import Agent
from map_generator import generate_map, get_valid_position, set_obstacles_on_potential, visualize_map
import time

WIDTH = 50
HEIGHT = 50
STEPS = 200
DECAY = 1
N = 10   # agent count 고정
OBSTACLE_RATIO = 0.2  # 장애물 비율

def entropy(potential):
    p = potential.flatten()
    p = p / np.sum(p)
    return -np.sum(p * np.log(p + 1e-10))

def visited(potential):
    return sum(potential> 1e-6)

def run_sim(alpha, beta, map_grid=None, target_pos=None, seed=0):
    """
    시뮬레이션 실행
    
    Returns:
        찾은 경우: 도달한 스텝 수
        못 찾은 경우: STEPS
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    potential = np.zeros((HEIGHT, WIDTH)) + 1e-6

    # 장애물 설정
    if map_grid is not None:
        potential = set_obstacles_on_potential(potential, map_grid)

    agents = [
        Agent(
            y=random.randint(0, 5),
            x=random.randint(WIDTH - 6, WIDTH - 1),
            height=HEIGHT,
            width=WIDTH,
            alpha=alpha,
            beta=beta
        )
        for _ in range(N)
    ]

    for t in range(STEPS):
        potential *= DECAY
        
        # 장애물 설정 유지 (매 스텝마다 복원)
        if map_grid is not None:
            potential = set_obstacles_on_potential(potential, map_grid)

        for agent in agents:
            agent.step(potential, agents)
            
            # 목표 도달 체크
            if target_pos is not None:
                if agent.y == target_pos[0] and agent.x == target_pos[1]:
                    return t  # 도달 스텝 수 반환

    return STEPS  # 못 찾으면 최대값 반환

# =======================
# alpha sweep
# =======================

# 맵 생성
map_grid = generate_map(WIDTH, HEIGHT, OBSTACLE_RATIO, seed=42)
target_y, target_x = get_valid_position(map_grid, seed=43)
target_pos = (target_y, target_x)

print(f"Map created with {np.sum(map_grid)} obstacles ({np.sum(map_grid)/(WIDTH*HEIGHT)*100:.1f}%)")
print(f"Target position: {target_pos}")

# alpha-beta sweep (2D)
alphas = np.linspace(0.5, 40, 10)
betas = np.linspace(0.5, 40, 10)

Hs = np.zeros((len(betas), len(alphas)))

for i, b in enumerate(betas):
    for j, a in enumerate(alphas):
        vals = []
        for s in range(10):
            result = run_sim(a, b, map_grid=map_grid, target_pos=target_pos, seed=s)
            vals.append(result)
        Hs[i, j] = np.mean(vals)

# plot heatmap with text annotations
plt.figure(figsize=(12, 8))
im = plt.imshow(Hs, extent=[alphas.min(), alphas.max(), betas.min(), betas.max()],
                aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(im, label='Steps to Goal')

# 각 셀에 숫자 표시
for i in range(len(betas)):
    for j in range(len(alphas)):
        plt.text(alphas[j], betas[i], f'{Hs[i, j]:.1f}',
                ha='center', va='center', color='white', fontsize=14, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='none'))

plt.xlabel("Alpha")
plt.ylabel("Beta")
plt.title("Goal Reach Time vs Alpha and Beta")
plt.tight_layout()
plt.savefig("goal_reach_time_alpha_beta.png", dpi=200)
plt.close()

print("Heatmap saved: goal_reach_time_alpha_beta.png")

# =======================
# 맵 시각화
# =======================
visualize_map(map_grid, target_pos, save_path="map_visualization.png")
