import numpy as np
from map_generator import generate_map, get_valid_position
import random

WIDTH = 50
HEIGHT = 50
OBSTACLE_RATIO = 0.2
N = 10  # agent count

# 시드 고정으로 맵과 에이전트 위치 일관성 유지
random.seed(42)
np.random.seed(42)

# 맵 생성
map_grid = generate_map(WIDTH, HEIGHT, OBSTACLE_RATIO, seed=42)
target_pos = get_valid_position(map_grid, seed=43)

print(f"Map created with {np.sum(map_grid)} obstacles ({np.sum(map_grid)/(WIDTH*HEIGHT)*100:.1f}%)")
print(f"Target position: {target_pos}")

# 에이전트 초기 위치 생성 (고정)
agents_positions = [
    (random.randint(0, 5), random.randint(WIDTH - 6, WIDTH - 1))
    for _ in range(N)
]

print(f"Agent positions: {agents_positions}")

# 저장
np.save("fixed_map.npy", map_grid)
np.save("fixed_target.npy", target_pos)
np.save("fixed_agents.npy", agents_positions)

print("Fixed map, target, and agents saved: fixed_map.npy, fixed_target.npy, fixed_agents.npy")