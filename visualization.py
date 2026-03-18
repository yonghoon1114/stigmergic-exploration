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
RUNS_PER_PAIR = 10  # seeds per (alpha,beta) pair
ALPHA_SAMPLES = 8
BETA_SAMPLES = 8
def entropy(potential):
    p = potential.flatten()
    p = p / np.sum(p)
    return -np.sum(p * np.log(p + 1e-10))

def visited(potential):
    return sum(potential> 1e-6)

def run_sim(alpha, beta, map_grid=None, target_pos=None, agents_positions=None):
    """
    시뮬레이션 실행
    
    Returns:
        찾은 경우: 도달한 스텝 수
        못 찾은 경우: STEPS
    """
    potential = np.zeros((HEIGHT, WIDTH)) + 1e-6

    # 장애물 설정
    if map_grid is not None:
        potential = set_obstacles_on_potential(potential, map_grid)

    if agents_positions is not None:
        agents = [
            Agent(
                y=int(pos[0]),
                x=int(pos[1]),
                height=HEIGHT,
                width=WIDTH,
                alpha=alpha,
                beta=beta
            )
            for pos in agents_positions
        ]
    else:
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

# 고정 맵과 에이전트 위치 로드
map_grid = np.load("fixed_map.npy")
target_pos = tuple(np.load("fixed_target.npy"))
agents_positions = np.load("fixed_agents.npy")

print(f"Fixed map loaded with {np.sum(map_grid)} obstacles ({np.sum(map_grid)/(WIDTH*HEIGHT)*100:.1f}%)")
print(f"Target position: {target_pos}")
print(f"Agent positions: {agents_positions}")

# alpha-beta sweep (2D) -- reduced resolution for speed
alphas = np.linspace(0.5, 25, ALPHA_SAMPLES)
betas = np.linspace(0.5, 25, BETA_SAMPLES)

Hs = np.zeros((len(betas), len(alphas)), dtype=np.float32)
times = np.zeros_like(Hs)
total_pairs = len(alphas) * len(betas)
completed_pairs = 0
cumulative_time = 0.0

for i, b in enumerate(betas):
    for j, a in enumerate(alphas):
        vals = []
        t0 = time.perf_counter()
        for s in range(RUNS_PER_PAIR):
            result = run_sim(a, b, map_grid=map_grid, target_pos=target_pos, agents_positions=agents_positions)
            vals.append(result)
        Hs[i, j] = np.mean(vals)
        elapsed = time.perf_counter() - t0
        times[i, j] = elapsed
        # update progress counters and print a concise ETA line
        completed_pairs += 1
        cumulative_time += elapsed
        avg_time = cumulative_time / completed_pairs
        remaining = total_pairs - completed_pairs
        eta_seconds = remaining * avg_time
        pct = completed_pairs / total_pairs * 100
        print(f"Progress {completed_pairs}/{total_pairs} ({pct:.1f}%) a={a:.2f} b={b:.2f} time={elapsed:.2f}s eta~{eta_seconds/60:.1f}min", flush=True)

# plot heatmap (use sampled indices; tick labels show actual alpha/beta values)
plt.figure(figsize=(12, 8))
im = plt.imshow(Hs, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(im, label='Steps to Goal')

# helper to select tick positions and labels (max 10 ticks)
def _ticks_and_labels(arr, max_ticks=10):
    n = len(arr)
    if n <= max_ticks:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_ticks, dtype=int)
    labels = [f"{arr[k]:.1f}" for k in idx]
    return idx, labels

xticks, xticklabels = _ticks_and_labels(alphas)
yticks, yticklabels = _ticks_and_labels(betas)
plt.xticks(xticks, xticklabels, rotation=45)
plt.yticks(yticks, yticklabels)
plt.xlabel("Alpha")
plt.ylabel("Beta")
plt.title("Goal Reach Time vs Alpha and Beta")
plt.tight_layout()
plt.savefig("goal_reach_time_alpha_beta4.png", dpi=200)
plt.close()

print("Heatmap saved: goal_reach_time_alpha_beta3.png")

# Save results as CSV (long format)
import csv
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['alpha', 'beta', 'mean_steps', 'time_sec'])
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            writer.writerow([f"{a:.6f}", f"{b:.6f}", f"{Hs[i,j]:.6f}", f"{times[i,j]:.6f}"])

# 콘솔에 표 출력 (alpha-beta 조합별 평균 스텝과 시간)
print("\n=== 실험 결과 표 (Alpha vs Beta) ===")
print(f"{'Beta \\ Alpha':<10}", end="")
for a in alphas:
    print(f"{a:<8.1f}", end="")
print()
print("-" * (10 + 8 * len(alphas)))
for i, b in enumerate(betas):
    print(f"{b:<10.1f}", end="")
    for j, a in enumerate(alphas):
        steps = Hs[i, j]
        time_val = times[i, j]
        print(f"{steps:<4.1f}({time_val:<4.2f})", end=" ")
    print()

print("\n(형식: 스텝수(시간초))")

# Save a PNG table (rows: beta, cols: alpha) for quick visual lookup
fig, ax = plt.subplots(figsize=(max(8, ALPHA_SAMPLES * 0.6), max(6, BETA_SAMPLES * 0.4)))
ax.axis('off')
cell_text = [[f"{v:.1f}" for v in row] for row in Hs]
row_labels = [f"{b:.1f}" for b in betas]
col_labels = [f"{a:.1f}" for a in alphas]
the_table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(8)
the_table.scale(1, 1.2)
plt.title('Mean Steps Table (rows: beta, cols: alpha)')
plt.tight_layout()
plt.savefig('results_table.png', dpi=200)
plt.close()

print('Results saved: results2.csv, results_table2.png, goal_reach_time_alpha_beta2.png')

# 저장: 배열로 결과 확인 가능하게 함
np.save("Hs.npy", Hs)
np.save("alphas.npy", alphas)
np.save("betas.npy", betas)
np.save("times.npy", times)
print("Arrays saved: Hs.npy, alphas.npy, betas.npy, times.npy")

# =======================
# 맵 시각화
# =======================
visualize_map(map_grid, target_pos, save_path="map_visualization2.png")
