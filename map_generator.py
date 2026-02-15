import numpy as np
import matplotlib.pyplot as plt


def generate_map(width, height, obstacle_ratio=0.2, seed=None):
    """장애물이 있는 맵 생성
    
    Args:
        width: 맵 가로 길이
        height: 맵 세로 길이
        obstacle_ratio: 장애물 비율 (0~1)
        seed: 랜덤 시드
    
    Returns:
        map_grid: 0=자유공간, 1=장애물
    """
    if seed is not None:
        np.random.seed(seed)
    
    map_grid = np.random.choice([0, 1], size=(height, width), p=[1-obstacle_ratio, obstacle_ratio])
    return map_grid


def get_valid_position(map_grid, seed=None):
    """맵에서 장애물이 아닌 위치 반환
    
    Args:
        map_grid: 맵 그리드
        seed: 랜덤 시드
    
    Returns:
        (y, x): 유효한 위치
    """
    if seed is not None:
        np.random.seed(seed)
    
    height, width = map_grid.shape
    while True:
        y = np.random.randint(0, height)
        x = np.random.randint(0, width)
        if map_grid[y, x] == 0:  # 장애물 아닐 때
            return y, x


def set_obstacles_on_potential(potential, map_grid, obstacle_value=1e6):
    """맵의 장애물 위치에 높은 potential 값 설정
    
    Args:
        potential: potential 배열
        map_grid: 맵 그리드
        obstacle_value: 장애물에 설정할 potential 값
    
    Returns:
        potential: 수정된 potential
    """
    potential[map_grid == 1] = obstacle_value
    return potential


def visualize_map(map_grid, target_pos, save_path="map_visualization.png"):
    """맵과 목표 위치를 시각화
    
    Args:
        map_grid: 맵 그리드
        target_pos: (y, x) 목표 위치
        save_path: 저장 경로
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(map_grid, cmap='gray', origin='lower')  # 0=흰색(자유), 1=검은색(장애물)
    plt.scatter([target_pos[1]], [target_pos[0]], c='red', s=200, marker='*', label='Target')
    plt.title("Map with Obstacles")
    plt.colorbar(label='0=Free, 1=Obstacle')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    print(f"Map visualization saved: {save_path}")
