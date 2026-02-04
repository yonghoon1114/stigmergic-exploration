# import numpy as np
# import random

# class Agent:

#     def __init__(self, y, x, height, width):

#         self.y = y
#         self.x = x
#         self.height = height
#         self.width = width

#     def step(self, potential):
#         odds = self.getOdds(potential)

#         (dy, dx) = random.choices([*odds.keys()],
#             weights = [*odds.values()],k=1)[0]

#         self.y = int(np.clip(self.y + dy, 0, self.height - 1))
#         self.x = int(np.clip(self.x + dx, 0, self.width - 1))

#         potential[self.y, self.x] += 1
    
#     def getOdds(self, potential):
#         HEIGHT, WIDTH = potential.shape
#         weights = {}
#         for dy in [-1, 0, 1]:
#             for dx in [-1, 0, 1]:
#                 ny = self.y + dy
#                 nx = self.x + dx

#                 # 자기 자신 칸은 제외 (원하면 포함해도 됨)
#                 if dy == 0 and dx == 0:
#                     continue

#                 # 경계 체크: "이동 결과 좌표"가 유효한지
#                 if 0 <= ny < HEIGHT and 0 <= nx < WIDTH:
#                     weights[(dy, dx)] = np.exp(-potential[ny, nx])
#                 else:
#                     # 맵 밖은 매우 큰 포텐셜 (사실상 못 가게)
#                     weights[(dy, dx)] = np.exp(-1e9)

#         total = sum(weights.values())
#         for k in weights:
#             weights[k] /= total
    
#         return weights

import numpy as np
import random

class Agent:

    def __init__(self, y, x, height, width, alpha):

        self.y = y
        self.x = x
        self.height = height
        self.width = width
        self.alpha = alpha

    def step(self, potential, agents):
        
        odds = self.getOdds(potential, agents)

        (dy, dx) = random.choices(
            list(odds.keys()),
            weights=list(odds.values()),
            k=1
        )[0]

        self.y = int(np.clip(self.y + dy, 0, self.height - 1))
        self.x = int(np.clip(self.x + dx, 0, self.width - 1))

        potential[self.y, self.x] += 1

    
    def getOdds(self, potential, agents):

        HEIGHT, WIDTH = potential.shape

        r = 5
        k = 2
        beta = 1.0   # repulsion weight

        weights = {}

        explore = 1.0 / (potential + 1e-3)

        gy, gx = np.gradient(explore)

        for dy in [-1,0,1]:
            for dx in [-1,0,1]:

                if dy == 0 and dx == 0:
                    continue

                ny = self.y + dy
                nx = self.x + dx

                if not (0 <= ny < HEIGHT and 0 <= nx < WIDTH):
                    weights[(dy,dx)] = 0
                    continue

                rep = 0.0

                for other in agents:
                    if other is self:
                        continue

                    d = np.hypot(other.y - ny, other.x - nx)

                    if d < r:
                        rep += k/(d+1e-3)

                grad = gx[ny,nx]*dx + gy[ny,nx]*dy

                v = self.alpha * explore[ny,nx] - beta * rep + grad
                v = np.clip(v, -50, 50)

                weights[(dy, dx)] = np.exp(v)

        return weights
