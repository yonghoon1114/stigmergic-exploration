import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import animation

WIDTH, HEIGHT = 50, 50
NUM_AGENTS = 10
STEPS = 100

potential = np.zeros((HEIGHT, WIDTH))
agents = [
    [random.randint(0, HEIGHT-1), random.randint(0, WIDTH-1)]
    for _ in range(NUM_AGENTS)
]

fig, ax = plt.subplots()

def update(step):
    ax.clear()

    for agent in agents:
        dy, dx = random.choice([(0,1),(0,-1),(1,0),(-1,0)])
        agent[0] = np.clip(agent[0] + dy, 0, HEIGHT-1)
        agent[1] = np.clip(agent[1] + dx, 0, WIDTH-1)
        potential[agent[0], agent[1]] += 1

    ax.imshow(potential, cmap="hot", origin="lower")
    ys = [a[0] for a in agents]
    xs = [a[1] for a in agents]
    ax.scatter(xs, ys, c="cyan", s=30)
    ax.set_title(f"Step {step}")
    ax.set_xticks([])
    ax.set_yticks([])

ani = animation.FuncAnimation(fig, update, frames=STEPS)
ani.save("simulationRAND.gif", writer="pillow")
plt.savefig("final_resultRAND.png", dpi=200, bbox_inches="tight")
plt.close()