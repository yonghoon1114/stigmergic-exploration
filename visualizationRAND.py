import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import animation
from agents import Agent

WIDTH = 100
HEIGHT = 100
NUM_AGENTS = 10
STEPS = 200

potential = np.zeros((HEIGHT, WIDTH))

agents = [
    Agent(
        random.randint(0, HEIGHT - 1),
        random.randint(0, WIDTH - 1),
        HEIGHT,
        WIDTH,
        alpha=10
    )
    for _ in range(NUM_AGENTS)
]

fig, ax = plt.subplots()

def update(step):
    ax.clear()

    for agent in agents:
        agent.step(potential, agents)

    ax.imshow(potential, cmap="hot", origin="lower")

    ys = [a.y for a in agents]
    xs = [a.x for a in agents]
    ax.scatter(xs, ys, c="cyan", s=30)

    ax.set_title(f"Step {step}")
    ax.set_xticks([])
    ax.set_yticks([])

ani = animation.FuncAnimation(
    fig,
    update,
    frames=STEPS,
    interval=50
)

ani.save("swarm.gif", writer="pillow", dpi=150)

plt.close()

print("saved swarm.gif")
