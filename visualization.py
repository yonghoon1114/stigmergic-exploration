import numpy as np
import matplotlib.pyplot as plt
import random
from agents import Agent

WIDTH = 50
HEIGHT = 50
NUM_AGENTS = 10
STEPS = 100

potential = np.zeros((HEIGHT, WIDTH))

agents = [
    Agent(
        random.randint(0, HEIGHT - 1),
        random.randint(0, WIDTH - 1),
        HEIGHT,
        WIDTH
    )
    for _ in range(NUM_AGENTS)
]

fig, ax = plt.subplots()

for step in range(STEPS):
    ax.clear()

    for agent in agents:
        agent.step(potential)

    ax.imshow(potential, cmap="hot", origin="lower")

    ys = [a.y for a in agents]
    xs = [a.x for a in agents]
    ax.scatter(xs, ys, c="cyan", s=30)

    ax.set_title(f"Step {step}")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.pause(0.01)

plt.show()

plt.savefig("final_result.png", dpi=200, bbox_inches="tight")