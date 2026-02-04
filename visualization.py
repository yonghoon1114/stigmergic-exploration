import numpy as np
import matplotlib.pyplot as plt
import random
from agents import Agent
import time

WIDTH = 100
HEIGHT = 100
STEPS = 200
DECAY = 1
N = 10   # agent count 고정

def entropy(potential):
    p = potential.flatten()
    p = p / np.sum(p)
    return -np.sum(p * np.log(p + 1e-10))

def run_sim(alpha, seed=0):

    random.seed(seed)
    np.random.seed(seed)

    potential = np.zeros((HEIGHT, WIDTH)) + 1e-6

    agents = [
        Agent(
            random.randint(0, HEIGHT - 1),
            random.randint(0, WIDTH - 1),
            HEIGHT,
            WIDTH,
            alpha
        )
        for _ in range(N)
    ]

    for t in range(STEPS):
        potential *= DECAY
        for agent in agents:
            agent.step(potential, agents)

    return entropy(potential)

# =======================
# alpha sweep
# =======================

alphas = np.linspace(0.5, 300, 25)
Hs = []

for a in alphas:
    vals = []
    for s in range(3):
        vals.append(run_sim(a,seed=s))
    Hs.append(np.mean(vals))

# =======================
# plot
# =======================

plt.plot(alphas, Hs, marker="o")
plt.xlabel("Alpha")
plt.ylabel("Entropy")
plt.title("Entropy vs Alpha")

plt.tight_layout()
plt.savefig("entropy_vs_alpha.png", dpi=200)
plt.close()
