# Name: Ava Cai
# Date: 11/3/26
# Assignment: Project 3
# Purpose of the program: Run the finch speciation model across selection strengths.
# AI DECLARATION: AI was not used in the creation of this program.

import numpy as np
import matplotlib.pyplot as plt

from Finch_Model import S_VALUES, run

np.random.seed(1) # fixed random seed so the results are reproducible

# Two graphs side by side: L frequency on the left, divergence on the right
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Table header: generation it stopped at, then the final values from each graph
print(f"{'s':>6} {'generations':>12} {'L in A':>8} {'L in B':>8} {'divergence':>11}")

# Run the model once for each selection strength and plot both populations
for s in S_VALUES:
    # history columns: generation, L frequency in A, L frequency in B, divergence
    # run() stops once the populations are diverged enough and returns that
    # generation (or np.nan if they never got there)
    gen, history = run(s)

    # Same color for both populations at this s: solid = A, dashed = B
    # Population A gets the next color from matplotlib's default color cycle
    line, = ax1.plot(history[:, 0], history[:, 1], label=f"s = {s}")
    # Population B reuses A's color; no label so each s appears once in the legend
    ax1.plot(history[:, 0], history[:, 2], color=line.get_color(), ls="--")

    # Divergence between the two populations, same color as this s on the left
    ax2.plot(history[:, 0], history[:, 3], color=line.get_color(), label=f"s = {s}")

    # One table row per s, using the last row of history (where the run stopped)
    last = history[-1]
    print(f"{s:>6} {gen:>12.0f} {last[1]:>8.3f} {last[2]:>8.3f} {last[3]:>11.3f}")

# Axis labels, title, and legend for the L frequency graph
ax1.set_xlabel("Generation")
ax1.set_ylabel("Frequency of L")
ax1.set_title("L frequency (solid = population A, dashed = population B)")
ax1.legend()

# Axis labels, title, and legend for the divergence graph
ax2.set_xlabel("Generation")
ax2.set_ylabel("Divergence")
ax2.set_title("Divergence between populations")
ax2.legend()

fig.tight_layout() # keep the two graphs' labels from overlapping
fig.savefig("finch_graphs.png", dpi=150) # save both graphs as one image
plt.show() # open the graphs in a window