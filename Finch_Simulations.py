# Name: Ava Cai
# Date: 11/3/26
# Assignment: Project 3
# Purpose of the program: Run the finch speciation model across selection strengths and population sizes.
# AI DECLARATION: AI was not used in the creation of this program.

import numpy as np
import matplotlib.pyplot as plt

from Finch_Model import S_VALUES, N, run

np.random.seed(1) # fixed random seed so the results are reproducible

N_VALUES = [10, 50, 200, 1000, 5000] # population sizes to sweep
S_FOR_N_SWEEP = 0.05 # s held fixed during the population size sweep
# Weak selection, so random drift has a chance to matter at small sizes

def sweep_selection_strength():
    """Run the model once per selection strength s, then graph and print a table."""
    # Two graphs side by side: L frequency on the left, divergence on the right
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Table header: generation it stopped at, then the final values from each graph
    print(f"\nSelection strength sweep (N = {N})")
    print(f"{'s':>6} {'generations':>12} {'L in A':>8} {'L in B':>8} {'divergence':>11}")

    # Run the model once for each s and plot both populations
    for s in S_VALUES:
        # history columns: generation, L frequency in A, L frequency in B, divergence
        # run() stops once the populations are diverged enough and returns that
        # generation (or np.nan if they never got there)
        gen, history = run(s) # population size stays at the default N

        # Same color for both populations at this s: solid = A, dashed = B
        # Population A gets the next color from matplotlib's default color cycle
        line, = ax1.plot(history[:, 0], history[:, 1], label=f"s = {s}")
        # Population B reuses A's color; no label so each s appears once in the legend
        # Learned this from: https://www.geeksforgeeks.org/python/matplotlib-pyplot-colors-in-python/
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

    fig.suptitle(f"Selection strength sweep (N = {N})") # title over both graphs, says which sweep this is
    fig.tight_layout() # keep the two graphs' labels from overlapping
    fig.savefig("finch_graphs_s.png", dpi=150) # save both graphs as one image

def sweep_population_size():
    """Run the model once per population size N, then graph and print a table."""
    # One graph: divergence over time
    fig, ax = plt.subplots(figsize=(7, 5))

    # Table header: generation it stopped at, then the final values when the run stopped
    print(f"\nPopulation size sweep (s = {S_FOR_N_SWEEP})")
    print(f"{'N':>6} {'generations':>12} {'L in A':>8} {'L in B':>8} {'divergence':>11}")

    # Run the model once for each n and plot the divergence
    for n in N_VALUES:
        # history columns: generation, L frequency in A, L frequency in B, divergence
        # run() stops once the populations are diverged enough and returns that
        # generation (or np.nan if they never got there)
        gen, history = run(S_FOR_N_SWEEP, n=n) # s stays fixed

        # Divergence between the two populations; each n gets the next default color
        ax.plot(history[:, 0], history[:, 3], label=f"N = {n}")

        # One table row per n, using the last row of history (where the run stopped)
        last = history[-1]
        print(f"{n:>6} {gen:>12.0f} {last[1]:>8.3f} {last[2]:>8.3f} {last[3]:>11.3f}")

    # Axis labels, title, and legend for the divergence graph
    ax.set_xlabel("Generation")
    ax.set_ylabel("Divergence")
    ax.set_title(f"Divergence between populations (s = {S_FOR_N_SWEEP})")
    ax.legend()

    fig.tight_layout() # keep labels from getting cut off
    fig.savefig("finch_graphs_N.png", dpi=150) # save the graph as an image

sweep_selection_strength()
sweep_population_size()
plt.show() # open both figures in windows
