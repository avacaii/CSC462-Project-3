# Name: Ava Cai
# Date: 11/3/26
# Assignment: Project 3
# Purpose of the program: Model of the Speciation of Finches
# AI DECLARATION: AI was not used in the creation of this program.

import numpy as np 
import matplotlib.pyplot as plt 

# Parameters
S_VALUES = [0.01, 0.05, 0.1, 0.3, 0.59, 0.7] # selection strengths to sweep
# 0.59 is the measured value from the 2004-05 drought (Lamichhaney et al. 2016),
# described as exceptionally strong.
# The other values are chosen to cover weak to strong selection.

H = 0.5 # dominance (0.5 = heterozygotes intermediate,
# if has both the large and small allele will have medium sized beak)

P0 = 0.5 # starting frequency of L (large-beak allele) in both populations

N = 200 # breeding individuals per population

GENE_DIV_THRESHOLD = 0.9 # value at which the populations count as very different

REPLICATES = 200 # runs per s value (drift makes each run different)

MAX_GENERATIONS = 5000 # cap the max number of generations that the model runs

SEED = 1 # fixed random seed so the results are reproducible


def select(p, s, h, favor_L):
    """One generation of selection. Returns the L frequency after selection."""
    q = 1 - p # frequency of S (small-beak allele); p + q = 1

    # Fitness of each genotype (1.0 = best, smaller = fewer surviving offspring).
    # The heterozygote LS loses h * s, so with h = 0.5 it is halfway between.
    if favor_L: # Population A: selection against S
        w_LL, w_LS, w_SS = 1.0, 1 - h * s, 1 - s
    else: # Population B: selection against L
        w_LL, w_LS, w_SS = 1 - s, 1 - h * s, 1.0

    # Mean fitness: each genotype's fitness weighted by its Hardy-Weinberg
    # frequency (LL = p^2, LS = 2pq, SS = q^2)
    w_bar = p**2 * w_LL + 2 * p * q * w_LS + q**2 * w_SS

    # New L frequency = share of L copies among survivors.
    # LL carries two L copies (p^2), LS carries one (half of 2pq = pq).
    # Dividing by w_bar rescales so the frequencies still add to 1.
    return (p**2 * w_LL + p * q * w_LS) / w_bar


def drift(p, n, rng):
    """Binomial sampling of 2N gene copies."""
    # Diploid birds -> n individuals carry 2n copies of the gene.
    # Randomly draw how many of those copies are L (each is L with chance p),
    # then convert the count back to a frequency. This random chance is
    # genetic drift, so small populations wander more.
    return rng.binomial(2 * n, p) / (2 * n)

def gene_div(p_a, p_b):
    """Calculates the genetic divergence for two equal-sized populations at one biallelic locus."""
    p_bar = (p_a + p_b) / 2 # average L frequency across both populations
    denom = 4 * p_bar * (1 - p_bar) # scales the result to the 0-1 range
    # Squared difference between the populations, scaled. 0 = identical,
    # 1 = completely different (one all L, the other all S).
    # If p_bar is 0 or 1 both populations are the same, so return 0
    # instead of dividing by zero.
    return 0.0 if denom == 0 else (p_a - p_b) ** 2 / denom

def run(s, rng, n=N, record=False):
    """Simulate until genetic dvergence value crosses the threshold.

    Returns the generation it crossed (or np.nan if never),
    and the trajectory if record=True.
    """
    p_a = p_b = P0 # both populations start with the same L frequency
    traj = [(0, p_a, p_b, gene_div(p_a, p_b))] # history: (generation, p_a, p_b, divergence)
    for gen in range(1, MAX_GENERATIONS + 1):
        # Each generation: selection first, then drift from random breeding.
        # Population A favors large beaks, population B favors small beaks.
        p_a = drift(select(p_a, s, H, favor_L=True), n, rng)
        p_b = drift(select(p_b, s, H, favor_L=False), n, rng)
        f = gene_div(p_a, p_b) # how different the two populations are now
        if record: # only save every generation when we want to plot it
            traj.append((gen, p_a, p_b, f))
        if f >= GENE_DIV_THRESHOLD: # populations are diverged enough, stop here
            return gen, np.array(traj)
    return np.nan, np.array(traj) # never crossed the threshold within MAX_GENERATIONS

