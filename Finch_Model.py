# Name: Ava Cai
# Date: 11/3/26
# Assignment: Project 3
# Purpose of the program: Model of the Speciation of Finches
# AI DECLARATION: AI was not used in the creation of this program.

import numpy as np 

# Parameters
S_VALUES = [0.01, 0.05, 0.1, 0.3, 0.59, 0.8] # selection strengths to sweep
# 0.59 is the measured value from the 2004-05 drought, described as exceptionally strong.
# Learned this from: https://omia.org/OMIA001992/48881/
# The other values are chosen to cover weak to strong selection.

H = 0.5 # dominance (0.5 = heterozygotes intermediate,
# if has both the large and small allele will have medium sized beak)
# Learned this from; https://www.nature.com/scitable/topicpage/genetic-dominance-genotype-phenotype-relationships-489/

P0 = 0.5 # starting frequency of L (large-beak allele) in both populations; other allele is S (small-beak allele)

N = 200 # breeding individuals per population

GENE_DIV_THRESHOLD = 0.9 # value at which the populations count as very different

REPLICATES = 200 # runs per s value (drift makes each run different)

MAX_GENERATIONS = 5000 # cap the max number of generations that the model runs

def select(p, s, h, favor_L):
    """One generation of selection. Returns the L frequency after selection."""
    # Learned this from: 
    # https://nij.ojp.gov/nij-hosted-online-training-courses/population-genetics-and-
    # statistics-forensic-analysts/population-theory/hardy-weinberg-principle
    q = 1 - p # frequency of S (small-beak allele); p + q = 1

    # Fitness of each genotype (1.0 = best, smaller = fewer surviving offspring).
    # The heterozygote LS loses h * s, so with h = 0.5 it is halfway between.
    if favor_L: # Population A: selection against S
        w_LL, w_LS, w_SS = 1.0, 1 - h * s, 1 - s
    else: # Population B: selection against L
        w_LL, w_LS, w_SS = 1 - s, 1 - h * s, 1.0

    # Mean fitness: each genotype's fitness weighted by its Hardy-Weinberg
    # frequency (LL = p^2, LS = 2pq, SS = q^2)
    # Learned this from: 
    # https://nij.ojp.gov/nij-hosted-online-training-courses/population-genetics-and-
    # statistics-forensic-analysts/population-theory/hardy-weinberg-principle
    w_bar = p**2 * w_LL + 2 * p * q * w_LS + q**2 * w_SS

    # New L frequency = share of L copies among survivors.
    # LL carries two L copies (p^2), LS carries one (half of 2pq = pq).
    # Dividing by w_bar rescales so the frequencies still add to 1.
    return (p**2 * w_LL + p * q * w_LS) / w_bar


def drift(p, n):
    """Binomial sampling of 2N gene copies."""
    # Diploid birds, n individuals carry 2n copies of the gene.
    # Randomly draw how many of those copies are L (each is L with chance p),
    # then convert the count back to a frequency. This random chance is
    # genetic drift, so small populations wander more.
    # Learned to use this from: https://www.geeksforgeeks.org/numpy/binomial-distribution-in-numpy/
    return np.random.binomial(2 * n, p) / (2 * n)

def gene_div(p_a, p_b):
    """Calculates the genetic divergence for two equal-sized populations at one gene locus."""
    # Learned this from: https://en.wikipedia.org/wiki/Fixation_index  
    p_bar = (p_a + p_b) / 2 # average L frequency across both populations
    denom = 4 * p_bar * (1 - p_bar) # scales the result to the 0-1 range
    # Squared difference between the populations, scaled. 
    # 0 = identical, 1 = completely different (one all L, the other all S).
    # If p_bar is 0 or 1 both populations are the same, return 0 instead of dividing by 0.
    return 0.0 if denom == 0 else (p_a - p_b) ** 2 / denom

def run(s, n=N):
    """
    Simulate until genetic divergence value crosses the threshold.
    Returns the generation it crossed (or np.nan if never) and the trajectory.
    """
    p_a = p_b = P0 # both populations start with the same L frequency
    history = [(0, p_a, p_b, gene_div(p_a, p_b))] # history: (generation, p_a, p_b, divergence)

    # Run simulation until we reach max generation cap 
    for gen in range(1, MAX_GENERATIONS + 1):
        # Gene selection first, then drift from random breeding.
        # Population A favors large beaks, population B favors small beaks.
        p_a = drift(select(p_a, s, H, favor_L=True), n)
        p_b = drift(select(p_b, s, H, favor_L=False), n)
        diff = gene_div(p_a, p_b) # how different the two populations are now

        # record each generation and the genetic frequencies
        history.append((gen, p_a, p_b, diff))

        # populations are diverged enough, stop here
        if diff >= GENE_DIV_THRESHOLD: 
            return gen, np.array(history)
        
    # Return this if never crossed the threshold within MAX_GENERATIONS
    # np.nan means "Not a Number", constant representing undefined or unrepresentable numerical data 
    # Learned this from: https://www.geeksforgeeks.org/python/check-for-nan-values-in-python/
    return np.nan, np.array(history) 