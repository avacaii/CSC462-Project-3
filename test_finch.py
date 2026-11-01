# Name: Ava Cai
# Date: 11/3/26
# Assignment: Project 3
# Purpose of the program: Tests for the finch speciation model.
# AI DECLARATION: AI was not used in the creation of this program.

import numpy as np
from pytest import approx # approx() allows tiny rounding differences when comparing decimals

from Finch_Model import P0, H, GENE_DIV_THRESHOLD, select, drift, gene_div, run

# Run with: .venv/bin/python -m pytest
# pytest finds this file because its name starts with test_, and runs every
# function whose name starts with test_
# Learned this from: https://www.geeksforgeeks.org/python/getting-started-with-pytest/ 

# Test select() 
def test_select_no_selection_no_change():
    # s = 0 means every genotype has the same fitness, so p stays the same
    # assert passes if the condition is True; if it is False, pytest marks the test as failed
    assert select(0.5, 0, H, favor_L=True) == approx(0.5)
    assert select(0.5, 0, H, favor_L=False) == approx(0.5)

def test_select_hand_calculated_value():
    # p = 0.5, s = 0.5, h = 0.5, favoring L:
    # fitness LL = 1, LS = 0.75, SS = 0.5
    # w_bar = 0.25*1 + 0.5*0.75 + 0.25*0.5 = 0.75
    # new p = (0.25*1 + 0.25*0.75) / 0.75 = 0.5833...
    assert select(0.5, 0.5, 0.5, favor_L=True) == approx(0.4375 / 0.75)

def test_select_direction():
    # Population A (favors L) gains L, population B (favors S) loses L
    assert select(0.5, 0.1, H, favor_L=True) > 0.5
    assert select(0.5, 0.1, H, favor_L=False) < 0.5

def test_select_fixed_allele_stays_fixed():
    # If L is already at 0 or 1, selection has nothing to act on
    assert select(0.0, 0.5, H, favor_L=True) == approx(0.0)
    assert select(1.0, 0.5, H, favor_L=False) == approx(1.0)


# Test drift()
def test_drift_result_is_valid_frequency():
    # Always between 0 and 1, and a whole number of copies out of 2n
    np.random.seed(1) # fixed random seed 
    n = 50
    # Check many draws, since one draw could pass by luck
    for _ in range(100):
        p = drift(0.5, n)
        assert 0 <= p <= 1
        # p * 2n is the number of L copies, which has to be a whole number
        assert p * 2 * n == approx(round(p * 2 * n))

def test_drift_fixed_allele_stays_fixed():
    # With no L (or all L) there is nothing to drift
    assert drift(0.0, 50) == 0.0
    assert drift(1.0, 50) == 1.0


# Test gene_div()
def test_gene_div_identical_populations():
    # Same L frequency in both populations means no divergence
    assert gene_div(0.4, 0.4) == 0.0

def test_gene_div_completely_different():
    # One all L, the other all S
    assert gene_div(1.0, 0.0) == approx(1.0)

def test_gene_div_both_fixed_same_allele():
    # p_bar is 0 or 1, which would divide by 0
    assert gene_div(0.0, 0.0) == 0.0
    assert gene_div(1.0, 1.0) == 0.0

def test_gene_div_hand_calculated_value():
    # p_bar = 0.5, denom = 4*0.5*0.5 = 1, (0.75 - 0.25)^2 = 0.25
    assert gene_div(0.75, 0.25) == approx(0.25)

def test_gene_div_order_does_not_matter():
    # Swapping population A and B should give the same divergence
    assert gene_div(0.8, 0.1) == approx(gene_div(0.1, 0.8))


# Test run()
def test_run_starting_row():
    # Generation 0: both populations at P0, no divergence yet
    np.random.seed(1)
    _, history = run(0.3)
    # history[0] is the first row: generation, L in A, L in B, divergence
    assert list(history[0]) == [0, P0, P0, 0.0]

def test_run_strong_selection_diverges():
    # Strong selection should cross the threshold, and stop right there
    np.random.seed(1)
    gen, history = run(0.8)
    assert not np.isnan(gen) # it did cross (np.nan would mean it never did)
    # history[-1] is the last row; column 0 is the generation, column 3 the divergence
    assert history[-1, 0] == gen
    assert history[-1, 3] >= GENE_DIV_THRESHOLD
    # Every generation before the last was still below the threshold
    # history[:-1, 3] is the divergence column for every row except the last
    assert np.all(history[:-1, 3] < GENE_DIV_THRESHOLD)

def test_run_one_row_per_generation():
    # Generations go 0, 1, 2, ... with no gaps
    np.random.seed(1)
    gen, history = run(0.3)
    # np.arange(gen + 1) is 0, 1, ..., gen; compare it to the generation column
    assert np.array_equal(history[:, 0], np.arange(gen + 1))