"""Tests for visualization functions (coleman_coalitions.visualization) — headless."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import coleman_coalitions as cc


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close('all')


@pytest.fixture(scope="module")
def coalition_analysis(trad12_full):
    coalitions = cc.feasible_coalitions(trad12_full)
    outputs = cc.coalition_trad(trad12_full, coalitions)
    # Append full-collective entry required by optimal_coalition
    full_key = str(list(range(trad12_full['n'])))
    if full_key not in outputs:
        outputs[full_key] = trad12_full
    summary, TPM = cc.optimal_coalition(outputs)
    return outputs, summary, TPM


class TestDrawCoalitionMap:
    def test_returns_figure(self, coalition_analysis):
        outputs, summary, TPM = coalition_analysis
        fig = cc.draw_coalition_map(outputs, summary, TPM)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_does_not_raise(self, coalition_analysis):
        outputs, summary, TPM = coalition_analysis
        cc.draw_coalition_map(outputs, summary, TPM)


class TestDrawOnlyStrongest:
    def test_returns_figure(self, coalition_analysis):
        outputs, summary, TPM = coalition_analysis
        fig = cc.draw_strongest_transitions(outputs, summary, TPM)
        assert isinstance(fig, matplotlib.figure.Figure)
