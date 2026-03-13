"""Tests for the iterative solver (coleman_coalitions.solver)."""
import copy

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import coleman_coalitions as cc
from coleman_coalitions.solver import SolverConvergenceError


class TestSolverOutputShapes:
    def test_trad12_shapes(self, trad12_full):
        n, q, m = trad12_full['n'], trad12_full['q'], trad12_full['m']
        assert trad12_full['r'].shape == (n,)
        assert trad12_full['v'].shape == (q,)
        assert trad12_full['w'].shape == (m,)

    def test_techno_shapes(self, techno_full):
        n, q, m = techno_full['n'], techno_full['q'], techno_full['m']
        assert techno_full['r'].shape == (n,)
        assert techno_full['v'].shape == (q,)
        assert techno_full['w'].shape == (m,)


class TestSolverOutputValues:
    def test_r_sums_to_one_trad12(self, trad12_full):
        np.testing.assert_allclose(np.sum(trad12_full['r']), 1.0, atol=1e-6)

    def test_v_sums_to_one_trad12(self, trad12_full):
        np.testing.assert_allclose(np.sum(trad12_full['v']), 1.0, atol=1e-6)

    def test_r_all_non_negative_trad12(self, trad12_full):
        assert np.all(trad12_full['r'] >= 0)

    def test_v_all_non_negative_trad12(self, trad12_full):
        assert np.all(trad12_full['v'] >= 0)

    def test_w_sums_to_one_techno(self, techno_full):
        np.testing.assert_allclose(np.sum(techno_full['w']), 1.0, atol=1e-6)

    def test_w_all_non_negative_techno(self, techno_full):
        assert np.all(techno_full['w'] >= 0)

    def test_trad12_w_equals_ones(self, trad12_full):
        """In identity-a (trad) case resource values are trivially all equal to 1."""
        np.testing.assert_allclose(trad12_full['w'], np.ones(trad12_full['m']), atol=1e-6)


class TestSolverStability:
    def test_trad12_reproducible_with_same_seed(self, trad12_inputs):
        """Same seed → same result."""
        for seed in (0, 42, 99):
            np.random.seed(seed)
            r1 = cc.solve(copy.deepcopy(trad12_inputs))['r']
            np.random.seed(seed)
            r2 = cc.solve(copy.deepcopy(trad12_inputs))['r']
            np.testing.assert_array_equal(r1, r2)

    def test_trad12_converges_to_same_fixed_point(self, trad12_inputs):
        """Different seeds → same equilibrium (unique fixed point)."""
        results = []
        for seed in (0, 1, 2, 7, 42):
            np.random.seed(seed)
            r = cc.solve(copy.deepcopy(trad12_inputs))['r']
            results.append(r)
        for r in results[1:]:
            np.testing.assert_allclose(r, results[0], atol=1e-4,
                err_msg="Solver converges to different fixed points from different seeds")

    def test_techno_converges_to_same_fixed_point(self, techno_inputs):
        results = []
        for seed in (0, 1, 42):
            np.random.seed(seed)
            r = cc.solve(copy.deepcopy(techno_inputs))['r']
            results.append(r)
        for r in results[1:]:
            np.testing.assert_allclose(r, results[0], atol=1e-4)


class TestSolverConvergenceError:
    def test_raises_on_max_iter_exceeded(self):
        """Solver should raise SolverConvergenceError when max_iter is too tight."""
        inputs = cc.standard_variables_techno()
        with pytest.raises(SolverConvergenceError):
            cc.solve(inputs, max_iter=1, tol=1e-15)


class TestSolverKnownStructure:
    def test_trad12_actors_have_distinct_power(self, trad12_full):
        """The trad12 interest/control structure produces distinct actor powers."""
        r = trad12_full['r']
        assert len(set(np.round(r, 3))) > 1, "All actors identical power — unexpected"

    def test_techno2_all_actors_have_positive_power(self, techno2_full):
        """All actors control at least one resource, so all should have positive power."""
        assert np.all(techno2_full['r'] > 0)
        np.testing.assert_allclose(np.sum(techno2_full['r']), 1.0, atol=1e-4)
