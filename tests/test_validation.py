"""
tests/test_validation.py

External validation of coleman_coalitions using two independent strategies.

Strategy 1 — Mathematical invariants from Coleman (1973):
  Hard constraints that must hold at equilibrium regardless of implementation.
  These verify the code actually solves the correct mathematical problem.
  Examples:
    - r, v, w each sum to 1 (normalization)
    - v = normalize(r @ x) at convergence (equilibrium self-consistency)
    - C = a @ c, B = x @ a, z = B @ c (definitional equalities)
    - sum_j |c_AE[j,i]| = 1 for every event i (column normalization)
    - P_p in [0, 1] (probability bounds)
    - w = ones and C = c for identity-a (trad) case

Strategy 2 — Cross-implementation comparison (loop vs vectorised):
  Both deprecated/ColemanFunctions.py (explicit nested-loop reference, Coleman
  1973) and the new vectorised coleman_coalitions package are run on the same
  inputs with the same random seed.  Every derived variable that does not depend
  on p_h is compared numerically at atol=1e-8.

  Note on p_h / d_i: ColemanFunctions.ExpectedValueOfCollectivity sums p_hj
  along axis=0 (column sum: "how much all actors benefit from j") rather than
  axis=1 (row sum: "how much actor h benefits from all j").  The row-sum
  interpretation is consistent with both the formula stated in Coleman (1973)
  and the function's own docstring.  p_h and d_i are therefore NOT included
  in the cross-implementation comparison; they are validated via Strategy 1
  (the formula tests in test_analysis.py).
"""
import copy

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deprecated'))

import coleman_coalitions as cc
import ColemanFunctions as cf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_new(example_fn: callable, seed: int = 42) -> dict:
    """Run the new package on a named example function; return the full outputs dict."""
    np.random.seed(seed)
    inputs = example_fn()
    cc.run_full_analysis(inputs)
    return inputs


def _run_old(old_example_fn: callable, seed: int = 42) -> dict:
    """Run the deprecated package; unwrap [value, description] pairs into plain values."""
    np.random.seed(seed)
    raw = old_example_fn()
    raw.update(cf.RunFullAnalysis(raw))
    return {k: (v[0] if isinstance(v, list) else v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trad12_new():
    return _run_new(cc.standard_variables_trad12)


@pytest.fixture(scope="module")
def techno_new():
    return _run_new(cc.standard_variables_techno)


@pytest.fixture(scope="module")
def techno2_new():
    return _run_new(cc.standard_variables_techno2)


@pytest.fixture(scope="module")
def trad12_old():
    return _run_old(cf.standard_variables_trad12)


@pytest.fixture(scope="module")
def techno_old():
    return _run_old(cf.standard_variables_techno)


# ---------------------------------------------------------------------------
# Strategy 1: Mathematical invariants
# ---------------------------------------------------------------------------

class TestEquilibriumConditionsTrad12:
    """Trad (identity-a) branch: structural invariants that follow from Coleman's
    equilibrium definition when a = I."""

    def test_w_equals_ones(self, trad12_new):
        """When a is the identity, every resource maps to exactly one event, so
        resource values are all equal (normalised to 1 each)."""
        np.testing.assert_allclose(
            trad12_new['w'], np.ones(trad12_new['m']), atol=1e-6)

    def test_C_equals_c_when_a_is_identity(self, trad12_new):
        """C = a @ c = I @ c = c when a is the identity matrix."""
        np.testing.assert_allclose(trad12_new['C'], trad12_new['c'], atol=1e-10)

    def test_v_self_consistent_at_equilibrium(self, trad12_new):
        """At convergence: v = normalize(r @ x) must hold (solver update rule)."""
        r, x = trad12_new['r'], trad12_new['x']
        v_check = r @ x
        v_check /= np.sum(v_check)
        np.testing.assert_allclose(trad12_new['v'], v_check, atol=1e-5)

    def test_r_self_consistent_at_equilibrium(self, trad12_new):
        """At convergence: r = v @ c (unnormalized), which sums to 1 because c is
        row-normalised and v is a probability vector."""
        r_check = trad12_new['v'] @ trad12_new['c']
        np.testing.assert_allclose(trad12_new['r'], r_check, atol=1e-5)


class TestEquilibriumConditionsTechno:
    """Full (non-identity-a) branch: equilibrium self-consistency checks."""

    def test_v_self_consistent_at_equilibrium(self, techno_new):
        """At convergence: normalize(r @ x) = v."""
        r, x = techno_new['r'], techno_new['x']
        v_check = r @ x
        v_check /= np.sum(v_check)
        np.testing.assert_allclose(techno_new['v'], v_check, atol=1e-4)

    def test_w_self_consistent_at_equilibrium(self, techno_new):
        """At convergence: normalize(v @ a) = w."""
        v, a = techno_new['v'], techno_new['a']
        w_check = v @ a
        w_check /= np.sum(w_check)
        np.testing.assert_allclose(techno_new['w'], w_check, atol=1e-4)

    def test_r_self_consistent_at_equilibrium(self, techno_new):
        """At convergence: normalize(w @ c) = r."""
        w, c = techno_new['w'], techno_new['c']
        r_check = w @ c
        r_check /= np.sum(r_check)
        np.testing.assert_allclose(techno_new['r'], r_check, atol=1e-4)


class TestDefinitionalEqualities:
    """Variables defined as explicit matrix products must equal those products exactly."""

    def test_C_is_a_at_c(self, techno_new):
        np.testing.assert_allclose(
            techno_new['C'], techno_new['a'] @ techno_new['c'], atol=1e-12)

    def test_B_is_x_at_a(self, trad12_new):
        np.testing.assert_allclose(
            trad12_new['B'], trad12_new['x'] @ trad12_new['a'], atol=1e-12)

    def test_z_is_B_at_c(self, trad12_new):
        np.testing.assert_allclose(
            trad12_new['z'], trad12_new['B'] @ trad12_new['c'], atol=1e-12)

    def test_P_p_formula(self, trad12_new):
        expected = 0.5 + 0.5 * np.sum(trad12_new['c_AE'], axis=0)
        np.testing.assert_allclose(trad12_new['P_p'], expected, atol=1e-12)

    def test_d_formula(self, trad12_new):
        expected = trad12_new['r'] @ trad12_new['y']
        np.testing.assert_allclose(trad12_new['d'], expected, atol=1e-12)

    def test_R_formula(self, trad12_new):
        np.testing.assert_allclose(
            trad12_new['R'], np.sum(np.abs(trad12_new['d'])), atol=1e-12)

    def test_p_h_is_row_sum_of_p_hj(self, trad12_new):
        """p_h[h] = sum_j p_hj[h, j] + 0.5.  Summing over axis=1 (over sources j)
        gives how much actor h benefits from the collectivity, consistent with
        Coleman (1973)."""
        expected = np.sum(trad12_new['p_hj'], axis=1) + 0.5
        np.testing.assert_allclose(trad12_new['p_h'], expected, atol=1e-12)


class TestColumnNormalizationInvariant:
    """sum_j |c_AE[j,i]| = 1 for every event i.

    Derivation: c_AE[j,i] = y[j,i] * r[j] / v[i].
    Summing absolute values over j: sum_j |y[j,i]| * r[j] / v[i].
    At equilibrium v[i] = sum_j |y[j,i]| * r[j], so the ratio equals 1.
    """

    def test_trad12(self, trad12_new):
        np.testing.assert_allclose(
            np.sum(np.abs(trad12_new['c_AE']), axis=0), 1.0, atol=1e-5)

    def test_techno(self, techno_new):
        np.testing.assert_allclose(
            np.sum(np.abs(techno_new['c_AE']), axis=0), 1.0, atol=1e-5)

    def test_techno2(self, techno2_new):
        np.testing.assert_allclose(
            np.sum(np.abs(techno2_new['c_AE']), axis=0), 1.0, atol=1e-5)


class TestSimpleHandCheckableCases:
    """Two-actor, two-event system whose equilibrium can be solved analytically.

    Setup (identity a, so trad branch):
      y = [[1, 0], [0, 1]]   — actor 0 cares only about event 0, actor 1 only about event 1
      a = I                  — resources map one-to-one to events
      c = [[0.6, 0.4],       — resource 0: actor 0 holds 60%, actor 1 holds 40%
           [0.3, 0.7]]       — resource 1: actor 0 holds 30%, actor 1 holds 70%

    Analytical solution:
      At equilibrium v = r (since x = y = I means r @ x = r, normalized → r).
      Equilibrium condition r = v @ c = r @ c:
        r0 = 0.6 r0 + 0.3 r1  →  0.4 r0 = 0.3 r1  →  r0/r1 = 3/4
      With r0 + r1 = 1:  r = [3/7, 4/7]  ≈  [0.4286, 0.5714].

      c_AE[j,i] = y[j,i] * r[j] / v[i].  Because y is the identity and v = r:
        c_AE = I  (each actor fully controls their own event).
      Therefore P_p = 0.5 + 0.5 * [1, 1] = [1, 1].
    """

    R0 = 3.0 / 7.0
    R1 = 4.0 / 7.0

    @pytest.fixture(scope="class")
    def analytic_2x2(self):
        y = np.array([[1.0, 0.0], [0.0, 1.0]])
        a = np.eye(2)
        c = np.array([[0.6, 0.4], [0.3, 0.7]])
        inputs = cc.setup(2, 2, 2, y, a, c)
        np.random.seed(0)
        cc.run_full_analysis(inputs)
        return inputs

    def test_actor_power(self, analytic_2x2):
        np.testing.assert_allclose(
            analytic_2x2['r'], [self.R0, self.R1], atol=1e-6)

    def test_event_value(self, analytic_2x2):
        np.testing.assert_allclose(
            analytic_2x2['v'], [self.R0, self.R1], atol=1e-6)

    def test_P_p_equals_one(self, analytic_2x2):
        """Each actor exclusively controls and fully supports their event → P_p = 1."""
        np.testing.assert_allclose(analytic_2x2['P_p'], [1.0, 1.0], atol=1e-6)

    def test_constitutional_control_equals_c(self, analytic_2x2):
        """C = a @ c = I @ c = c."""
        np.testing.assert_allclose(analytic_2x2['C'], analytic_2x2['c'], atol=1e-10)


# ---------------------------------------------------------------------------
# Strategy 2: Cross-implementation comparison (loop vs vectorised)
# ---------------------------------------------------------------------------

class TestCrossImplementationTrad12:
    """Compare every derived variable between the deprecated loop implementation
    and the new vectorised implementation, using the trad12 example with the
    same random seed.  Variables that depend on p_h are excluded (see module
    docstring for the rationale)."""

    SEED = 42

    @pytest.fixture(scope="class")
    def both(self):
        new = _run_new(cc.standard_variables_trad12, seed=self.SEED)
        old = _run_old(cf.standard_variables_trad12, seed=self.SEED)
        return new, old

    def test_r(self, both):
        new, old = both
        np.testing.assert_allclose(new['r'], old['r'], atol=1e-8,
            err_msg="Actor power r differs between loop and vectorised implementations")

    def test_v(self, both):
        new, old = both
        np.testing.assert_allclose(new['v'], old['v'], atol=1e-8,
            err_msg="Event value v differs between loop and vectorised implementations")

    def test_w(self, both):
        new, old = both
        np.testing.assert_allclose(new['w'], old['w'], atol=1e-8,
            err_msg="Resource value w differs")

    def test_C(self, both):
        new, old = both
        np.testing.assert_allclose(new['C'], old['C'], atol=1e-8,
            err_msg="Constitutional control C differs")

    def test_c_AE(self, both):
        new, old = both
        np.testing.assert_allclose(new['c_AE'], old['c_AE'], atol=1e-8,
            err_msg="Actor-event control c_AE differs")

    def test_B(self, both):
        new, old = both
        np.testing.assert_allclose(new['B'], old['B'], atol=1e-8,
            err_msg="Derived interest B differs")

    def test_c_AR(self, both):
        new, old = both
        np.testing.assert_allclose(new['c_AR'], old['c_AR'], atol=1e-8,
            err_msg="Actor-resource control c_AR differs")

    def test_z(self, both):
        new, old = both
        np.testing.assert_allclose(new['z'], old['z'], atol=1e-8,
            err_msg="Actor-actor control z differs")

    def test_P_p(self, both):
        new, old = both
        np.testing.assert_allclose(new['P_p'], old['P_p'], atol=1e-8,
            err_msg="Positive-outcome probability P_p differs")

    def test_d(self, both):
        new, old = both
        np.testing.assert_allclose(new['d'], old['d'], atol=1e-8,
            err_msg="Directed power d differs")

    def test_R(self, both):
        new, old = both
        np.testing.assert_allclose(new['R'], old['R'], atol=1e-8,
            err_msg="Total power R differs")

    def test_p_hj(self, both):
        new, old = both
        np.testing.assert_allclose(new['p_hj'], old['p_hj'], atol=1e-8,
            err_msg="Increment matrix p_hj differs")


class TestCrossImplementationTechno:
    """Same cross-validation for the techno (full-branch) example."""

    SEED = 42

    @pytest.fixture(scope="class")
    def both(self):
        new = _run_new(cc.standard_variables_techno, seed=self.SEED)
        old = _run_old(cf.standard_variables_techno, seed=self.SEED)
        return new, old

    def test_r(self, both):
        new, old = both
        np.testing.assert_allclose(new['r'], old['r'], atol=1e-8)

    def test_v(self, both):
        new, old = both
        np.testing.assert_allclose(new['v'], old['v'], atol=1e-8)

    def test_w(self, both):
        new, old = both
        np.testing.assert_allclose(new['w'], old['w'], atol=1e-8)

    def test_C(self, both):
        new, old = both
        np.testing.assert_allclose(new['C'], old['C'], atol=1e-8)

    def test_F(self, both):
        new, old = both
        np.testing.assert_allclose(new['F'], old['F'], atol=1e-8)

    def test_c_AE(self, both):
        new, old = both
        np.testing.assert_allclose(new['c_AE'], old['c_AE'], atol=1e-8)

    def test_B(self, both):
        new, old = both
        np.testing.assert_allclose(new['B'], old['B'], atol=1e-8)

    def test_c_AR(self, both):
        new, old = both
        np.testing.assert_allclose(new['c_AR'], old['c_AR'], atol=1e-8)

    def test_z(self, both):
        new, old = both
        np.testing.assert_allclose(new['z'], old['z'], atol=1e-8)

    def test_P_p(self, both):
        new, old = both
        np.testing.assert_allclose(new['P_p'], old['P_p'], atol=1e-8)

    def test_d(self, both):
        new, old = both
        np.testing.assert_allclose(new['d'], old['d'], atol=1e-8)

    def test_R(self, both):
        new, old = both
        np.testing.assert_allclose(new['R'], old['R'], atol=1e-8)
