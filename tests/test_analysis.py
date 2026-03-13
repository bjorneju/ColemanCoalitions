"""Tests for derived-variable analysis functions (coleman_coalitions.analysis)."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import coleman_coalitions as cc


class TestConstitutionalControl:
    def test_shape(self, trad12_full):
        q, n = trad12_full['q'], trad12_full['n']
        assert trad12_full['C'].shape == (q, n)

    def test_equals_a_at_c(self, trad12_full):
        """C is defined as a @ c; verify the matrix product holds exactly."""
        np.testing.assert_allclose(trad12_full['C'], trad12_full['a'] @ trad12_full['c'], atol=1e-12)

    def test_techno_shape(self, techno_full):
        q, n = techno_full['q'], techno_full['n']
        assert techno_full['C'].shape == (q, n)


class TestFractionOfResources:
    def test_shape(self, techno_full):
        m, q = techno_full['m'], techno_full['q']
        assert techno_full['F'].shape == (m, q)

    def test_values_non_negative(self, techno_full):
        assert np.all(techno_full['F'] >= -1e-10)


class TestControlActorEvent:
    def test_shape(self, trad12_full):
        n, q = trad12_full['n'], trad12_full['q']
        assert trad12_full['c_AE'].shape == (n, q)

    def test_column_abs_sums_to_one(self, trad12_full):
        """sum_j |c_AE[j,i]| = 1 for every event i (normalization identity).

        Proof: sum_j |y[j,i]| * r[j] / v[i] = v[i] / v[i] = 1, because
        v[i] = sum_j |y[j,i]| * r[j] at equilibrium.
        """
        col_abs_sums = np.sum(np.abs(trad12_full['c_AE']), axis=0)
        np.testing.assert_allclose(col_abs_sums, 1.0, atol=1e-5)

    def test_techno_column_abs_sums_to_one(self, techno_full):
        col_abs_sums = np.sum(np.abs(techno_full['c_AE']), axis=0)
        np.testing.assert_allclose(col_abs_sums, 1.0, atol=1e-5)


class TestDerivedInterest:
    def test_shape(self, techno_full):
        n, m = techno_full['n'], techno_full['m']
        assert techno_full['B'].shape == (n, m)

    def test_equals_x_at_a(self, techno_full):
        """B is defined as x @ a; verify the matrix product holds exactly."""
        np.testing.assert_allclose(
            techno_full['B'], techno_full['x'] @ techno_full['a'], atol=1e-12)

    def test_rows_sum_to_one(self, techno_full):
        """Each actor's derived interest distributes fully across resources."""
        np.testing.assert_allclose(
            np.sum(techno_full['B'], axis=1), np.ones(techno_full['n']), atol=1e-6)


class TestControlActorActor:
    def test_shape(self, trad12_full):
        n = trad12_full['n']
        assert trad12_full['z'].shape == (n, n)

    def test_z_equals_c_AA(self, trad12_full):
        np.testing.assert_array_equal(trad12_full['z'], trad12_full['c_AA'])

    def test_equals_B_at_c(self, trad12_full):
        """z is defined as B @ c; verify the matrix product holds exactly."""
        np.testing.assert_allclose(
            trad12_full['z'], trad12_full['B'] @ trad12_full['c'], atol=1e-12)


class TestPositiveOutcome:
    def test_shape(self, trad12_full):
        assert trad12_full['P_p'].shape == (trad12_full['q'],)

    def test_values_in_zero_one(self, trad12_full):
        P_p = trad12_full['P_p']
        assert np.all(P_p >= 0.0 - 1e-6)
        assert np.all(P_p <= 1.0 + 1e-6)

    def test_techno_values_in_zero_one(self, techno_full):
        P_p = techno_full['P_p']
        assert np.all(P_p >= 0.0 - 1e-6)
        assert np.all(P_p <= 1.0 + 1e-6)

    def test_formula(self, trad12_full):
        """P_p[i] = 0.5 + 0.5 * sum_j c_AE[j, i]."""
        expected = 0.5 + 0.5 * np.sum(trad12_full['c_AE'], axis=0)
        np.testing.assert_allclose(trad12_full['P_p'], expected, atol=1e-12)


class TestDirectedAndTotalPower:
    def test_d_shape(self, trad12_full):
        assert trad12_full['d'].shape == (trad12_full['q'],)

    def test_d_formula(self, trad12_full):
        """d = r @ y by definition."""
        expected = trad12_full['r'] @ trad12_full['y']
        np.testing.assert_allclose(trad12_full['d'], expected, atol=1e-12)

    def test_R_is_scalar(self, trad12_full):
        assert isinstance(trad12_full['R'], float)

    def test_R_non_negative(self, trad12_full):
        assert trad12_full['R'] >= 0.0

    def test_R_bounded_by_one(self, trad12_full):
        assert trad12_full['R'] <= 1.0 + 1e-9

    def test_R_formula(self, trad12_full):
        """R = sum |d| by definition."""
        np.testing.assert_allclose(trad12_full['R'], np.sum(np.abs(trad12_full['d'])), atol=1e-12)


class TestExpectedValueFunctions:
    def test_p_hj_shape(self, trad12_full):
        n = trad12_full['n']
        assert trad12_full['p_hj'].shape == (n, n)

    def test_p_h_shape(self, trad12_full):
        assert trad12_full['p_h'].shape == (trad12_full['n'],)

    def test_p_h_formula(self, trad12_full):
        """p_h[h] = sum_j p_hj[h, j] + 0.5 (row sum over sources j, not column sum)."""
        expected = np.sum(trad12_full['p_hj'], axis=1) + 0.5
        np.testing.assert_allclose(trad12_full['p_h'], expected, atol=1e-12)

    def test_d_i_is_float(self, trad12_full):
        assert isinstance(trad12_full['d_i'], float)


class TestRunFullAnalysis:
    def test_all_keys_present_trad12(self, trad12_full):
        expected = {'r', 'v', 'w', 'C', 'c_AE', 'B', 'c_AR', 'z', 'c_AA',
                    'c_EE', 'P_p', 'p_hj', 'p_h', 'd_i', 'd', 'R'}
        assert expected.issubset(trad12_full.keys())

    def test_all_keys_present_techno(self, techno_full):
        expected = {'r', 'v', 'w', 'F', 'C', 'c_AE', 'B', 'c_AR', 'z', 'P_p', 'R'}
        assert expected.issubset(techno_full.keys())

    def test_all_matrix_values_are_ndarrays(self, trad12_full):
        for key in ('r', 'v', 'c_AE', 'B', 'P_p', 'p_hj', 'p_h', 'd', 'C', 'z'):
            assert isinstance(trad12_full[key], np.ndarray), \
                f"inputs['{key}'] is {type(trad12_full[key])}, expected ndarray"
