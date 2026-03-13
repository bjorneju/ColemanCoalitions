"""Tests for core setup and normalization functions (coleman_coalitions.core)."""
import copy

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import coleman_coalitions as cc
from coleman_coalitions.core import normalize_matrix_row


class TestNormalizeMatrixRow:
    def test_2d_rows_sum_to_one(self):
        m = np.array([[3.0, 1.0], [1.0, 4.0]])
        result = normalize_matrix_row(m)
        np.testing.assert_allclose(np.sum(np.abs(result), axis=1), np.ones(2))

    def test_1d_sums_to_one(self):
        v = np.array([2.0, 3.0, 5.0])
        result = normalize_matrix_row(v)
        np.testing.assert_allclose(np.sum(np.abs(result)), 1.0)

    def test_already_normalized_unchanged(self):
        m = np.array([[0.6, 0.4], [0.3, 0.7]])
        np.testing.assert_allclose(normalize_matrix_row(m), m)

    def test_does_not_modify_original(self):
        m = np.array([[3.0, 1.0], [1.0, 4.0]])
        original = m.copy()
        normalize_matrix_row(m)
        np.testing.assert_array_equal(m, original)

    def test_negative_values_use_abs_sum(self):
        m = np.array([[0.4, -0.3, 0.1, -0.2]])
        result = normalize_matrix_row(m)
        np.testing.assert_allclose(np.sum(np.abs(result)), 1.0)

    def test_all_zero_row_left_unchanged(self):
        m = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = normalize_matrix_row(m)
        np.testing.assert_array_equal(result[0], np.zeros(2))


class TestSetup:
    def test_returns_all_required_keys(self, trad12_inputs):
        assert {'n', 'q', 'm', 'y', 'x', 's', 'a', 'c'}.issubset(trad12_inputs.keys())

    def test_x_is_abs_of_y(self, trad12_inputs):
        np.testing.assert_allclose(trad12_inputs['x'], np.abs(trad12_inputs['y']))

    def test_s_is_sign_of_y(self, trad12_inputs):
        np.testing.assert_array_equal(trad12_inputs['s'], np.sign(trad12_inputs['y']))

    def test_dimensions_consistent(self, trad12_inputs):
        n, q, m = trad12_inputs['n'], trad12_inputs['q'], trad12_inputs['m']
        assert trad12_inputs['y'].shape == (n, q)
        assert trad12_inputs['a'].shape == (q, m)
        assert trad12_inputs['c'].shape == (m, n)

    def test_techno_dimensions(self, techno_inputs):
        assert techno_inputs['n'] == 3
        assert techno_inputs['q'] == 2
        assert techno_inputs['m'] == 4

    def test_values_are_plain_ndarrays(self, trad12_inputs):
        """New API returns plain ndarrays, not [value, description] pairs."""
        for key in ('y', 'x', 's', 'a', 'c'):
            assert isinstance(trad12_inputs[key], np.ndarray), \
                f"inputs['{key}'] is {type(trad12_inputs[key])}, expected ndarray"


class TestCheckParameters:
    def test_unnormalized_y_gets_normalized(self):
        y = np.array([[3.0, 1.0], [1.0, 4.0], [2.0, 2.0]])
        a = np.eye(2)
        c = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
        inputs = cc.setup(3, 2, 2, y, a, c)
        cc.check_parameters(inputs)
        np.testing.assert_allclose(np.sum(np.abs(inputs['y']), axis=1), np.ones(3), atol=1e-10)

    def test_unnormalized_a_gets_normalized(self):
        y = np.array([[0.5, 0.5], [0.3, 0.7]])
        a = np.array([[2.0, 2.0], [1.0, 3.0]])  # rows sum to 4 and 4
        c = np.array([[0.6, 0.4], [0.4, 0.6]])
        inputs = cc.setup(2, 2, 2, y, a, c)
        cc.check_parameters(inputs)
        np.testing.assert_allclose(np.sum(inputs['a'], axis=1), np.ones(2), atol=1e-10)

    def test_already_normalized_y_unchanged(self, trad12_inputs):
        inputs = copy.deepcopy(trad12_inputs)
        y_before = inputs['y'].copy()
        cc.check_parameters(inputs)
        np.testing.assert_allclose(inputs['y'], y_before, atol=1e-10)
