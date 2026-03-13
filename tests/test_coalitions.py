"""Tests for coalition analysis functions (coleman_coalitions.coalitions)."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import coleman_coalitions as cc


@pytest.fixture(scope="module")
def trad12_coalitions(trad12_full):
    return cc.feasible_coalitions(trad12_full)


@pytest.fixture(scope="module")
def trad12_coalition_outputs(trad12_full, trad12_coalitions):
    outputs = cc.coalition_trad(trad12_full, trad12_coalitions)
    # Append the full-collective analysis as the last key (required by value_of_coalition)
    import ast
    full_key = str(list(range(trad12_full['n'])))
    if full_key not in outputs:
        outputs[full_key] = trad12_full
    return outputs


class TestFeasibleCoalitions:
    def test_returns_dict(self, trad12_coalitions):
        assert isinstance(trad12_coalitions, dict)

    def test_non_empty(self, trad12_coalitions):
        assert len(trad12_coalitions) > 0

    def test_each_coalition_has_members_and_control(self, trad12_coalitions):
        for key, val in trad12_coalitions.items():
            members, control = val
            assert isinstance(members, list)
            assert len(members) >= 2
            assert isinstance(control, (int, float, np.floating))

    def test_each_coalition_has_majority_control(self, trad12_full, trad12_coalitions):
        for key, val in trad12_coalitions.items():
            _, control = val
            assert control > 0.5, f"Coalition {key} has control {control} <= 0.5"

    def test_small_example_n3(self):
        """3 actors with clear majority structure — verify enumeration is correct."""
        np.random.seed(42)
        y = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
        c = np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3]])  # (m=2, n=3)
        a = np.eye(2)
        inputs = cc.setup(3, 2, 2, y, a, c)
        coalitions = cc.feasible_coalitions(inputs)
        for key, (members, control) in coalitions.items():
            assert control > 0.5


class TestCoalitionTrad:
    def test_returns_dict(self, trad12_coalition_outputs):
        assert isinstance(trad12_coalition_outputs, dict)

    def test_keys_match_feasible_coalitions(self, trad12_coalitions, trad12_coalition_outputs):
        # All feasible coalition keys should be present (full-collective key is added by fixture)
        assert set(trad12_coalitions.keys()).issubset(set(trad12_coalition_outputs.keys()))

    def test_each_output_has_full_analysis(self, trad12_coalitions, trad12_coalition_outputs):
        expected = {'r', 'v', 'P_p'}
        for key in trad12_coalitions:
            assert expected.issubset(set(trad12_coalition_outputs[key].keys())), \
                f"Coalition {key} missing keys: {expected - set(trad12_coalition_outputs[key].keys())}"


class TestValueOfCoalition:
    @pytest.fixture(scope="class")
    def value(self, trad12_coalition_outputs):
        return cc.value_of_coalition(trad12_coalition_outputs)

    def test_returns_dict(self, value):
        assert isinstance(value, dict)

    def test_keys_are_numeric_values(self, value):
        for key, entry in value.items():
            assert 'to coalition' in entry
            assert 'to collective' in entry
            assert isinstance(entry['to coalition'], (int, float, np.floating))

    def test_coalition_plus_opposition_equals_collective(self, value):
        """'to coalition' + 'to opposition' must equal 'to collective' (they partition actors)."""
        for key, entry in value.items():
            total = entry['to coalition'] + entry['to opposition']
            np.testing.assert_allclose(total, entry['to collective'], atol=1e-10,
                err_msg=f"Coalition {key}: coalition + opposition != collective")


class TestOptimalCoalition:
    @pytest.fixture(scope="class")
    def result(self, trad12_coalition_outputs):
        return cc.optimal_coalition(trad12_coalition_outputs)

    def test_returns_summary_and_tpm(self, result):
        summary, TPM = result
        assert isinstance(summary, dict)
        assert isinstance(TPM, list)

    def test_tpm_rows_sum_to_at_most_one(self, result):
        _, TPM = result
        tpm_arr = np.array(TPM)
        assert np.all(np.sum(tpm_arr, axis=1) <= 1.0 + 1e-9)

    def test_tpm_non_negative(self, result):
        _, TPM = result
        assert np.all(np.array(TPM) >= -1e-9)


class TestWinningCoalitions:
    @pytest.fixture(scope="class")
    def winning_and_tpm(self, trad12_coalition_outputs):
        _, TPM = cc.optimal_coalition(trad12_coalition_outputs)
        return cc.winning_coalitions(TPM), TPM

    def test_returns_list(self, winning_and_tpm):
        winning, _ = winning_and_tpm
        assert isinstance(winning, list)

    def test_winning_values_are_binary(self, winning_and_tpm):
        winning, _ = winning_and_tpm
        assert all(w in (0, 1) for w in winning)

    def test_winning_coalitions_are_sinks(self, winning_and_tpm):
        """A coalition marked winning must have zero out-flow in the TPM."""
        winning, TPM = winning_and_tpm
        tpm_arr = np.array(TPM)
        for i, w in enumerate(winning):
            if w == 1:
                assert np.sum(tpm_arr[i]) == 0.0, \
                    f"Coalition {i} marked winning but has out-flow in TPM"
