"""Smoke tests for the new coleman/ package — verifies the public API works end-to-end."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import coleman_coalitions as coleman


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close('all')


class TestPackageImport:
    def test_key_symbols_importable(self):
        assert callable(coleman.setup)
        assert callable(coleman.run_full_analysis)
        assert callable(coleman.run_coalition_analysis)
        assert callable(coleman.solve)
        assert callable(coleman.draw_coalition_map)
        assert issubclass(coleman.SolverConvergenceError, RuntimeError)


class TestEndToEndTrad12:
    def test_full_pipeline(self):
        np.random.seed(42)
        inputs = coleman.standard_variables_trad12()
        coleman.run_full_analysis(inputs)

        assert 'r' in inputs
        assert 'v' in inputs
        assert 'P_p' in inputs
        assert 'R' in inputs
        r = inputs['r']
        np.testing.assert_allclose(np.sum(r), 1.0, atol=1e-4)
        assert np.all(r > 0)

    def test_coalition_pipeline(self):
        np.random.seed(42)
        inputs = coleman.standard_variables_trad12()
        full, coalitions, coal_outputs = coleman.run_coalition_analysis(inputs)

        assert len(coalitions) > 0
        for key, (members, control) in coalitions.items():
            assert control > 0.5

        val = coleman.value_of_coalition(coal_outputs)
        summary, TPM = coleman.optimal_coalition(coal_outputs)
        winning = coleman.winning_coalitions(TPM)
        assert isinstance(winning, list)


class TestEndToEndTechno:
    def test_full_pipeline(self):
        np.random.seed(42)
        inputs = coleman.standard_variables_techno()
        coleman.run_full_analysis(inputs)
        assert 'w' in inputs
        w = inputs['w']
        np.testing.assert_allclose(np.sum(w), 1.0, atol=1e-4)


class TestNewVisualizations:
    @pytest.fixture(scope='class')
    def coalition_data(self):
        np.random.seed(42)
        inputs = coleman.standard_variables_trad12()
        _, _, coal_outputs = coleman.run_coalition_analysis(inputs)
        summary, TPM = coleman.optimal_coalition(coal_outputs)
        return inputs, coal_outputs, summary, TPM

    def test_draw_coalition_map(self, coalition_data):
        _, coal_outputs, summary, TPM = coalition_data
        fig = coleman.draw_coalition_map(coal_outputs, summary, TPM)
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_draw_strongest_transitions(self, coalition_data):
        _, coal_outputs, summary, TPM = coalition_data
        fig = coleman.draw_strongest_transitions(coal_outputs, summary, TPM)
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_draw_power_distribution(self, coalition_data):
        inputs, _, _, _ = coalition_data
        fig = coleman.draw_power_distribution(inputs)
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_draw_event_values(self, coalition_data):
        inputs, _, _, _ = coalition_data
        fig = coleman.draw_event_values(inputs)
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)


class TestOutputFormat:
    def test_values_are_plain_ndarrays(self):
        """New API returns plain ndarrays, not [value, description] pairs."""
        np.random.seed(42)
        inputs = coleman.standard_variables_trad12()
        coleman.run_full_analysis(inputs)
        # All matrix values should be ndarrays, not lists
        for key in ('r', 'v', 'c_AE', 'B', 'P_p'):
            assert isinstance(inputs[key], np.ndarray), \
                f"inputs['{key}'] is {type(inputs[key])}, expected ndarray"

    def test_scalars_are_floats(self):
        np.random.seed(42)
        inputs = coleman.standard_variables_trad12()
        coleman.run_full_analysis(inputs)
        assert isinstance(inputs['R'], float)
        assert isinstance(inputs['d_i'], float)
