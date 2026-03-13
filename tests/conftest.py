import copy

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import coleman_coalitions as cc


@pytest.fixture(scope="session")
def trad12_inputs():
    return cc.standard_variables_trad12()


@pytest.fixture(scope="session")
def techno_inputs():
    return cc.standard_variables_techno()


@pytest.fixture(scope="session")
def techno2_inputs():
    return cc.standard_variables_techno2()


@pytest.fixture(scope="session")
def trad12_full(trad12_inputs):
    np.random.seed(42)
    inputs = copy.deepcopy(trad12_inputs)
    cc.run_full_analysis(inputs)
    return inputs


@pytest.fixture(scope="session")
def techno_full(techno_inputs):
    np.random.seed(42)
    inputs = copy.deepcopy(techno_inputs)
    cc.run_full_analysis(inputs)
    return inputs


@pytest.fixture(scope="session")
def techno2_full(techno2_inputs):
    np.random.seed(42)
    inputs = copy.deepcopy(techno2_inputs)
    cc.run_full_analysis(inputs)
    return inputs
