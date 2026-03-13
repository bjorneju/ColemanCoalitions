"""
coleman_coalitions/core.py

Core setup, normalisation, and parameter-checking functions.

The central data structure throughout the package is a plain Python dict mapping
string keys (e.g. 'y', 'c', 'r') to numpy ndarrays (or scalars for 'n', 'q', 'm').
This module builds and validates that dict from raw input matrices.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


# Human-readable descriptions for all output variables, used by io.py.
DESCRIPTIONS: dict[str, str] = {
    'n':     'Number of actors',
    'q':     'Number of events',
    'm':     'Number of resources',
    'y':     'Directed interests of actor j in event i [y(n,q)]',
    'x':     'Absolute interests of actor j in event i [x(n,q)]',
    's':     'Sign of interests of actor j in event i [s(n,q)]',
    'a':     'Resource requirements of event i for resource k [a(q,m)]',
    'c':     'Control of resource k by actor j [c(m,n)]',
    'r':     'Total power of actor j [r(n)]',
    'v':     'Value of each event i [v(q)]',
    'w':     'Value of each resource k [w(m)]',
    'C':     'Constitutional control over event i by actor j [C(q,n)]',
    'F':     'Fraction of resource k used towards event i [F(m,q)]',
    'B':     'Derived interest of actor j in resource k [B(n,m)]',
    'c_AE':  'Final directed control of actor j over event i [c_AE(n,q)]',
    'c_AR':  'Final control of actor j over resource k [c_AR(n,m)]',
    'c_AA':  'Control of actor j by actor h [c_AA(n,n)]',
    'z':     'Control of actor j by actor h [z(n,n)]',
    'c_EE':  'Control of event j by event i [c_EE(m,q)]',
    'P_p':   'Probability of positive outcome for event i [P_p(q)]',
    'p_hj':  'Increment in expected realization of interests of actor h from j [p_hj(n,n)]',
    'p_h':   'Expected value of collectivity for actor h [p_h(n)]',
    'd_i':   'Expected weighted realization of interests',
    'd':     'Directed power of collectivity on event i [d(q)]',
    'R':     'Total external power of collectivity',
}


def normalize_matrix_row(matrix: ndarray) -> ndarray:
    """Normalise a matrix so each row sums to 1 by absolute value.

    For a 2-D input each row is divided by the sum of its absolute values.
    For a 1-D input the entire vector is normalised.
    Rows (or the vector) that are all-zero are left unchanged.

    Returns a copy; the input is not modified.
    """
    normed = matrix.copy()
    if normed.ndim == 2:
        # Compute row-wise L1 norm and divide, skipping zero rows
        row_sums = np.sum(np.abs(normed), axis=1, keepdims=True)
        # Avoid division by zero: only normalise non-zero rows
        nonzero = row_sums != 0
        normed = np.where(nonzero, normed / np.where(nonzero, row_sums, 1.0), normed)
    elif normed.ndim == 1:
        total = np.sum(np.abs(normed))
        if total != 0:
            normed = normed / total
    return normed


def setup(n: int, q: int, m: int, y: ndarray, a: ndarray, c: ndarray) -> dict:
    """Build the base parameter dict from raw input matrices.

    Parameters
    ----------
    n : int   -- number of actors
    q : int   -- number of events
    m : int   -- number of resources
    y : array-like (n, q)
        Directed interests: sign indicates preferred direction (+1 or -1),
        magnitude indicates strength of interest.
    a : array-like (q, m)
        Resource requirements: a[i, k] is the fraction of resource k required
        by event i.  Should be row-normalised (each row sums to 1).
    c : array-like (m, n)
        Resource control: c[k, j] is actor j's control over resource k.
        Should be row-normalised.

    Returns
    -------
    dict  with keys 'n', 'q', 'm', 'y', 'x', 's', 'a', 'c'.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    return {
        'n': n,
        'q': q,
        'm': m,
        'y': y,
        'x': np.abs(y),   # absolute interests (magnitudes only)
        's': np.sign(y),  # direction of each interest (+1 / -1 / 0)
        'a': a,
        'c': c,
    }


def check_parameters(inputs: dict) -> dict:
    """Normalise interest, resource-requirement and control matrices if needed.

    Coleman's model requires y (row-wise by |abs|), a (row-wise), and c (row-wise)
    to each be row-normalised.  This function detects and fixes any violation,
    then updates 'x' and 's' to match the normalised 'y'.

    Modifies ``inputs`` in-place and returns it.
    """
    # Normalise directed interests if any row does not sum to 1 by absolute value
    if not np.allclose(np.sum(np.abs(inputs['y']), axis=1), 1.0):
        inputs['y'] = normalize_matrix_row(inputs['y'])
        inputs['x'] = np.abs(inputs['y'])  # keep x in sync with normalised y

    # Normalise resource-requirement matrix (row sums should equal 1)
    if not np.allclose(np.sum(inputs['a'], axis=1), 1.0):
        inputs['a'] = normalize_matrix_row(inputs['a'])

    # Normalise resource-control matrix (row sums should equal 1)
    if not np.allclose(np.sum(inputs['c'], axis=1), 1.0):
        inputs['c'] = normalize_matrix_row(inputs['c'])

    return inputs


# ---------------------------------------------------------------------------
# Canonical example datasets
# ---------------------------------------------------------------------------

def standard_variables_techno() -> dict:
    """Standard inputs for the 'techno' example (Coleman 1973, problem 1).

    3 actors, 2 events, 4 resources.
    The 'techno' structure means resources are distinct from events (a is not identity).
    """
    # c[k, j]: control of resource k by actor j  (shape m=4, n=3)
    c = np.array([
        [1.000, 0.000, 0.000],
        [0.500, 0.250, 0.250],
        [0.333, 0.500, 0.167],
        [0.100, 0.200, 0.700],
    ])
    # y[j, i]: directed interests of actor j in event i  (shape n=3, q=2)
    y = np.array([
        [0.75, 0.25],
        [0.25, 0.75],
        [0.10, 0.90],
    ])
    # a[i, k]: resource requirements of event i for resource k  (shape q=2, m=4)
    a = np.array([
        [0.400, 0.333, 0.167, 0.100],
        [0.250, 0.500, 0.125, 0.125],
    ])
    n, q, m = y.shape[0], y.shape[1], a.shape[1]
    return setup(n, q, m, y, a, c)


def standard_variables_techno2() -> dict:
    """Standard inputs for the 'techno2' example.

    3 actors, 2 events, 3 resources.  Identity control matrix (each actor
    controls exactly one resource), non-identity a.
    """
    c = np.eye(3)  # each actor exclusively controls one resource
    y = np.array([
        [0.500, 0.500],
        [0.333, 0.667],
        [0.750, 0.250],
    ])
    a = np.array([
        [0.500, 0.250, 0.250],
        [0.500, 0.333, 0.167],
    ])
    n, q, m = y.shape[0], y.shape[1], a.shape[1]
    return setup(n, q, m, y, a, c)


def standard_variables_trad12() -> dict:
    """Standard inputs for the 'trad12' example (Coleman 1973).

    3 actors, 4 events, 4 resources.  The 'trad' (traditional) structure uses
    an identity a matrix, meaning each resource maps directly to one event.
    Mixed-sign interests make this a richer test of the solver.
    """
    # Control matrix is given as actor-by-resource (ct), then transposed to (m, n)
    ct = np.array([
        [0.3, 0.4, 0.4, 0.25],
        [0.2, 0.3, 0.4, 0.35],
        [0.5, 0.3, 0.2, 0.40],
    ])
    c = ct.T  # c[k, j]: control of resource k by actor j  (shape m=4, n=3)

    # Mixed positive/negative interests: actor 3 opposes events 1-3
    y = np.array([
        [ 0.4,  0.2,  0.1,  0.3],
        [-0.3,  0.3,  0.2,  0.2],
        [-0.1, -0.3, -0.5,  0.1],
    ])
    n, q = y.shape
    m = q
    a = np.eye(q)  # identity: resource k used exclusively by event k
    return setup(n, q, m, y, a, c)
