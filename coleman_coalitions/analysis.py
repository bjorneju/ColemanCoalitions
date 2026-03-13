"""
coleman_coalitions/analysis.py

Derived-variable functions and the full analysis pipeline.

All functions accept and return plain dicts of ndarrays (no [value, description] wrapping).
Each function documents its mathematical formula explicitly so vectorised expressions can
be cross-checked against Coleman (1973) without needing to trace individual loop indices.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray

from .core import check_parameters
from .solver import solve


# ---------------------------------------------------------------------------
# Derived-variable functions
# ---------------------------------------------------------------------------

def constitutional_control(inputs: dict) -> dict:
    """Compute constitutional control of each event by each actor.

    Formula: C = a @ c
      where a[i,k] is the resource requirement of event i for resource k,
      and c[k,j] is actor j's control over resource k.
    Result shape: (q, n).
    """
    return {'C': inputs['a'] @ inputs['c']}


def fraction_of_resources(inputs: dict) -> dict:
    """Compute the fraction of each resource directed towards each event.

    Formula: F[k, i] = a[i, k] * v[i] / w[k]
      Vectorised: F = (a.T * v[None, :]) / w[:, None]
      where a.T has shape (m, q), v is broadcast across rows,
      and w normalises each row by resource value.
    Result shape: (m, q).
    """
    a, v, w = inputs['a'], inputs['v'], inputs['w']
    # a.T[k, i] = a[i, k]; multiply each column i by v[i], divide each row k by w[k]
    F: ndarray = (a.T * v[np.newaxis, :]) / w[:, np.newaxis]
    return {'F': F}


def control_actor_event(inputs: dict) -> dict:
    """Compute the final directed control of each actor over each event.

    Formula: c_AE[j, i] = y[j, i] * r[j] / v[i]
      Vectorised: c_AE = (y * r[:, None]) / v[None, :]
      r[:, None] broadcasts actor power across events (columns);
      v[None, :] broadcasts event value across actors (rows).
    Result shape: (n, q).
    """
    y, r, v = inputs['y'], inputs['r'], inputs['v']
    # Scale each row j by actor power r[j], then normalise each column i by event value v[i]
    c_AE: ndarray = (y * r[:, np.newaxis]) / v[np.newaxis, :]
    return {'c_AE': c_AE}


def derived_interest(inputs: dict) -> dict:
    """Compute the derived interest of each actor in each resource.

    Formula: B[j, k] = sum_i x[j, i] * a[i, k]
      This is a dot product: B = x @ a
      where x[j, i] is actor j's absolute interest in event i,
      and a[i, k] is how much event i draws on resource k.
    Result shape: (n, m).
    """
    # x @ a: (n, q) @ (q, m) -> (n, m)
    B: ndarray = inputs['x'] @ inputs['a']
    return {'B': B}


def control_actor_resource(inputs: dict) -> dict:
    """Compute the final control of each actor over each resource.

    Formula: c_AR[j, k] = B[j, k] * r[j] / w[k]
      Vectorised: c_AR = (B * r[:, None]) / w[None, :]
      Analogous to control_actor_event: scale by actor power, normalise by resource value.
    Result shape: (n, m).
    """
    B, r, w = inputs['B'], inputs['r'], inputs['w']
    c_AR: ndarray = (B * r[:, np.newaxis]) / w[np.newaxis, :]
    return {'c_AR': c_AR}


def control_actor_actor(inputs: dict) -> dict:
    """Compute the direct control of each actor over every other actor.

    Formula: z[j, h] = sum_k B[j, k] * c[k, h]
      This is a dot product: z = B @ c
      where B[j, k] is actor j's derived interest in resource k,
      and c[k, h] is actor h's control over resource k.
    Result shape: (n, n).  Also stored as 'c_AA' (alias).
    """
    # B @ c: (n, m) @ (m, n) -> (n, n)
    z: ndarray = inputs['B'] @ inputs['c']
    return {'z': z, 'c_AA': z}


def control_event_event(inputs: dict) -> dict:
    """Compute control of events by events (mediated through resources).

    Formula: c_EE = c @ x
      c[k, j] * x[j, i] summed over j gives how much resource k is
      oriented towards event i via actor interests.
    Result shape: (m, q).
    """
    return {'c_EE': inputs['c'] @ inputs['x']}


def positive_outcome(inputs: dict) -> dict:
    """Compute the probability of a positive outcome for each event.

    Formula: P_p[i] = 0.5 + 0.5 * sum_j c_AE[j, i]
      0.5 is the baseline (no information); the signed sum of directed control
      shifts the probability up (positive net control) or down.
    Result shape: (q,).
    """
    return {'P_p': 0.5 + 0.5 * np.sum(inputs['c_AE'], axis=0)}


def increment_expected_realization(inputs: dict) -> dict:
    """Compute the increment in expected realization of interests of actor h from actor j.

    Formula: p_hj[h, j] = 0.5 * sum_i s[h,i] * x[h,i] * |c_AE[j,i]| * s[j,i]
      Vectorised as a matrix product:
        A[h, i] = s[h, i] * x[h, i]       (interest strength weighted by direction)
        D[j, i] = |c_AE[j, i]| * s[j, i]  (absolute control, signed by actor direction)
        p_hj = 0.5 * A @ D.T
      The (h, j) entry accumulates how much actor j's control aligns with actor h's
      interests across all events.
    Result shape: (n, n).
    """
    s, x, c_AE = inputs['s'], inputs['x'], inputs['c_AE']
    # A: how much each actor cares about each event (magnitude, signed by preference direction)
    A: ndarray = s * x                    # (n, q)
    # D: how strongly each actor pushes each event in their own preferred direction
    D: ndarray = np.abs(c_AE) * s        # (n, q)
    # A @ D.T: sums over events i for every (h, j) pair -> (n, n)
    p_hj: ndarray = 0.5 * (A @ D.T)
    return {'p_hj': p_hj}


def expected_value_of_collectivity(inputs: dict) -> dict:
    """Compute expected value of the collectivity for each actor.

    Formula: p_h[h] = sum_j p_hj[h, j] + 0.5
      Summing across all actors j gives actor h's total expected benefit;
      0.5 is the baseline (random outcome with no collective action).
    Result shape: (n,).
    """
    return {'p_h': np.sum(inputs['p_hj'], axis=1) + 0.5}


def expected_weighted_realization(inputs: dict) -> dict:
    """Compute the power-weighted expected realization of interests across all actors.

    Formula: d_i = sum_h p_h[h] * r[h]
      Weights each actor's expected value by their power share, giving a
      collective-level measure of interest satisfaction.
    Returns a scalar float.
    """
    return {'d_i': float(np.dot(inputs['p_h'], inputs['r']))}


def directed_power_of_collectivity(inputs: dict) -> dict:
    """Compute the directed power of the collectivity on each event.

    Formula: d[i] = sum_j r[j] * y[j, i]  =  r @ y
      The power-weighted sum of interests; positive values indicate net
      collective push towards a positive outcome for event i.
    Result shape: (q,).
    """
    return {'d': inputs['r'] @ inputs['y']}


def total_power_of_collectivity(inputs: dict) -> dict:
    """Compute the total (unsigned) external power of the collectivity.

    Formula: R = sum_i |d[i]|
      Sums absolute directed power across all events; measures how strongly
      the collectivity pushes outcomes in any direction.
    Returns a scalar float.
    """
    return {'R': float(np.sum(np.abs(inputs['d'])))}


def matching_attitudes(y: ndarray) -> float:
    """Scalar measure of how well actor attitudes align across events.

    For each event, computes the fraction of actors sharing the majority sign,
    then averages and rescales to [0, 1].  A value of 1 means all actors agree
    on all events; 0 means maximally split attitudes.

    Parameters
    ----------
    y : ndarray (n, q)  -- directed interests (signs matter, magnitudes ignored)
    """
    s = np.sign(y)
    n = float(s.shape[0])
    # Mean absolute consensus per event, then average across events
    avg = np.mean(np.abs(np.sum(s, axis=0) / n))
    # Theoretical minimum: 1/n (odd n) or 0 (even n, perfectly split possible)
    minimum = 1.0 / n if n % 2 == 1 else 0.0
    return (avg - minimum) / (1.0 - minimum)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(
    inputs: dict,
    tol: float = 1e-6,
    max_iter: int = 500,
    correction: float = 0.01,
) -> dict:
    """Run the complete Coleman analysis pipeline.

    Normalises parameters, solves for equilibrium (r, v, w), then computes
    all derived variables in order.  Updates and returns ``inputs`` in-place.

    Parameters
    ----------
    inputs : dict
        Base parameter dict from ``setup()``.
    tol : float
        Convergence tolerance passed to the solver.
    max_iter : int
        Maximum solver iterations before raising ``SolverConvergenceError``.
    correction : float
        Damping factor for the full (non-identity-a) solver branch.
    """
    inputs.update(check_parameters(inputs))
    inputs.update(solve(inputs, tol=tol, max_iter=max_iter, correction=correction))
    inputs.update(fraction_of_resources(inputs))
    inputs.update(control_actor_event(inputs))
    inputs.update(derived_interest(inputs))
    inputs.update(control_actor_resource(inputs))
    inputs.update(control_event_event(inputs))
    inputs.update(control_actor_actor(inputs))
    inputs.update(constitutional_control(inputs))
    inputs.update(positive_outcome(inputs))
    inputs.update(increment_expected_realization(inputs))
    inputs.update(expected_value_of_collectivity(inputs))
    inputs.update(expected_weighted_realization(inputs))
    inputs.update(directed_power_of_collectivity(inputs))
    inputs.update(total_power_of_collectivity(inputs))
    return inputs
