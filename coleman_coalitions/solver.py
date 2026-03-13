"""
coleman_coalitions/solver.py

Iterative solver for actor power (r), event value (v), and resource value (w).

The solver implements Coleman's (1973) exchange model equilibrium.  Two branches
are used depending on whether the resource-requirement matrix a is the identity:

  Identity-a branch ('trad' examples):
    Resources map one-to-one to events, so resource values (w) equal 1.
    Only r and v need to be solved iteratively.

  Full branch ('techno' examples):
    r, v, and w are all coupled and solved together with damped iteration.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray

from .core import normalize_matrix_row


class SolverConvergenceError(RuntimeError):
    """Raised when the Coleman exchange solver fails to converge."""


def solve(
    inputs: dict,
    tol: float = 1e-6,
    max_iter: int = 500,
    correction: float = 0.01,
) -> dict:
    """Solve for equilibrium power (r), event value (v), and resource value (w).

    Implements Coleman's (1973) iterative exchange model:
      - Actor power r is proportional to the value of resources the actor controls.
      - Event value v is proportional to the power of actors interested in the event.
      - Resource value w is proportional to the value of events requiring the resource.

    Parameters
    ----------
    inputs : dict
        Must contain 'c' (m,n), 'a' (q,m), 'x' (n,q), 'n', 'q', 'm'.
    tol : float
        Convergence tolerance (sum of absolute per-element errors across all variables).
    max_iter : int
        Maximum number of iterations before raising SolverConvergenceError.
    correction : float
        Step-size damping factor applied in the full branch to improve stability.
        Smaller values converge more slowly but more reliably.

    Returns
    -------
    dict  with keys 'r', 'v', 'w'  (all plain 1-D ndarrays).

    Raises
    ------
    SolverConvergenceError
        If the solver does not converge within max_iter iterations.
    """
    c = inputs['c']  # (m, n): resource control by actor
    a = inputs['a']  # (q, m): resource requirements of each event
    x = inputs['x']  # (n, q): absolute (unsigned) interests
    n = inputs['n']
    m = inputs['m']
    q = inputs['q']

    # Initialise from random starting points to avoid dependence on trivial guesses
    r: ndarray = normalize_matrix_row(np.random.rand(n))  # actor power
    w: ndarray = normalize_matrix_row(np.random.rand(m))  # resource value
    v: ndarray = normalize_matrix_row(np.random.rand(q))  # event value

    # Detect whether a is the identity matrix (trad examples)
    is_identity_a = (a.shape[0] == a.shape[1] and np.allclose(a, np.eye(a.shape[0])))

    if is_identity_a:
        # ----------------------------------------------------------------
        # Identity-a branch: resources map one-to-one to events, so w = 1.
        # Only r and v are iterated.
        #   v_new = normalise(r @ x)    [event value from actor interests]
        #   r_new = v_new @ c           [actor power from resource control]
        # ----------------------------------------------------------------
        w = np.ones(m)  # resource values are all 1 in the trad case

        for _ in range(max_iter):
            v_ny: ndarray = normalize_matrix_row(r @ x)  # new candidate for v
            r = v_ny @ c                                  # update r immediately
            error_v = v_ny - v
            v = v + error_v                               # update v

            if np.sum(np.abs(error_v)) < tol:
                break
        else:
            raise SolverConvergenceError(
                f'Solver did not converge after {max_iter} iterations '
                f'(identity-a branch, tol={tol}).'
            )

    else:
        # ----------------------------------------------------------------
        # Full branch: r, v, w are mutually coupled.
        # Damped iteration (correction factor) improves stability:
        #   r_new = normalise(w @ c)   [power from resource ownership]
        #   w_new = normalise(v @ a)   [resource value from event value]
        #   v_new = normalise(r @ x)   [event value from actor interests]
        # Each variable is updated as: x = x_new + error * correction
        # where error = candidate_from_one_formula - candidate_from_another.
        # ----------------------------------------------------------------
        for _ in range(max_iter):
            r_ny: ndarray = normalize_matrix_row(w @ c)   # r candidate via resources
            w_ny: ndarray = normalize_matrix_row(v @ a)   # w candidate via events
            v_ny: ndarray = normalize_matrix_row(r @ x)   # v candidate via interests

            r_new: ndarray = w_ny @ c   # r via updated w
            w_new: ndarray = v_ny @ a   # w via updated v
            v_new: ndarray = r_ny @ x   # v via updated r

            # Errors measure the discrepancy between the two update paths
            error_r = r_ny - r_new
            error_w = w_ny - w_new
            error_v = v_ny - v_new

            # Damped update: only move a fraction (correction) toward the new estimate
            r = r_new + error_r * correction
            w = w_new + error_w * correction
            v = v_new + error_v * correction

            corre = (np.sum(np.abs(error_r))
                     + np.sum(np.abs(error_w))
                     + np.sum(np.abs(error_v)))
            if corre < tol:
                break
        else:
            raise SolverConvergenceError(
                f'Solver did not converge after {max_iter} iterations '
                f'(full branch, tol={tol}).'
            )

    return {'r': r, 'v': v, 'w': w}
