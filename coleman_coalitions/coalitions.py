"""
coleman_coalitions/coalitions.py

Coalition enumeration, analysis, and optimal-coalition computation.

A 'feasible coalition' is any subset of actors (size >= 2) whose combined
control of the first resource exceeds 50%.  For each feasible coalition the
full Coleman analysis is run treating the coalition as its own sub-collective,
then the value of each coalition to its members, the opposition, and the
collective as a whole is computed.

The transition probability matrix (TPM) encodes which coalition each current
coalition would rationally switch to (if any), and sink nodes in the TPM are
identified as 'winning' (stable) coalitions.
"""
from __future__ import annotations

import ast
import copy
import warnings
from collections import OrderedDict as odict

import numpy as np
from numpy import ndarray

from .core import setup
from .analysis import run_full_analysis


# ---------------------------------------------------------------------------
# Coalition enumeration
# ---------------------------------------------------------------------------

def feasible_coalitions(inputs: dict) -> odict:
    """Enumerate all coalitions of size >= 2 that hold majority control (> 0.5).

    Generates all 2^n subsets of actors, retains those with at least two members
    and combined control (sum of first resource row across members) exceeding 0.5.

    Parameters
    ----------
    inputs : dict  -- must contain 'n' and 'c'

    Returns
    -------
    OrderedDict  mapping str(member_list) -> [member_list, coalition_control]

    Notes
    -----
    Coalition control is computed as the sum of the first row of ``c`` across
    coalition members — a proxy for combined resource influence (see Coleman 1973 §8).

    The enumeration is exponential in n (2^n subsets). A warning is raised for
    n >= 15 (2^15 = 32 768 subsets) and a hard limit applies at n > 18 (2^18 =
    262 144 subsets), beyond which memory and runtime become prohibitive.
    """
    n: int = inputs['n']
    c: ndarray = inputs['c']

    # Warn before the hard limit so users can catch this in advance
    if n >= 15:
        warnings.warn(
            f'feasible_coalitions: n={n} generates 2^{n} = {2**n:,} candidate subsets. '
            f'This may be slow. The hard limit is n=18.',
            stacklevel=2,
        )

    if n > 18:
        raise ValueError(
            f'feasible_coalitions: n={n} exceeds the hard limit of 18. '
            f'2^{n} = {2**n:,} subsets cannot be enumerated in reasonable time/memory.'
        )

    # Build a (2^n, n) binary matrix where row i encodes membership of subset i
    if n < 9:
        # For small n, np.unpackbits is fastest
        LOLI: ndarray = np.unpackbits(
            np.arange(2 ** n).astype(np.uint8)[:, None], axis=1
        )[:, -n:]
    else:
        # For n 9-18, build the binary table from format strings
        LOLI = np.array([
            [int(b) for b in format(i, f'0{n}b')]
            for i in range(2 ** n)
        ])

    # Multiply each row (binary membership mask) by actor indices 1..n
    actors: ndarray = np.arange(1, n + 1)
    coalitions: ndarray = actors * LOLI  # (2^n, n); non-members have value 0

    result: odict = odict()

    for i in range(2 ** n):
        # Recover list of 0-indexed member positions
        coal: list[int] = [j for j in range(n) if coalitions[i, j] > 0]
        if len(coal) < 2:
            continue  # skip singletons and the empty set
        # Coalition control: sum first row of c across member columns
        control: float = float(sum(c[0, cc] for cc in coal))
        if control > 0.5:
            result[str(coal)] = [coal, control]

    return result


# ---------------------------------------------------------------------------
# Coalition analysis
# ---------------------------------------------------------------------------

def coalition_trad(inputs: dict, coalitions: odict) -> dict:
    """Run the full Coleman analysis for each coalition as a sub-collective.

    For each coalition, a new system is constructed using only the coalition
    members' interests and the columns of c corresponding to those members.
    The full analysis (solve + all derived variables) is run on each subsystem.

    Parameters
    ----------
    inputs : dict      -- full-system inputs
    coalitions : dict  -- from feasible_coalitions()

    Returns
    -------
    dict  mapping coalition key -> full analysis dict for that sub-collective
    """
    q: int = inputs['q']
    m: int = inputs['m']
    y: ndarray = inputs['y']
    c: ndarray = inputs['c']
    a: ndarray = inputs['a']
    outputs: dict = {}

    for key, (idx, _) in coalitions.items():
        n_coal = len(idx)
        # Subset interests (rows) and control columns for the coalition members
        subsystem = setup(n_coal, q, m, y[idx, :], a, c[:, idx])
        run_full_analysis(subsystem)
        outputs[key] = subsystem

    return outputs


def coalition_techno_equal_control(inputs: dict, coalitions: odict) -> dict:
    """Techno analysis treating each coalition as a composite actor with equal internal control.

    Each coalition becomes a single 'meta-resource' in a new techno-style system.
    Within each coalition, control is distributed equally among members (1/|coalition|).
    This is the methodologically preferred approach for techno-style coalition analysis.

    Parameters
    ----------
    inputs : dict      -- full-system inputs
    coalitions : dict  -- from feasible_coalitions()

    Returns
    -------
    dict  -- full analysis of the composite system
    """
    q: int = inputs['q']
    m: int = len(coalitions)   # one resource per coalition
    n: int = inputs['n']
    y: ndarray = inputs['y']
    cc: ndarray = inputs['c']   # original control matrix
    a: ndarray = np.zeros((q, m))
    c: ndarray = np.zeros((m, n))

    for i, (key, (idx, _)) in enumerate(coalitions.items()):
        members = float(len(idx))
        for actor_id in idx:
            # Equal share of coalition control for each member
            c[i, actor_id] = 1.0 / members
        for event_id in range(q):
            # Coalition's resource-requirement entry: sum of original control for these members
            a[event_id, i] = np.sum(cc[idx, event_id])

    result = setup(n, q, m, y, a, c)
    run_full_analysis(result)
    return result


# ---------------------------------------------------------------------------
# Coalition value
# ---------------------------------------------------------------------------

def value_for_opposition(
    coalition_outputs: dict,
    actor: int,
    current_key: str,
    full_key: str,
) -> float:
    """Compute the value of a coalition configuration to one specific actor.

    Value is defined as the probability-weighted alignment between the coalition's
    outcome probabilities and the actor's interests:
      value = sum_i (P_p[i] * 2 - 1) * y[actor, i] / q

    Parameters
    ----------
    coalition_outputs : dict  -- output of coalition_trad()
    actor : int               -- 0-indexed actor
    current_key : str         -- coalition key to evaluate
    full_key : str            -- key for the full-collective analysis
    """
    P: ndarray = coalition_outputs[current_key]['P_p']   # outcome probabilities (q,)
    y: ndarray = coalition_outputs[full_key]['y'][actor, :]  # actor interests (q,)
    # (P*2-1) maps [0,1] -> [-1,1]; dot with y gives interest-weighted alignment
    return float(np.sum(np.multiply(P * 2 - 1, y)) / y.size)


def value_of_coalition(coalition_outputs: dict) -> dict:
    """Compute the value of each coalition to its members, opposition, and collective.

    For each coalition, the value to every actor (member and non-member) is computed,
    then aggregated into coalition-level, opposition-level, and collective-level totals.

    Returns
    -------
    dict  mapping coalition key -> {
        'to actors': list[float],      -- value to each actor (all n actors)
        'to coalition': float,         -- sum over member actors
        'to opposition': float,        -- sum over non-member actors
        'to collective': float,        -- sum over all actors
    }
    """
    keys: list[str] = list(coalition_outputs.keys())
    full_key: str = keys[-1]   # last entry is the full-collective analysis
    n: int = coalition_outputs[full_key]['n']
    result: dict = {}

    for key in keys:
        members: list[int] = ast.literal_eval(key)
        # Value to each actor under the current coalition's outcome distribution
        vals: list[float] = [
            value_for_opposition(coalition_outputs, i, key, full_key)
            for i in range(n)
        ]
        val_coal = sum(vals[i] for i in members)
        result[key] = {
            'to actors':     vals,
            'to coalition':  val_coal,
            'to opposition': sum(vals) - val_coal,
            'to collective': sum(vals),
        }

    return result


# ---------------------------------------------------------------------------
# Optimal coalition / TPM
# ---------------------------------------------------------------------------

def optimal_coalition(coalition_outputs: dict) -> tuple[dict, list[list[float]]]:
    """For each coalition, find the alternative coalition that best improves member payoffs.

    Constructs a transition probability matrix (TPM) where TPM[i, j] > 0 means
    coalition i would benefit from switching to coalition j (all members improve).
    Rows are normalised to sum to at most 1.

    Returns
    -------
    summary : dict
        Per-coalition breakdown:
          - 'change': raw improvement scores to each potential coalition
          - 'self':   current value to each actor under this coalition
          - 'best':   value to each actor under the identified best transition
    TPM : list[list[float]]
        Row-normalised transition probability matrix.
    """
    coalition_names: list[str] = list(coalition_outputs.keys())
    num_coalitions: int = len(coalition_names)
    members_list: list[list[int]] = [ast.literal_eval(k) for k in coalition_names]
    # Pad value lists to the size of the largest coalition for uniform storage
    tot_members: int = max(len(m) for m in members_list)

    # coalition_choice[i][j] = improvement score if coalition i moves to j (0 if no improvement)
    coalition_choice: list[list[float]] = [[0.0] * num_coalitions for _ in range(num_coalitions)]
    savevals:  list[list[float]] = [[0.0] * tot_members for _ in range(num_coalitions)]
    savevals2: list[list[float]] = [[0.0] * tot_members for _ in range(num_coalitions)]

    for c_idx in range(num_coalitions):
        # Compute each actor's value under the current coalition
        current_value: list[float] = [
            value_for_opposition(
                coalition_outputs, i, coalition_names[c_idx], coalition_names[-1]
            )
            for i in range(tot_members)
        ]
        savevals[c_idx] = copy.deepcopy(current_value)

        best_improvement = -1.0
        for n_idx in range(num_coalitions):
            potential_members = members_list[n_idx]
            potential_value: list[float] = [0.0] * tot_members
            for i in potential_members:
                potential_value[i] = value_for_opposition(
                    coalition_outputs, i, coalition_names[n_idx], coalition_names[-1])

            # Only consider the switch if every current member would improve
            all_improve = all(
                potential_value[i] > current_value[i]
                for i in potential_members
            )
            new_improvement = 0.0
            if all_improve:
                new_improvement = sum(
                    potential_value[i] - current_value[i]
                    for i in potential_members
                )

            if new_improvement >= best_improvement and new_improvement > 0:
                best_improvement = new_improvement
                savevals2[c_idx] = copy.deepcopy(potential_value)
                coalition_choice[c_idx][n_idx] = new_improvement

    summary: dict = {
        coalition_names[i]: {
            'change': coalition_choice[i],
            'self':   savevals[i],
            'best':   savevals2[i],
        }
        for i in range(num_coalitions)
    }

    # Normalise rows of the TPM to turn improvement scores into probabilities
    TPM: list[list[float]] = [row[:] for row in coalition_choice]
    for i in range(num_coalitions):
        row_sum = sum(TPM[i])
        if row_sum > 0:
            TPM[i] = [v / row_sum for v in TPM[i]]

    return summary, TPM


# ---------------------------------------------------------------------------
# Winning coalitions
# ---------------------------------------------------------------------------

def winning_coalitions(TPM: list[list[float]]) -> list[int]:
    """Return a binary list: 1 if coalition is a 'sink' (never dominated), 0 otherwise.

    A coalition is 'winning' if:
      - At least one other coalition transitions into it (source[i] > 0), AND
      - It never transitions out to another coalition (sink[i] == 0).
    Sink nodes in the TPM represent stable equilibria — no coalition of members
    can rationally defect to improve their payoffs.
    """
    arr: ndarray = np.array(TPM)
    source = np.sum(arr, axis=0)   # total in-flow to each coalition
    sink   = np.sum(arr, axis=1)   # total out-flow from each coalition
    return [1 if (sink[i] == 0 and source[i] > 0) else 0
            for i in range(arr.shape[0])]


def is_winning_minimal(coalition_names: list[str], winning: list[int]) -> int:
    """Return 1 if all winning coalitions are minimal, 0 otherwise.

    A winning coalition is minimal if no proper subset of it is also a winning coalition.
    Minimality is a desirable property: it means no redundant members are included.
    """
    for i, w in enumerate(winning):
        if w == 1:
            members_i = set(ast.literal_eval(coalition_names[i]))
            for j in range(len(winning)):
                members_j = set(ast.literal_eval(coalition_names[j]))
                # If i strictly contains j, then i is not minimal
                if members_i > members_j:
                    return 0
    return 1


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

def run_coalition_analysis(
    inputs: dict,
    tol: float = 1e-6,
    max_iter: int = 500,
    correction: float = 0.01,
) -> tuple[dict, odict, dict]:
    """Run the full system analysis followed by coalition analysis.

    Calls ``run_full_analysis`` on the full system, enumerates feasible coalitions,
    then runs ``coalition_trad`` on each.

    Parameters
    ----------
    inputs : dict
        Base parameter dict from ``setup()``.
    tol : float
        Solver convergence tolerance.
    max_iter : int
        Maximum solver iterations.
    correction : float
        Solver damping factor.

    Returns
    -------
    full_inputs : dict         -- full system with all derived variables
    coalitions : OrderedDict   -- feasible coalitions
    coalition_outputs : dict   -- per-coalition analysis dicts
    """
    run_full_analysis(inputs, tol=tol, max_iter=max_iter, correction=correction)
    coalitions: odict = feasible_coalitions(inputs)
    coalition_outputs: dict = coalition_trad(inputs, coalitions)
    return inputs, coalitions, coalition_outputs
