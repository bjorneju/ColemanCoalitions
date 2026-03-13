# -*- coding: utf-8 -*-
"""
ColemanFunctions.py

Implementation of James Coleman's mathematical framework for collective action
and coalition analysis, based on:
  Coleman, J.S. (1973). The Mathematics of Collective Action. Aldine, Chicago.
  Coleman, J.S. (1970). The Benefits of Coalition. Public Choice, 8, 45-61.

All matrices use numpy ndarray (not the deprecated np.matrix).
Matrix multiplication is written explicitly with @ (matmul operator).
"""

import ast
import copy
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import OrderedDict as odict


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def normalize_matrix_row(matrix):
    """Normalize a 2-D matrix so each row sums to 1 (by absolute value),
    or normalize a 1-D vector so it sums to 1."""
    normed = matrix.copy()
    if normed.ndim == 2:
        for i in range(normed.shape[0]):
            row_sum = np.sum(np.abs(normed[i, :]))
            if row_sum != 0:
                normed[i, :] = normed[i, :] / row_sum
    elif normed.ndim == 1:
        total = np.sum(np.abs(normed))
        if total != 0:
            normed = normed / total
    return normed


# ---------------------------------------------------------------------------
# Input / setup functions
# ---------------------------------------------------------------------------

def SetUp(n, q, m, y, a, c):
    """Build the base input dict from raw matrices.

    Parameters
    ----------
    n : int   — number of actors
    q : int   — number of events
    m : int   — number of resources
    y : ndarray (n, q) — directed interests (sign = direction, magnitude = strength)
    a : ndarray (q, m) — resource requirements of each event
    c : ndarray (m, n) — control of resource k by actor j
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    return {
        'n': [n,   'number of actors [n]'],
        'q': [q,   'Number of events [q]'],
        'm': [m,   'Number of resources [m]'],
        'y': [y,   'Directed interests of actor j in event i [y(n,q)]'],
        'x': [np.abs(y), 'Real interests of actor j in event i [x(n,q)]'],
        's': [np.sign(y), 'Directedness of interests of actor j in event i [s(n,q)]'],
        'a': [a,   'Requirement of event i of resource k [a(q,m)]'],
        'c': [c,   'Control of resource k by actor j [c(m,n)]'],
    }


def CheckParameters(inputs):
    """Normalize interest, resource-requirement and control matrices if needed."""
    y = inputs['y'][0]
    a = inputs['a'][0]
    c = inputs['c'][0]
    n = inputs['n'][0]
    q = inputs['q'][0]
    m = inputs['m'][0]

    if not np.allclose(np.sum(np.abs(y), axis=1), 1.0):
        inputs['y'][0] = normalize_matrix_row(y)
        inputs['x'][0] = np.abs(inputs['y'][0])

    if not np.allclose(np.sum(a, axis=1), 1.0):
        inputs['a'][0] = normalize_matrix_row(a)

    if not np.allclose(np.sum(c, axis=1), 1.0):
        inputs['c'][0] = normalize_matrix_row(c)

    return inputs


# ---------------------------------------------------------------------------
# Canonical example datasets
# ---------------------------------------------------------------------------

def standard_variables_techno():
    """Standard variables for 'techno' example (Coleman problem 1)."""
    c = np.array([
        [1.000, 0.000, 0.000],
        [0.500, 0.250, 0.250],
        [0.333, 0.500, 0.167],
        [0.100, 0.200, 0.700],
    ])  # (m=4, n=3)

    y = np.array([
        [0.75, 0.25],
        [0.25, 0.75],
        [0.10, 0.90],
    ])  # (n=3, q=2)
    x = normalize_matrix_row(np.abs(y))

    a = np.array([
        [0.400, 0.333, 0.167, 0.100],
        [0.250, 0.500, 0.125, 0.125],
    ])  # (q=2, m=4)

    n = x.shape[0]  # 3
    q = x.shape[1]  # 2
    m = a.shape[1]  # 4
    return SetUp(n, q, m, y, a, c)


def standard_variables_techno2():
    """Standard variables for 'techno2' example."""
    c = np.eye(3)  # (m=3, n=3)

    y = np.array([
        [0.500, 0.500],
        [0.333, 0.667],
        [0.750, 0.250],
    ])  # (n=3, q=2)
    x = normalize_matrix_row(np.abs(y))

    a = np.array([
        [0.500, 0.250, 0.250],
        [0.500, 0.333, 0.167],
    ])  # (q=2, m=3)

    n = x.shape[0]
    q = x.shape[1]
    m = a.shape[1]
    return SetUp(n, q, m, y, a, c)


def standard_variables_trad12():
    """Standard variables from 'trad12' example (Coleman)."""
    ct = np.array([
        [0.3, 0.4, 0.4, 0.25],
        [0.2, 0.3, 0.4, 0.35],
        [0.5, 0.3, 0.2, 0.40],
    ])  # (n=3, m=4)
    c = ct.T  # (m=4, n=3)

    y = np.array([
        [ 0.4,  0.2,  0.1,  0.3],
        [-0.3,  0.3,  0.2,  0.2],
        [-0.1, -0.3, -0.5,  0.1],
    ])  # (n=3, q=4)
    x = normalize_matrix_row(np.abs(y))

    n = x.shape[0]  # 3
    q = x.shape[1]  # 4
    m = q            # resources = events (identity a)
    a = np.eye(q)   # (q, m)

    return SetUp(n, q, m, y, a, c)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solver_rwv(inputs):
    """Iteratively solve for actor power (r), event value (v), resource value (w).

    Uses Coleman's exchange model equilibrium conditions.
    Returns dict with keys 'r', 'v', 'w'.
    """
    c = inputs['c'][0]
    a = inputs['a'][0]
    x = inputs['x'][0]
    n = inputs['n'][0]
    m = inputs['m'][0]
    q = inputs['q'][0]

    # Initial random guesses
    r = normalize_matrix_row(np.random.rand(n))
    w = normalize_matrix_row(np.random.rand(m))
    v = normalize_matrix_row(np.random.rand(q))

    tol = 1e-6
    max_iter = 100
    correction = 0.01

    i = 1
    corre = 1.0
    is_identity_a = (a.shape[0] == a.shape[1] and np.allclose(a, np.eye(a.shape[0])))

    if is_identity_a:
        # Resource-free case: w is trivially 1
        w = np.ones(m)
        while corre > tol and i < max_iter:
            v_ny = normalize_matrix_row(r @ x)    # (n,)@(n,q) -> (q,)
            r    = v_ny @ c                        # (q,)@(q,n) -> (n,) [since m=q]
            error_v = v_ny - v
            v = v + error_v
            corre = np.sum(np.abs(error_v))
            i += 1
        if i >= max_iter:
            print('warning! solver did not converge!')
    else:
        while corre > tol and i < max_iter:
            r_ny = normalize_matrix_row(w @ c)    # (m,)@(m,n) -> (n,)
            w_ny = normalize_matrix_row(v @ a)    # (q,)@(q,m) -> (m,)
            v_ny = normalize_matrix_row(r @ x)    # (n,)@(n,q) -> (q,)

            r = w_ny @ c                          # (m,)@(m,n) -> (n,)
            w = v_ny @ a                          # (q,)@(q,m) -> (m,)
            v = r_ny @ x                          # (n,)@(n,q) -> (q,)

            error_r = r_ny - r
            error_w = w_ny - w
            error_v = v_ny - v
            r = r + error_r * correction
            w = w + error_w * correction
            v = v + error_v * correction

            corre = (np.sum(np.abs(error_r))
                     + np.sum(np.abs(error_w))
                     + np.sum(np.abs(error_v)))
            i += 1
        if i >= max_iter:
            print('warning! solver did not converge!')

    return {
        'r': [r, 'Total power of actor j [r(n)]'],
        'v': [v, 'Value of each event i [v(q)]'],
        'w': [w, 'Value of each resource k [w(m)]'],
    }


# ---------------------------------------------------------------------------
# Derived variable functions
# ---------------------------------------------------------------------------

def ConstitutionalControl(inputs):
    """Constitutional control of event i by actor j: C = a @ c  [shape (q,n)]."""
    C = inputs['a'][0] @ inputs['c'][0]
    return {'C': [C, 'Constitutional control over event i by actor j [C(q,n)]']}


def FractionOfResources(inputs):
    """Fraction of resource k used towards event i: F[k,i] = a[i,k]*v[i]/w[k]."""
    a = inputs['a'][0]
    v = inputs['v'][0]
    w = inputs['w'][0]
    m = inputs['m'][0]
    q = inputs['q'][0]
    F = np.zeros((m, q))
    for k in range(m):
        for i in range(q):
            F[k, i] = a[i, k] * v[i] / w[k]
    return {'F': [F, 'Fraction of resource k used towards event i [F(m,q)]']}


def Control_ActorEvent(inputs):
    """Final directed control of actor j over event i: c_AE[j,i] = y[j,i]*r[j]/v[i]."""
    y   = inputs['y'][0]
    r   = inputs['r'][0]
    v   = inputs['v'][0]
    n   = inputs['n'][0]
    q   = inputs['q'][0]
    c_AE = np.zeros((n, q))
    for j in range(n):
        for i in range(q):
            c_AE[j, i] = y[j, i] * r[j] / v[i]
    return {'c_AE': [c_AE, 'Final control of actor j of each event i [c_AE(n,q)]']}


def DerivedInterest(inputs):
    """Derived interest of actor j in resource k: B[j,k] = sum_i x[j,i]*a[i,k]."""
    x = inputs['x'][0]
    a = inputs['a'][0]
    n = inputs['n'][0]
    m = inputs['m'][0]
    q = inputs['q'][0]
    B = np.zeros((n, m))
    for j in range(n):
        for k in range(m):
            B[j, k] = np.sum(x[j, :] * a[:, k])
    return {'B': [B, 'Derived interest of each actor j in each resource k [B(n,m)]']}


def Control_ActorResource(inputs):
    """Final control of actor j over resource k: c_AR[j,k] = B[j,k]*r[j]/w[k]."""
    B = inputs['B'][0]
    r = inputs['r'][0]
    w = inputs['w'][0]
    n = inputs['n'][0]
    m = inputs['m'][0]
    c_AR = np.zeros((n, m))
    for j in range(n):
        for k in range(m):
            c_AR[j, k] = B[j, k] * r[j] / w[k]
    return {'c_AR': [c_AR, 'Final control of actor j over resource k [c_AR(n,m)]']}


def Control_ActorActor(inputs):
    """Direct control of actor h over actor j: z[j,h] = sum_k B[j,k]*c[k,h]."""
    B = inputs['B'][0]
    c = inputs['c'][0]
    n = inputs['n'][0]
    m = inputs['m'][0]
    z = np.zeros((n, n))
    for j in range(n):
        for h in range(n):
            z[j, h] = np.sum(B[j, :] * c[:, h])
    return {
        'z':    [z, 'Final control of actor j by actor h [z(n,n)]'],
        'c_AA': [z, 'Final control of actor j by actor h [c_AA(n,n)]'],
    }


def Control_EventEvent(inputs):
    """Control of event i by event j: c_EE = c @ x  [shape (m,q)]."""
    c_EE = inputs['c'][0] @ inputs['x'][0]
    return {'c_EE': [c_EE, 'Control of event i by event j [c_EE(m,q)]']}


def PositiveOutcome(inputs):
    """Probability of positive outcome for each event: P_p = 0.5 + 0.5*sum_j(c_AE)."""
    c_AE = inputs['c_AE'][0]
    P_p = 0.5 + 0.5 * np.sum(c_AE, axis=0)
    return {'P_p': [P_p, 'Probability of positive outcome of event i [P_p(q)]']}


def Increment_ExpectedRealization(inputs):
    """Increment in expected realization of interests of actor h from actor j.

    p_hj[h,j] = 0.5 * sum_i s[h,i]*x[h,i]*|c_AE[j,i]|*s[j,i]
    """
    s    = inputs['s'][0]
    x    = inputs['x'][0]
    c_AE = inputs['c_AE'][0]
    q    = inputs['q'][0]
    n    = inputs['n'][0]
    p_hj = np.zeros((n, n))
    for j in range(n):
        for h in range(n):
            for i in range(q):
                p_hj[h, j] += s[h, i] * x[h, i] * abs(c_AE[j, i]) * s[j, i]
            p_hj[h, j] *= 0.5
    return {'p_hj': [p_hj, 'Increment in expected realization of interests of actor h from j']}


def ExpectedValueOfCollectivity(inputs):
    """Expected value of collectivity for each actor: p_h = sum_j(p_hj) + 0.5."""
    p_hj = inputs['p_hj'][0]
    p_h  = np.sum(p_hj, axis=0) + 0.5
    return {'p_h': [p_h, 'Expected value of collectivity [p_h(n)]']}


def ExpectedWeightedRealization(inputs):
    """Expected weighted realization of interests across all actors: d_i = sum_j(p_h*r)."""
    p_h = inputs['p_h'][0]
    r   = inputs['r'][0]
    d_i = np.sum(np.multiply(p_h, r))
    return {'d_i': [d_i, 'Expected weighted realization of interests of all actors']}


def DirectedPowerOfCollectivity(inputs):
    """Directed power of collectivity on each event: d = r @ y  [shape (q,)]."""
    r = inputs['r'][0]
    y = inputs['y'][0]
    d = r @ y
    return {'d': [d, 'Directed power of collectivity on event [d(q)]']}


def TotalPowerOfCollectivity(inputs):
    """Total external power of collectivity: R = sum|d|."""
    d = inputs['d'][0]
    R = np.sum(np.abs(d))
    return {'R': [R, 'Total external power of collectivity']}


def MatchingAttitudes(y):
    """Measure of how well actor attitudes match across events."""
    s = np.sign(y)
    n = float(len(s))
    avg = np.mean(np.abs(np.sum(s, axis=0) / n))
    minimum = 1 / n if n % 2 == 1 else 0.0
    rang = 1 - minimum
    return (avg - minimum) / rang


# ---------------------------------------------------------------------------
# Coalition functions
# ---------------------------------------------------------------------------

def FeasibleCoalitions(inputs):
    """Enumerate all feasible (majority) coalitions of size >= 2.

    Returns an OrderedDict mapping coalition-string keys to [member_list, control].
    """
    n = inputs['n'][0]
    c = inputs['c'][0]

    if n < 9:
        LOLI = np.unpackbits(
            np.arange(2 ** n).astype(np.uint8)[:, None], axis=1
        )[:, -n:]
    elif n < 19:
        LOLI = []
        for i in range(2 ** n):
            bits = format(i, '#0' + str(n + 2) + 'b')
            LOLI.append([int(b) for b in bits[2:]])
        LOLI = np.array(LOLI)
    else:
        raise ValueError('FeasibleCoalitions: n must be <= 18 (too many coalitions for n>18)')

    actors     = np.arange(1, n + 1)
    coalitions = actors * LOLI
    CoalitionDict = odict()

    for i in range(2 ** n):
        coal = [j for j in range(n) if coalitions[i, j] > 0]
        if len(coal) < 2:
            continue
        # Coalition control = sum of first resource row across members
        # (proxy for combined influence; see note in literature_notes.md §8)
        coalition_control = sum(c[0, cc] for cc in coal)
        if coalition_control > 0.5:
            CoalitionDict[str(coal)] = [coal, coalition_control]

    return CoalitionDict


def CoalitionTrad(inputs, coalitions):
    """Run full Coleman analysis for each feasible coalition as a sub-collective."""
    q = inputs['q'][0]
    m = inputs['m'][0]
    y = inputs['y'][0]
    c = inputs['c'][0]
    a = inputs['a'][0]
    outputs = {}
    for key, (idx, _) in coalitions.items():
        n_coal = len(idx)
        yy = y[idx, :]
        cc = c[:, idx]
        subsystem = SetUp(n_coal, q, m, yy, a, cc)
        subsystem.update(RunFullAnalysis(subsystem))
        outputs[key] = subsystem
    return outputs


def CoalitionTechno(inputs, coalitions):
    """Run techno-style analysis treating each coalition as a single composite actor."""
    q  = inputs['q'][0]
    m  = len(coalitions)
    n  = inputs['n'][0]
    y  = inputs['y'][0]
    cc = inputs['c'][0]
    a  = np.zeros((q, m))
    c  = np.zeros((m, n))

    total_control = 0.0
    for i, (key, (idx, _)) in enumerate(coalitions.items()):
        coalition_control = 0.0
        for actor_id in idx:
            c[i, actor_id] = np.sum(cc[:, actor_id]) / q
            coalition_control += c[i, actor_id]
        total_control += coalition_control
        for event_id in range(q):
            a[event_id, i] = coalition_control / max(total_control, 1e-12)

    result = SetUp(n, q, m, y, a, c)
    result.update(RunFullAnalysis(result))
    return result


def CoalitionTechno_EqualControl(inputs, coalitions):
    """Techno analysis with equal within-coalition control distribution."""
    q  = inputs['q'][0]
    m  = len(coalitions)
    n  = inputs['n'][0]
    y  = inputs['y'][0]
    cc = inputs['c'][0]
    a  = np.zeros((q, m))
    c  = np.zeros((m, n))

    for i, (key, (idx, _)) in enumerate(coalitions.items()):
        members = float(len(idx))
        for actor_id in idx:
            c[i, actor_id] = 1.0 / members
        for event_id in range(q):
            a[event_id, i] = np.sum(cc[idx, event_id])

    result = SetUp(n, q, m, y, a, c)
    result.update(RunFullAnalysis(result))
    return result


def RunCoalitionAnalysis_noOutput(n, q, m, y, a, c):
    """Run the full coalition analysis and return trad coalition outputs."""
    outputs = SetUp(n, q, m, y, a, c)
    outputs.update(RunFullAnalysis(outputs))
    coalitions = FeasibleCoalitions(outputs)
    return CoalitionTrad(outputs, coalitions)


def ValueForOpposition(inputs, actor, current_key, full_key):
    """Value of current coalition configuration to a specific actor."""
    P  = inputs[current_key]['P_p'][0]
    y  = inputs[full_key]['y'][0][actor, :]
    return np.sum(np.multiply(P * 2 - 1, y)) / float(y.size)


def ValueOfCoalition(inputs):
    """Compute the value of each coalition to its members, opposition, and collective."""
    keys    = list(inputs.keys())
    n       = inputs[keys[-1]]['n'][0]
    full_key = keys[-1]
    coal_value = {}

    for key in keys:
        members = ast.literal_eval(key)   # e.g. "[0, 1, 2]" -> [0, 1, 2]
        vals = [ValueForOpposition(inputs, i, key, full_key) for i in range(n)]
        val_coal = sum(vals[i] for i in members)
        coal_value[key] = {
            'to actors':     vals,
            'to coalition':  val_coal,
            'to opposition': sum(vals) - val_coal,
            'to collective': sum(vals),
        }
    return coal_value


def OptimalCoalition(inputs):
    """For each coalition, find the coalition that maximises member payoffs.

    Returns (summary dict, TPM transition probability matrix as list-of-lists).
    """
    coalition_names  = list(inputs.keys())
    num_coalitions   = len(coalition_names)
    members_list     = [ast.literal_eval(k) for k in coalition_names]
    tot_members      = max(len(m) for m in members_list)

    coalition_choice = [[0] * num_coalitions for _ in range(num_coalitions)]
    savevals  = [[0] * tot_members for _ in range(num_coalitions)]
    savevals2 = [[0] * tot_members for _ in range(num_coalitions)]

    for c_idx in range(num_coalitions):
        current_members = members_list[c_idx]
        current_value   = [0.0] * tot_members
        for i in range(tot_members):
            current_value[i] = ValueForOpposition(
                inputs, i, coalition_names[c_idx], coalition_names[-1])
        savevals[c_idx] = copy.deepcopy(current_value)

        best_improvement = -1.0
        for n_idx in range(num_coalitions):
            potential_members = members_list[n_idx]
            potential_value   = [0.0] * tot_members
            for i in potential_members:
                potential_value[i] = ValueForOpposition(
                    inputs, i, coalition_names[n_idx], coalition_names[-1])

            # Only consider if all current members improve
            all_improve = all(potential_value[i] > current_value[i]
                              for i in potential_members)
            new_improvement = 0.0
            if all_improve:
                new_improvement = sum(potential_value[i] - current_value[i]
                                      for i in potential_members)

            if new_improvement >= best_improvement and new_improvement > 0:
                best_improvement = new_improvement
                savevals2[c_idx] = copy.deepcopy(potential_value)
                coalition_choice[c_idx][n_idx] = new_improvement

    summary = {
        coalition_names[i]: {
            'change': coalition_choice[i],
            'self':   savevals[i],
            'best':   savevals2[i],
        }
        for i in range(num_coalitions)
    }

    # Build transition probability matrix
    TPM = [row[:] for row in coalition_choice]
    for i in range(num_coalitions):
        row_sum = sum(TPM[i])
        if row_sum > 0:
            TPM[i] = [v / row_sum for v in TPM[i]]

    return summary, TPM


def WinningCoalitions(TPM):
    """Identify sink nodes in the TPM (winning / stable coalitions).

    A coalition is 'winning' if it receives transitions but never sends any
    (i.e. no coalition beats it).
    """
    TPM_arr  = np.array(TPM)
    num_coal = TPM_arr.shape[0]
    source   = np.sum(TPM_arr, axis=0)
    sink     = np.sum(TPM_arr, axis=1)
    return [1 if (sink[i] == 0 and source[i] > 0) else 0
            for i in range(num_coal)]


def IsWinningMinimal(coalition_names, winning):
    """Check whether all winning coalitions are minimal (no proper subset also wins)."""
    for i, w in enumerate(winning):
        if w == 1:
            for j, _ in enumerate(winning):
                if set(coalition_names[i]) > set(coalition_names[j]):
                    return 0
    return 1


# ---------------------------------------------------------------------------
# Analysis pipelines
# ---------------------------------------------------------------------------

def Trad12ExampleAnalysis():
    """Full analysis on the canonical trad12 example."""
    outputs = standard_variables_trad12()
    outputs.update(solver_rwv(outputs))
    outputs.update(ConstitutionalControl(outputs))
    outputs.update(DerivedInterest(outputs))
    outputs.update(Control_ActorActor(outputs))
    outputs.update(Control_ActorEvent(outputs))
    outputs.update(Control_EventEvent(outputs))
    outputs.update(PositiveOutcome(outputs))
    outputs.update(Increment_ExpectedRealization(outputs))
    outputs.update(ExpectedValueOfCollectivity(outputs))
    outputs.update(ExpectedWeightedRealization(outputs))
    outputs.update(DirectedPowerOfCollectivity(outputs))
    outputs.update(TotalPowerOfCollectivity(outputs))
    return outputs


def TechnoExampleAnalysis():
    """Full analysis on the canonical techno example."""
    outputs = standard_variables_techno()
    outputs.update(solver_rwv(outputs))
    outputs.update(FractionOfResources(outputs))
    outputs.update(Control_ActorEvent(outputs))
    outputs.update(DerivedInterest(outputs))
    outputs.update(Control_ActorResource(outputs))
    outputs.update(Control_ActorActor(outputs))
    return outputs


def RunFullAnalysis(outputs):
    """Run the complete Coleman analysis pipeline."""
    outputs.update(CheckParameters(outputs))
    outputs.update(solver_rwv(outputs))
    outputs.update(FractionOfResources(outputs))
    outputs.update(Control_ActorEvent(outputs))
    outputs.update(DerivedInterest(outputs))
    outputs.update(Control_ActorResource(outputs))
    outputs.update(Control_EventEvent(outputs))
    outputs.update(Control_ActorActor(outputs))
    outputs.update(ConstitutionalControl(outputs))
    outputs.update(PositiveOutcome(outputs))
    outputs.update(Increment_ExpectedRealization(outputs))
    outputs.update(ExpectedValueOfCollectivity(outputs))
    outputs.update(ExpectedWeightedRealization(outputs))
    outputs.update(DirectedPowerOfCollectivity(outputs))
    outputs.update(TotalPowerOfCollectivity(outputs))
    return outputs


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def DrawCoalitionMap(inputs, summary, TPM):
    """Draw a directed graph of coalition transitions.

    Returns (fig, node_sizes_raw, node_sizes_scaled).
    Safe to call in headless environments (does not call plt.show()).
    """
    TPM_arr   = np.array(TPM)
    num_coal  = TPM_arr.shape[0]
    names     = list(inputs.keys())

    conns      = [(names[i], names[j], TPM_arr[i, j])
                  for i in range(num_coal) for j in range(num_coal)
                  if TPM_arr[i, j] > 0]
    node_sizes = [np.sum(summary[coal]['self']) for coal in names]
    nS = 1000 * (0.1 + (node_sizes - np.min(node_sizes))
                 / max(np.max(node_sizes) - np.min(node_sizes), 1e-12))
    winner = WinningCoalitions(TPM)

    G = nx.DiGraph()
    G.add_nodes_from(names)
    G.add_weighted_edges_from(conns)
    weights = [3 * G[u][v]['weight'] for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos=nx.kamada_kawai_layout(G), ax=ax,
            with_labels=True, node_size=nS, width=weights,
            arrowsize=20, node_color=winner, alpha=0.4, font_size=14)
    return fig, node_sizes, nS


def DrawOnlyStrongest(inputs, summary, TPM):
    """Draw only the strongest (most likely) transition from each coalition.

    Returns (fig, node_sizes_raw, node_sizes_scaled).
    Safe to call in headless environments.
    """
    TPM_arr  = np.array(TPM)
    num_coal = TPM_arr.shape[0]
    names    = list(inputs.keys())

    conns = [(names[i], names[j], TPM_arr[i, j])
             for i in range(num_coal) for j in range(num_coal)
             if TPM_arr[i, j] == np.max(TPM_arr[i]) and TPM_arr[i, j] > 0]
    node_sizes = [np.sum(summary[coal]['self']) for coal in names]
    nS = 5000 * (0.1 + (node_sizes - np.min(node_sizes))
                 / max(np.max(node_sizes) - np.min(node_sizes), 1e-12))
    winner = WinningCoalitions(TPM)

    G = nx.DiGraph()
    G.add_nodes_from(names)
    G.add_weighted_edges_from(conns)
    weights = [3 * G[u][v]['weight'] for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos=nx.spring_layout(G), ax=ax,
            with_labels=True, node_size=nS, width=weights,
            arrowsize=20, node_color=winner, alpha=0.4, font_size=14)
    return fig, node_sizes, nS


def imagesc_varyInterests(data, titles, mask):
    """Heatmap grid for varying interest scenarios."""
    n_plots = len(data)
    cols = 4
    rows = max(1, (n_plots + cols - 1) // cols)  # dynamic rows
    iterator = list(range(len(data[0])))

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    for i in range(n_plots):
        ax = fig.add_subplot(rows, cols, i + 1)
        plotdata = [[data[i][a][b] * mask[a][b] for a in iterator] for b in iterator]
        im = ax.imshow(plotdata, extent=[-1, 1, -1, 1], vmin=-1, vmax=1, cmap='seismic')
        ax.set_title(titles[i])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def CalculateMetrics(orig_outputs, outputs, metrics):
    """Calculate difference between final directed control and interest."""
    if not metrics:
        metrics = {'control': []}
    control = np.mean(outputs['c_AE'][0] - orig_outputs['y'][0], axis=1)
    metrics['control'].append(control.tolist())
    return metrics


# ---------------------------------------------------------------------------
# Output / I/O functions
# ---------------------------------------------------------------------------

def _output_variable(heading, data, f, newlines):
    if np.size(data) > 1:
        data = np.asarray(data)
        print(heading, file=f)
        if data.ndim == 1:
            np.savetxt(f, data.reshape(1, -1), fmt='%.3f')
        else:
            for row in data:
                np.savetxt(f, np.asarray(row).reshape(1, -1), fmt='%.3f')
    else:
        print(f"{heading}: {float(data):.3f}", file=f)
    for _ in range(newlines):
        print('', file=f)


def TechnoOutput_file(filepath, title, inputs):
    """Write full techno analysis to a timestamped text file."""
    filename = os.path.join(filepath, time.strftime('%Y%m%d') + '_' + title + '.txt')
    with open(filename, 'w+') as f:
        header = f"{'*'*44}\n  {title}\n  Created: {time.strftime('%d/%m/%Y %H:%M:%S')}\n{'*'*44}\n"
        print(header, file=f)
        for key, val in inputs.items():
            _output_variable(val[1], val[0], f, 1)
        print(header, file=f)


def OrderedOutput_file(filepath, title, inputs, order):
    """Write selected variables (in given order) to a timestamped text file."""
    filename = os.path.join(filepath, time.strftime('%Y%m%d') + '_' + title + '.txt')
    with open(filename, 'w+') as f:
        header = f"{'*'*44}\n  {title}\n  Created: {time.strftime('%d/%m/%Y %H:%M:%S')}\n{'*'*44}\n"
        print(header, file=f)
        for key in order:
            _output_variable(inputs[key][1], inputs[key][0], f, 1)
        print(header, file=f)


def Outputvariables_screen(heading, data, newlines):
    """Print a variable to stdout."""
    data = np.asarray(data)
    if data.size > 1:
        print(heading)
        print(np.array2string(data, precision=3))
    else:
        print(f"{heading}: {float(data):.3f}")
    for _ in range(newlines):
        print()


def OrderedOutput_screen(inputs, order):
    """Print selected variables to stdout."""
    for key in order:
        Outputvariables_screen(inputs[key][1], inputs[key][0], 1)
