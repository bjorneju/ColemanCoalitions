"""
paper_examples.py — Reproduce all worked examples from the draft manuscript
"Coalition paper - draft manus summer 2019"

Examples 1–4: 3 actors, 3 issues  (attitude / interest / control tables)
Example 5:    4 actors, 3 issues  — cycling loop, no stable coalition
Example 6:    4 actors, 4 issues  — breaking the loop by adding one issue
Example 7:    4 actors, 3 issues  — path-dependent winning coalition
Example 8:    5 actors, 3 issues  — 1 000 MC sims, min-winning not always best
Example 9:    4 actors, 5 issues  — 1 000 MC sims, varying control distribution

All figures are saved to figures/paper_examples/.
Run with:  python paper_examples.py
"""

from __future__ import annotations

import os
import ast
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import coleman_coalitions as cc
from coleman_coalitions.solver import SolverConvergenceError

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTDIR = os.path.join("figures", "paper_examples")
os.makedirs(OUTDIR, exist_ok=True)

RNG = np.random.default_rng(42)   # reproducible Monte Carlo runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(
    directed_issues_actors: list,   # (q x n) signed interest  [issues × actors]
    c_issues_actors: list,          # (q x n) control          [issues × actors]
) -> dict:
    """Build an inputs dict from paper-convention matrices (issues × actors).

    In the API, y is (n × q) and c is (m × n) = (q × n).  So we transpose y
    but pass c directly (the paper's control table rows already index issues).
    """
    y_ia = np.array(directed_issues_actors, dtype=float)   # (q, n)
    c_ia = np.array(c_issues_actors, dtype=float)          # (q, n) = (m, n)
    q, n = y_ia.shape
    y = y_ia.T          # (n, q)
    c = c_ia            # (m, n), m = q
    a = np.eye(q)       # identity: resource k ↔ event k
    inputs = cc.setup(n, q, q, y, a, c)
    cc.check_parameters(inputs)
    return inputs


def _run(inputs: dict) -> tuple[dict, dict, list]:
    """Full analysis + coalition analysis.  Returns (coalition_outputs, summary, TPM)."""
    cc.run_full_analysis(inputs)
    coalitions = cc.feasible_coalitions(inputs)
    coalition_outputs = cc.coalition_trad(inputs, coalitions)
    summary, TPM = cc.optimal_coalition(coalition_outputs)
    return coalition_outputs, summary, TPM


def _winning_names(coalition_outputs: dict, TPM: list, actor_names: list[str]) -> list[str]:
    names = list(coalition_outputs.keys())
    winning = cc.winning_coalitions(TPM)
    return [
        '+'.join(actor_names[i] for i in ast.literal_eval(k))
        for k, w in zip(names, winning) if w
    ]


def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(OUTDIR, filename)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def _print_header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


def _uniform_control(q: int, n: int) -> list:
    """Uniform control: each actor controls 1/n of each issue."""
    return [[1 / n] * n for _ in range(q)]


# ---------------------------------------------------------------------------
# Example 1 — No conflict, no gain
# ---------------------------------------------------------------------------
_print_header("Example 1: No conflict, no gain")

# All attitudes positive, equal interests, equal control
_d1 = [[1/3, 1/3, 1/3],   # I
        [1/3, 1/3, 1/3],   # II
        [1/3, 1/3, 1/3]]   # III (directed = attitude × interest, all +)

inputs1 = _make_inputs(_d1, _uniform_control(3, 3))
co1, sm1, TPM1 = _run(inputs1)

print("Feasible coalitions:", list(co1.keys()))
print("Winners:", _winning_names(co1, TPM1, list('ABC')))

fig = cc.draw_coalition_table(co1, sm1, TPM1,
                              actor_names=list('ABC'),
                              title='Example 1 — No conflict, no gain')
_save(fig, 'ex1_table.png')

fig = cc.draw_coalition_map(co1, sm1, TPM1, actor_labels=list('ABC'))
fig.suptitle('Example 1 — coalition map', fontweight='bold')
_save(fig, 'ex1_coalition_map.png')


# ---------------------------------------------------------------------------
# Example 2 — Conflicting attitudes, symmetric
# ---------------------------------------------------------------------------
_print_header("Example 2: Conflicting attitudes")

# Attitudes (Table 6): A(+,+,−), B(+,−,+), C(−,+,+)
# interest uniform 1/3, so directed = attitude × 1/3
_att2 = np.array([[+1, +1, -1],   # issue I
                   [+1, -1, +1],   # issue II
                   [-1, +1, +1]],  # issue III
                  dtype=float)
_d2 = (_att2 * (1/3)).tolist()

inputs2 = _make_inputs(_d2, _uniform_control(3, 3))
co2, sm2, TPM2 = _run(inputs2)

print("Feasible coalitions:", list(co2.keys()))
print("Winners:", _winning_names(co2, TPM2, list('ABC')))

fig = cc.draw_coalition_table(co2, sm2, TPM2,
                              actor_names=list('ABC'),
                              title='Example 2 — Conflicting attitudes')
_save(fig, 'ex2_table.png')

fig = cc.draw_coalition_map(co2, sm2, TPM2, actor_labels=list('ABC'))
fig.suptitle('Example 2 — coalition map', fontweight='bold')
_save(fig, 'ex2_coalition_map.png')


# ---------------------------------------------------------------------------
# Example 3 — Agreement on most issues determines the winner
# ---------------------------------------------------------------------------
_print_header("Example 3: Agreement on most issues wins (A+C expected)")

# Attitudes (Table 10): A(−,+,−), B(+,−,+), C(−,+,+)
# A & B oppose on all 3; A & C agree on I and II; B & C agree on III
_att3 = np.array([[-1, +1, -1],   # I
                   [+1, -1, +1],   # II
                   [-1, +1, +1]],  # III
                  dtype=float)
_d3 = (_att3 * (1/3)).tolist()

inputs3 = _make_inputs(_d3, _uniform_control(3, 3))
co3, sm3, TPM3 = _run(inputs3)

print("Feasible coalitions:", list(co3.keys()))
print("Winners:", _winning_names(co3, TPM3, list('ABC')))

fig = cc.draw_coalition_table(co3, sm3, TPM3,
                              actor_names=list('ABC'),
                              title='Example 3 — Agreement on most issues wins')
_save(fig, 'ex3_table.png')

fig = cc.draw_strongest_transitions(co3, sm3, TPM3, actor_labels=list('ABC'))
fig.suptitle('Example 3 — strongest transitions', fontweight='bold')
_save(fig, 'ex3_transitions.png')


# ---------------------------------------------------------------------------
# Example 4 — Intensity of interest can override alignment
# ---------------------------------------------------------------------------
_print_header("Example 4: Interest intensity overrides alignment (B+C expected)")

# Same attitudes as Example 3; interest weights: I=0.10, II=0.20, III=0.70
_interest4 = np.array([[0.10, 0.10, 0.10],   # I
                        [0.20, 0.20, 0.20],   # II
                        [0.70, 0.70, 0.70]])  # III (all actors weight issues equally)
_att3_arr = np.array([[-1, +1, -1],
                       [+1, -1, +1],
                       [-1, +1, +1]], dtype=float)
_d4 = (_att3_arr * _interest4).tolist()

inputs4 = _make_inputs(_d4, _uniform_control(3, 3))
co4, sm4, TPM4 = _run(inputs4)

print("Feasible coalitions:", list(co4.keys()))
print("Winners:", _winning_names(co4, TPM4, list('ABC')))

fig = cc.draw_coalition_table(co4, sm4, TPM4,
                              actor_names=list('ABC'),
                              title='Example 4 — Interest intensity overrides alignment')
_save(fig, 'ex4_table.png')

fig = cc.draw_strongest_transitions(co4, sm4, TPM4, actor_labels=list('ABC'))
fig.suptitle('Example 4 — strongest transitions', fontweight='bold')
_save(fig, 'ex4_transitions.png')


# ---------------------------------------------------------------------------
# Example 5 — Cycling loop: no stable coalition
# ---------------------------------------------------------------------------
_print_header("Example 5: Cycling loop — no stable coalition")

# Directed interest matrix (Table 16) in issues × actors convention
_d5 = [[ 0.21, -0.15,  0.27, -0.49],   # I
        [ 0.06, -0.44,  0.57, -0.19],   # II
        [-0.73,  0.41,  0.16,  0.32]]   # III

# Control matrix (Table 17): same row for all issues
_ctrl5 = [[0.49, 0.07, 0.38, 0.06],
           [0.49, 0.07, 0.38, 0.06],
           [0.49, 0.07, 0.38, 0.06]]

inputs5 = _make_inputs(_d5, _ctrl5)
co5, sm5, TPM5 = _run(inputs5)

print("Feasible coalitions:", list(co5.keys()))
print("Winners:", _winning_names(co5, TPM5, list('ABCD')))

fig = cc.draw_coalition_table(co5, sm5, TPM5,
                              actor_names=list('ABCD'),
                              title='Example 5 — Cycling loop (no stable coalition)')
_save(fig, 'ex5_table.png')

fig = cc.draw_coalition_map(co5, sm5, TPM5, actor_labels=list('ABCD'), figsize=(10, 7))
fig.suptitle('Example 5 — coalition transitions (cycling)', fontweight='bold')
_save(fig, 'ex5_coalition_map.png')

fig = cc.draw_strongest_transitions(co5, sm5, TPM5, actor_labels=list('ABCD'), figsize=(10, 7))
fig.suptitle('Example 5 — strongest transitions (cycling)', fontweight='bold')
_save(fig, 'ex5_transitions.png')


# ---------------------------------------------------------------------------
# Example 6 — Breaking the loop by adding a new issue
# ---------------------------------------------------------------------------
_print_header("Example 6: Breaking the loop — add Issue IV where B, C, D agree")

# Take Example 5's interests, append Issue IV where A strongly opposes
# and B, C, D all support.  Renormalise rows to keep |y| summing to 1.
_d5_arr = np.array(_d5)             # (3, 4) issues × actors
_new_issue = np.array([[-0.50, 0.15, 0.50, 0.20]])   # (1, 4)
_d6_raw = np.vstack([_d5_arr, _new_issue])            # (4, 4)

# Row-normalise per actor (columns here = actors)
# Normalisation must be applied column-wise (each actor's values across issues)
col_abs = np.sum(np.abs(_d6_raw), axis=0, keepdims=True)
_d6 = (_d6_raw / col_abs).tolist()

_ctrl6 = [[0.49, 0.07, 0.38, 0.06],
           [0.49, 0.07, 0.38, 0.06],
           [0.49, 0.07, 0.38, 0.06],
           [0.49, 0.07, 0.38, 0.06]]   # same control distribution for Issue IV

inputs6 = _make_inputs(_d6, _ctrl6)
co6, sm6, TPM6 = _run(inputs6)

print("Feasible coalitions:", list(co6.keys()))
print("Winners:", _winning_names(co6, TPM6, list('ABCD')))

fig = cc.draw_coalition_table(co6, sm6, TPM6,
                              actor_names=list('ABCD'),
                              title='Example 6 — Loop broken by adding Issue IV')
_save(fig, 'ex6_table.png')

fig = cc.draw_strongest_transitions(co6, sm6, TPM6, actor_labels=list('ABCD'), figsize=(10, 7))
fig.suptitle('Example 6 — strongest transitions (loop broken)', fontweight='bold')
_save(fig, 'ex6_transitions.png')


# ---------------------------------------------------------------------------
# Example 7 — Path dependence
# ---------------------------------------------------------------------------
_print_header("Example 7: Path dependence")

# Directed interest (Table 20) issues × actors
_d7 = [[ 0.05,  0.24,  0.32,  0.24],   # I
        [ 0.20, -0.53, -0.23, -0.43],   # II
        [ 0.75,  0.23, -0.45,  0.32]]   # III

# Control (Table 19): same distribution across all issues
_ctrl7 = [[0.33, 0.16, 0.21, 0.30],
           [0.33, 0.16, 0.21, 0.30],
           [0.33, 0.16, 0.21, 0.30]]

inputs7 = _make_inputs(_d7, _ctrl7)
co7, sm7, TPM7 = _run(inputs7)

print("Feasible coalitions:", list(co7.keys()))
print("Winners:", _winning_names(co7, TPM7, list('ABCD')))

fig = cc.draw_coalition_table(co7, sm7, TPM7,
                              actor_names=list('ABCD'),
                              title='Example 7 — Path dependence')
_save(fig, 'ex7_table.png')

fig = cc.draw_coalition_map(co7, sm7, TPM7, actor_labels=list('ABCD'), figsize=(12, 8))
fig.suptitle('Example 7 — coalition map (multiple sinks = path dependence)',
             fontweight='bold')
_save(fig, 'ex7_coalition_map.png')

fig = cc.draw_strongest_transitions(co7, sm7, TPM7, actor_labels=list('ABCD'), figsize=(12, 8))
fig.suptitle('Example 7 — strongest transitions', fontweight='bold')
_save(fig, 'ex7_transitions.png')

# Trace dominant paths from two starting coalitions
_names7 = list(co7.keys())
_actor7 = list('ABCD')
_TPM7_arr = np.array(TPM7)


def _trace(TPM_arr, names, actor_names, start_key, max_steps=15):
    """Follow strongest transition from start_key until reaching a sink or loop."""
    path_keys = [start_key]
    current = names.index(start_key)
    for _ in range(max_steps):
        row = TPM_arr[current]
        if row.max() == 0:
            break
        nxt = int(row.argmax())
        if names[nxt] in path_keys:
            break
        path_keys.append(names[nxt])
        current = nxt
    return ['+'.join(actor_names[i] for i in ast.literal_eval(k)) for k in path_keys]


# Starting from C+D (key '[2, 3]') and A+C+D (key '[0, 2, 3]')
for start_key, label in [('[2, 3]', 'C+D'), ('[0, 2, 3]', 'A+C+D')]:
    if start_key in _names7:
        path = _trace(_TPM7_arr, _names7, _actor7, start_key)
        print(f"  Path from {label}: {' → '.join(path)}")
    else:
        print(f"  Note: coalition {label} not feasible with these parameters.")


# ---------------------------------------------------------------------------
# Example 8 — Minimum winning not always best (5 actors, Monte Carlo)
# ---------------------------------------------------------------------------
_print_header("Example 8: Min-winning not always best — 1 000 simulations (5 actors)")

# Control fixed: A=0.05, B=0.40, C=0.26, D=0.25, E=0.04  (Wikipedia Riker example)
_ctrl8_row = [0.05, 0.40, 0.26, 0.25, 0.04]
_ctrl8 = [_ctrl8_row, _ctrl8_row, _ctrl8_row]   # 3 issues, same control

N_SIM = 1_000
_n8, _q8 = 5, 3
_actor8 = list('ABCDE')

win_counts8: dict[str, int] = {}
n_conv8 = 0

for _ in range(N_SIM):
    # Random directed interest matrix: uniform in [-1, 1], then row-normalised
    raw = RNG.uniform(-1, 1, size=(_q8, _n8))
    col_abs = np.sum(np.abs(raw), axis=0, keepdims=True)
    col_abs[col_abs == 0] = 1
    y_ia = (raw / col_abs).tolist()

    try:
        inp = _make_inputs(y_ia, _ctrl8)
        co, sm, TPM = _run(inp)
        for key, w in zip(co.keys(), cc.winning_coalitions(TPM)):
            if w:
                members = ast.literal_eval(key)
                label = '+'.join(_actor8[i] for i in members)
                win_counts8[label] = win_counts8.get(label, 0) + 1
        n_conv8 += 1
    except (SolverConvergenceError, Exception):
        pass

print(f"  Converged: {n_conv8}/{N_SIM}")
print("  Win counts:", dict(sorted(win_counts8.items(), key=lambda x: -x[1])[:8]))

# Bar chart of winning frequencies
_labels8 = sorted(win_counts8, key=lambda k: -win_counts8[k])
_counts8 = [win_counts8[k] for k in _labels8]

fig, ax = plt.subplots(figsize=(max(8, len(_labels8) * 0.7 + 1), 5))
bars = ax.bar(_labels8, _counts8, color='steelblue', alpha=0.85)
# Highlight the Riker minimum-winning coalition (C+D)
for bar, lbl in zip(bars, _labels8):
    if lbl == 'C+D':
        bar.set_color('darkorange')
ax.set_xlabel('Coalition')
ax.set_ylabel(f'Wins (out of {n_conv8} simulations)')
ax.set_title('Example 8 — Winning coalition frequency (control fixed, interests random)',
             fontweight='bold')
ax.tick_params(axis='x', rotation=45)
blue_patch = mpatches.Patch(color='steelblue', alpha=0.85, label='Other coalitions')
orange_patch = mpatches.Patch(color='darkorange', label='C+D (Riker min-winning)')
ax.legend(handles=[blue_patch, orange_patch])
fig.tight_layout()
_save(fig, 'ex8_mc_bar.png')


# ---------------------------------------------------------------------------
# Example 9 — Varying control (4 actors, 5 issues, fixed interests)
# ---------------------------------------------------------------------------
_print_header("Example 9: Varying control — 1 000 simulations (4 actors)")

# Directed interest fixed (Table 23): issues × actors
_d9 = [[ 0.10, -0.22, -0.13,  0.09],   # I
        [ 0.20,  0.33, -0.25,  0.09],   # II
        [-0.20, -0.11,  0.38, -0.27],   # III
        [-0.10,  0.11,  0.13,  0.18],   # IV
        [ 0.40,  0.22,  0.13,  0.36]]   # V

_n9, _q9 = 4, 5
_actor9 = list('ABCD')

win_counts9: dict[str, int] = {}
n_conv9 = 0

for _ in range(N_SIM):
    # Random control: one distribution per issue, each sums to 1
    raw_ctrl = RNG.uniform(0, 1, size=(_q9, _n9))
    row_sums = raw_ctrl.sum(axis=1, keepdims=True)
    c_rand = (raw_ctrl / row_sums).tolist()

    try:
        inp = _make_inputs(_d9, c_rand)
        co, sm, TPM = _run(inp)
        for key, w in zip(co.keys(), cc.winning_coalitions(TPM)):
            if w:
                members = ast.literal_eval(key)
                label = '+'.join(_actor9[i] for i in members)
                win_counts9[label] = win_counts9.get(label, 0) + 1
        n_conv9 += 1
    except (SolverConvergenceError, Exception):
        pass

print(f"  Converged: {n_conv9}/{N_SIM}")
print("  Win counts:", dict(sorted(win_counts9.items(), key=lambda x: -x[1])[:8]))

_labels9 = sorted(win_counts9, key=lambda k: -win_counts9[k])
_counts9 = [win_counts9[k] for k in _labels9]

fig, ax = plt.subplots(figsize=(max(8, len(_labels9) * 0.7 + 1), 5))
ax.bar(_labels9, _counts9, color='mediumseagreen', alpha=0.85)
ax.set_xlabel('Coalition')
ax.set_ylabel(f'Wins (out of {n_conv9} simulations)')
ax.set_title('Example 9 — Winning coalition frequency (interests fixed, control random)',
             fontweight='bold')
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
_save(fig, 'ex9_mc_bar.png')

# ---------------------------------------------------------------------------
print(f"\nAll figures saved to: {OUTDIR}/")
