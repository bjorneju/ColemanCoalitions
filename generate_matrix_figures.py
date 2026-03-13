"""
generate_matrix_figures.py — Heatmap figures for all input matrices in the paper examples.

For each example, produces:
  - Control matrix (Blues colormap, values sum to 1 across actors)
  - Directed interest matrix (RdBu colormap, signed, attitude × interest magnitude)

Examples with random inputs show only the fixed matrix:
  - Example 8: control fixed, interest random  → only control heatmap
  - Example 9: interest fixed, control random  → only directed interest heatmap

Saves all figures to figures/paper_examples/.
Run with:  python generate_matrix_figures.py
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = os.path.join("figures", "paper_examples")
os.makedirs(OUTDIR, exist_ok=True)

# ── Labels ────────────────────────────────────────────────────────────────────
ISSUES_3 = ["Issue I", "Issue II", "Issue III"]
ISSUES_4 = ["Issue I", "Issue II", "Issue III", "Issue IV"]
ISSUES_5 = ["Issue I", "Issue II", "Issue III", "Issue IV", "Issue V"]
ACTORS_3 = ["A", "B", "C"]
ACTORS_4 = ["A", "B", "C", "D"]
ACTORS_5 = ["A", "B", "C", "D", "E"]


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _heatmap(ax, data: np.ndarray, row_labels: list, col_labels: list,
             title: str, cmap: str, vmin: float, vmax: float) -> None:
    """Draw an annotated heatmap on *ax*."""
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_rows, n_cols = data.shape
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Actors", fontsize=11, labelpad=8)
    ax.set_ylabel("Issues", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=16)

    # Cell annotations — choose text colour based on background brightness
    span = vmax - vmin if vmax != vmin else 1.0
    for i in range(n_rows):
        for j in range(n_cols):
            norm = (data[i, j] - vmin) / span          # 0→1
            # For diverging maps the centre is white; use distance from 0.5
            if cmap == "RdBu":
                darkness = abs(norm - 0.5) * 2         # 0 = white, 1 = dark
            else:
                darkness = norm                         # sequential: high = dark
            color = "white" if darkness > 0.60 else "black"
            ax.text(j, i, f"{data[i, j]:.2f}",
                    ha="center", va="center", fontsize=10, color=color)


def plot_pair(ctrl: list, d_interest: list | None,
              actor_labels: list, issue_labels: list,
              ctrl_title: str = "Control matrix",
              di_title: str = "Directed interest matrix",
              filename: str | None = None) -> None:
    """Side-by-side control + directed-interest heatmaps (or single panel)."""
    n = 1 if d_interest is None else 2
    w = 4.8 * n + 0.6
    h = max(3.2, 0.75 * len(issue_labels) + 1.8)
    fig, axes = plt.subplots(1, n, figsize=(w, h))
    if n == 1:
        axes = [axes]

    # Control matrix (always shown)
    ctrl_arr = np.array(ctrl)
    _heatmap(axes[0], ctrl_arr, issue_labels, actor_labels,
             ctrl_title, cmap="Blues", vmin=0, vmax=ctrl_arr.max() * 1.05 or 1)

    # Directed interest matrix (when provided)
    if d_interest is not None:
        di_arr = np.array(d_interest)
        absmax = max(np.abs(di_arr).max(), 1e-6)
        _heatmap(axes[1], di_arr, issue_labels, actor_labels,
                 di_title, cmap="RdBu", vmin=-absmax, vmax=absmax)

    fig.tight_layout()
    if filename:
        path = os.path.join(OUTDIR, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved → {path}")


def plot_single(data: list, row_labels: list, col_labels: list,
                title: str, cmap: str,
                filename: str) -> None:
    """Single heatmap panel (for cases with only one fixed matrix)."""
    arr = np.array(data)
    w = max(4.5, 1.1 * len(col_labels) + 1.0)
    h = max(3.2, 0.75 * len(row_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(w, h))
    absmax = max(np.abs(arr).max(), 1e-6) if cmap == "RdBu" else None
    vmin = -absmax if cmap == "RdBu" else 0
    vmax = absmax if cmap == "RdBu" else (arr.max() * 1.05 or 1)
    _heatmap(ax, arr, row_labels, col_labels, title, cmap, vmin, vmax)
    fig.tight_layout()
    path = os.path.join(OUTDIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Matrix data (mirroring paper_examples.py) ─────────────────────────────────

def uniform_ctrl(q, n):
    return [[1 / n] * n for _ in range(q)]


# ── Example 1 — uniform, all positive ────────────────────────────────────────
print("Example 1")
plot_pair(
    ctrl=uniform_ctrl(3, 3),
    d_interest=[[1/3, 1/3, 1/3]] * 3,
    actor_labels=ACTORS_3, issue_labels=ISSUES_3,
    filename="ex1_matrices.png",
)

# ── Example 2 — conflicting attitudes, uniform interest ───────────────────────
print("Example 2")
att2 = np.array([[+1, +1, -1],
                 [+1, -1, +1],
                 [-1, +1, +1]], dtype=float)
plot_pair(
    ctrl=uniform_ctrl(3, 3),
    d_interest=(att2 * (1 / 3)).tolist(),
    actor_labels=ACTORS_3, issue_labels=ISSUES_3,
    filename="ex2_matrices.png",
)

# ── Example 3 — A flips attitude on issue I ───────────────────────────────────
print("Example 3")
att3 = np.array([[-1, +1, -1],
                 [+1, -1, +1],
                 [-1, +1, +1]], dtype=float)
plot_pair(
    ctrl=uniform_ctrl(3, 3),
    d_interest=(att3 * (1 / 3)).tolist(),
    actor_labels=ACTORS_3, issue_labels=ISSUES_3,
    filename="ex3_matrices.png",
)

# ── Example 4 — same attitudes, skewed interest weights ──────────────────────
print("Example 4")
interest4 = np.array([[0.10, 0.10, 0.10],
                      [0.20, 0.20, 0.20],
                      [0.70, 0.70, 0.70]])
plot_pair(
    ctrl=uniform_ctrl(3, 3),
    d_interest=(att3 * interest4).tolist(),
    actor_labels=ACTORS_3, issue_labels=ISSUES_3,
    filename="ex4_matrices.png",
)

# ── Example 5 — cycling loop, 4 actors ───────────────────────────────────────
print("Example 5")
di5 = [[ 0.21, -0.15,  0.27, -0.49],
       [ 0.06, -0.44,  0.57, -0.19],
       [-0.73,  0.41,  0.16,  0.32]]
ctrl5 = [[0.49, 0.07, 0.38, 0.06]] * 3
plot_pair(ctrl5, di5, actor_labels=ACTORS_4, issue_labels=ISSUES_3,
          filename="ex5_matrices.png")

# ── Example 6 — loop broken by adding Issue IV ───────────────────────────────
print("Example 6")
d5_arr = np.array(di5)
new_issue = np.array([[-0.50, 0.15, 0.50, 0.20]])
d6_raw = np.vstack([d5_arr, new_issue])
col_abs = np.sum(np.abs(d6_raw), axis=0, keepdims=True)
di6 = (d6_raw / col_abs).tolist()
ctrl6 = [[0.49, 0.07, 0.38, 0.06]] * 4
plot_pair(ctrl6, di6, actor_labels=ACTORS_4, issue_labels=ISSUES_4,
          filename="ex6_matrices.png")

# ── Example 7 — path dependence ──────────────────────────────────────────────
print("Example 7")
di7 = [[ 0.05,  0.24,  0.32,  0.24],
       [ 0.20, -0.53, -0.23, -0.43],
       [ 0.75,  0.23, -0.45,  0.32]]
ctrl7 = [[0.33, 0.16, 0.21, 0.30]] * 3
plot_pair(ctrl7, di7, actor_labels=ACTORS_4, issue_labels=ISSUES_3,
          filename="ex7_matrices.png")

# ── Example 8 — control fixed, interest random: show control only ─────────────
print("Example 8")
ctrl8 = [[0.05, 0.40, 0.26, 0.25, 0.04]] * 3
plot_single(ctrl8, row_labels=ISSUES_3, col_labels=ACTORS_5,
            title="Control matrix (fixed; interests drawn randomly)",
            cmap="Blues", filename="ex8_control.png")

# ── Example 9 — directed interest fixed, control random: show DI only ─────────
print("Example 9")
di9 = [[ 0.10, -0.22, -0.13,  0.09],
       [ 0.20,  0.33, -0.25,  0.09],
       [-0.20, -0.11,  0.38, -0.27],
       [-0.10,  0.11,  0.13,  0.18],
       [ 0.40,  0.22,  0.13,  0.36]]
plot_single(di9, row_labels=ISSUES_5, col_labels=ACTORS_4,
            title="Directed interest matrix (fixed; control drawn randomly)",
            cmap="RdBu", filename="ex9_directed_interest.png")

print(f"\nAll matrix figures saved to: {OUTDIR}/")
