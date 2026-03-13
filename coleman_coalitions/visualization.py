"""
coleman_coalitions/visualization.py

Visualization functions for Coleman coalition analysis results.

All functions return matplotlib Figure objects and do NOT call plt.show().
This makes them safe for headless/server environments and easy to embed in
notebooks or save to file without triggering a display.
"""
from __future__ import annotations

import ast
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import matplotlib.figure
import networkx as nx

from .coalitions import winning_coalitions


def draw_coalition_map(
    coalition_outputs: dict,
    summary: dict,
    TPM: list[list[float]],
    figsize: tuple[int, int] = (12, 8),
) -> matplotlib.figure.Figure:
    """Draw a directed graph of all coalition transitions.

    Each node is a feasible coalition; edge weight encodes the transition
    probability from the TPM.  Node size reflects the coalition's total
    self-value; winning (sink) coalitions are highlighted.

    Parameters
    ----------
    coalition_outputs : dict   -- output of coalition_trad()
    summary : dict             -- output of optimal_coalition()
    TPM : list[list[float]]    -- row-normalised transition probability matrix
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    TPM_arr: ndarray = np.array(TPM)
    num_coal: int = TPM_arr.shape[0]
    names: list[str] = list(coalition_outputs.keys())

    # Build edge list from non-zero entries of the TPM
    conns: list[tuple] = [
        (names[i], names[j], TPM_arr[i, j])
        for i in range(num_coal) for j in range(num_coal)
        if TPM_arr[i, j] > 0
    ]

    # Node size proportional to total actor value (self-value of the coalition)
    node_sizes: ndarray = np.array([np.sum(summary[coal]['self']) for coal in names])
    node_range = node_sizes.max() - node_sizes.min()
    # Scale to [100, 1100] so all nodes are visible
    nS: ndarray = 1000 * (0.1 + (node_sizes - node_sizes.min()) / max(node_range, 1e-12))

    # Winning coalitions are coloured differently (non-zero value in nx.draw node_color)
    winner: list[int] = winning_coalitions(TPM)

    G: nx.DiGraph = nx.DiGraph()
    G.add_nodes_from(names)
    G.add_weighted_edges_from(conns)
    # Line width scaled by edge weight (transition probability)
    weights: list[float] = [3 * G[u][v]['weight'] for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(
        G, pos=nx.kamada_kawai_layout(G), ax=ax,
        with_labels=True, node_size=nS, width=weights,
        arrowsize=20, node_color=winner, alpha=0.4, font_size=14,
    )
    ax.set_title('Coalition transition map')
    return fig


def draw_strongest_transitions(
    coalition_outputs: dict,
    summary: dict,
    TPM: list[list[float]],
    figsize: tuple[int, int] = (12, 8),
) -> matplotlib.figure.Figure:
    """Draw only the strongest (most likely) outgoing transition from each coalition.

    Produces a cleaner graph than draw_coalition_map by showing only the dominant
    edge from each node — useful for quickly identifying transition chains.

    Parameters
    ----------
    coalition_outputs : dict
    summary : dict
    TPM : list[list[float]]
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    TPM_arr: ndarray = np.array(TPM)
    num_coal: int = TPM_arr.shape[0]
    names: list[str] = list(coalition_outputs.keys())

    # Keep only the maximum-weight outgoing edge per node
    conns: list[tuple] = [
        (names[i], names[j], TPM_arr[i, j])
        for i in range(num_coal) for j in range(num_coal)
        if TPM_arr[i, j] == np.max(TPM_arr[i]) and TPM_arr[i, j] > 0
    ]

    node_sizes: ndarray = np.array([np.sum(summary[coal]['self']) for coal in names])
    node_range = node_sizes.max() - node_sizes.min()
    nS: ndarray = 5000 * (0.1 + (node_sizes - node_sizes.min()) / max(node_range, 1e-12))
    winner: list[int] = winning_coalitions(TPM)

    G: nx.DiGraph = nx.DiGraph()
    G.add_nodes_from(names)
    G.add_weighted_edges_from(conns)
    weights: list[float] = [3 * G[u][v]['weight'] for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(
        G, pos=nx.spring_layout(G, seed=42), ax=ax,
        with_labels=True, node_size=nS, width=weights,
        arrowsize=20, node_color=winner, alpha=0.4, font_size=14,
    )
    ax.set_title('Strongest coalition transitions')
    return fig


def draw_interest_heatmap(
    data: list,
    titles: list[str],
    mask: list,
    figsize: tuple[int, int] | None = None,
    matrix_type: str = 'interest',
) -> matplotlib.figure.Figure:
    """Grid of heatmaps for varying-interest scenarios.

    Each panel shows a 2-D interest or control matrix for one scenario,
    masked and colour-coded by sign.  Useful for exploring how outcomes
    change as actor interests vary across a parameter sweep.

    Parameters
    ----------
    data : list of 2-D array-like  -- one matrix per scenario
    titles : list[str]             -- panel titles (one per scenario)
    mask : 2-D array-like          -- binary mask applied to each data matrix
    figsize : tuple or None        -- defaults to (5*cols, 5*rows)
    matrix_type : str
        ``'interest'`` (default): directed interest, vmin=-1, vmax=1,
        diverging red–blue colormap.
        ``'control'``: control values, vmin=0, vmax=1, light-to-dark gray
        colormap.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if matrix_type == 'control':
        cmap, vmin, vmax = 'Greys', 0.0, 1.0
    else:
        cmap, vmin, vmax = 'seismic', -1.0, 1.0

    n_plots: int = len(data)
    cols: int = 4
    rows: int = max(1, (n_plots + cols - 1) // cols)
    if figsize is None:
        figsize = (5 * cols, 5 * rows)
    iterator = list(range(len(data[0])))
    n: int = len(iterator)
    # Cell-boundary positions for grid lines within the [-1, 1] extent
    grid_ticks: ndarray = np.linspace(-1, 1, n + 1)

    fig = plt.figure(figsize=figsize)
    im = None
    for idx in range(n_plots):
        ax = fig.add_subplot(rows, cols, idx + 1)
        # Apply the mask element-wise (e.g. zero out diagonal or off-diagonal entries)
        plotdata = [[data[idx][a][b] * mask[a][b] for a in iterator] for b in iterator]
        im = ax.imshow(plotdata, extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(titles[idx])
        # Grid lines at cell boundaries using minor ticks
        ax.set_xticks(grid_ticks, minor=True)
        ax.set_yticks(grid_ticks, minor=True)
        ax.grid(True, which='minor', color='k', linewidth=0.5)
        ax.tick_params(which='minor', length=0)

    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    return fig


def draw_power_distribution(
    inputs: dict,
    figsize: tuple[int, int] = (8, 4),
) -> matplotlib.figure.Figure:
    """Bar chart of actor power distribution.

    Parameters
    ----------
    inputs : dict   -- must contain 'r' (actor power) and 'n' (number of actors)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    r: ndarray = inputs['r']
    n: int = inputs['n']
    labels: list[str] = [f'Actor {i + 1}' for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, r, color='steelblue', alpha=0.8)
    ax.set_ylabel('Power (r)')
    ax.set_title('Actor power distribution')
    ax.set_ylim(0, max(r) * 1.2)
    return fig


def draw_event_values(
    inputs: dict,
    figsize: tuple[int, int] = (8, 4),
) -> matplotlib.figure.Figure:
    """Bar chart of event value distribution.

    Parameters
    ----------
    inputs : dict   -- must contain 'v' (event values) and 'q' (number of events)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    v: ndarray = inputs['v']
    q: int = inputs['q']
    labels: list[str] = [f'Event {i + 1}' for i in range(q)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, v, color='darkorange', alpha=0.8)
    ax.set_ylabel('Value (v)')
    ax.set_title('Event value distribution')
    ax.set_ylim(0, max(v) * 1.2)
    return fig


def draw_coalition_table(
    coalition_outputs: dict,
    summary: dict,
    TPM: list[list[float]] | None = None,
    actor_names: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    title: str = 'Coalition value table',
) -> matplotlib.figure.Figure:
    """Render coalition values as a formatted matplotlib table.

    Produces a table matching the paper's layout: one row per feasible
    coalition, columns for each actor's value, plus aggregate columns for
    the coalition, opposition, and collectivity.  Winning (sink) coalitions
    are highlighted in green when ``TPM`` is supplied.

    Parameters
    ----------
    coalition_outputs : dict   -- output of coalition_trad()
    summary : dict             -- output of optimal_coalition()
    TPM : list[list[float]] or None
        If provided, winning coalitions are highlighted green.
    actor_names : list[str] or None
        Labels for each actor.  Defaults to ['A', 'B', 'C', ...].
    figsize : tuple or None
        Figure size.  Auto-sized from the number of rows/columns if None.
    title : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    names: list[str] = list(coalition_outputs.keys())
    n_coalitions: int = len(names)
    n_actors: int = len(summary[names[0]]['self'])

    if actor_names is None:
        actor_names = [chr(65 + i) for i in range(n_actors)]

    winning: list[int] = winning_coalitions(TPM) if TPM is not None else [0] * n_coalitions

    col_labels: list[str] = (
        ['Coalition']
        + [f'Actor {a}' for a in actor_names]
        + ['Coalition', 'Opposition', 'Collectivity']
    )
    n_cols: int = len(col_labels)

    rows: list[list[str]] = []
    for key in names:
        members: list[int] = ast.literal_eval(key)
        s: list[float] = summary[key]['self']
        coal_val: float = sum(s[i] for i in members)
        opp_val: float = sum(s) - coal_val
        coll_val: float = sum(s)
        label: str = '+'.join(actor_names[i] for i in members)
        rows.append(
            [label]
            + [f'{v:.2f}' for v in s]
            + [f'{coal_val:.2f}', f'{opp_val:.2f}', f'{coll_val:.2f}']
        )

    if figsize is None:
        figsize = (max(6, n_cols * 1.5), max(2, n_coalitions * 0.55 + 1.0))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # Style header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor('#2c5f8a')
        cell.set_text_props(color='white', fontweight='bold')

    # Style data rows: green for winners, white otherwise; bold aggregate cols
    coal_col: int = n_actors + 1   # 'Coalition' aggregate column index
    for i, (key, w) in enumerate(zip(names, winning)):
        bg = '#d4edda' if w else '#ffffff'
        for j in range(n_cols):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg)
        tbl[i + 1, coal_col].set_text_props(fontweight='bold')

    ax.set_title(title, pad=16, fontweight='bold')
    return fig
