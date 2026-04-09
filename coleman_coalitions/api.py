"""
coleman_coalitions/api.py

High-level, JSON-friendly API for programmatic access and web application integration.

This module bridges the Coleman analysis library (which uses numpy arrays and plain dicts)
and external consumers that need JSON-serializable inputs and outputs — such as REST APIs,
Streamlit apps, or Jupyter widgets.

Typical usage
-------------
>>> from coleman_coalitions.api import AnalysisConfig, analyze
>>> config = AnalysisConfig(
...     interest_matrix=[[ 0.4,  0.2,  0.1,  0.3],
...                      [-0.3,  0.3,  0.2,  0.2],
...                      [-0.1, -0.3, -0.5,  0.1]],
...     control_matrix=[[0.3, 0.2, 0.5],       # actor × resource (rows = actors)
...                     [0.2, 0.3, 0.4],
...                     [0.5, 0.3, 0.2],
...                     [0.25, 0.35, 0.40]],
...     actor_labels=["Labour", "Greens", "Business"],
...     event_labels=["Policy A", "Policy B", "Policy C", "Policy D"],
... )
>>> result, raw = analyze(config)
>>> print(result.power)                    # actor power shares
>>> print(result.winning_coalition_indices)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .core import setup
from .analysis import run_full_analysis
from .analysis import matching_attitudes as _matching_attitudes
from .coalitions import (
    feasible_coalitions,
    coalition_trad,
    value_of_coalition,
    optimal_coalition,
    winning_coalitions,
    is_winning_minimal,
)


# ── Input configuration ────────────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    """Input configuration for a Coleman coalition analysis.

    All matrices are plain Python lists of lists so this dataclass is
    JSON-serializable.  Matrices use actor-row orientation (actors as rows) for
    both the interest and control inputs; the library internally transposes the
    control matrix to its (resource × actor) form.
    """

    interest_matrix: list[list[float]]
    """Actor × Event interest matrix — shape (n_actors, n_events).

    Positive values express preference for a positive outcome on that event;
    negative values express preference for a negative outcome.  Magnitudes
    indicate relative interest strength.  Rows are row-normalised automatically.
    """

    control_matrix: list[list[float]]
    """Actor × Resource control matrix — shape (n_actors, n_resources).

    Each entry is actor i's degree of control over resource k.  Rows are
    row-normalised automatically.  Internally transposed to (n_resources × n_actors)
    for the Coleman model.
    """

    resource_matrix: Optional[list[list[float]]] = None
    """Event × Resource requirements matrix — shape (n_events, n_resources).

    Each entry is how much event i draws on resource k.  Rows are row-normalised
    automatically.  If ``None``, an identity matrix is used (every event requires
    exactly one dedicated resource).
    """

    actor_labels: Optional[list[str]] = None
    """Human-readable label for each actor.  Defaults to 'A1', 'A2', …"""

    event_labels: Optional[list[str]] = None
    """Human-readable label for each event.  Defaults to 'E1', 'E2', …"""

    resource_labels: Optional[list[str]] = None
    """Human-readable label for each resource.  Defaults to 'R1', 'R2', …"""

    run_coalitions: bool = True
    """Whether to run the coalition analysis pipeline (feasible coalitions,
    transition probability matrix, winning coalitions).  Set to ``False`` to
    skip coalition enumeration and return only the full-system analysis."""

    tol: float = 1e-6
    """Convergence tolerance for the equilibrium solver."""

    max_iter: int = 500
    """Maximum solver iterations before raising ``SolverConvergenceError``."""


# ── Output data structures ─────────────────────────────────────────────────────

@dataclass
class CoalitionInfo:
    """Results for a single feasible coalition."""

    members: list[int]
    """Zero-indexed actor positions in this coalition."""

    member_labels: list[str]
    """Display labels for each coalition member."""

    control: float
    """Combined control of the first resource (> 0.5 means the coalition holds
    majority control and is therefore feasible)."""

    actor_values: list[float]
    """Value of this coalition's outcome distribution to *every* actor
    (all n actors, not just members).  Non-member (opposition) values are
    typically negative when their preferred outcomes are blocked."""

    value_to_coalition: float
    """Aggregate value to coalition members (sum of member entries in
    ``actor_values``)."""

    value_to_opposition: float
    """Aggregate value to non-members."""

    value_to_collective: float
    """Aggregate value to all actors (coalition + opposition)."""

    is_winning: bool = False
    """``True`` if this coalition is a stable equilibrium — a sink node in the
    transition probability matrix (no rational defection is possible)."""


@dataclass
class AnalysisResult:
    """Full results of a Coleman coalition analysis.

    All numeric data is stored as plain Python lists or floats so this
    dataclass can be serialised to JSON without further transformation.
    """

    # ── Core solved variables ────────────────────────────────────────────────
    power: list[float]
    """Actor power shares (r).  Each entry is actor i's equilibrium power.
    Values sum to 1."""

    event_values: list[float]
    """Event values (v).  Reflects how intensely each event is contested.
    Values sum to 1."""

    resource_values: list[float]
    """Resource values (w).  For identity-resource systems these are all equal
    (trivially ones after normalisation)."""

    # ── Derived measures ─────────────────────────────────────────────────────
    constitutional_control: list[list[float]]
    """Constitutional control matrix C — shape (n_events, n_actors).
    C[i, j] is actor j's structural control over event i via resource ownership."""

    actor_event_control: list[list[float]]
    """Final directed actor-event control matrix c_AE — shape (n_actors, n_events).
    Positive entries indicate push towards a positive outcome; negative towards negative."""

    outcome_probabilities: list[float]
    """Probability of a positive outcome for each event P_p — values in [0, 1].
    0.5 represents an evenly contested event."""

    expected_collective_value: list[float]
    """Expected value of the collectivity for each actor p_h.
    Higher values indicate more beneficial collective outcomes for that actor."""

    expected_weighted_realization: float
    """Power-weighted average expected realization across all actors (d_i). Scalar."""

    directed_power: list[float]
    """Directed power of the collectivity on each event (d).
    Sign indicates the net direction of collective push."""

    total_power: float
    """Total unsigned external power of the collectivity R ∈ [0, 1].
    Higher values indicate stronger collective orientation."""

    matching_attitudes: float
    """Attitude alignment measure ∈ [0, 1].  1 means all actors agree on all
    event directions; 0 means maximally split."""

    # ── Resource-level variables ─────────────────────────────────────────────
    fraction_of_resources: list[list[float]]
    """Fraction of each resource directed towards each event (F) — shape (n_resources, n_events).
    F[k, i] is the fraction of resource k that is allocated towards event i in equilibrium."""

    derived_interests: list[list[float]]
    """Derived interest of each actor in each resource (B) — shape (n_actors, n_resources).
    B[j, k] = sum_i x[j,i] * a[i,k]: how much actor j cares about resource k,
    mediated through their interests in events that require that resource."""

    actor_resource_control: list[list[float]]
    """Final control of each actor over each resource (c_AR) — shape (n_actors, n_resources).
    c_AR[j, k] = B[j,k] * r[j] / w[k]: actor j's effective control over resource k
    after exchange, weighted by their power and the resource's value."""

    actor_actor_control: list[list[float]]
    """Control of each actor over every other actor (z / c_AA) — shape (n_actors, n_actors).
    z[j, h] = sum_k B[j,k] * c[k,h]: how much actor j's interests are advanced by
    actor h's resource control — a measure of indirect influence."""

    event_event_control: list[list[float]]
    """Control of events by events mediated through resources (c_EE) — shape (n_resources, n_events).
    c_EE = c @ x: how much control over each event flows through each resource."""

    realization_increments: list[list[float]]
    """Increment in expected realization of actor h's interests due to actor j (p_hj)
    — shape (n_actors, n_actors).  p_hj[h, j] measures how much actor j's directed
    control advances (positive) or opposes (negative) actor h's interests."""

    # ── Coalition results ────────────────────────────────────────────────────
    coalitions: Optional[list[CoalitionInfo]] = None
    """All feasible coalitions with their analysis.  ``None`` if
    ``run_coalitions=False``."""

    winning_coalition_indices: Optional[list[int]] = None
    """Indices into ``coalitions`` for the winning (stable) coalitions.
    An empty list means no stable coalition exists (cycling).
    ``None`` if ``run_coalitions=False``."""

    are_winning_minimal: Optional[bool] = None
    """``True`` if every winning coalition is minimal (contains no redundant
    members).  ``None`` if ``run_coalitions=False``."""

    transition_matrix: Optional[list[list[float]]] = None
    """Row-normalised transition probability matrix (TPM).  TPM[i][j] > 0
    means all members of coalition i would improve by switching to j.
    ``None`` if ``run_coalitions=False``."""

    # ── Labels ───────────────────────────────────────────────────────────────
    actor_labels: list[str] = field(default_factory=list)
    event_labels: list[str] = field(default_factory=list)
    resource_labels: list[str] = field(default_factory=list)


@dataclass
class RawAnalysisData:
    """Internal numpy-based data retained for use with visualization functions.

    Not JSON-serializable.  Intended for use with the ``draw_*`` functions in
    ``coleman_coalitions.visualization``.

    Example
    -------
    >>> result, raw = analyze(config)
    >>> from coleman_coalitions import draw_coalition_map
    >>> fig = draw_coalition_map(raw.coalition_outputs, raw.summary, raw.tpm)
    """

    inputs: dict
    """Full analysis dict (all derived variables as numpy arrays)."""

    coalition_outputs: Optional[dict] = None
    """Per-coalition analysis dicts as returned by ``coalition_trad()``."""

    summary: Optional[dict] = None
    """Per-coalition transition summary as returned by ``optimal_coalition()``."""

    tpm: Optional[list[list[float]]] = None
    """Raw TPM as returned by ``optimal_coalition()``."""


# ── Main entry point ───────────────────────────────────────────────────────────

def analyze(config: AnalysisConfig) -> tuple[AnalysisResult, RawAnalysisData]:
    """Run the full Coleman analysis from a webapp-friendly configuration.

    Converts ``config`` into the Coleman model's internal numpy representation,
    runs the equilibrium solver and all derived-variable computations, and
    optionally runs the coalition analysis pipeline.

    Parameters
    ----------
    config : AnalysisConfig

    Returns
    -------
    result : AnalysisResult
        All outputs as plain Python types (JSON-serializable).
    raw : RawAnalysisData
        Internal numpy dicts for use with ``coleman_coalitions.visualization``.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent.
    SolverConvergenceError
        If the equilibrium solver fails to converge within ``config.max_iter`` steps.
    """
    # ── Convert inputs to numpy ──────────────────────────────────────────────
    y = np.array(config.interest_matrix, dtype=float)       # (n, q)
    c_user = np.array(config.control_matrix, dtype=float)   # (n, m) — user orientation
    c = c_user.T                                             # (m, n) — model orientation

    n_actors, n_events = y.shape
    n_resources = c.shape[0]

    if config.resource_matrix is None:
        if n_resources != n_events:
            raise ValueError(
                f"When resource_matrix is None an identity matrix is used, which requires "
                f"n_resources == n_events.  Got n_resources={n_resources}, n_events={n_events}.  "
                f"Either supply a resource_matrix or give control_matrix {n_events} column(s)."
            )
        a = np.eye(n_events)
    else:
        a = np.array(config.resource_matrix, dtype=float)
        if a.shape != (n_events, n_resources):
            raise ValueError(
                f"resource_matrix shape {a.shape} does not match expected "
                f"({n_events}, {n_resources}) = (n_events, n_resources)."
            )

    actor_labels = config.actor_labels or [f"A{i + 1}" for i in range(n_actors)]
    event_labels = config.event_labels or [f"E{i + 1}" for i in range(n_events)]
    resource_labels = config.resource_labels or [f"R{i + 1}" for i in range(n_resources)]

    # ── Build and run full analysis ───────────────────────────────────────────
    inputs = setup(n_actors, n_events, n_resources, y, a, c)
    run_full_analysis(inputs, tol=config.tol, max_iter=config.max_iter)

    # ── Optional coalition analysis ───────────────────────────────────────────
    coal_infos: Optional[list[CoalitionInfo]] = None
    winning_indices: Optional[list[int]] = None
    are_minimal: Optional[bool] = None
    tpm: Optional[list[list[float]]] = None
    coal_outputs: Optional[dict] = None
    summary_dict: Optional[dict] = None

    if config.run_coalitions:
        feasible = feasible_coalitions(inputs)
        coal_outputs = coalition_trad(inputs, feasible)
        values = value_of_coalition(coal_outputs)
        summary_dict, tpm = optimal_coalition(coal_outputs)
        winning = winning_coalitions(tpm)
        minimal = is_winning_minimal(list(coal_outputs.keys()), winning)

        coal_infos = []
        for i, (key, (members, control)) in enumerate(feasible.items()):
            v = values.get(key, {})
            coal_infos.append(CoalitionInfo(
                members=members,
                member_labels=[actor_labels[j] for j in members],
                control=control,
                actor_values=v.get("to actors", []),
                value_to_coalition=v.get("to coalition", 0.0),
                value_to_opposition=v.get("to opposition", 0.0),
                value_to_collective=v.get("to collective", 0.0),
                is_winning=bool(winning[i]),
            ))

        winning_indices = [i for i, w in enumerate(winning) if w]
        are_minimal = bool(minimal)

    # ── Assemble result ───────────────────────────────────────────────────────
    result = AnalysisResult(
        power=inputs["r"].tolist(),
        event_values=inputs["v"].tolist(),
        resource_values=inputs["w"].tolist(),
        constitutional_control=inputs["C"].tolist(),
        actor_event_control=inputs["c_AE"].tolist(),
        outcome_probabilities=inputs["P_p"].tolist(),
        expected_collective_value=inputs["p_h"].tolist(),
        expected_weighted_realization=float(inputs["d_i"]),
        directed_power=inputs["d"].tolist(),
        total_power=float(inputs["R"]),
        matching_attitudes=float(_matching_attitudes(inputs["y"])),
        fraction_of_resources=inputs["F"].tolist(),
        derived_interests=inputs["B"].tolist(),
        actor_resource_control=inputs["c_AR"].tolist(),
        actor_actor_control=inputs["z"].tolist(),
        event_event_control=inputs["c_EE"].tolist(),
        realization_increments=inputs["p_hj"].tolist(),
        coalitions=coal_infos,
        winning_coalition_indices=winning_indices,
        are_winning_minimal=are_minimal,
        transition_matrix=tpm,
        actor_labels=actor_labels,
        event_labels=event_labels,
        resource_labels=resource_labels,
    )

    raw = RawAnalysisData(
        inputs=inputs,
        coalition_outputs=coal_outputs,
        summary=summary_dict,
        tpm=tpm,
    )

    return result, raw
