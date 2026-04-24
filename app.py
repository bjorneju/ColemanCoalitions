"""
app.py — Interactive Streamlit webapp for Coleman Coalition Analysis.

Run with:
    pip install "coleman_coalitions[webapp]"
    streamlit run app.py
"""
from __future__ import annotations

import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import coleman_coalitions as cc
from coleman_coalitions.api import (
    AnalysisConfig,
    AnalysisResult,
    RawAnalysisData,
    analyze,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Coleman Coalition Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Coleman Coalition Analyzer")
st.caption(
    "Based on Coleman (1973) *The Mathematics of Collective Action*. "
    "Enter actor interests and resource control to solve for equilibrium power, "
    "coalition stability, and collective outcomes."
)


# ── Preset examples ────────────────────────────────────────────────────────────
# All paper examples use identity resource matrices (each issue backed by one resource).
# Matrices are in library convention: interest_matrix = (n_actors × n_events),
# control_matrix = (n_actors × n_resources) — rows will be normalised automatically.

def _build_presets() -> dict:
    unif3 = (np.ones((3, 3)) / 3).tolist()

    # ── Examples 1–4: 3 actors (A, B, C), 3 issues ──────────────────────────
    e1_y = unif3  # all positive, uniform → no conflict
    e1_c = unif3

    att2 = np.array([[+1,+1,-1],[+1,-1,+1],[-1,+1,+1]], dtype=float) / 3
    e2_y = att2.T.tolist()   # (n×q): actor A → [+I,+II,−III], etc.
    e2_c = unif3

    att3 = np.array([[-1,+1,-1],[+1,-1,+1],[-1,+1,+1]], dtype=float) / 3
    e3_y = att3.T.tolist()
    e3_c = unif3

    int4 = np.array([[.10,.10,.10],[.20,.20,.20],[.70,.70,.70]])
    att3_arr = np.array([[-1,+1,-1],[+1,-1,+1],[-1,+1,+1]], dtype=float)
    e4_y = (att3_arr * int4).T.tolist()
    e4_c = unif3

    # ── Examples 5–7: 4 actors (A, B, C, D), identity resources ─────────────
    d5 = np.array([[ 0.21,-0.15, 0.27,-0.49],
                   [ 0.06,-0.44, 0.57,-0.19],
                   [-0.73, 0.41, 0.16, 0.32]])        # (3,4) issues×actors
    ctrl5 = np.array([[0.49,0.07,0.38,0.06]]*3)       # (3,4) resources×actors
    e5_y = d5.T.tolist()        # (4,3) actors×issues
    e5_c = ctrl5.T.tolist()     # (4,3) actors×resources

    d6_raw = np.vstack([d5, np.array([[-0.50,0.15,0.50,0.20]])])  # (4,4)
    d6 = d6_raw / np.sum(np.abs(d6_raw), axis=0, keepdims=True)
    ctrl6 = np.array([[0.49,0.07,0.38,0.06]]*4)
    e6_y = d6.T.tolist()
    e6_c = ctrl6.T.tolist()

    d7 = np.array([[ 0.05, 0.24, 0.32, 0.24],
                   [ 0.20,-0.53,-0.23,-0.43],
                   [ 0.75, 0.23,-0.45, 0.32]])
    ctrl7 = np.array([[0.33,0.16,0.21,0.30]]*3)
    e7_y = d7.T.tolist()
    e7_c = ctrl7.T.tolist()

    abc  = ["A", "B", "C"]
    abcd = ["A", "B", "C", "D"]
    iss3 = ["Issue I", "Issue II", "Issue III"]
    iss4 = ["Issue I", "Issue II", "Issue III", "Issue IV"]

    return {
        "Example 1 — No conflict, no gain": dict(
            description=(
                "All actors have identical, positive attitudes on every issue and equal "
                "control. No actor gains from forming a coalition — the outcome is the same "
                "regardless of who cooperates."
            ),
            actor_labels=abc, event_labels=iss3, resource_labels=iss3,
            interest_matrix=e1_y, control_matrix=e1_c, resource_matrix=None,
        ),
        "Example 2 — Symmetric conflict": dict(
            description=(
                "Each actor disagrees with the other two on exactly one issue. "
                "All two-actor coalitions are equally valuable, so the first movers win — "
                "but it does not matter which coalition forms."
            ),
            actor_labels=abc, event_labels=iss3, resource_labels=iss3,
            interest_matrix=e2_y, control_matrix=e2_c, resource_matrix=None,
        ),
        "Example 3 — Agreement on most issues wins": dict(
            description=(
                "A and C agree on two of three issues; all other pairs agree on only one. "
                "The A+C coalition dominates and is the unique stable outcome."
            ),
            actor_labels=abc, event_labels=iss3, resource_labels=iss3,
            interest_matrix=e3_y, control_matrix=e3_c, resource_matrix=None,
        ),
        "Example 4 — Interest intensity overrides alignment": dict(
            description=(
                "Same attitudes as Example 3, but Issue III carries 70 % of each actor's "
                "interest weight. This reverses the winning coalition: B+C now wins because "
                "they agree on the issue that matters most."
            ),
            actor_labels=abc, event_labels=iss3, resource_labels=iss3,
            interest_matrix=e4_y, control_matrix=e4_c, resource_matrix=None,
        ),
        "Example 5 — Cycling loop (no stable coalition)": dict(
            description=(
                "With 4 actors and heterogeneous interests, no coalition is stable: the "
                "transition graph cycles endlessly. No winning coalition exists."
            ),
            actor_labels=abcd, event_labels=iss3, resource_labels=iss3,
            interest_matrix=e5_y, control_matrix=e5_c, resource_matrix=None,
        ),
        "Example 6 — Loop broken by adding Issue IV": dict(
            description=(
                "Adding a fourth issue on which B, C, and D agree breaks the cycle from "
                "Example 5 and produces a unique stable winning coalition."
            ),
            actor_labels=abcd, event_labels=iss4, resource_labels=iss4,
            interest_matrix=e6_y, control_matrix=e6_c, resource_matrix=None,
        ),
        "Example 7 — Path dependence": dict(
            description=(
                "Multiple stable winning coalitions coexist. Which one actually forms "
                "depends on the starting point — a canonical illustration of path dependence."
            ),
            actor_labels=abcd, event_labels=iss3, resource_labels=iss3,
            interest_matrix=e7_y, control_matrix=e7_c, resource_matrix=None,
        ),
    }


PRESETS = _build_presets()

# ── Default labels and matrices for the custom mode ────────────────────────────
# Norwegian political parties example (5 actors, 4 events, 3 resources).
# Events are coded so that positive = more of the policy (e.g. stronger climate
# action, higher taxes/welfare, more rural support, more open immigration).
# Control shares are calibrated from the 2021 Storting election among these five
# parties (Høyre 36, KrF 3, Ap 48, Sp 28, MDG 3 seats out of 118 total).

_DEF_ACTORS    = ["Høyre", "KrF", "Arbeiderpartiet", "Senterpartiet", "MDG"]
_DEF_EVENTS    = ["Klimapolitikk", "Skatt og velferd", "Distriktspolitikk", "Innvandring"]
_DEF_RESOURCES = ["Stortingsseter", "Medieinnflytelse", "Velgeroppslutning"]

_DEF_Y = [
    [ 0.20, -0.50, -0.20, -0.30],  # Høyre:          mild climate, anti-tax, anti-district, restrictive imm.
    [ 0.30,  0.10,  0.30,  0.10],  # KrF:            moderate climate+district, some welfare, open imm.
    [ 0.30,  0.50,  0.20,  0.00],  # Arbeiderpartiet: pro-climate+welfare+district, neutral imm.
    [ 0.10,  0.20,  0.60, -0.30],  # Senterpartiet:   weak climate, some welfare, strong district, restrictive
    [ 0.70,  0.20,  0.00,  0.30],  # MDG:            strongly pro-climate, some welfare, neutral, open imm.
]

_DEF_C = [
    [0.31, 0.28, 0.25],  # Høyre:           significant seats+media, moderate voter base
    [0.03, 0.08, 0.05],  # KrF:             few seats, limited media and voters
    [0.41, 0.30, 0.35],  # Arbeiderpartiet: most seats, strong media and voter base
    [0.24, 0.20, 0.25],  # Senterpartiet:   significant seats and voter base
    [0.03, 0.14, 0.10],  # MDG:             few seats, notable media presence
]

_DEF_A = [
    [0.15, 0.35, 0.50],  # Klimapolitikk:    driven by public opinion and media
    [0.55, 0.20, 0.25],  # Skatt og velferd: primarily a parliamentary decision
    [0.40, 0.20, 0.40],  # Distriktspolitikk: seats + voter base in rural areas
    [0.25, 0.45, 0.30],  # Innvandring:       media framing + parliamentary majority
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
) -> plt.Figure:
    arr = np.array(matrix)
    h = max(2.5, len(row_labels) * 0.65 + 0.8)
    w = max(3.0, len(col_labels) * 0.85 + 1.2)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.75)
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(
                    j, i, f"{arr[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(arr[i, j]) > 0.5 * (vmax or 1) else "black",
                )
    ax.set_title(title, fontsize=10, pad=8)
    plt.tight_layout()
    return fig


def _bar(values: list[float], labels: list[str], title: str, color: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(3, len(labels) * 0.7 + 1), 2.8))
    ax.bar(labels, values, color=color, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Value", fontsize=8)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    return fig


def _narrative(text: str, label: str = "What does this mean?") -> None:
    with st.expander(label, expanded=False):
        st.markdown(text)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System setup")

    mode = st.radio(
        "Input mode",
        ["📋  Load preset example", "✏️  Create your own system"],
        index=1,
        label_visibility="collapsed",
    )
    st.divider()

    # ── Preset mode ───────────────────────────────────────────────────────────
    if mode == "📋  Load preset example":
        example_name = st.selectbox("Choose example", list(PRESETS.keys()), label_visibility="collapsed")
        preset = PRESETS[example_name]
        st.info(preset["description"])

        actor_labels    = preset["actor_labels"]
        event_labels    = preset["event_labels"]
        resource_labels = preset["resource_labels"]
        use_identity    = preset["resource_matrix"] is None
        n_actors        = len(actor_labels)
        n_events        = len(event_labels)
        n_resources     = len(resource_labels)

    # ── Custom mode ───────────────────────────────────────────────────────────
    else:
        n_actors    = int(st.number_input("Actors",           min_value=2, max_value=18, value=5, step=1))
        n_events    = int(st.number_input("Events / issues",  min_value=1, max_value=20, value=4, step=1))
        use_identity = st.checkbox(
            "Events = Resources (identity)",
            value=False,
            help="When unchecked, you can define a custom Event × Resource requirements matrix.",
        )
        if use_identity:
            n_resources = n_events
        else:
            n_resources = int(st.number_input("Resources", min_value=1, max_value=20, value=3, step=1))

        st.subheader("Actor labels")
        _actor_defaults = _DEF_ACTORS + [f"Actor {i+1}" for i in range(len(_DEF_ACTORS), 18)]
        actor_labels = [
            st.text_input(f"Actor {i+1}", value=_actor_defaults[i], key=f"al_{i}")
            for i in range(n_actors)
        ]

        st.subheader("Event labels")
        _event_defaults = _DEF_EVENTS + [f"Event {i+1}" for i in range(len(_DEF_EVENTS), 20)]
        event_labels = [
            st.text_input(f"Event {i+1}", value=_event_defaults[i], key=f"el_{i}")
            for i in range(n_events)
        ]

        if not use_identity:
            st.subheader("Resource labels")
            _res_defaults = _DEF_RESOURCES + [f"Resource {i+1}" for i in range(len(_DEF_RESOURCES), 20)]
            resource_labels = [
                st.text_input(f"Resource {i+1}", value=_res_defaults[i], key=f"rl_{i}")
                for i in range(n_resources)
            ]
        else:
            resource_labels = event_labels[:]


# ── Matrix editors ─────────────────────────────────────────────────────────────
st.header("Input matrices")

# Keys encode both mode and dimensions so editors reset on any structural change.
if mode == "📋  Load preset example":
    _safe_name = example_name.replace(" ", "_").replace("—", "").replace(",", "")
    _key_sfx = f"preset_{_safe_name}"
else:
    _key_sfx = f"custom_{n_actors}_{n_events}_{n_resources}"

tab_names = ["Interests (y)", "Control (c)"]
if not use_identity:
    tab_names.append("Resources (a)")
tabs = st.tabs(tab_names)

# ── Interest matrix ───────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown(
        "**Actor × Event interest matrix.**  "
        "Positive = preference for a positive outcome; negative = preference for a negative outcome.  "
        "Magnitudes reflect interest strength. Rows are normalised automatically."
    )
    if mode == "📋  Load preset example":
        _y_init = preset["interest_matrix"]
    else:
        # Pad / trim the default matrix to the requested dimensions
        _y_init = [
            [(_DEF_Y[i][j] if i < len(_DEF_Y) and j < len(_DEF_Y[0]) else 1/n_events)
             for j in range(n_events)]
            for i in range(n_actors)
        ]
    y_df = st.data_editor(
        pd.DataFrame(_y_init, index=actor_labels, columns=event_labels),
        key=f"y_{_key_sfx}", use_container_width=True,
    )

# ── Control matrix ────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown(
        "**Actor × Resource control matrix.**  "
        "Each row shows one actor's degree of control over each resource.  "
        "Rows are normalised automatically."
    )
    if mode == "📋  Load preset example":
        _c_init = preset["control_matrix"]
    else:
        _c_init = [
            [(_DEF_C[i][j] if i < len(_DEF_C) and j < len(_DEF_C[0]) else 1/n_actors)
             for j in range(n_resources)]
            for i in range(n_actors)
        ]
    c_df = st.data_editor(
        pd.DataFrame(_c_init, index=actor_labels, columns=resource_labels),
        key=f"c_{_key_sfx}", use_container_width=True,
    )

# ── Resource-requirements matrix (only when not identity) ─────────────────────
a_df: pd.DataFrame | None = None
if not use_identity:
    with tabs[2]:
        st.markdown(
            "**Event × Resource requirements matrix.**  "
            "Each entry is how much event i draws on resource k.  "
            "Rows are normalised automatically."
        )
        if mode == "📋  Load preset example":
            _a_init = preset["resource_matrix"]  # None for all paper examples
        else:
            _a_init = [
                [(_DEF_A[i][j] if i < len(_DEF_A) and j < len(_DEF_A[0]) else 1/n_resources)
                 for j in range(n_resources)]
                for i in range(n_events)
            ]
        if _a_init is not None:
            a_df = st.data_editor(
                pd.DataFrame(_a_init, index=event_labels, columns=resource_labels),
                key=f"a_{_key_sfx}", use_container_width=True,
            )


# ── Clear stale results when inputs change ──────────────────────────────────────
_hash_parts = [_key_sfx, str(y_df.values.round(8).tolist()), str(c_df.values.round(8).tolist())]
if a_df is not None:
    _hash_parts.append(str(a_df.values.round(8).tolist()))
_current_config_hash = hashlib.md5("|".join(_hash_parts).encode()).hexdigest()

if st.session_state.get("config_hash") != _current_config_hash:
    st.session_state.pop("result", None)
    st.session_state.pop("raw", None)
    st.session_state["config_hash"] = _current_config_hash

# ── Run button ─────────────────────────────────────────────────────────────────
st.divider()
_, btn_col, _ = st.columns([2, 1, 2])
run_clicked = btn_col.button("Run analysis", type="primary", use_container_width=True)

if run_clicked:
    try:
        config = AnalysisConfig(
            interest_matrix=y_df.values.tolist(),
            control_matrix=c_df.values.tolist(),
            resource_matrix=a_df.values.tolist() if a_df is not None else None,
            actor_labels=actor_labels,
            event_labels=event_labels,
            resource_labels=resource_labels,
        )
        with st.spinner("Solving…"):
            result, raw = analyze(config)

        st.session_state["result"] = result
        st.session_state["raw"] = raw

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()


# ── Results ────────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.info("Configure the system above, then click **Run analysis** to see results.")
    st.stop()

result: AnalysisResult = st.session_state["result"]
raw: RawAnalysisData = st.session_state["raw"]

a_lbls = result.actor_labels
e_lbls = result.event_labels
r_lbls = result.resource_labels

st.header("Results")

# ══════════════════════════════════════════════════════════════════════════════
# 1. EQUILIBRIUM POWER AND VALUES
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("1. Equilibrium power and values")

_narrative("""
Coleman's model assumes that actors engage in a perfect market for vote or control exchanges
— each actor gives up control over events they care about less in exchange for control over
events they care about more.  The system iterates until it reaches a **fixed-point
equilibrium** described by three vectors:

- **Actor power (r):** Each actor's total power after all exchanges have settled.  An actor's
  power reflects both the total amount of resources they control and how valuable those
  resources are.  Power is a summary measure: an actor with high power wields substantial
  influence over the outcomes that matter to others.

- **Event values (v):** How intensely each event is contested across the collectivity.  An
  event has high value when many actors care deeply about it and compete to control its
  outcome.  Events that are uncontested (everyone agrees, or nobody cares) have low value.

- **Resource values (w):** How much each resource is sought after in equilibrium.  Resources
  required by high-value events — events that many actors are willing to trade control for
  — acquire high value.  In the identity case (each event requires one dedicated resource),
  resource values coincide with event values.
""")

col_r, col_v, col_w = st.columns(3)

with col_r:
    fig = _bar(result.power, a_lbls, "Actor power (r)", "#4C78A8")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_v:
    fig = _bar(result.event_values, e_lbls, "Event values (v)", "#72B7B2")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_w:
    fig = _bar(result.resource_values, r_lbls, "Resource values (w)", "#54A24B")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 2. COLLECTIVE OUTCOMES
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("2. Collective outcomes")

_narrative("""
These quantities describe what the collectivity as a whole is likely to achieve, and how
beneficial those outcomes are to each actor.

- **Outcome probabilities (P_p):** The probability that each event resolves with a positive
  outcome, given the final directed control of all actors.  A value of 0.5 means the event
  is perfectly contested — neither side dominates.  Values above 0.5 indicate net collective
  push towards a positive resolution; values below indicate the reverse.  These probabilities
  follow directly from the signed sum of each actor's final directed control over the event.

- **Directed power (d):** The power-weighted sum of actor interests on each event.  It
  captures the net direction in which the collectivity is pushing.  Positive values mean
  most of the collective weight favours a positive outcome; negative values the reverse.
  The total unsigned directed power — **R** — sums these absolute values across all events
  and gives a single scalar measure of how strongly the collectivity acts.

- **Expected collective value (p_h):** For each actor, the probability-weighted expected
  realisation of their interests given the equilibrium outcome probabilities.  An actor
  with high p_h is likely to see the events resolve in their favour.  The baseline is 0.5
  (random outcomes with no collective action); values above 0.5 indicate net benefit from
  the current distribution of control.

- **Expected weighted realization (d_i):** A single summary scalar — the power-weighted
  average of p_h across all actors.  It captures how well the collectivity as a whole
  serves the interests of its members, weighting each actor's satisfaction by their power.

- **Matching attitudes:** A scalar measure ∈ [0, 1] of how well actor attitudes align across
  events.  When all actors agree on the direction of every event (they all want positive or
  all want negative outcomes), this equals 1.  When attitudes are maximally split, it
  approaches 0.  High attitude matching reduces the value of coalition formation, because
  acting collectively adds little when everyone already wants the same outcomes.
""")

m1, m2, m3 = st.columns(3)
m1.metric("Total external power (R)", f"{result.total_power:.4f}")
m2.metric("Expected weighted realization (d_i)", f"{result.expected_weighted_realization:.4f}")
m3.metric("Matching attitudes", f"{result.matching_attitudes:.4f}")

col_p, col_ph = st.columns(2)

with col_p:
    st.markdown("**Outcome probabilities and directed power per event**")
    p_df = pd.DataFrame(
        {
            "P(positive outcome)": [f"{p:.4f}" for p in result.outcome_probabilities],
            "Directed power (d)": [f"{d:.4f}" for d in result.directed_power],
        },
        index=e_lbls,
    )
    st.dataframe(p_df, use_container_width=True)

with col_ph:
    st.markdown("**Expected collective value and power per actor**")
    ph_df = pd.DataFrame(
        {
            "Expected value (p_h)": [f"{v:.4f}" for v in result.expected_collective_value],
            "Power (r)": [f"{r:.4f}" for r in result.power],
        },
        index=a_lbls,
    )
    st.dataframe(ph_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONTROL STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("3. Control structure")

_narrative("""
These matrices reveal how control is structured in the system — both before exchange
(constitutional control) and after exchange equilibrium has been reached (final control).

- **Constitutional control (C):** Actor j's structural control over event i *before any
  exchange*, determined purely by the initial resource ownership matrix.  C = a @ c, where
  a maps events to resources and c maps resources to actors.  This is the baseline from
  which logrolling begins; it answers the question "who holds the formal power to determine
  each outcome?"

- **Final actor–event control (c_AE):** The directed control each actor exerts over each
  event *after* the exchange equilibrium.  Positive entries indicate the actor is pushing
  for a positive outcome on that event; negative entries indicate the reverse.  This matrix
  incorporates both the actor's power and the direction of their interest, and its column
  sums directly determine the outcome probabilities.

- **Actor–resource control (c_AR):** Each actor's effective control over each resource after
  exchange, weighted by actor power and resource value.  An actor with high c_AR over a
  resource has effectively cornered that resource in the exchange market, reflecting the
  alignment of their derived interest with their equilibrium power.

- **Actor–actor control (z):** How much actor j's resource control advances actor h's
  interests, summed across all resources.  z[h, j] measures indirect influence: a high
  value means actor j — through the resources they control — tends to advance the outcomes
  that matter to actor h.  This can reveal hidden alliances or dependencies that are not
  apparent from the raw interest and control matrices.

- **Event–event control (c_EE):** How much control over each event flows through each
  resource.  c_EE = c @ x, and captures the indirect linkages between events via the
  resource base.

- **Fraction of resources (F):** The fraction of each resource that is allocated toward
  each event in equilibrium.  F[k, i] = a[i,k] * v[i] / w[k]: resources flow most strongly
  toward the events with the highest value, weighted by how much of each resource those
  events require.
""")

tabs_ctrl = st.tabs([
    "Constitutional control (C)",
    "Actor–event control (c_AE)",
    "Actor–resource control (c_AR)",
    "Actor–actor control (z)",
    "Event–event control (c_EE)",
    "Fraction of resources (F)",
])

with tabs_ctrl[0]:
    C_arr = np.array(result.constitutional_control)
    rows_identical = C_arr.shape[0] > 1 and np.allclose(C_arr, C_arr[[0], :], atol=1e-8)

    if rows_identical:
        st.info(
            "All events share the same constitutional control vector "
            "(C = a @ c = c when the resource matrix is identity and all resources "
            "are controlled in equal proportions). Showing the shared vector instead of "
            "a redundant heatmap."
        )
        fig = _bar(
            C_arr[0].tolist(), a_lbls,
            "Constitutional control per actor  (identical for every event)",
            "#4C78A8",
        )
    else:
        fig = _heatmap(
            result.constitutional_control, e_lbls, a_lbls,
            "Constitutional control C = a @ c  (event × actor)",
            cmap="Blues", vmin=0, vmax=1,
        )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tabs_ctrl[1]:
    fig = _heatmap(
        result.actor_event_control, a_lbls, e_lbls,
        "Final actor–event control c_AE  (actor × event)",
        cmap="RdBu_r", vmin=-1, vmax=1,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tabs_ctrl[2]:
    fig = _heatmap(
        result.actor_resource_control, a_lbls, r_lbls,
        "Actor–resource control c_AR  (actor × resource)",
        cmap="Blues", vmin=0, vmax=None,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tabs_ctrl[3]:
    fig = _heatmap(
        result.actor_actor_control, a_lbls, a_lbls,
        "Actor–actor control z  (actor × actor)",
        cmap="RdBu_r", vmin=-1, vmax=1,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tabs_ctrl[4]:
    fig = _heatmap(
        result.event_event_control, r_lbls, e_lbls,
        "Event–event control c_EE = c @ x  (resource × event)",
        cmap="Blues", vmin=0, vmax=None,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tabs_ctrl[5]:
    fig = _heatmap(
        result.fraction_of_resources, r_lbls, e_lbls,
        "Fraction of resources F  (resource × event)",
        cmap="YlOrRd", vmin=0, vmax=None,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 4. ACTOR INTERESTS AND BENEFIT FROM COLLECTIVITY
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("4. Actor interests in resources and mutual benefit")

_narrative("""
These quantities connect actor interests to the resource structure of the collectivity,
and show how much each actor benefits from (or is hindered by) the actions of every other.

- **Derived interests in resources (B):** Actor j's interest in resource k, derived from
  their interests in events: B[j, k] = sum_i x[j,i] * a[i,k].  An actor cares about a
  resource to the extent that the events they care about depend on it.  These derived
  interests determine which resources actors will compete over in the exchange market.

- **Realization increments (p_hj):** How much actor j's directed control advances or
  opposes actor h's interests.  p_hj[h, j] > 0 means actor j's final control pushes events
  in directions that align with actor h's preferences; p_hj[h, j] < 0 means they work at
  cross-purposes.  The row sums (summing over all j) plus 0.5 give the expected collective
  value p_h for each actor.  This matrix reveals the pairwise benefit structure of the
  collectivity — who helps whom, and who harms whom.
""")

col_b, col_phj = st.columns(2)

with col_b:
    fig = _heatmap(
        result.derived_interests, a_lbls, r_lbls,
        "Derived interests B  (actor × resource)",
        cmap="Blues", vmin=0, vmax=None,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_phj:
    fig = _heatmap(
        result.realization_increments, a_lbls, a_lbls,
        "Realization increments p_hj  (actor × actor)",
        cmap="RdBu_r", vmin=None, vmax=None,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 5. COALITION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if result.coalitions:
    st.subheader("5. Coalition analysis")

    _narrative("""
A **feasible coalition** is any subset of two or more actors whose combined control of the
first resource exceeds 50% — i.e. a majority coalition.  For each feasible coalition, the
full Coleman analysis is run treating only the coalition members as the collectivity.  This
gives the outcome probabilities and outcome values that would obtain if only the coalition
members acted together.

- **Value to coalition members:** The sum of each member's expected realization of interests
  under the coalition's outcome distribution.  Higher values indicate that the coalition's
  joint action advances its members' interests.

- **Value to the opposition:** The sum of expected realization for non-members.  Typically
  negative — when a coalition acts in its members' interests, non-members are often harmed
  by the externalities of that action.

- **Value to the collectivity:** The sum across all actors.  This measures the overall
  welfare effect of the coalition's joint action.

- **Winning (stable) coalitions:** A coalition is *winning* if no member can rationally
  defect to another feasible coalition that would benefit all of that new coalition's members.
  In graph terms, it is a sink node in the transition probability matrix — all arrows point
  in but none point out.  If no winning coalition exists, the system may cycle endlessly
  between coalitions, with no stable equilibrium.

- **Minimality:** A winning coalition is *minimal* if no proper subset of it is also a
  winning coalition.  Minimal winning coalitions contain no redundant members — every member
  is necessary to keep the coalition winning.  This corresponds to Riker's "size principle":
  coalitions tend toward the smallest size that can still win.
""")

    rows = []
    for coal in result.coalitions:
        rows.append({
            "Coalition": " + ".join(coal.member_labels),
            "Control": round(coal.control, 3),
            "→ Members": round(coal.value_to_coalition, 4),
            "→ Opposition": round(coal.value_to_opposition, 4),
            "→ Collective": round(coal.value_to_collective, 4),
            "Stable": "✓" if coal.is_winning else "",
        })

    coal_table = pd.DataFrame(rows)
    winning_row_idx = [i for i, c in enumerate(result.coalitions) if c.is_winning]

    def _highlight_winning(row):  # type: ignore[return]
        return (
            ["background-color: #d4edda"] * len(row)
            if row.name in winning_row_idx
            else [""] * len(row)
        )

    st.dataframe(
        coal_table.style.apply(_highlight_winning, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    if result.winning_coalition_indices:
        winning_names = [
            " + ".join(result.coalitions[i].member_labels)
            for i in result.winning_coalition_indices
        ]
        st.success(f"Winning coalition(s): **{', '.join(winning_names)}**")
        if result.are_winning_minimal:
            st.info("All winning coalitions are minimal (no redundant members).")
        else:
            st.warning("Some winning coalitions contain redundant members.")
    else:
        st.warning(
            "No stable winning coalition found — the transition graph may cycle."
        )

    # ── Per-actor value breakdown ──────────────────────────────────────────────
    with st.expander("Per-actor value for each coalition"):
        actor_val_rows = []
        for coal in result.coalitions:
            row = {"Coalition": " + ".join(coal.member_labels)}
            for j, lbl in enumerate(a_lbls):
                val = coal.actor_values[j] if j < len(coal.actor_values) else 0.0
                row[lbl] = round(val, 4)
            actor_val_rows.append(row)
        st.dataframe(pd.DataFrame(actor_val_rows), use_container_width=True, hide_index=True)

    # ── Transition maps ────────────────────────────────────────────────────────
    if raw.coalition_outputs is not None and raw.summary is not None and raw.tpm is not None:
        st.subheader("Coalition transition maps")
        _narrative("""
The transition map is a directed graph where each node is a feasible coalition and each
arrow indicates a potential switch.  An arrow from coalition X to coalition Y means that
*all* members of X would benefit by switching to Y.  Arrow thickness encodes the transition
probability (proportional to the relative gain from switching).

**Winning coalitions** (stable equilibria, shown in green) are sink nodes — all arrows
point into them, none point out.  When the system has no sink, it cycles: a situation
analogous to a voting paradox, where no coalition can claim to be the uniquely rational
outcome.

The **strongest-transitions** map shows only the single most likely destination for each
coalition, making the dominant dynamic easier to read.
""")
        col_map1, col_map2 = st.columns(2)

        with col_map1:
            st.markdown("**All transitions**")
            try:
                fig_all = cc.draw_coalition_map(raw.coalition_outputs, raw.summary, raw.tpm)
                st.pyplot(fig_all, use_container_width=True)
                plt.close(fig_all)
            except Exception as e:
                st.caption(f"Could not render map: {e}")

        with col_map2:
            st.markdown("**Strongest transitions only**")
            try:
                fig_str = cc.draw_strongest_transitions(raw.coalition_outputs, raw.summary, raw.tpm)
                st.pyplot(fig_str, use_container_width=True)
                plt.close(fig_str)
            except Exception as e:
                st.caption(f"Could not render map: {e}")
