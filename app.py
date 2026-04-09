"""
app.py — Interactive Streamlit webapp for Coleman Coalition Analysis.

Users configure actors, events, and resources (number and labels), enter
interest and control matrices via spreadsheet-style editors, and receive an
analysis report — power distributions, coalition stability, transition
probabilities, and more — live in the browser.

Run with:
    pip install "coleman_coalitions[webapp]"
    streamlit run app.py

Or install manually:
    pip install streamlit pandas
    streamlit run app.py
"""
from __future__ import annotations

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


# ── Sidebar: system configuration ─────────────────────────────────────────────
with st.sidebar:
    st.header("System setup")

    n_actors = int(st.number_input("Actors", min_value=2, max_value=18, value=3, step=1))
    n_events = int(st.number_input("Events / issues", min_value=1, max_value=20, value=3, step=1))

    use_identity = st.checkbox(
        "Events = Resources (identity)",
        value=True,
        help=(
            "When checked (default), each event is backed by exactly one dedicated "
            "resource. Uncheck to specify a custom Event × Resource requirements matrix."
        ),
    )

    if use_identity:
        n_resources = n_events
    else:
        n_resources = int(
            st.number_input("Resources", min_value=1, max_value=20, value=n_events, step=1)
        )

    st.subheader("Actor labels")
    actor_labels = [
        st.text_input(f"Actor {i + 1}", value=f"A{i + 1}", key=f"al_{i}")
        for i in range(n_actors)
    ]

    st.subheader("Event labels")
    event_labels = [
        st.text_input(f"Event {i + 1}", value=f"E{i + 1}", key=f"el_{i}")
        for i in range(n_events)
    ]

    if not use_identity:
        st.subheader("Resource labels")
        resource_labels = [
            st.text_input(f"Resource {i + 1}", value=f"R{i + 1}", key=f"rl_{i}")
            for i in range(n_resources)
        ]
    else:
        resource_labels = event_labels[:]


# ── Matrix editors ─────────────────────────────────────────────────────────────
st.header("Input matrices")

# Key suffix encodes dimensions so editors reset automatically when dimensions change
dim_key = f"{n_actors}_{n_events}_{n_resources}"

tab_names = ["Interests (y)", "Control (c)"]
if not use_identity:
    tab_names.append("Resources (a)")
tabs = st.tabs(tab_names)

with tabs[0]:
    st.markdown(
        "**Actor × Event interest matrix.**  "
        "Positive values = preference for a positive outcome; "
        "negative values = preference for a negative outcome.  "
        "Magnitudes reflect interest strength. Rows are normalised automatically."
    )
    y_default = pd.DataFrame(
        np.ones((n_actors, n_events)) / n_events,
        index=actor_labels,
        columns=event_labels,
    )
    y_df = st.data_editor(y_default, key=f"y_{dim_key}", use_container_width=True)

with tabs[1]:
    st.markdown(
        "**Actor × Resource control matrix.**  "
        "Each row shows one actor's degree of control over each resource.  "
        "Rows are normalised automatically."
    )
    c_default = pd.DataFrame(
        np.ones((n_actors, n_resources)) / n_actors,
        index=actor_labels,
        columns=resource_labels,
    )
    c_df = st.data_editor(c_default, key=f"c_{dim_key}", use_container_width=True)

a_df: pd.DataFrame | None = None
if not use_identity:
    with tabs[2]:
        st.markdown(
            "**Event × Resource requirements matrix.**  "
            "Each entry is how much event i draws on resource k.  "
            "Rows are normalised automatically."
        )
        a_default = pd.DataFrame(
            np.ones((n_events, n_resources)) / n_resources,
            index=event_labels,
            columns=resource_labels,
        )
        a_df = st.data_editor(a_default, key=f"a_{dim_key}", use_container_width=True)


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
