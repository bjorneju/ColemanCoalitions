"""
example.py — Minimal working example for coleman_coalitions

Demonstrates:
  1. Setting up inputs (built-in dataset or custom matrices)
  2. Running the full Coleman analysis
  3. Running coalition analysis
  4. Plotting results

Install dependencies first:
    pip install -e .
  or:
    pip install numpy matplotlib networkx
"""

import matplotlib.pyplot as plt
import coleman_coalitions as cc

# ---------------------------------------------------------------------------
# 1. Load a built-in example dataset
#    - standard_variables_trad12 : 3 actors, 4 events, identity resource matrix
#    - standard_variables_techno : 3 actors, 2 events, 4 resources
#    - standard_variables_techno2: 3 actors, 2 events, identity control matrix
# ---------------------------------------------------------------------------
inputs = cc.standard_variables_trad12()

print(f"Actors: {inputs['n']},  Events: {inputs['q']},  Resources: {inputs['m']}")
print("Directed interests (y):\n", inputs['y'])
print("Resource control (c):\n", inputs['c'])

# ---------------------------------------------------------------------------
# 2. (Optional) Define your own inputs with cc.setup()
#
# import numpy as np
# n, q, m = 3, 4, 4
# y = np.array([[ 0.4,  0.2,  0.1,  0.3],   # actor interests (n x q)
#               [-0.3,  0.3,  0.2,  0.2],
#               [-0.1, -0.3, -0.5,  0.1]])
# a = np.eye(q)                               # resource requirements (q x m)
# c = np.array([[0.3, 0.2, 0.5],             # resource control (m x n)
#               [0.4, 0.3, 0.3],
#               [0.4, 0.4, 0.2],
#               [0.25,0.35,0.40]])
# inputs = cc.setup(n, q, m, y, a, c)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. Run the full Coleman analysis
#    Solves for equilibrium power (r), event values (v), resource values (w),
#    then computes all derived variables in one call.
# ---------------------------------------------------------------------------
cc.run_full_analysis(inputs)

print("\n--- Key results ---")
print(f"Actor power (r):          {inputs['r'].round(4)}")
print(f"Event values (v):         {inputs['v'].round(4)}")
print(f"Outcome probabilities:    {inputs['P_p'].round(4)}")
print(f"Expected collectivity:    {inputs['p_h'].round(4)}")
print(f"Total external power (R): {inputs['R']:.4f}")

# ---------------------------------------------------------------------------
# 4. Run coalition analysis
#    Enumerates feasible coalitions (majority control, size >= 2),
#    runs the Coleman model for each sub-collective,
#    and computes the transition probability matrix (TPM).
# ---------------------------------------------------------------------------
coalitions = cc.feasible_coalitions(inputs)
print(f"\nFeasible coalitions ({len(coalitions)}):", list(coalitions.keys()))

coalition_outputs = cc.coalition_trad(inputs, coalitions)
summary, TPM = cc.optimal_coalition(coalition_outputs)

winning = cc.winning_coalitions(TPM)
winning_names = [name for name, w in zip(coalition_outputs.keys(), winning) if w]
print("Winning (stable) coalitions:", winning_names)

# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

# Power and event-value distributions
fig1 = cc.draw_power_distribution(inputs)
fig2 = cc.draw_event_values(inputs)

# Coalition transition graphs
fig3 = cc.draw_coalition_map(coalition_outputs, summary, TPM)
fig4 = cc.draw_strongest_transitions(coalition_outputs, summary, TPM)

plt.show()