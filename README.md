# Coleman Coalitions

A Python implementation of James Coleman's mathematical framework for collective action
and coalition analysis, based on:

> Coleman, J.S. (1973). *The Mathematics of Collective Action*. Aldine, Chicago.
> Coleman, J.S. (1970). The Benefits of Coalition. *Public Choice*, 8, 45–61.

The package extends Coleman's classical theory with new concepts regarding the value of
coalitions within a collective. Developed in relation to work by G. Hernes and B.E. Juel.

---

## Installation

**With pip (editable install):**
```bash
pip install -e .
```

**With conda (recommended):**
```bash
conda env create -f environment.yml
conda activate coleman-coalitions
```

**Dependencies:** `numpy`, `matplotlib`, `networkx`.

---

## Quick start

```python
from coleman_coalitions import (
    setup, standard_variables_trad12,
    run_full_analysis, run_coalition_analysis,
    value_of_coalition, optimal_coalition, winning_coalitions,
    draw_coalition_map, print_analysis,
)

# 1. Load a built-in example (3 actors, 4 events, 4 resources)
inputs = standard_variables_trad12()

# 2. Solve the full Coleman system (power r, event values v, resource values w,
#    and all derived control variables)
run_full_analysis(inputs)

print(f"Actor power: {inputs['r']}")
print(f"Event values: {inputs['v']}")

# 3. Coalition analysis
full_inputs, coalitions, coal_outputs = run_coalition_analysis(inputs)
print(f"Feasible coalitions: {list(coalitions.keys())}")

# 4. Value of each coalition
val = value_of_coalition(coal_outputs)
for key, v in val.items():
    print(f"{key}: to coalition={v['to coalition']:.3f}, to collective={v['to collective']:.3f}")

# 5. Find winning (stable) coalitions
summary, TPM = optimal_coalition(coal_outputs)
winners = winning_coalitions(TPM)
print(f"Winning coalitions: {[k for k, w in zip(coalitions, winners) if w]}")

# 6. Visualise (returns Figure; call fig.savefig() or plt.show() as needed)
fig = draw_coalition_map(coal_outputs, summary, TPM)
fig.savefig("coalition_map.png", bbox_inches="tight")
```

**Custom system:**
```python
import numpy as np
from coleman_coalitions import setup, run_full_analysis

y = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])   # (n=3, q=2) interests
a = np.eye(2)                                           # (q=2, m=2) identity resources
c = np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]])      # (m=2, n=3) resource control

inputs = setup(n=3, q=2, m=2, y=y, a=a, c=c)
run_full_analysis(inputs)
```

---

## Package structure

```
coleman_coalitions/
├── core.py          # setup(), check_parameters(), standard example datasets
├── solver.py        # Iterative equilibrium solver (r, v, w)
├── analysis.py      # Derived variables + run_full_analysis()
├── coalitions.py    # Enumeration, value, TPM, run_coalition_analysis()
├── visualization.py # Plots (coalition map, power bars, heatmaps)
└── io.py            # write_analysis(), print_analysis()

deprecated/
└── ColemanFunctions.py   # Original monolithic implementation (kept for reference)
```

---

## Key concepts

| Symbol | Meaning |
|--------|---------|
| `y` (n, q) | Directed interests of each actor in each event |
| `c` (m, n) | Control of each resource by each actor |
| `a` (q, m) | Resource requirements of each event |
| `r` (n)    | Actor power (equilibrium) |
| `v` (q)    | Event value (equilibrium) |
| `w` (m)    | Resource value (equilibrium) |
| `P_p` (q)  | Probability of positive outcome per event |

---

## Citation

If you use this package in academic work, please cite the related article (reference TBD)
as well as Coleman (1973). Contact: bjorneju@gmail.com
