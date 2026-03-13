"""
coleman_coalitions — Python implementation of Coleman's mathematical framework for
collective action and coalition analysis.

Based on:
  Coleman, J.S. (1973). The Mathematics of Collective Action. Aldine, Chicago.
  Coleman, J.S. (1970). The Benefits of Coalition. Public Choice, 8, 45-61.

Quick start
-----------
>>> from coleman_coalitions import setup, run_full_analysis, run_coalition_analysis
>>> from coleman_coalitions import standard_variables_trad12
>>> inputs = standard_variables_trad12()
>>> run_full_analysis(inputs)
>>> inputs, coalitions, coalition_outputs = run_coalition_analysis(inputs)
"""

# Core
from .core import (
    setup,
    check_parameters,
    normalize_matrix_row,
    standard_variables_techno,
    standard_variables_techno2,
    standard_variables_trad12,
    DESCRIPTIONS,
)

# Solver
from .solver import solve, SolverConvergenceError

# Analysis
from .analysis import (
    constitutional_control,
    fraction_of_resources,
    control_actor_event,
    derived_interest,
    control_actor_resource,
    control_actor_actor,
    control_event_event,
    positive_outcome,
    increment_expected_realization,
    expected_value_of_collectivity,
    expected_weighted_realization,
    directed_power_of_collectivity,
    total_power_of_collectivity,
    matching_attitudes,
    run_full_analysis,
)

# Coalitions
from .coalitions import (
    feasible_coalitions,
    coalition_trad,
    coalition_techno_equal_control,
    value_of_coalition,
    value_for_opposition,
    optimal_coalition,
    winning_coalitions,
    is_winning_minimal,
    run_coalition_analysis,
)

# Visualization
from .visualization import (
    draw_coalition_map,
    draw_strongest_transitions,
    draw_coalition_table,
    draw_interest_heatmap,
    draw_power_distribution,
    draw_event_values,
)

# I/O
from .io import write_analysis, print_analysis

__all__ = [
    # core
    'setup', 'check_parameters', 'normalize_matrix_row',
    'standard_variables_techno', 'standard_variables_techno2', 'standard_variables_trad12',
    'DESCRIPTIONS',
    # solver
    'solve', 'SolverConvergenceError',
    # analysis
    'constitutional_control', 'fraction_of_resources', 'control_actor_event',
    'derived_interest', 'control_actor_resource', 'control_actor_actor',
    'control_event_event', 'positive_outcome', 'increment_expected_realization',
    'expected_value_of_collectivity', 'expected_weighted_realization',
    'directed_power_of_collectivity', 'total_power_of_collectivity',
    'matching_attitudes', 'run_full_analysis',
    # coalitions
    'feasible_coalitions', 'coalition_trad', 'coalition_techno_equal_control',
    'value_of_coalition', 'value_for_opposition', 'optimal_coalition',
    'winning_coalitions', 'is_winning_minimal', 'run_coalition_analysis',
    # visualization
    'draw_coalition_map', 'draw_strongest_transitions', 'draw_interest_heatmap',
    'draw_power_distribution', 'draw_event_values', 'draw_coalition_table',
    # io
    'write_analysis', 'print_analysis',
]
