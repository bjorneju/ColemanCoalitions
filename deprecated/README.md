# deprecated/

This folder contains the original monolithic implementation of the Coleman framework,
kept here for reference and backwards compatibility while the codebase migrated to the
modular `coleman_coalitions/` package.

## ColemanFunctions.py

The original single-file implementation. It uses a `[value, description]` tuple-wrapping
convention for all variables (e.g. `inputs['r'][0]` to get the value) and CamelCase
function names. Both conventions are superseded by the new package.

### Functions present here but not in coleman_coalitions/

- `CalculateMetrics(orig_outputs, outputs, metrics)` — computes the difference between
  final directed control (`c_AE`) and original interests (`y`), used for post-hoc
  analysis of attitude–control divergence. Not ported because it was specific to a
  particular workflow in the original article.
- `CoalitionTechno(inputs, coalitions)` — a proportional-control variant of the coalition
  techno analysis. The new package only includes the equal-control variant
  (`coalition_techno_equal_control`), which is the methodologically preferred approach.

### Do not rely on this file for new work

Use `coleman_coalitions` instead. This file will be removed in a future version.
