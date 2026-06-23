# Model Class

::: pdhcg.model.Model
    options:
      show_root_heading: true
      show_source: true
      show_bases: false
      merge_init_into_class: false
      docstring_section_style: table
      members:
        - __init__
        - setObjectiveVector
        - setObjectiveConstant
        - setObjectiveMatrix
        - setObjectiveMatrixLowRank
        - setObjectiveMatrixLowRankMiddle
        - setConstraintMatrix
        - setConstraintLowerBound
        - setConstraintUpperBound
        - setVariableLowerBound
        - setVariableUpperBound
        - setWarmStart
        - clearWarmStart
        - setParam
        - setParams
        - getParam
        - optimize

## Cone constraints

The `Model` class is QP-only. Problems with second-order, rotated second-order,
or exponential cone constraints use `solve_once` directly (imported from
`pdhcg._core`) with a `cones=` kwarg. Each cone is a dict:

| Key | Type | Notes |
|---|---|---|
| `type` | str | `"soc"`, `"rsoc"`, or `"exp"`. |
| `start_idx` | int | Index of the first slot in `x`. |
| `v_dim` | int | Length of the `v` block. Omit for `"exp"` (always 3 slots: `x`, `y`, `z`). |
| `is_fixed` | list[bool], optional | Per-slot pin mask, length `v_dim + 2` (SOC/RSOC) or `3` (exp). |

Slot layout per cone:

- `soc`: `v[0..v_dim-1], w, z` with `||v||^2 + w^2 <= z^2`, `z >= 0`.
- `rsoc`: `v[0..v_dim-1], s, t` with `||v||^2 <= 2 s t`, `s, t >= 0`.
- `exp`: `x, y, z` with `y * exp(x / y) <= z`, `y > 0`.

```python
from pdhcg._core import solve_once

cones = [
    {"type": "soc",  "start_idx": 0, "v_dim": 2},
    {"type": "exp",  "start_idx": 4, "is_fixed": [False, True, False]},
]
res = solve_once(Q, R, A, c, c0, lb, ub, cl, cu, cones=cones)
```

See [quickstart](quickstart.md#quick-start-with-cone-constraints) for a runnable example.
