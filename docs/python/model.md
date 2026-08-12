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
        - setVariableCones
        - setAffineConeConstraints
        - setVariableLowerBound
        - setVariableUpperBound
        - setWarmStart
        - clearWarmStart
        - setParam
        - setParams
        - getParam
        - optimize

## Cone constraints

Use the columnar `ConeSpec` API for both variable and affine cones. Its metadata
is stored in contiguous NumPy arrays, so even millions of cone blocks do not
require one Python dict per cone.

| Field | Type | Notes |
|---|---|---|
| `types` | scalar or `int32[K]` | `ConeType.SOC`, `RSOC`, `EXP`, `POWER`, or `PSD`. |
| `starts` | `int32[K]` | First variable index or affine row of each block. |
| `v_dims` | scalar or `int32[K]` | Length of `v`, or matrix order for PSD; defaults to 1. |
| `power_alphas` | scalar or `float64[K]` | Required in `(0, 1)` for power cones. |
| `fixed_mask` | optional `uint8[N]` | Ambient-coordinate mask for fixed variable slots. Values come from the primal warm start. |

Slot layout per cone:

- `soc`: `v[0..v_dim-1], w, z` with `||v||^2 + w^2 <= z^2`, `z >= 0`.
- `rsoc`: `v[0..v_dim-1], s, t` with `||v||^2 <= 2 s t`, `s, t >= 0`.
- `exp`: `x, y, z` with `y * exp(x / y) <= z`, `y > 0`.
- `power`: `x, y, z` with `x^alpha * y^(1-alpha) >= |z|`, `x, y >= 0`.
- `psd`: `svec(X)` for an order-`v_dim` symmetric matrix `X >= 0`. `svec`
  stores the lower triangle in column-major order, leaves diagonal entries
  unchanged, and multiplies off-diagonal entries by `sqrt(2)`.

PSD blocks do not support entries in `fixed_mask`; express fixed matrix entries
as ordinary linear equalities instead.
In distributed solves, each PSD block remains on one GPU; permutation and
partitioning never split its `svec` coordinates across devices.

```python
import numpy as np
from pdhcg import ConeSpec, ConeType, Model

cones = ConeSpec(
    types=[ConeType.SOC, ConeType.EXP],
    starts=np.array([0, 4], dtype=np.int32),
    v_dims=[2, 1],
)
model = Model(objective_vector=c, constraint_matrix=A, variable_cones=cones)
```

Scalar metadata broadcasts. For example, one million adjacent exponential
cones can be described without a Python loop:

```python
num_cones = 1_000_000
cones = ConeSpec(
    ConeType.EXP,
    3 * np.arange(num_cones, dtype=np.int32),
)
```

`solve_once(..., cones=cones)` accepts the same object. Cone arguments accept
`ConeSpec` only; the former per-cone `list[dict]` input is not supported.

See [quickstart](quickstart.md#quick-start-with-cone-constraints) for a runnable example.

Native affine cone constraints `F x + g in K` are available directly on
`Model` through the `affine_cone_matrix`, `affine_cone_offset`, and
`affine_cones` constructor arguments, or `setAffineConeConstraints`. Here,
`ConeSpec.starts` indexes rows of `F`, and the blocks must cover every row of
`F` exactly once.

```python
model = Model(
    objective_vector=c,
    affine_cone_matrix=F,
    affine_cone_offset=g,
    affine_cones=ConeSpec(ConeType.SOC, np.array([0], dtype=np.int32)),
)
```
