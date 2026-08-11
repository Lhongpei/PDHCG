# Migrating to 0.3

Version 0.3 adds quadratic conic models and changes the C and Python model
construction APIs. This page covers the source changes needed by existing 0.2
callers.

## C API

`create_qp_problem` has six new trailing arguments:

```c
qp_problem_t *create_qp_problem(
    const double *objective_c,
    const matrix_desc_t *Q_desc,
    const matrix_desc_t *R_desc,
    const matrix_desc_t *D_desc,
    const matrix_desc_t *A_desc,
    const double *con_lb,
    const double *con_ub,
    const double *var_lb,
    const double *var_ub,
    const double *objective_constant,
    int num_var_cones,
    const cone_spec_t *var_cones,
    const matrix_desc_t *affine_cone_matrix_desc,
    const double *affine_cone_offset,
    int num_affine_cones,
    const cone_spec_t *affine_cones);
```

An existing QP with no cones only needs the six neutral arguments appended:

```c
qp_problem_t *problem = create_qp_problem(
    c, Q, R, D, A, con_lb, con_ub, var_lb, var_ub, objective_constant,
    0, NULL, NULL, NULL, 0, NULL);
```

For conic models, use `cone_spec_t` arrays as described in the
[C API overview](c/overview.md). Variable-cone indices refer to variables;
affine-cone indices refer to rows of the separately supplied
`affine_cone_matrix_desc` (`F`), and `affine_cone_offset` has one entry per row
of `F`. Affine cone blocks must cover every row of `F`.

`pdhcg_postsolve` now returns nonzero after a complete primal-dual recovery and
zero when postsolve fails or full dual recovery is unavailable:

```c
if (!pdhcg_postsolve(info, result, original_problem)) {
    /* Handle postsolve failure. */
}
```

`qp_problem_t` and `pdhg_parameters_t` gained conic fields. Do not depend on
their old binary layout. Recompile downstream code and initialize parameters
through `set_default_parameters` before overriding individual fields.

## Python API

Cone metadata is now columnar. Replace a list of dictionaries with one
`ConeSpec`:

```python
import numpy as np
from pdhcg import ConeSpec, ConeType

cones = ConeSpec(
    types=np.array([ConeType.SOC, ConeType.POWER], dtype=np.int32),
    starts=np.array([0, 4], dtype=np.int32),
    v_dims=np.array([2, 1], dtype=np.int32),
    power_alphas=np.array([0.0, 0.4]),
)
```

Pass this object as `variable_cones` or `affine_cones` when constructing a
`Model`. Legacy `list[dict]` inputs intentionally raise `TypeError`.

CVXPY support is optional:

```bash
pip install "pdhcg[cvxpy]"
```

Import the backend once before selecting PDHCG as the solver:

```python
import cvxpy as cp
import pdhcg.cvxpy_backend  # Registers solver="PDHCG".
```

## Executable Location

A source build places the command-line executable at `build/pdhcg`. Installed
packages place it in the installation prefix's `bin` directory.
