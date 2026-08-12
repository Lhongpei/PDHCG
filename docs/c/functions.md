# C API Functions

## create_qp_problem

```c
qp_problem_t *create_qp_problem(
    const double *objective_c,
    const matrix_desc_t *Q_desc,
    const matrix_desc_t *R_desc,
    const matrix_desc_t *D_desc,
    const matrix_desc_t *A_desc,
    const double *con_lb, const double *con_ub,
    const double *var_lb, const double *var_ub,
    const double *objective_constant,
    int num_var_cones,
    const cone_spec_t *var_cones,
    const matrix_desc_t *affine_cone_matrix_desc,
    const double *affine_cone_offset,
    int num_affine_cones,
    const cone_spec_t *affine_cones
);
```

Creates a QP problem of the form
`min 0.5 * x^T (Q + R^T D R) x + c^T x` subject to
`con_lb <= A x <= con_ub`, `F x + affine_cone_offset in K`,
`var_lb <= x <= var_ub`, and optional variable cone blocks. The affine cone
blocks must be disjoint and cover every row of `F`.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `objective_c` | Linear objective coefficients (size n) |
| `Q_desc` | Sparse quadratic matrix descriptor (can be NULL) |
| `R_desc` | Low-rank factor descriptor, shape `k x n` (can be NULL) |
| `D_desc` | Middle matrix in `R^T D R`, shape `k x k` (can be NULL). |
| `A_desc` | Constraint matrix descriptor |
| `con_lb` | Constraint lower bounds (size m) |
| `con_ub` | Constraint upper bounds (size m) |
| `var_lb` | Variable lower bounds (size n) |
| `var_ub` | Variable upper bounds (size n) |
| `objective_constant` | Constant term in objective (can be NULL) |
| `num_var_cones` | Number of variable cone blocks |
| `var_cones` | Array of variable `cone_spec_t` descriptors, or NULL when the count is zero |
| `affine_cone_matrix_desc` | Matrix `F` in the native affine cone constraint; NULL when no affine cones are present |
| `affine_cone_offset` | Offset vector with one entry per row of `F`; NULL means zero |
| `num_affine_cones` | Number of affine cone blocks covering `F` |
| `affine_cones` | Array of affine `cone_spec_t` descriptors, or NULL when the count is zero |

**Returns:** Pointer to allocated `qp_problem_t`, or NULL on error.

---

## set_start_values

```c
void set_start_values(
    qp_problem_t *prob,
    const double *primal,
    const double *dual
);
```

Sets initial primal and dual solutions for warm starting. Passing `NULL` clears
the corresponding warm start, while values pinned by `set_cone_fixed` remain
part of the model and are preserved.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `prob` | QP problem pointer |
| `primal` | Primal solution vector (size n, can be NULL) |
| `dual` | Dual solution vector (size `m + p`, ordered as `[dual_A, dual_F]`; can be NULL) |

Rejects a primal warm start that changes a value already pinned by
`set_cone_fixed`.

---

## set_cone_fixed

```c
int set_cone_fixed(
    qp_problem_t *prob,
    int cone_idx,
    int slot,
    double value
);
```

Pins one slot of cone `cone_idx` to `value`. Allocates the `is_fixed` flag array on first use and also writes `primal_start[start_idx + slot] = value` so the projection sees the constant. During preprocessing, that slot is also converted to equal lower and upper bounds. Typical use: fix the `y` slot of an exponential cone (e.g. Fisher-market entropy term with `y = 1`).

Variable SOC, rotated-SOC, exponential, and power cones support every fixed-slot pattern whose intersection with the cone is nonempty. The solver validates the section before preprocessing and rejects empty or non-finite sections. Projection and stationarity residuals use the same weighted fixed-section operator, including diagonal quadratic objectives and large SOC/rotated-SOC blocks. `set_cone_fixed` rejects PSD blocks.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `prob` | QP problem pointer |
| `cone_idx` | Cone index in `[0, num_cones)` |
| `slot` | Slot offset within the cone (0-based) |
| `value` | Fixed value |

**Returns:** 0 on success, nonzero on error (bad indices or no cones).

---

## solve_qp_problem

```c
pdhcg_result_t *solve_qp_problem(
    const qp_problem_t *prob,
    const pdhg_parameters_t *params
);
```

Solves the QP problem using the PDHCG algorithm.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `prob` | QP problem pointer |
| `params` | Solver parameters |

**Returns:** Pointer to `pdhcg_result_t` containing solution information.

---

## solve_qp_problem_distributed

```c
pdhcg_result_t *solve_qp_problem_distributed(
    const pdhg_parameters_t *params,
    const qp_problem_t *original_problem
);
```

Solves the QP problem using the distributed multi-GPU PDHCG algorithm.

!!! note "Availability"
    Distributed execution requires `-DPDHCG_COMPILE_DISTRIBUTED=ON`. In a
    non-distributed build the API remains available but returns `NULL` with an
    explanatory error.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `params` | Solver parameters (including `partition_method`, `permute_method`, `grid_size`, and `permute_block_size`) |
| `original_problem` | QP problem pointer (only required on rank 0; can be NULL on other ranks) |

**Returns:** Pointer to `pdhcg_result_t` containing solution information (valid on all ranks; only rank 0 writes output).

---

## set_default_parameters

```c
void set_default_parameters(pdhg_parameters_t *params);
```

Fills the parameter struct with default values.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `params` | Pointer to parameters struct to fill |

---

## pdhcg_result_free

```c
void pdhcg_result_free(pdhcg_result_t *results);
```

Frees memory allocated for the result structure.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `results` | Result pointer to free |

---

## qp_problem_free

```c
void qp_problem_free(qp_problem_t *prob);
```

Frees memory allocated for the QP problem structure.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `prob` | Problem pointer to free |
