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
    const double *objective_constant
);
```

Creates a QP problem of the form
`min 0.5 * x^T (Q + R^T D R) x + c^T x` subject to `con_lb <= A x <= con_ub` and `var_lb <= x <= var_ub`.

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

Sets initial primal and dual solutions for warm starting.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `prob` | QP problem pointer |
| `primal` | Primal solution vector (size n, can be NULL) |
| `dual` | Dual solution vector (size m, can be NULL) |

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
    This function is only available when PDHCG is compiled with `-DPDHCG_COMPILE_DISTRIBUTED=ON`.

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
