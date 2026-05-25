
### C Interface

PDHCG-II provides a C API for directly solving QPs in memory, defined in header file `include/pdhcg.h`.

#### Functions and Parameters

The C API involves two main functions:

```c
qp_problem_t *create_qp_problem(
    const double *objective_c,           // objective vector (length n)
    const matrix_desc_t *Q_desc,         // sparse quadratic matrix (n x n)
    const matrix_desc_t *R_desc,         // low-rank factor (k x n)
    const matrix_desc_t *D_desc,         // middle matrix in R^T D R (k x k)
    const matrix_desc_t *A_desc,         // constraint matrix (m x n)
    const double *con_lb,                // constraint lower bounds (length m)
    const double *con_ub,                // constraint upper bounds (length m)
    const double *var_lb,                // variable lower bounds (length n)
    const double *var_ub,                // variable upper bounds (length n)
    const double *objective_constant     // scalar objective offset
);

pdhcg_result_t* solve_qp_problem(
    const qp_problem_t* prob,
    const pdhg_parameters_t* params    // NULL → use default parameters
);
```

The objective minimized is `0.5 * x^T (Q + R^T D R) x + c^T x + c0`. `Q`, `R`, and `D` are all optional; `D` defaults to identity, recovering the standard `Q + R^T R` form.

`create_qp_problem` parameters:
- `objective_c`: Objective vector. If `NULL`, defaults to all zeros.
- `Q_desc`: Matrix descriptor. Supports `matrix_dense`, `matrix_csr`, `matrix_csc`, `matrix_coo`. Pass `NULL` to omit.
- `R_desc`: Matrix descriptor for the low-rank factor (shape `k x n`). Same supported formats. Pass `NULL` to omit.
- `D_desc`: Matrix descriptor for the middle matrix in `R^T D R` (shape `k x k`). Same supported formats. May be diagonal, sparse, dense, or indefinite — the runtime auto-detects the cheapest representation in `preprocess_qp_problem`. Pass `NULL` for `D = I`.
- `A_desc`: Matrix descriptor. Supports `matrix_dense`, `matrix_csr`, `matrix_csc`, `matrix_coo`.
- `con_lb`: Constraint lower bounds. If `NULL`, defaults to all `-INFINITY`.
- `con_ub`: Constraint upper bounds. If `NULL`, defaults to all `+INFINITY`.
- `var_lb`: Variable lower bounds. If `NULL`, defaults to all `-INFINITY`.
- `var_ub`: Variable upper bounds. If `NULL`, defaults to all `+INFINITY`.
- `objective_constant`: Scalar constant term added to the objective value. If `NULL`, defaults to `0.0`.


`solve_qp_problem` parameters:
- `prob`: An QP problem built with `create_qp_problem`.
- `params`: Solver parameters. If `NULL`, the solver will use default parameters.

#### Distributed Solving

For multi-GPU distributed solving, use `solve_qp_problem_distributed()` instead of `solve_qp_problem()`:

```c
pdhcg_result_t* solve_qp_problem_distributed(
    const pdhg_parameters_t* params,
    const qp_problem_t* original_problem    // only required on rank 0
);
```

This requires PDHCG to be compiled with `-DPDHCG_COMPILE_DISTRIBUTED=ON` and launched via `mpirun`.

#### Example: Solving a Small QP
```c
#include "pdhcg.h"
#include <math.h>
#include <stdio.h>

int main() {
    int m = 3; // number of constraints
    int n = 2; // number of variables

    // 1. Define Dense Constraint Matrix A
    double A[3][2] = {
        {1.0, 1.0},
        {-1.0, 2.0},
        {2.0, 1.0}
    };

    matrix_desc_t A_desc;
    A_desc.m = m; A_desc.n = n;
    A_desc.fmt = matrix_dense;
    A_desc.zero_tolerance = 0.0;
    A_desc.data.dense.A = &A[0][0];

    // 2. Define Quadratic Objective Matrix Q
    // Minimize 0.5 * (4*x0^2 + 2*x1^2) -> Q = diag(4, 2)
    double Q[2][2] = {
        {4.0, 0.0},
        {0.0, 2.0}
    };

    matrix_desc_t Q_desc;
    Q_desc.m = n; Q_desc.n = n;
    Q_desc.fmt = matrix_dense;
    Q_desc.zero_tolerance = 0.0;
    Q_desc.data.dense.A = &Q[0][0];

    // 3. Linear Objective coefficients c
    double c[2] = {-2.0, -6.0};

    // 4. Constraint bounds: l <= A x <= u
    double l[3] = {-INFINITY, -INFINITY, -INFINITY};
    double u[3] = {2.0, 2.0, 3.0};

    // 5. Variable bounds: x >= 0
    double lb[2] = {0.0, 0.0};
    double ub[2] = {INFINITY, INFINITY};

    // 6. Build the QP problem
    // Note: We pass NULL for R_desc (low-rank factor), D_desc (middle matrix),
    // and objective_constant.
    qp_problem_t* prob = create_qp_problem(
        c,          // objective_c
        &Q_desc,    // Q_desc
        NULL,       // R_desc
        NULL,       // D_desc (NULL -> D = I)
        &A_desc,    // A_desc
        l,          // con_lb
        u,          // con_ub
        lb,         // var_lb
        ub,         // var_ub
        NULL        // objective_constant
    );

    // 7. Solve (NULL → use default parameters)
    pdhcg_result_t* res = solve_qp_problem(prob, NULL);

    // 8. Output results
    printf("Termination reason: %d\n", res->termination_reason);
    printf("Primal objective: %.6f\n", res->primal_objective_value);
    printf("Dual objective:   %.6f\n", res->dual_objective_value);
    printf("Solution:\n");
    for (int j = 0; j < res->num_variables; ++j) {
        printf("  x[%d] = %.6f\n", j, res->primal_solution[j]);
    }

    // 9. Cleanup
    qp_problem_free(prob);
    pdhcg_result_free(res);

    return 0;
}
```
