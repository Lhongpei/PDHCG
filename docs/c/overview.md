# C API Overview

PDHCG provides a C API for integration with other languages and applications.

## Header Files

```c
#include "pdhcg.h"       // Main API functions
#include "pdhcg_types.h" // Type definitions
```

## Quick Example

```c
#include "pdhcg.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Define problem dimensions
    int n = 2;  // variables
    int m = 3;  // constraints

    // Objective vector
    double c[] = {-2.0, -6.0};

    // Constraint matrix (CSR format)
    int row_ptr[] = {0, 2, 4, 6};
    int col_ind[] = {0, 1, 0, 1, 0, 1};
    double vals[] = {1.0, 1.0, -1.0, 2.0, 2.0, 1.0};

    matrix_desc_t A_desc = {
        .m = m,
        .n = n,
        .fmt = matrix_csr,
        .zero_tolerance = 0.0,
        .data.csr.nnz = 6,
        .data.csr.row_ptr = row_ptr,
        .data.csr.col_ind = col_ind,
        .data.csr.vals = vals
    };

    // Bounds
    double con_lb[] = {-1e30, -1e30, -1e30};
    double con_ub[] = {2.0, 2.0, 3.0};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {1e30, 1e30};

    // Create problem (NULL for Q, R, and D -> linear problem)
    qp_problem_t *prob = create_qp_problem(
        c, NULL, NULL, NULL, &A_desc,
        con_lb, con_ub, var_lb, var_ub, NULL,
        0, NULL, NULL, NULL, 0, NULL
    );

    // Set parameters
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 1;

    // Solve
    pdhcg_result_t *result = solve_qp_problem(prob, &params);

    // Print results
    printf("Status: %d\n", result->termination_reason);
    printf("Objective: %f\n", result->primal_objective_value);
    printf("Iterations: %d\n", result->total_count);

    // Cleanup
    pdhcg_result_free(result);
    qp_problem_free(prob);

    return 0;
}
```

## Distributed / Multi-GPU Solving

PDHCG supports distributed solving across multiple GPUs via MPI and NCCL. The public header declares:

```c
pdhcg_result_t *solve_qp_problem_distributed(const pdhg_parameters_t *params,
                                             const qp_problem_t *original_problem);
```

Use `solve_qp_problem_distributed()` in place of `solve_qp_problem()`, compile with
`-DPDHCG_COMPILE_DISTRIBUTED=ON`, and launch your program with `mpirun` (or
`mpiexec`). A non-distributed build keeps the symbol for API compatibility and
returns `NULL` with an explanatory error.

See the [C API Functions](functions.md) reference for details, and the [Examples](../examples.md) page for usage examples.

## Main Functions

### Problem Creation

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
`min 0.5 * x^T (Q + R^T D R) x + c^T x  s.t.  con_lb <= A x <= con_ub,  var_lb <= x <= var_ub`,
with optional variable cones and native affine constraints
`F x + affine_cone_offset in K`. Affine cone starts are relative to rows of
`F`, and the blocks must cover `F` completely. The problem is built from matrix
descriptors. `Q_desc` (sparse quadratic), `R_desc` (low-rank factor, shape
`k x n`), and `D_desc` (rank-by-rank middle matrix in `R^T D R`) are all
optional — pass `NULL` to omit any of them. `D_desc` defaults to identity,
recovering the standard `Q + R^T R` formulation; it may be diagonal, sparse,
dense, or indefinite, and the runtime auto-detects the cheapest representation.

### Setting Start Values

```c
void set_start_values(
    qp_problem_t *prob,
    const double *primal,
    const double *dual
);
```

Sets initial primal and dual solutions for warm starting. A `NULL` primal clears
free-coordinate warm starts but preserves values pinned by `set_cone_fixed`.

### Solving

```c
pdhcg_result_t *solve_qp_problem(
    const qp_problem_t *prob,
    const pdhg_parameters_t *params
);
```

Solves the QP problem and returns the results.

### Default Parameters

```c
void set_default_parameters(pdhg_parameters_t *params);
```

Fills the parameter struct with default values.

### Cleanup

```c
void pdhcg_result_free(pdhcg_result_t *results);
void qp_problem_free(qp_problem_t *prob);
```

Frees allocated memory for results and problems.
