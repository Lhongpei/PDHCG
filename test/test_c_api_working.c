/*
 * Test program to verify C API works correctly.
 *
 * This demonstrates that the C API works while Python binding hangs.
 *
 * Compile:
 *   gcc -o test_c_api test_c_api.c -I../include -L../build -lpdhcg -Wl,-rpath,../build -lm
 *
 * Run:
 *   ./test_c_api
 */

#include "pdhcg.h"
#include "pdhcg_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main()
{
    printf("="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "=\n");
    printf("Testing PDHCG C API\n");
    printf("="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "=\n\n");

    /* Problem: min x + 2y s.t. x + y = 1, x,y >= 0 */
    printf("Problem: min x + 2y s.t. x + y = 1\n");
    printf("Expected: x ≈ 1, y ≈ 0, obj ≈ 1\n\n");

    /* A = [1, 1] (CSR format) */
    double val[] = {1.0, 1.0};
    int col_ind[] = {0, 1};
    int row_ptr[] = {0, 2};

    matrix_desc_t A_desc;
    A_desc.m = 1;
    A_desc.n = 2;
    A_desc.fmt = matrix_csr;
    A_desc.zero_tolerance = 0.0;
    A_desc.data.csr.nnz = 2;
    A_desc.data.csr.row_ptr = row_ptr;
    A_desc.data.csr.col_ind = col_ind;
    A_desc.data.csr.vals = val;

    double c[] = {1.0, 2.0};
    double lb[] = {0.0, 0.0};
    double ub[] = {1e30, 1e30}; /* Use large numbers instead of inf */
    double cl[] = {1.0};
    double cu[] = {1.0};

    printf("Creating problem...\n");
    qp_problem_t *prob =
        create_qp_problem(c, NULL, NULL, NULL, &A_desc, cl, cu, lb, ub, NULL, 0, NULL, NULL, NULL, 0, NULL);
    if (!prob)
    {
        printf("FAIL: create_qp_problem failed\n");
        return 1;
    }
    printf("Problem created successfully\n");
    printf("  num_variables: %d\n", prob->num_variables);
    printf("  num_constraints: %d\n", prob->num_constraints);

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.presolve = true; /* Test with presolve enabled */
    params.verbose = 1;

    printf("\nCalling solve_qp_problem (with presolve)...\n");
    clock_t start = clock();
    pdhcg_result_t *result = solve_qp_problem(prob, &params);
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    if (!result)
    {
        printf("FAIL: solve_qp_problem returned NULL\n");
        qp_problem_free(prob);
        return 1;
    }

    printf("\n");
    printf("="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "=\n");
    printf("Results (completed in %.3f seconds)\n", elapsed);
    printf("="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "=\n");
    printf("Status: %d (OPTIMAL=%d)\n", result->termination_reason, TERMINATION_REASON_OPTIMAL);

    if (result->primal_solution)
    {
        printf("Primal X: [%.6f, %.6f]\n", result->primal_solution[0], result->primal_solution[1]);
    }
    if (result->dual_solution)
    {
        printf("Dual Y: [%.6f]\n", result->dual_solution[0]);
    }
    printf("Objective: %.6f\n", result->primal_objective_value);

    /* Verification */
    printf("\n");
    printf("="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "=\n");
    printf("Verification\n");
    printf("="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "=\n");

    int success = (result->termination_reason == TERMINATION_REASON_OPTIMAL);
    double x0 = result->primal_solution ? result->primal_solution[0] : -1;
    double x1 = result->primal_solution ? result->primal_solution[1] : -1;
    double y = result->dual_solution ? result->dual_solution[0] : -1;

    /* Check primal solution */
    if (x0 > 0.9 && x0 < 1.1 && x1 >= 0.0 && x1 < 0.1)
    {
        printf("Primal solution: PASS (x≈1, y≈0)\n");
    }
    else
    {
        printf("Primal solution: FAIL (expected x≈1, y≈0, got x=%.4f, y=%.4f)\n", x0, x1);
        success = 0;
    }

    /* Check dual solution */
    if (y > 0.9 && y < 1.1)
    {
        printf("Dual solution: PASS (y≈1)\n");
    }
    else
    {
        printf("Dual solution: FAIL (expected y≈1, got y=%.4f)\n", y);
        success = 0;
    }

    /* Check objective */
    if (result->primal_objective_value > 0.9 && result->primal_objective_value < 1.1)
    {
        printf("Objective: PASS (obj≈1)\n");
    }
    else
    {
        printf("Objective: FAIL (expected obj≈1, got obj=%.4f)\n", result->primal_objective_value);
        success = 0;
    }

    pdhcg_result_free(result);
    qp_problem_free(prob);

    printf("\n");
    printf("="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "=\n");
    if (success)
    {
        printf("OVERALL: PASS - C API works correctly!\n");
    }
    else
    {
        printf("OVERALL: FAIL - C API has issues\n");
    }
    printf("="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "="
           "=\n");

    return success ? 0 : 1;
}
