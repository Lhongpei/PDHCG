/*
 * E2E test for off-diagonal (sparse) Q on non-cone vars coupled to a SOC cone.
 * Exercises the SPARSE_Q path through BB with dispatch_cone_projection in the
 * inner loop. Variables: (a, b, v, w, z); Q couples (a, b); (v, w, z) is K_soc.
 */

#include "pdhcg.h"
#include "pdhcg_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    int n = 5;

    double aval[] = {1.0, 1.0, 1.0, -1.0, 1.0, -1.0};
    int acol[] = {0, 1, 0, 2, 1, 3};
    int arow[] = {0, 1, 2, 4, 6};
    matrix_desc_t A = {0};
    A.m = 4;
    A.n = n;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 6;
    A.data.csr.row_ptr = arow;
    A.data.csr.col_ind = acol;
    A.data.csr.vals = aval;

    double qval[] = {1.0, 0.5, 0.5, 1.0};
    int qcol[] = {0, 1, 0, 1};
    int qrow[] = {0, 2, 4, 4, 4, 4};
    matrix_desc_t Q = {0};
    Q.m = n;
    Q.n = n;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = 4;
    Q.data.csr.row_ptr = qrow;
    Q.data.csr.col_ind = qcol;
    Q.data.csr.vals = qval;

    double c[] = {-3.0, -4.0, 0.0, 0.0, 1.0};
    double var_lb[] = {-1e30, -1e30, -1e30, -1e30, -1e30};
    double var_ub[] = {1e30, 1e30, 1e30, 1e30, 1e30};
    double con_lb[] = {3.0, 4.0, 0.0, 0.0};
    double con_ub[] = {3.0, 4.0, 0.0, 0.0};

    cone_spec_t cones[] = {{.type = CONE_STANDARD_SOC, .start_idx = 2, .v_dim = 1, .is_fixed = NULL}};

    qp_problem_t *prob =
        create_qp_problem(c, &Q, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 1, cones, 0, NULL, NULL);
    if (!prob)
    {
        fprintf(stderr, "create_qp_problem failed\n");
        return 1;
    }

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 1;
    params.termination_criteria.eps_optimal_relative = 1e-7;
    params.termination_criteria.eps_feasible_relative = 1e-7;
    params.termination_criteria.iteration_limit = 100000;
    params.termination_criteria.time_sec_limit = 30.0;
    pdhcg_result_t *res = solve_qp_problem(prob, &params);
    if (!res)
    {
        qp_problem_free(prob);
        return 1;
    }

    double a = res->primal_solution[0], b = res->primal_solution[1];
    double v = res->primal_solution[2], w = res->primal_solution[3], z = res->primal_solution[4];
    double cone_lhs = v * v + w * w, cone_rhs = z * z;
    double cone_viol = cone_lhs - cone_rhs;
    printf(
        "\nstatus=%d iter=%d obj=%.6f\n", (int)res->termination_reason, res->total_count, res->primal_objective_value);
    printf("a=%.6f b=%.6f v=%.6f w=%.6f z=%.6f\n", a, b, v, w, z);
    printf("cone violation (v^2+w^2-z^2) = %.3e  (expect 0)\n", cone_viol);
    printf("expected: a=3, b=4, v=3, w=4, z=5; obj=-1.5\n");

    int pass = (res->termination_reason == TERMINATION_REASON_OPTIMAL) && fabs(a - 3.0) < 1e-4 &&
        fabs(b - 4.0) < 1e-4 && fabs(v - 3.0) < 1e-4 && fabs(w - 4.0) < 1e-4 && fabs(z - 5.0) < 1e-4 &&
        fabs(cone_viol) < 1e-4 && fabs(res->primal_objective_value - (-1.5)) < 1e-3;
    printf("%s\n", pass ? "PASS" : "FAIL");

    pdhcg_result_free(res);
    qp_problem_free(prob);
    return pass ? 0 : 1;
}
