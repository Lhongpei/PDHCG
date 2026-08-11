/*
 * QCQP -> conic QP demo via RSOC.
 *   Original: min -x + y^2  s.t.  x^2 + y^2 <= 1
 *   Lifted:   vars (x, y, v1, v2, s, t)
 *             v1 - sqrt(2) x = 0, v2 - sqrt(2) y = 0, s = 1, t = 1
 *             (v1, v2, s, t) in K_rsoc
 *   Expected: x*=1, y*=0, obj*=-1.
 */

#include "pdhcg.h"
#include "pdhcg_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    const double SQRT2 = 1.4142135623730951;
    int n = 6; /* x=0, y=1, v1=2, v2=3, s=4, t=5 */

    /* A (4 rows, sparse CSR):
       r0: sqrt(2) x - v1 = 0  -> A[0]=(0, sqrt2), (2, -1)
       r1: sqrt(2) y - v2 = 0  -> A[1]=(1, sqrt2), (3, -1)
       r2: s = 1               -> A[2]=(4, 1)
       r3: t = 1               -> A[3]=(5, 1) */
    double aval[] = {SQRT2, -1.0, SQRT2, -1.0, 1.0, 1.0};
    int acol[] = {0, 2, 1, 3, 4, 5};
    int arow[] = {0, 2, 4, 5, 6};
    matrix_desc_t A = {0};
    A.m = 4;
    A.n = n;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 6;
    A.data.csr.row_ptr = arow;
    A.data.csr.col_ind = acol;
    A.data.csr.vals = aval;

    /* Q (diag with Q_yy = 2 only, so 0.5 * 2 * y^2 = y^2). */
    double qval[] = {2.0};
    int qcol[] = {1};
    int qrow[] = {0, 0, 1, 1, 1, 1, 1};
    matrix_desc_t Q = {0};
    Q.m = n;
    Q.n = n;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = 1;
    Q.data.csr.row_ptr = qrow;
    Q.data.csr.col_ind = qcol;
    Q.data.csr.vals = qval;

    double c[] = {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double var_lb[] = {-1e30, -1e30, -1e30, -1e30, -1e30, -1e30};
    double var_ub[] = {1e30, 1e30, 1e30, 1e30, 1e30, 1e30};
    double con_lb[] = {0.0, 0.0, 1.0, 1.0};
    double con_ub[] = {0.0, 0.0, 1.0, 1.0};

    cone_spec_t cones[] = {{.type = CONE_ROTATED_SOC, .start_idx = 2, .v_dim = 2, .is_fixed = NULL}};

    qp_problem_t *prob =
        create_qp_problem(c, &Q, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 1, cones, NULL, NULL, 0, NULL);
    if (!prob)
    {
        fprintf(stderr, "create_qp_problem failed\n");
        return 1;
    }

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 1;
    params.termination_criteria.eps_optimal_relative = 1e-8;
    params.termination_criteria.eps_feasible_relative = 1e-8;
    params.termination_criteria.iteration_limit = 100000;
    params.termination_criteria.time_sec_limit = 30.0;
    pdhcg_result_t *res = solve_qp_problem(prob, &params);
    if (!res)
    {
        qp_problem_free(prob);
        return 1;
    }

    double x = res->primal_solution[0], y = res->primal_solution[1];
    double v1 = res->primal_solution[2], v2 = res->primal_solution[3];
    double s = res->primal_solution[4], t = res->primal_solution[5];
    double q_lhs = x * x + y * y;
    double cone_viol_v = v1 * v1 + v2 * v2 - 2.0 * s * t;

    printf(
        "\nstatus=%d iter=%d obj=%.6f\n", (int)res->termination_reason, res->total_count, res->primal_objective_value);
    printf("x=%.6f y=%.6f  (expect x=1, y=0)\n", x, y);
    printf("v1=%.6f v2=%.6f s=%.6f t=%.6f\n", v1, v2, s, t);
    printf("original QC residual (x^2+y^2 - 1) = %.3e  (expect <= 0)\n", q_lhs - 1.0);
    printf("RSOC slack (||v||^2 - 2st) = %.3e  (expect <= 0)\n", cone_viol_v);

    int pass = (res->termination_reason == TERMINATION_REASON_OPTIMAL) && fabs(x - 1.0) < 1e-4 && fabs(y) < 1e-4 &&
        fabs(res->primal_objective_value - (-1.0)) < 1e-4 && (q_lhs - 1.0 < 1e-5) && (cone_viol_v < 1e-5);
    printf("%s\n", pass ? "PASS" : "FAIL");

    pdhcg_result_free(res);
    qp_problem_free(prob);
    return pass ? 0 : 1;
}
