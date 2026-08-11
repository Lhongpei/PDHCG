/*
 * Compare two formulations of the same QCQP:
 *   min -x + y^2  s.t. x^2 + y^2 <= 1
 *
 * Version A (overlap):    y has Q (y^2 in obj) AND y is linearly coupled to cone slot v2.
 * Version B (separated):  new aux y_q holds Q (y_q^2 in obj); y is ONLY in the cone link;
 *                         y_q - y = 0 bridges them.
 *
 * Same optimum (x*=1, y*=0, obj=-1) — comparison is the inner iteration count.
 */

#include "pdhcg.h"
#include "pdhcg_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int solve_and_report(const char *name, qp_problem_t *prob)
{
    if (!prob)
    {
        fprintf(stderr, "[%s] create_qp_problem failed\n", name);
        return 1;
    }
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 0;
    params.termination_criteria.eps_optimal_relative = 1e-8;
    params.termination_criteria.eps_feasible_relative = 1e-8;
    params.termination_criteria.iteration_limit = 200000;
    params.termination_criteria.time_sec_limit = 60.0;
    pdhcg_result_t *res = solve_qp_problem(prob, &params);
    if (!res)
    {
        qp_problem_free(prob);
        return 1;
    }
    double x = res->primal_solution[0], y = res->primal_solution[1];
    printf("[%-12s] status=%d iter=%d obj=%.8f  x=%.6f y=%.6f\n",
           name,
           (int)res->termination_reason,
           res->total_count,
           res->primal_objective_value,
           x,
           y);
    pdhcg_result_free(res);
    qp_problem_free(prob);
    return 0;
}

/* Version A: vars (x, y, v1, v2, s, t); Q on y; y links to cone via sqrt(2)*y - v2 = 0. */
static qp_problem_t *build_overlap(void)
{
    const double SQRT2 = 1.4142135623730951;
    /* rows: sqrt2 x - v1 = 0; sqrt2 y - v2 = 0; s = 1; t = 1 */
    double aval[] = {SQRT2, -1.0, SQRT2, -1.0, 1.0, 1.0};
    int acol[] = {0, 2, 1, 3, 4, 5};
    int arow[] = {0, 2, 4, 5, 6};
    matrix_desc_t A = {0};
    A.m = 4;
    A.n = 6;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 6;
    A.data.csr.row_ptr = arow;
    A.data.csr.col_ind = acol;
    A.data.csr.vals = aval;

    double qval[] = {2.0}; /* Q on y slot (col 1) */
    int qcol[] = {1};
    int qrow[] = {0, 0, 1, 1, 1, 1, 1};
    matrix_desc_t Q = {0};
    Q.m = 6;
    Q.n = 6;
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
    return create_qp_problem(
        c, &Q, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 1, cones, NULL, NULL, 0, NULL);
}

/* Version B: vars (x, y, y_q, v1, v2, s, t); Q on y_q (NOT on y);
   add row y_q - y = 0 bridging them; cone link is sqrt(2)*y - v2 = 0 as before. */
static qp_problem_t *build_separated(void)
{
    const double SQRT2 = 1.4142135623730951;
    /* rows:
       0: sqrt2 x - v1 = 0
       1: sqrt2 y - v2 = 0
       2: y_q - y = 0           ← new
       3: s = 1
       4: t = 1                                                                 */
    double aval[] = {SQRT2, -1.0, SQRT2, -1.0, -1.0, 1.0, 1.0, 1.0};
    int acol[] = {0, 3, 1, 4, 1, 2, 5, 6};
    int arow[] = {0, 2, 4, 6, 7, 8};
    matrix_desc_t A = {0};
    A.m = 5;
    A.n = 7;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 8;
    A.data.csr.row_ptr = arow;
    A.data.csr.col_ind = acol;
    A.data.csr.vals = aval;

    double qval[] = {2.0}; /* Q on y_q slot (col 2) */
    int qcol[] = {2};
    int qrow[] = {0, 0, 0, 1, 1, 1, 1, 1};
    matrix_desc_t Q = {0};
    Q.m = 7;
    Q.n = 7;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = 1;
    Q.data.csr.row_ptr = qrow;
    Q.data.csr.col_ind = qcol;
    Q.data.csr.vals = qval;

    double c[] = {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double var_lb[] = {-1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30};
    double var_ub[] = {1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30};
    double con_lb[] = {0.0, 0.0, 0.0, 1.0, 1.0};
    double con_ub[] = {0.0, 0.0, 0.0, 1.0, 1.0};
    cone_spec_t cones[] = {{.type = CONE_ROTATED_SOC, .start_idx = 3, .v_dim = 2, .is_fixed = NULL}};
    return create_qp_problem(
        c, &Q, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 1, cones, NULL, NULL, 0, NULL);
}

int main(void)
{
    int rcA = solve_and_report("overlap", build_overlap());
    int rcB = solve_and_report("separated", build_separated());
    return rcA | rcB;
}
