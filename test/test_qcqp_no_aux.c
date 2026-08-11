/*
 * Same QCQP (min -x + y^2  s.t.  x^2 + y^2 <= 1) tested two ways:
 *   A) lifted with aux v_i = sqrt(2) x_i: cone slots have NO Q  -> closed form path
 *   B) NO aux: cone slots ARE (x, y, s, t) with s=t=1/2, Q lives on cone slot y
 *      -> kernel MUST bisect (w_y = 1 + tau*2 != 1)
 *
 * Answers whether bisection convergence is fine when Q genuinely lives on cone slots.
 * Optimum x=1, y=0, obj=-1 in both cases.
 */

#include "pdhcg.h"
#include "pdhcg_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int solve_and_report(const char *name, qp_problem_t *prob, double eps)
{
    if (!prob)
    {
        fprintf(stderr, "[%s] create failed\n", name);
        return 1;
    }
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 0;
    params.termination_criteria.eps_optimal_relative = eps;
    params.termination_criteria.eps_feasible_relative = eps;
    params.termination_criteria.iteration_limit = 500000;
    params.termination_criteria.time_sec_limit = 30.0;
    pdhcg_result_t *res = solve_qp_problem(prob, &params);
    if (!res)
    {
        qp_problem_free(prob);
        return 1;
    }
    double x = res->primal_solution[0], y = res->primal_solution[1];
    printf("[%-15s eps=%.0e] status=%d iter=%6d obj=%.8f  x=%.6f y=%.6f\n",
           name,
           eps,
           (int)res->termination_reason,
           res->total_count,
           res->primal_objective_value,
           x,
           y);
    pdhcg_result_free(res);
    qp_problem_free(prob);
    return 0;
}

/* A: aux lift. vars = (x, y, v1, v2, s, t); v_i = sqrt(2) x_i; s = t = 1. */
static qp_problem_t *build_aux(void)
{
    const double SQRT2 = 1.4142135623730951;
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

    double qval[] = {2.0};
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
    return create_qp_problem(c, &Q, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 1, cones, 0, NULL, NULL);
}

/* B: NO aux. vars = (x, y, s, t); (x,y,s,t) in K_rsoc, s = t = 1/2 -> x^2 + y^2 <= 1.
   Q on cone slot y (index 1). Q_yy = 2 -> weight w_y = 1 + tau*2 != 1. Kernel bisection. */
static qp_problem_t *build_no_aux(void)
{
    /* rows:   0: s = 1/2
               1: t = 1/2                                                                    */
    double aval[] = {1.0, 1.0};
    int acol[] = {2, 3};
    int arow[] = {0, 1, 2};
    matrix_desc_t A = {0};
    A.m = 2;
    A.n = 4;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 2;
    A.data.csr.row_ptr = arow;
    A.data.csr.col_ind = acol;
    A.data.csr.vals = aval;

    /* Q_yy = 2 on cone slot (index 1) */
    double qval[] = {2.0};
    int qcol[] = {1};
    int qrow[] = {0, 0, 1, 1, 1};
    matrix_desc_t Q = {0};
    Q.m = 4;
    Q.n = 4;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = 1;
    Q.data.csr.row_ptr = qrow;
    Q.data.csr.col_ind = qcol;
    Q.data.csr.vals = qval;

    double c[] = {-1.0, 0.0, 0.0, 0.0};
    double var_lb[] = {-1e30, -1e30, -1e30, -1e30};
    double var_ub[] = {1e30, 1e30, 1e30, 1e30};
    /* s=1, t=1/2 -> 2*s*t=1, so x^2+y^2 <= 1 */
    double con_lb[] = {1.0, 0.5};
    double con_ub[] = {1.0, 0.5};
    /* cone is (v0=x, v1=y, s, t) with v_dim=2. Q lives on v1. */
    cone_spec_t cones[] = {{.type = CONE_ROTATED_SOC, .start_idx = 0, .v_dim = 2, .is_fixed = NULL}};
    return create_qp_problem(c, &Q, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 1, cones, 0, NULL, NULL);
}

int main(void)
{
    double eps_list[] = {1e-4, 1e-6, 1e-8};
    for (int i = 0; i < 3; ++i)
    {
        double eps = eps_list[i];
        solve_and_report("aux", build_aux(), eps);
        solve_and_report("no_aux", build_no_aux(), eps);
    }
    return 0;
}
