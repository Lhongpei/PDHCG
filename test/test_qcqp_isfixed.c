/*
 * Same QCQP (min -x + y^2 s.t. x^2 + y^2 <= 1, optimum x=1, y=0, obj=-1) lifted two ways:
 *
 *  A) pin t via linear equality (t = 1 row + s = 1 row in A matrix; current transform style)
 *  B) pin t via is_fixed slot (t row removed; cone projection treats t as constant 1)
 *
 *  Both also pin s = 1 the same way (RSOC needs it). The point: do is_fixed slots converge
 *  faster than linear-equality-pinned slots at tight tolerance?
 */

#include "pdhcg.h"
#include "pdhcg_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int solve_and_report(const char *name, qp_problem_t *prob, int t_slot, int s_slot)
{
    if (!prob)
    {
        fprintf(stderr, "[%s] create failed\n", name);
        return 1;
    }

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 1;
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
    double t = res->primal_solution[t_slot], s = res->primal_solution[s_slot];
    printf("[%-15s] status=%d iter=%6d obj=%.8f  x=%.6f y=%.6f  s=%.6f t=%.6f\n",
           name,
           (int)res->termination_reason,
           res->total_count,
           res->primal_objective_value,
           x,
           y,
           s,
           t);
    pdhcg_result_free(res);
    qp_problem_free(prob);
    return 0;
}

/* Version A: vars (x, y, v1, v2, s, t); pin s=1 AND t=1 via linear equalities. */
static qp_problem_t *build_linear_pin(void)
{
    const double SQRT2 = 1.4142135623730951;
    /* rows: 0: sqrt2 x - v1 = 0
             1: sqrt2 y - v2 = 0
             2: s = 1
             3: t = 1                                          */
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

/* Version B: vars (x, y, v1, v2, s, t); pin s=1, t=1 via is_fixed cone slots.
   The s=1 and t=1 LINEAR rows are REMOVED — cone slots [4] and [5] are constants. */
static qp_problem_t *build_isfixed_pin(void)
{
    const double SQRT2 = 1.4142135623730951;
    /* rows: 0: sqrt2 x - v1 = 0
             1: sqrt2 y - v2 = 0
       (NO s/t pin rows — those are is_fixed)                  */
    double aval[] = {SQRT2, -1.0, SQRT2, -1.0};
    int acol[] = {0, 2, 1, 3};
    int arow[] = {0, 2, 4};
    matrix_desc_t A = {0};
    A.m = 2;
    A.n = 6;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 4;
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
    double con_lb[] = {0.0, 0.0};
    double con_ub[] = {0.0, 0.0};
    /* is_fixed pattern over the 4 cone slots (v1, v2, s, t): mark s and t. */
    static const char fix_pattern[4] = {0, 0, 1, 1};
    cone_spec_t cones[] = {{.type = CONE_ROTATED_SOC, .start_idx = 2, .v_dim = 2, .is_fixed = fix_pattern}};
    qp_problem_t *prob =
        create_qp_problem(c, &Q, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 1, cones, 0, NULL, NULL);
    if (!prob)
        return NULL;
    /* primal_start must carry the pin values: s=1 at slot 4, t=1 at slot 5. */
    double primal_start[6] = {0, 0, 0, 0, 1.0, 1.0};
    set_start_values(prob, primal_start, NULL);
    return prob;
}

int main(void)
{
    /* s slot = index 4, t slot = index 5 in both versions */
    int a = solve_and_report("linear-pin", build_linear_pin(), /*t*/ 5, /*s*/ 4);
    int b = solve_and_report("is_fixed-pin", build_isfixed_pin(), /*t*/ 5, /*s*/ 4);
    return a | b;
}
