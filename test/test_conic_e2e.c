/*
 * E2E tests for create_qp_problem covering the three supported cone types:
 *   Test 1: standard SOC, recovers (v, w, z) = (3, 4, 5).
 *   Test 2: rotated SOC, recovers s = t = 3/sqrt(2) at v = 3.
 *   Test 3: exponential cone, recovers x = log 2 with y = 1, z = 2.
 */

#include "pdhcg.h"
#include "pdhcg_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int approx_eq(double a, double b, double tol)
{
    return fabs(a - b) <= tol * (1.0 + fabs(b));
}

static int run_test_standard_soc(void)
{
    printf("[Test 1] Standard SOC: min z s.t. v=3, w=4, (v,w,z) in K_soc\n");

    double val[] = {1.0, 1.0};
    int col_ind[] = {0, 1};
    int row_ptr[] = {0, 1, 2};
    matrix_desc_t A;
    memset(&A, 0, sizeof(A));
    A.m = 2;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 2;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = val;

    double c[] = {0.0, 0.0, 1.0};
    double var_lb[] = {-1e30, -1e30, -1e30};
    double var_ub[] = {1e30, 1e30, 1e30};
    double con_lb[] = {3.0, 4.0};
    double con_ub[] = {3.0, 4.0};

    cone_spec_t cones[] = {
        {.type = CONE_STANDARD_SOC, .start_idx = 0, .v_dim = 1},
    };
    qp_problem_t *prob =
        create_qp_problem(c, NULL, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 1, cones, 0, NULL, NULL);
    if (!prob)
    {
        printf("  FAIL: create_qp_problem returned NULL\n");
        return 0;
    }

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 1;
    params.termination_criteria.eps_optimal_relative = 1e-7;
    params.termination_criteria.eps_feasible_relative = 1e-7;

    pdhcg_result_t *res = solve_qp_problem(prob, &params);
    if (!res)
    {
        printf("  FAIL: solve_qp_problem returned NULL\n");
        qp_problem_free(prob);
        return 0;
    }

    int ok = (res->termination_reason == TERMINATION_REASON_OPTIMAL);
    double v = res->primal_solution[0], w = res->primal_solution[1], z = res->primal_solution[2];
    printf(
        "  status=%d obj=%.6f v=%.6f w=%.6f z=%.6f\n", res->termination_reason, res->primal_objective_value, v, w, z);

    ok = ok && approx_eq(v, 3.0, 1e-3) && approx_eq(w, 4.0, 1e-3) && approx_eq(z, 5.0, 1e-3);
    printf("  %s (expected v=3, w=4, z=5)\n", ok ? "PASS" : "FAIL");

    pdhcg_result_free(res);
    qp_problem_free(prob);
    return ok;
}

static int run_test_rotated_soc(void)
{
    printf("\n[Test 2] Rotated SOC: min s+t s.t. v=3, (v,s,t) in K_rsoc\n");

    double val[] = {1.0};
    int col_ind[] = {0};
    int row_ptr[] = {0, 1};
    matrix_desc_t A;
    memset(&A, 0, sizeof(A));
    A.m = 1;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 1;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = val;

    double c[] = {0.0, 1.0, 1.0};
    double var_lb[] = {-1e30, -1e30, -1e30};
    double var_ub[] = {1e30, 1e30, 1e30};
    double con_lb[] = {3.0};
    double con_ub[] = {3.0};

    cone_spec_t cones[] = {
        {.type = CONE_ROTATED_SOC, .start_idx = 0, .v_dim = 1},
    };
    qp_problem_t *prob =
        create_qp_problem(c, NULL, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 1, cones, 0, NULL, NULL);
    if (!prob)
    {
        printf("  FAIL: create_qp_problem returned NULL\n");
        return 0;
    }

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 1;
    params.termination_criteria.eps_optimal_relative = 1e-7;
    params.termination_criteria.eps_feasible_relative = 1e-7;

    pdhcg_result_t *res = solve_qp_problem(prob, &params);
    if (!res)
    {
        printf("  FAIL: solve_qp_problem returned NULL\n");
        qp_problem_free(prob);
        return 0;
    }

    int ok = (res->termination_reason == TERMINATION_REASON_OPTIMAL);
    double v = res->primal_solution[0], s = res->primal_solution[1], t = res->primal_solution[2];
    double expected = 3.0 / sqrt(2.0);
    printf("  status=%d obj=%.6f v=%.6f s=%.6f t=%.6f (expected s=t=%.6f, obj=%.6f)\n",
           res->termination_reason,
           res->primal_objective_value,
           v,
           s,
           t,
           expected,
           2.0 * expected);

    ok = ok && approx_eq(v, 3.0, 1e-3) && approx_eq(s, expected, 5e-3) && approx_eq(t, expected, 5e-3);
    printf("  %s\n", ok ? "PASS" : "FAIL");

    pdhcg_result_free(res);
    qp_problem_free(prob);
    return ok;
}

static int run_test_exp_cone(void)
{
    printf("\n[Test 3] Exp cone (linear obj):  min -x s.t. y=1, z=2, (x,y,z) in K_exp\n");

    double val[] = {1.0, 1.0};
    int col_ind[] = {1, 2};
    int row_ptr[] = {0, 1, 2};
    matrix_desc_t A;
    memset(&A, 0, sizeof(A));
    A.m = 2;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 2;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = val;

    double c[] = {-1.0, 0.0, 0.0};
    double var_lb[] = {-1e30, -1e30, -1e30};
    double var_ub[] = {1e30, 1e30, 1e30};
    double con_lb[] = {1.0, 2.0};
    double con_ub[] = {1.0, 2.0};

    cone_spec_t cones[] = {
        {.type = CONE_EXPONENTIAL, .start_idx = 0, .v_dim = 1},
    };
    qp_problem_t *prob =
        create_qp_problem(c, NULL, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 1, cones, 0, NULL, NULL);
    if (!prob)
    {
        printf("  FAIL: create_qp_problem returned NULL\n");
        return 0;
    }

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 1;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    params.termination_criteria.time_sec_limit = 60.0;

    pdhcg_result_t *res = solve_qp_problem(prob, &params);
    if (!res)
    {
        printf("  FAIL: solve_qp_problem returned NULL\n");
        qp_problem_free(prob);
        return 0;
    }

    double x = res->primal_solution[0], y = res->primal_solution[1], z = res->primal_solution[2];
    int ok = (res->termination_reason == TERMINATION_REASON_OPTIMAL);
    double expected_x = log(2.0);
    double feas_gap = (y > 0.0) ? (y * exp(x / y) - z) : INFINITY;
    printf("  status=%d  x=%.6f y=%.6f z=%.6f  (expected x=%.6f, y=1, z=2)\n",
           res->termination_reason,
           x,
           y,
           z,
           expected_x);
    printf("  feasibility gap y*exp(x/y) - z = %.3e\n", feas_gap);
    ok = ok && approx_eq(x, expected_x, 5e-3) && approx_eq(y, 1.0, 5e-3) && approx_eq(z, 2.0, 5e-3);
    printf("  %s\n", ok ? "PASS" : "FAIL");

    pdhcg_result_free(res);
    qp_problem_free(prob);
    return ok;
}

int main(void)
{
    int p1 = run_test_standard_soc();
    int p2 = run_test_rotated_soc();
    int p3 = run_test_exp_cone();
    printf("\n=== Summary ===\n");
    printf("Test 1 (Standard SOC): %s\n", p1 ? "PASS" : "FAIL");
    printf("Test 2 (Rotated SOC):  %s\n", p2 ? "PASS" : "FAIL");
    printf("Test 3 (Exp cone):     %s\n", p3 ? "PASS" : "FAIL");
    return (p1 && p2 && p3) ? 0 : 1;
}
