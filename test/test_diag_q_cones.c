/*
 * E2E tests for diagonal-Q PDHG with cone constraints.
 *   Test 1: standard SOC with diag Q, boundary-active solution.
 *   Test 2: rotated SOC with symmetric diag Q, interior unconstrained solution.
 *   Test 4: rotated SOC with asymmetric diag Q (Q_s != Q_t), boundary-active.
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

static int run_test_soc_diag_q(void)
{
    printf("\n[Test 1] Standard SOC + diag Q\n");
    printf("  min (1/2) v^2 + (1/2) z^2 - 3v - 4z  s.t.  w=4, (v,w,z) in K_soc\n");

    double aval[] = {1.0};
    int acol[] = {1};
    int arow[] = {0, 1};
    matrix_desc_t A;
    memset(&A, 0, sizeof(A));
    A.m = 1;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 1;
    A.data.csr.row_ptr = arow;
    A.data.csr.col_ind = acol;
    A.data.csr.vals = aval;

    double qval[] = {1.0, 1.0};
    int qcol[] = {0, 2};
    int qrow[] = {0, 1, 1, 2};
    matrix_desc_t Q;
    memset(&Q, 0, sizeof(Q));
    Q.m = 3;
    Q.n = 3;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = 2;
    Q.data.csr.row_ptr = qrow;
    Q.data.csr.col_ind = qcol;
    Q.data.csr.vals = qval;

    double c[] = {-3.0, 0.0, -4.0};
    double var_lb[] = {-1e30, -1e30, -1e30};
    double var_ub[] = {1e30, 1e30, 1e30};
    double con_lb[] = {4.0};
    double con_ub[] = {4.0};

    cone_spec_t cones[] = {{.type = CONE_STANDARD_SOC, .start_idx = 0, .v_dim = 1, .is_fixed = NULL}};

    qp_problem_t *prob =
        create_qp_problem(c, &Q, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 1, cones, 0, NULL, NULL);
    if (!prob)
    {
        fprintf(stderr, "  create_qp_problem returned NULL\n");
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
        fprintf(stderr, "  solve_qp_problem returned NULL\n");
        qp_problem_free(prob);
        return 1;
    }

    double v = res->primal_solution[0], w = res->primal_solution[1], z = res->primal_solution[2];
    double cone_lhs = v * v + w * w;
    double cone_rhs = z * z;
    double cone_viol = cone_lhs - cone_rhs;
    double lam_v = (v > 1e-9) ? (3.0 / v - 1.0) / 2.0 : 0.0;
    double lam_z = (z > 1e-9) ? (1.0 - 4.0 / z) / 2.0 : 0.0;
    double lam_mismatch = fabs(lam_v - lam_z);

    printf("  status=%d  iter=%d  obj=%.6f  v=%.6f w=%.6f z=%.6f\n",
           (int)res->termination_reason,
           res->total_count,
           res->primal_objective_value,
           v,
           w,
           z);
    printf("  cone violation (v^2+w^2-z^2) = %.3e  (expect 0)\n", cone_viol);
    printf("  KKT mismatch |lam(v)-lam(z)| = %.3e  (expect 0)\n", lam_mismatch);
    printf("  dual y_w = %.6f  (expect ~0.64 at KKT)\n", res->dual_solution[0]);

    int pass = (res->termination_reason == TERMINATION_REASON_OPTIMAL) && approx_eq(w, 4.0, 1e-5) &&
        fabs(cone_viol) < 1e-5 && lam_mismatch < 1e-4;
    printf("  %s\n", pass ? "PASS" : "FAIL");

    pdhcg_result_free(res);
    qp_problem_free(prob);
    return pass ? 0 : 1;
}

static int run_test_rsoc_diag_q(void)
{
    printf("\n[Test 2] Rotated SOC + diag Q symmetric (Q_s == Q_t)\n");
    printf("  min (1/2) v^2 + (1/2) s^2 + (1/2) t^2 - v - 3s - 3t  s.t.  (v,s,t) in K_rsoc\n");

    matrix_desc_t A;
    memset(&A, 0, sizeof(A));
    int arow_empty[] = {0};
    A.m = 0;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 0;
    A.data.csr.row_ptr = arow_empty;
    A.data.csr.col_ind = NULL;
    A.data.csr.vals = NULL;

    double qval[] = {1.0, 1.0, 1.0};
    int qcol[] = {0, 1, 2};
    int qrow[] = {0, 1, 2, 3};
    matrix_desc_t Q;
    memset(&Q, 0, sizeof(Q));
    Q.m = 3;
    Q.n = 3;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = 3;
    Q.data.csr.row_ptr = qrow;
    Q.data.csr.col_ind = qcol;
    Q.data.csr.vals = qval;

    double c[] = {-1.0, -3.0, -3.0};
    double var_lb[] = {-1e30, -1e30, -1e30};
    double var_ub[] = {1e30, 1e30, 1e30};

    cone_spec_t cones[] = {{.type = CONE_ROTATED_SOC, .start_idx = 0, .v_dim = 1, .is_fixed = NULL}};

    qp_problem_t *prob =
        create_qp_problem(c, &Q, NULL, NULL, &A, NULL, NULL, var_lb, var_ub, NULL, 1, cones, 0, NULL, NULL);
    if (!prob)
    {
        fprintf(stderr, "  create_qp_problem returned NULL\n");
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
        fprintf(stderr, "  solve_qp_problem returned NULL\n");
        qp_problem_free(prob);
        return 1;
    }

    double v = res->primal_solution[0], s = res->primal_solution[1], t = res->primal_solution[2];
    printf("  status=%d  iter=%d  obj=%.6f  v=%.6f s=%.6f t=%.6f\n",
           (int)res->termination_reason,
           res->total_count,
           res->primal_objective_value,
           v,
           s,
           t);
    double cone_viol = v * v - 2.0 * s * t;
    printf("  cone slack v^2 - 2st = %.3e  (expect <= 0)\n", cone_viol);

    int pass = (res->termination_reason == TERMINATION_REASON_OPTIMAL) && approx_eq(v, 1.0, 1e-4) &&
        approx_eq(s, 3.0, 1e-4) && approx_eq(t, 3.0, 1e-4) && (cone_viol < 1e-6);
    printf("  %s  (expected v=1, s=t=3, obj=-9.5)\n", pass ? "PASS" : "FAIL");

    pdhcg_result_free(res);
    qp_problem_free(prob);
    return pass ? 0 : 1;
}

static int run_test_rsoc_diag_q_asymmetric(void)
{
    printf("\n[Test 4] Rotated SOC + diag Q asymmetric (Q_s != Q_t), boundary active\n");
    printf("  min (1/2)(v^2 + s^2 + 4 t^2) - 4v - 4s - 4t  s.t.  (v,s,t) in K_rsoc\n");

    matrix_desc_t A;
    memset(&A, 0, sizeof(A));
    int arow_empty[] = {0};
    A.m = 0;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 0;
    A.data.csr.row_ptr = arow_empty;
    A.data.csr.col_ind = NULL;
    A.data.csr.vals = NULL;

    double qval[] = {1.0, 1.0, 4.0};
    int qcol[] = {0, 1, 2};
    int qrow[] = {0, 1, 2, 3};
    matrix_desc_t Q;
    memset(&Q, 0, sizeof(Q));
    Q.m = 3;
    Q.n = 3;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = 3;
    Q.data.csr.row_ptr = qrow;
    Q.data.csr.col_ind = qcol;
    Q.data.csr.vals = qval;

    double c[] = {-4.0, -4.0, -4.0};
    double var_lb[] = {-1e30, -1e30, -1e30};
    double var_ub[] = {1e30, 1e30, 1e30};

    cone_spec_t cones[] = {{.type = CONE_ROTATED_SOC, .start_idx = 0, .v_dim = 1, .is_fixed = NULL}};

    qp_problem_t *prob =
        create_qp_problem(c, &Q, NULL, NULL, &A, NULL, NULL, var_lb, var_ub, NULL, 1, cones, 0, NULL, NULL);
    if (!prob)
    {
        fprintf(stderr, "  create_qp_problem returned NULL\n");
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
        fprintf(stderr, "  solve_qp_problem returned NULL\n");
        qp_problem_free(prob);
        return 1;
    }

    double v = res->primal_solution[0];
    double s = res->primal_solution[1];
    double t = res->primal_solution[2];
    double cone_viol = v * v - 2.0 * s * t;

    /* Reference solution computed offline by Newton-bisection on the KKT system. */
    const double xi_ref = 0.11320819470084;
    const double v_ref = 3.26153501744331;
    const double s_ref = 4.28128575585834;
    const double t_ref = 1.24233831570958;
    const double obj_ref = -17.57031817800;

    printf("  status=%d  iter=%d  obj=%.6f  v=%.6f s=%.6f t=%.6f\n",
           (int)res->termination_reason,
           res->total_count,
           res->primal_objective_value,
           v,
           s,
           t);
    printf("  cone violation (v^2 - 2st) = %+.3e  (expect ~0, |.| <= 1e-5)\n", cone_viol);

    double xi = (v > 1e-9) ? (4.0 / v - 1.0) / 2.0 : 0.0;
    double kkt_s = s - 4.0 - 2.0 * xi * t;
    double kkt_t = 4.0 * t - 4.0 - 2.0 * xi * s;
    printf("  recovered xi=%.6f (ref %.6f)\n", xi, xi_ref);
    printf("  KKT residual s - 4 - 2 xi t   = %+.3e\n", kkt_s);
    printf("  KKT residual 4t - 4 - 2 xi s  = %+.3e\n", kkt_t);
    printf("  reference (v*,s*,t*) = (%.5f, %.5f, %.5f),  obj* = %.5f\n", v_ref, s_ref, t_ref, obj_ref);

    int pass = (res->termination_reason == TERMINATION_REASON_OPTIMAL) && (fabs(cone_viol) < 1e-5) && (xi >= -1e-6) &&
        approx_eq(v, v_ref, 1e-3) && approx_eq(s, s_ref, 1e-3) && approx_eq(t, t_ref, 1e-3) && (fabs(kkt_s) < 1e-3) &&
        (fabs(kkt_t) < 1e-3);
    printf("  %s\n", pass ? "PASS" : "FAIL");

    pdhcg_result_free(res);
    qp_problem_free(prob);
    return pass ? 0 : 1;
}

int main(void)
{
    int fails = 0;
    fails += run_test_soc_diag_q();
    fails += run_test_rsoc_diag_q();
    fails += run_test_rsoc_diag_q_asymmetric();
    printf("\n=== Summary: %s ===\n", fails == 0 ? "ALL PASSED" : "SOME FAILED");
    return fails;
}
