#include "pdhcg.h"
#include "preconditioner.h"
#include "solver_state.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

static void check_close(const char *name, double actual, double expected)
{
    if (!isfinite(actual) || fabs(actual - expected) > 1e-5 * fmax(1.0, fabs(expected)))
    {
        fprintf(stderr, "%s: got %.9g, expected %.9g\n", name, actual, expected);
        ++failures;
    }
}

static qp_problem_t *make_problem(const matrix_desc_t *Q, const matrix_desc_t *R, const matrix_desc_t *D)
{
    static const int a_row[] = {0, 0};
    static const double zero[] = {0.0, 0.0};
    static const double con_lb[] = {-INFINITY};
    static const double con_ub[] = {INFINITY};
    matrix_desc_t A = {.m = 1, .n = 2, .fmt = matrix_csr};
    A.data.csr.row_ptr = a_row;
    return create_qp_problem(zero, Q, R, D, &A, con_lb, con_ub, NULL, NULL, NULL, 0, NULL, NULL, NULL, 0, NULL);
}

static void check_information(const char *name, qp_problem_t *problem, double expected_norm, double expected_minimum)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 0;
    params.l_inf_ruiz_iterations = 0;
    params.curtis_reid_iterations = 0;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = false;
    params.sv_tol = 1e-8;

    rescale_info_t *rescale_info = rescale_problem(&params, problem);
    pdhg_solver_state_t *state = initialize_solver_state(&params, problem, rescale_info, NULL);

    char norm_name[96];
    char minimum_name[96];
    snprintf(norm_name, sizeof(norm_name), "%s norm", name);
    snprintf(minimum_name, sizeof(minimum_name), "%s minimum eigenvalue", name);
    check_close(norm_name, state->quadratic_objective_term->norm, expected_norm);
    check_close(minimum_name, state->quadratic_objective_term->nonconvexity, expected_minimum);

    pdhg_solver_state_free(state);
    rescale_info_free(rescale_info);
    qp_problem_free(problem);
}

int main(void)
{
    static const int q_full_row[] = {0, 2, 4};
    static const int q_full_col[] = {0, 1, 0, 1};
    static const double q_full_val[] = {2.0, 1.0, 1.0, 2.0};
    matrix_desc_t Q_full = {.m = 2, .n = 2, .fmt = matrix_csr};
    Q_full.data.csr.nnz = 4;
    Q_full.data.csr.row_ptr = q_full_row;
    Q_full.data.csr.col_ind = q_full_col;
    Q_full.data.csr.vals = q_full_val;
    check_information("sparse", make_problem(&Q_full, NULL, NULL), 3.0, 1.0);

    static const int r_row[] = {0, 2};
    static const int r_col[] = {0, 1};
    static const double r_val[] = {1.0, 1.0};
    static const int d_row[] = {0, 1};
    static const int d_col[] = {0};
    static const double d_negative_val[] = {-1.0};
    matrix_desc_t R = {.m = 1, .n = 2, .fmt = matrix_csr};
    R.data.csr.nnz = 2;
    R.data.csr.row_ptr = r_row;
    R.data.csr.col_ind = r_col;
    R.data.csr.vals = r_val;
    matrix_desc_t D_negative = {.m = 1, .n = 1, .fmt = matrix_csr};
    D_negative.data.csr.nnz = 1;
    D_negative.data.csr.row_ptr = d_row;
    D_negative.data.csr.col_ind = d_col;
    D_negative.data.csr.vals = d_negative_val;
    check_information("low-rank", make_problem(NULL, &R, &D_negative), 2.0, -2.0);

    static const int q_diag_row[] = {0, 1, 2};
    static const int q_diag_col[] = {0, 1};
    static const double q_diag_val[] = {3.0, 1.0};
    static const double d_twice_negative_val[] = {-2.0};
    matrix_desc_t Q_diag = {.m = 2, .n = 2, .fmt = matrix_csr};
    Q_diag.data.csr.nnz = 2;
    Q_diag.data.csr.row_ptr = q_diag_row;
    Q_diag.data.csr.col_ind = q_diag_col;
    Q_diag.data.csr.vals = q_diag_val;
    matrix_desc_t D_twice_negative = D_negative;
    D_twice_negative.data.csr.vals = d_twice_negative_val;
    check_information("sparse plus low-rank", make_problem(&Q_diag, &R, &D_twice_negative), sqrt(5.0), -sqrt(5.0));

    printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
