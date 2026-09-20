#include "pdhcg.h"
#include "preconditioner.h"
#include "solver_state.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

static void check_close(const char *name, double actual, double expected)
{
    if (!isfinite(actual) || fabs(actual - expected) > 1e-12 * fmax(1.0, fabs(expected)))
    {
        fprintf(stderr, "%s: got %.17g, expected %.17g\n", name, actual, expected);
        ++failures;
    }
}

static qp_problem_t *make_problem(int lowrank, int with_middle)
{
    static const int a_row[] = {0, 2};
    static const int a_col[] = {0, 1};
    static const double a_val[] = {1.0, 1.0};
    static const int q_row[] = {0, 2, 4};
    static const int q_col[] = {0, 1, 0, 1};
    static const double q_val[] = {16.0, 8.0, 8.0, 16.0};
    static const int r_row[] = {0, 2};
    static const int r_col[] = {0, 1};
    static const double r_val[] = {3.0, 4.0};
    static const int d_row[] = {0, 1};
    static const int d_col[] = {0};
    static const double d_val[] = {2.0};
    static const double zero[] = {0.0, 0.0};
    static const double con_lb[] = {-INFINITY};
    static const double con_ub[] = {INFINITY};

    matrix_desc_t A = {.m = 1, .n = 2, .fmt = matrix_csr};
    A.data.csr.nnz = 2;
    A.data.csr.row_ptr = a_row;
    A.data.csr.col_ind = a_col;
    A.data.csr.vals = a_val;

    matrix_desc_t Q = {.m = 2, .n = 2, .fmt = matrix_csr};
    Q.data.csr.nnz = 4;
    Q.data.csr.row_ptr = q_row;
    Q.data.csr.col_ind = q_col;
    Q.data.csr.vals = q_val;

    matrix_desc_t R = {.m = 1, .n = 2, .fmt = matrix_csr};
    R.data.csr.nnz = 2;
    R.data.csr.row_ptr = r_row;
    R.data.csr.col_ind = r_col;
    R.data.csr.vals = r_val;

    matrix_desc_t D = {.m = 1, .n = 1, .fmt = matrix_csr};
    D.data.csr.nnz = 1;
    D.data.csr.row_ptr = d_row;
    D.data.csr.col_ind = d_col;
    D.data.csr.vals = d_val;

    return create_qp_problem(zero,
                             lowrank ? NULL : &Q,
                             lowrank ? &R : NULL,
                             with_middle ? &D : NULL,
                             &A,
                             con_lb,
                             con_ub,
                             NULL,
                             NULL,
                             NULL,
                             0,
                             NULL,
                             NULL,
                             NULL,
                             0,
                             NULL);
}

static pdhg_parameters_t scaling_parameters(non_diagonal_quadratic_mode_t mode, int ruiz)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.non_diagonal_quadratic_mode = mode;
    params.curtis_reid_iterations = 0;
    params.l_inf_ruiz_iterations = ruiz;
    params.has_pock_chambolle_alpha = !ruiz;
    params.pock_chambolle_alpha = 1.0;
    params.bound_objective_rescaling = false;
    return params;
}

static void check_scaling(int lowrank, int with_middle, int ruiz)
{
    qp_problem_t *problem = make_problem(lowrank, with_middle);
    pdhg_parameters_t params = scaling_parameters(NON_DIAGONAL_QUADRATIC_LINEARIZED, ruiz);
    rescale_info_t *info = rescale_problem(&params, problem);

    const double multiplier = with_middle ? 2.0 : 1.0;
    const double expected0 =
        lowrank ? (ruiz ? sqrt(12.0 * multiplier) : sqrt(1.0 + 21.0 * multiplier)) : (ruiz ? 4.0 : 5.0);
    const double expected1 =
        lowrank ? (ruiz ? sqrt(16.0 * multiplier) : sqrt(1.0 + 28.0 * multiplier)) : (ruiz ? 4.0 : 5.0);
    check_close(lowrank ? "low-rank variable scale 0" : "sparse variable scale 0", info->var_rescale[0], expected0);
    check_close(lowrank ? "low-rank variable scale 1" : "sparse variable scale 1", info->var_rescale[1], expected1);

    rescale_info_free(info);
    qp_problem_free(problem);
}

static void check_inner_unchanged(void)
{
    qp_problem_t *problem = make_problem(0, 0);
    pdhg_parameters_t params = scaling_parameters(NON_DIAGONAL_QUADRATIC_INNER, 1);
    rescale_info_t *info = rescale_problem(&params, problem);
    check_close("inner variable scale 0", info->var_rescale[0], 1.0);
    check_close("inner variable scale 1", info->var_rescale[1], 1.0);
    rescale_info_free(info);
    qp_problem_free(problem);
}

int main(void)
{
    check_scaling(0, 0, 1);
    check_scaling(0, 0, 0);
    check_scaling(1, 0, 1);
    check_scaling(1, 0, 0);
    check_scaling(1, 1, 1);
    check_scaling(1, 1, 0);
    check_inner_unchanged();
    printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
