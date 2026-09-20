#include "pdhcg.h"

#include <math.h>
#include <stdio.h>

static int solve_linearized_case(int with_sparse_q)
{
    static const int a_row[] = {0, 0};
    static const int r_row[] = {0, 2};
    static const int r_col[] = {0, 1};
    static const double r_val[] = {1.0, 1.0};
    static const int q_row[] = {0, 1, 2};
    static const int q_col[] = {0, 1};
    static const double q_val[] = {1.0, 1.0};

    matrix_desc_t A = {.m = 1, .n = 2, .fmt = matrix_csr};
    A.data.csr.row_ptr = a_row;
    matrix_desc_t R = {.m = 1, .n = 2, .fmt = matrix_csr};
    R.data.csr.nnz = 2;
    R.data.csr.row_ptr = r_row;
    R.data.csr.col_ind = r_col;
    R.data.csr.vals = r_val;
    matrix_desc_t Q = {.m = 2, .n = 2, .fmt = matrix_csr};
    Q.data.csr.nnz = 2;
    Q.data.csr.row_ptr = q_row;
    Q.data.csr.col_ind = q_col;
    Q.data.csr.vals = q_val;

    double c[] = {with_sparse_q ? -3.0 : -2.0, with_sparse_q ? -3.0 : -2.0};
    double con_lb[] = {-INFINITY};
    double con_ub[] = {INFINITY};
    double var_lb[] = {-1e30, -1e30};
    double var_ub[] = {1e30, 1e30};
    qp_problem_t *problem = create_qp_problem(
        c, with_sparse_q ? &Q : NULL, &R, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 0, NULL, NULL, NULL, 0, NULL);
    if (!problem)
        return 0;

    pdhg_parameters_t parameters;
    set_default_parameters(&parameters);
    parameters.verbose = 0;
    parameters.non_diagonal_quadratic_mode = NON_DIAGONAL_QUADRATIC_LINEARIZED;
    parameters.termination_criteria.eps_optimal_relative = 1e-7;
    parameters.termination_criteria.eps_feasible_relative = 1e-7;
    parameters.termination_criteria.time_sec_limit = 10.0;
    pdhcg_result_t *result = solve_qp_problem(problem, &parameters);

    int pass = result && result->termination_reason == TERMINATION_REASON_OPTIMAL;
    if (pass && with_sparse_q)
        pass = fabs(result->primal_solution[0] - 1.0) < 1e-4 && fabs(result->primal_solution[1] - 1.0) < 1e-4;
    else if (pass)
        pass = fabs(result->primal_solution[0] + result->primal_solution[1] - 2.0) < 1e-4;

    pdhcg_result_free(result);
    qp_problem_free(problem);
    return pass;
}

int main(void)
{
    int pass = solve_linearized_case(0) && solve_linearized_case(1);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
