#include "pdhcg.h"
#include "pdhcg_types.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int close_enough(double value, double expected, double tolerance)
{
    return fabs(value - expected) <= tolerance * (1.0 + fabs(expected));
}

static pdhcg_result_t *solve_tiny(qp_problem_t *problem)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 0;
    params.presolve = false;
    params.use_cone_preserving_scaling = false;
    params.termination_evaluation_frequency = 10;
    params.termination_criteria.eps_optimal_relative = 1e-7;
    params.termination_criteria.eps_feasible_relative = 1e-7;
    params.termination_criteria.time_sec_limit = 30.0;
    return solve_qp_problem(problem, &params);
}

static int check_variable_psd(void)
{
    const double sqrt_two = 1.41421356237309504880;
    const double objective[] = {0.0, 0.0, 1.0};
    const int row_ptr[] = {0, 1, 2};
    const int col_ind[] = {0, 1};
    const double values[] = {100.0, 0.01};
    const double rhs[] = {100.0, 0.02 * sqrt_two};
    matrix_desc_t A = {0};
    A.m = 2;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 2;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;
    cone_spec_t cone = {.type = CONE_PSD, .start_idx = 0, .v_dim = 2};

    qp_problem_t *problem =
        create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, NULL, NULL, NULL, 1, &cone, NULL, NULL, 0, NULL);
    pdhcg_result_t *result = problem ? solve_tiny(problem) : NULL;
    int passed = result && result->termination_reason == TERMINATION_REASON_OPTIMAL &&
        close_enough(result->primal_solution[0], 1.0, 2e-4) &&
        close_enough(result->primal_solution[1], 2.0 * sqrt_two, 2e-4) &&
        close_enough(result->primal_solution[2], 4.0, 3e-4) && result->relative_primal_residual < 2e-6 &&
        result->relative_dual_residual < 2e-6;
    if (!passed && result)
    {
        fprintf(stderr,
                "variable PSD: status=%d x=[%.9g %.9g %.9g], pr=%.3e du=%.3e gap=%.3e\n",
                (int)result->termination_reason,
                result->primal_solution[0],
                result->primal_solution[1],
                result->primal_solution[2],
                result->relative_primal_residual,
                result->relative_dual_residual,
                result->relative_objective_gap);
    }
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int check_affine_psd(void)
{
    const double sqrt_two = 1.41421356237309504880;
    const double objective[] = {1.0};
    const int row_ptr[] = {0, 1, 1, 1};
    const int col_ind[] = {0};
    const double values[] = {1.0};
    const double offset[] = {0.0, sqrt_two, 1.0};
    matrix_desc_t F = {0};
    F.m = 3;
    F.n = 1;
    F.fmt = matrix_csr;
    F.data.csr.nnz = 1;
    F.data.csr.row_ptr = row_ptr;
    F.data.csr.col_ind = col_ind;
    F.data.csr.vals = values;
    cone_spec_t cone = {.type = CONE_PSD, .start_idx = 0, .v_dim = 2};

    qp_problem_t *problem = create_qp_problem(
        objective, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL, &F, offset, 1, &cone);
    pdhcg_result_t *result = problem ? solve_tiny(problem) : NULL;
    int passed = result && result->termination_reason == TERMINATION_REASON_OPTIMAL &&
        close_enough(result->primal_solution[0], 1.0, 3e-4) && result->relative_primal_residual < 2e-6 &&
        result->relative_dual_residual < 2e-6;
    if (!passed && result)
    {
        fprintf(stderr,
                "affine PSD: status=%d x=%.9g pr=%.3e du=%.3e gap=%.3e\n",
                (int)result->termination_reason,
                result->primal_solution[0],
                result->relative_primal_residual,
                result->relative_dual_residual,
                result->relative_objective_gap);
    }
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int check_diagonal_q_psd(void)
{
    const double sqrt_two = 1.41421356237309504880;
    const double objective[] = {0.0, 0.0, -3.0};
    const int a_row_ptr[] = {0, 1, 2};
    const int a_col_ind[] = {0, 1};
    const double a_values[] = {1.0, 1.0};
    const double rhs[] = {1.0, 2.0 * sqrt_two};
    const int q_row_ptr[] = {0, 0, 0, 1};
    const int q_col_ind[] = {2};
    const double q_values[] = {1.0};
    matrix_desc_t A = {0};
    matrix_desc_t Q = {0};
    A.m = 2;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 2;
    A.data.csr.row_ptr = a_row_ptr;
    A.data.csr.col_ind = a_col_ind;
    A.data.csr.vals = a_values;
    Q.m = 3;
    Q.n = 3;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = 1;
    Q.data.csr.row_ptr = q_row_ptr;
    Q.data.csr.col_ind = q_col_ind;
    Q.data.csr.vals = q_values;
    cone_spec_t cone = {.type = CONE_PSD, .start_idx = 0, .v_dim = 2};

    qp_problem_t *problem =
        create_qp_problem(objective, &Q, NULL, NULL, &A, rhs, rhs, NULL, NULL, NULL, 1, &cone, NULL, NULL, 0, NULL);
    int uses_diagonal_q =
        problem && detect_q_type(problem->objective_sparse_matrix, NULL, problem->num_variables, 0) == PDHCG_DIAG_Q;
    pdhcg_result_t *result = problem ? solve_tiny(problem) : NULL;
    int passed = uses_diagonal_q && result && result->termination_reason == TERMINATION_REASON_OPTIMAL &&
        close_enough(result->primal_solution[2], 4.0, 5e-4) && result->relative_primal_residual < 3e-6 &&
        result->relative_dual_residual < 3e-6;
    if (!passed && result)
    {
        fprintf(stderr,
                "diagonal-Q PSD: status=%d x=[%.9g %.9g %.9g], pr=%.3e du=%.3e gap=%.3e\n",
                (int)result->termination_reason,
                result->primal_solution[0],
                result->primal_solution[1],
                result->primal_solution[2],
                result->relative_primal_residual,
                result->relative_dual_residual,
                result->relative_objective_gap);
    }
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int check_fixed_psd_rejected(void)
{
    const double objective[] = {0.0, 0.0, 0.0};
    const int row_ptr[] = {0, 0};
    const double rhs[] = {0.0};
    const char fixed[] = {0, 1, 0};
    matrix_desc_t A = {0};
    A.m = 1;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.row_ptr = row_ptr;
    cone_spec_t cone = {
        .type = CONE_PSD,
        .start_idx = 0,
        .v_dim = 2,
        .is_fixed = fixed,
    };
    qp_problem_t *problem =
        create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, NULL, NULL, NULL, 1, &cone, NULL, NULL, 0, NULL);
    int passed = problem == NULL;
    qp_problem_free(problem);

    cone.is_fixed = NULL;
    problem =
        create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, NULL, NULL, NULL, 1, &cone, NULL, NULL, 0, NULL);
    passed &= problem && set_cone_fixed(problem, 0, 1, 0.0) != 0;
    qp_problem_free(problem);
    return passed;
}

int main(void)
{
    int variable = check_variable_psd();
    int affine = check_affine_psd();
    int diagonal_q = check_diagonal_q_psd();
    int fixed_rejected = check_fixed_psd_rejected();
    printf("variable PSD:   %s\n", variable ? "PASS" : "FAIL");
    printf("affine PSD:     %s\n", affine ? "PASS" : "FAIL");
    printf("diagonal-Q PSD: %s\n", diagonal_q ? "PASS" : "FAIL");
    printf("fixed PSD:      %s\n", fixed_rejected ? "REJECTED" : "FAIL");
    return variable && affine && diagonal_q && fixed_rejected ? 0 : 1;
}
