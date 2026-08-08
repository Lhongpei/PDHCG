#include "pdhcg.h"
#include "pdhcg_types.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static qp_problem_t *make_one_variable_problem(double objective,
                                               const matrix_desc_t *A,
                                               const double *offset,
                                               const cone_spec_t *affine_cone)
{
    static const int q_row_ptr[] = {0, 0};
    matrix_desc_t Q;
    memset(&Q, 0, sizeof(Q));
    Q.m = 1;
    Q.n = 1;
    Q.fmt = matrix_csr;
    Q.data.csr.row_ptr = q_row_ptr;
    int num_affine_cones = affine_cone ? 1 : 0;
    return create_qp_problem(
        &objective, &Q, NULL, NULL, A, NULL, NULL, NULL, NULL, NULL, 0, NULL, num_affine_cones, affine_cone, offset);
}

static qp_problem_t *make_one_variable_quadratic_problem(
    double objective, double quadratic, const matrix_desc_t *A, const double *offset, const cone_spec_t *affine_cone)
{
    static const int q_row_ptr[] = {0, 1};
    static const int q_col_ind[] = {0};
    matrix_desc_t Q;
    memset(&Q, 0, sizeof(Q));
    Q.m = 1;
    Q.n = 1;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = 1;
    Q.data.csr.row_ptr = q_row_ptr;
    Q.data.csr.col_ind = q_col_ind;
    Q.data.csr.vals = &quadratic;
    int num_affine_cones = affine_cone ? 1 : 0;
    return create_qp_problem(
        &objective, &Q, NULL, NULL, A, NULL, NULL, NULL, NULL, NULL, 0, NULL, num_affine_cones, affine_cone, offset);
}

static pdhcg_result_t *solve_tiny(qp_problem_t *problem, norm_type_t norm, bool use_cone_preserving_scaling)
{
    pdhg_parameters_t parameters;
    set_default_parameters(&parameters);
    parameters.verbose = 0;
    parameters.presolve = false;
    parameters.optimality_norm = norm;
    parameters.use_cone_preserving_scaling = use_cone_preserving_scaling;
    parameters.termination_evaluation_frequency = 10;
    parameters.termination_criteria.eps_optimal_relative = 1e-7;
    parameters.termination_criteria.eps_feasible_relative = 1e-7;
    parameters.termination_criteria.iteration_limit = 1000000;
    parameters.termination_criteria.time_sec_limit = 30.0;
    return solve_qp_problem(problem, &parameters);
}

static int check_result(const char *name, pdhcg_result_t *result, double expected)
{
    if (!result || result->termination_reason != TERMINATION_REASON_OPTIMAL)
    {
        fprintf(stderr, "%s: expected OPTIMAL, got %d\n", name, result ? (int)result->termination_reason : -1);
        return 0;
    }
    double error = fabs(result->primal_solution[0] - expected);
    double objective_gap = fabs(result->primal_objective_value - result->dual_objective_value);
    if (error > 2e-4 * (1.0 + fabs(expected)) || result->relative_primal_residual > 2e-6 ||
        result->relative_dual_residual > 2e-6 || objective_gap > 2e-4 * (1.0 + fabs(expected)))
    {
        fprintf(stderr,
                "%s: x=%.17g expected=%.17g, rel_pr=%.3e rel_du=%.3e gap=%.3e\n",
                name,
                result->primal_solution[0],
                expected,
                result->relative_primal_residual,
                result->relative_dual_residual,
                objective_gap);
        return 0;
    }
    return 1;
}

static int run_soc(void)
{
    static const int row_ptr[] = {0, 1, 1, 2};
    static const int col_ind[] = {0, 0};
    static const double values[] = {10.0, 1.0};
    static const double offset[] = {0.0, 0.0, 9.0};
    matrix_desc_t F;
    memset(&F, 0, sizeof(F));
    F.m = 3;
    F.n = 1;
    F.fmt = matrix_csr;
    F.data.csr.nnz = 2;
    F.data.csr.row_ptr = row_ptr;
    F.data.csr.col_ind = col_ind;
    F.data.csr.vals = values;
    cone_spec_t cone = {.type = CONE_STANDARD_SOC, .start_idx = 0, .v_dim = 1};

    qp_problem_t *problem = make_one_variable_quadratic_problem(-2.0, 1.0, &F, offset, &cone);
    int setup_ok = problem != NULL;
    pdhcg_result_t *result = setup_ok ? solve_tiny(problem, NORM_TYPE_L_INF, false) : NULL;
    int passed = setup_ok && check_result("affine SOC", result, 1.0);
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int run_exp(void)
{
    static const int row_ptr[] = {0, 1, 1, 1};
    static const int col_ind[] = {0};
    static const double values[] = {1.0};
    static const double offset[] = {0.0, 1.0, 2.0};
    matrix_desc_t F;
    memset(&F, 0, sizeof(F));
    F.m = 3;
    F.n = 1;
    F.fmt = matrix_csr;
    F.data.csr.nnz = 1;
    F.data.csr.row_ptr = row_ptr;
    F.data.csr.col_ind = col_ind;
    F.data.csr.vals = values;
    cone_spec_t cone = {.type = CONE_EXPONENTIAL, .start_idx = 0, .v_dim = 1};

    qp_problem_t *problem = make_one_variable_problem(-1.0, &F, offset, &cone);
    int setup_ok = problem != NULL;
    pdhcg_result_t *result = setup_ok ? solve_tiny(problem, NORM_TYPE_L_INF, true) : NULL;
    int passed = setup_ok && check_result("affine exponential cone", result, log(2.0));
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int run_power(void)
{
    static const int row_ptr[] = {0, 0, 0, 1};
    static const int col_ind[] = {0};
    static const double values[] = {1.0};
    static const double offset[] = {1.0, 1.0, 0.0};
    matrix_desc_t F;
    memset(&F, 0, sizeof(F));
    F.m = 3;
    F.n = 1;
    F.fmt = matrix_csr;
    F.data.csr.nnz = 1;
    F.data.csr.row_ptr = row_ptr;
    F.data.csr.col_ind = col_ind;
    F.data.csr.vals = values;
    cone_spec_t cone = {.type = CONE_POWER, .start_idx = 0, .v_dim = 1, .power_alpha = 0.3};

    qp_problem_t *problem = make_one_variable_problem(-1.0, &F, offset, &cone);
    int setup_ok = problem != NULL;
    pdhcg_result_t *result = setup_ok ? solve_tiny(problem, NORM_TYPE_L2, true) : NULL;
    int passed = setup_ok && check_result("affine power cone", result, 1.0);
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int run_rsoc(void)
{
    static const int row_ptr[] = {0, 1, 1, 1};
    static const int col_ind[] = {0};
    static const double values[] = {1.0};
    static const double offset[] = {0.0, 1.0, 1.0};
    matrix_desc_t F;
    memset(&F, 0, sizeof(F));
    F.m = 3;
    F.n = 1;
    F.fmt = matrix_csr;
    F.data.csr.nnz = 1;
    F.data.csr.row_ptr = row_ptr;
    F.data.csr.col_ind = col_ind;
    F.data.csr.vals = values;
    cone_spec_t cone = {.type = CONE_ROTATED_SOC, .start_idx = 0, .v_dim = 1};

    qp_problem_t *problem = make_one_variable_problem(-1.0, &F, offset, &cone);
    int setup_ok = problem != NULL;
    pdhcg_result_t *result = setup_ok ? solve_tiny(problem, NORM_TYPE_L2, true) : NULL;
    int passed = setup_ok && check_result("affine rotated SOC", result, sqrt(2.0));
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int run_scalar_bound_infeasible(void)
{
    static const int row_ptr[] = {0, 0};
    static const double objective[] = {0.0};
    static const double equality[] = {-1.0};
    matrix_desc_t A;
    memset(&A, 0, sizeof(A));
    A.m = 1;
    A.n = 1;
    A.fmt = matrix_csr;
    A.data.csr.row_ptr = row_ptr;

    qp_problem_t *problem = create_qp_problem(
        objective, NULL, NULL, NULL, &A, equality, equality, NULL, NULL, NULL, 0, NULL, 0, NULL, NULL);
    if (!problem)
        return 0;

    pdhcg_result_t *result = solve_tiny(problem, NORM_TYPE_L_INF, true);
    int passed = result && result->termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE;
    if (!passed)
    {
        fprintf(stderr,
                "scalar bound: expected PRIMAL_INFEASIBLE, got %d\n",
                result ? (int)result->termination_reason : -1);
    }
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int run_reject_fixed_slots(void)
{
    static const int row_ptr[] = {0, 0, 0, 0};
    static const char is_fixed[] = {1, 0, 0};
    matrix_desc_t F;
    memset(&F, 0, sizeof(F));
    F.m = 3;
    F.n = 1;
    F.fmt = matrix_csr;
    F.data.csr.row_ptr = row_ptr;
    cone_spec_t cone = {
        .type = CONE_STANDARD_SOC,
        .start_idx = 0,
        .v_dim = 1,
        .is_fixed = is_fixed,
    };

    qp_problem_t *problem = make_one_variable_problem(0.0, &F, NULL, &cone);
    int rejected = problem == NULL;
    qp_problem_free(problem);
    return rejected;
}

int main(void)
{
    int soc = run_soc();
    int rsoc = run_rsoc();
    int exp = run_exp();
    int power = run_power();
    int scalar_bound = run_scalar_bound_infeasible();
    int fixed_slots = run_reject_fixed_slots();
    printf("affine SOC: %s\n", soc ? "PASS" : "FAIL");
    printf("affine RSOC: %s\n", rsoc ? "PASS" : "FAIL");
    printf("affine Exp: %s\n", exp ? "PASS" : "FAIL");
    printf("affine Power: %s\n", power ? "PASS" : "FAIL");
    printf("scalar bound infeasibility: %s\n", scalar_bound ? "PASS" : "FAIL");
    printf("affine fixed-slot rejection: %s\n", fixed_slots ? "PASS" : "FAIL");
    return (soc && rsoc && exp && power && scalar_bound && fixed_slots) ? 0 : 1;
}
