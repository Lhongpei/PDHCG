#include "pdhcg.h"
#include "pdhcg_types.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static qp_problem_t *make_unconstrained_power_problem(double alpha, const double objective[3])
{
    static const int row_ptr[] = {0};
    matrix_desc_t A;
    memset(&A, 0, sizeof(A));
    A.m = 0;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.row_ptr = row_ptr;

    cone_spec_t cone;
    memset(&cone, 0, sizeof(cone));
    cone.type = CONE_POWER;
    cone.start_idx = 0;
    cone.v_dim = 1;
    cone.power_alpha = alpha;
    return create_qp_problem(objective, NULL, NULL, NULL, &A, NULL, NULL, NULL, NULL, NULL, 1, &cone, 0, NULL, NULL);
}

static qp_problem_t *make_quadratic_power_problem(double alpha, const double center[3], const double weights[3])
{
    static const int empty_row_ptr[] = {0};
    static const int q_row_ptr[] = {0, 1, 2, 3};
    static const int q_col_ind[] = {0, 1, 2};
    double objective[3];
    for (int i = 0; i < 3; ++i)
        objective[i] = -weights[i] * center[i];

    matrix_desc_t A;
    memset(&A, 0, sizeof(A));
    A.m = 0;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.row_ptr = empty_row_ptr;

    matrix_desc_t Q;
    memset(&Q, 0, sizeof(Q));
    Q.m = 3;
    Q.n = 3;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = 3;
    Q.data.csr.row_ptr = q_row_ptr;
    Q.data.csr.col_ind = q_col_ind;
    Q.data.csr.vals = weights;

    cone_spec_t cone;
    memset(&cone, 0, sizeof(cone));
    cone.type = CONE_POWER;
    cone.start_idx = 0;
    cone.v_dim = 1;
    cone.power_alpha = alpha;
    return create_qp_problem(objective, &Q, NULL, NULL, &A, NULL, NULL, NULL, NULL, NULL, 1, &cone, 0, NULL, NULL);
}

static pdhcg_result_t *solve_tiny_with_norm(qp_problem_t *problem, norm_type_t optimality_norm)
{
    pdhg_parameters_t parameters;
    set_default_parameters(&parameters);
    parameters.verbose = 0;
    parameters.presolve = false;
    parameters.optimality_norm = optimality_norm;
    parameters.termination_evaluation_frequency = 10;
    parameters.termination_criteria.eps_optimal_relative = 1e-8;
    parameters.termination_criteria.eps_feasible_relative = 1e-8;
    parameters.termination_criteria.iteration_limit = 1000000;
    parameters.termination_criteria.time_sec_limit = 30.0;
    return solve_qp_problem(problem, &parameters);
}

static pdhcg_result_t *solve_tiny(qp_problem_t *problem)
{
    return solve_tiny_with_norm(problem, NORM_TYPE_L_INF);
}

static int check_solution(const char *name, pdhcg_result_t *result, const double expected[3], double tolerance)
{
    if (!result || result->termination_reason != TERMINATION_REASON_OPTIMAL)
    {
        fprintf(stderr, "%s: expected OPTIMAL, got %d\n", name, result ? (int)result->termination_reason : -1);
        return 0;
    }
    for (int i = 0; i < 3; ++i)
    {
        double error = fabs(result->primal_solution[i] - expected[i]);
        if (error > tolerance * (1.0 + fabs(expected[i])))
        {
            fprintf(stderr,
                    "%s: coordinate %d is %.17g, expected %.17g (error %.3e)\n",
                    name,
                    i,
                    result->primal_solution[i],
                    expected[i],
                    error);
            return 0;
        }
    }
    return 1;
}

static int run_full_cone_case(double alpha, norm_type_t optimality_norm)
{
    static const int row_ptr[] = {0, 2};
    static const int col_ind[] = {0, 1};
    static const double values[] = {1.0, 1.0};
    static const double rhs[] = {2.0};
    static const double objective[] = {0.0, 0.0, -1.0};
    const double primal_start[] = {1.0, 1.0, 0.0};
    const double dual_start[] = {0.0};

    matrix_desc_t A;
    memset(&A, 0, sizeof(A));
    A.m = 1;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 2;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;

    cone_spec_t cone;
    memset(&cone, 0, sizeof(cone));
    cone.type = CONE_POWER;
    cone.start_idx = 0;
    cone.v_dim = 1;
    cone.power_alpha = alpha;
    qp_problem_t *problem =
        create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, NULL, NULL, NULL, 1, &cone, 0, NULL, NULL);
    if (!problem)
        return 0;
    set_start_values(problem, primal_start, dual_start);

    pdhcg_result_t *result = solve_tiny_with_norm(problem, optimality_norm);
    double om = 1.0 - alpha;
    double expected[3] = {2.0 * alpha, 2.0 * om, pow(2.0 * alpha, alpha) * pow(2.0 * om, om)};
    int passed = check_solution("full power cone", result, expected, 2e-5);
    if (result && result->total_count == 0)
    {
        fprintf(stderr, "full power cone: nonoptimal feasible warm start was accepted at iteration zero\n");
        passed = 0;
    }
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int run_fixed_case(const char *name,
                          double alpha,
                          const double objective[3],
                          const char fixed[3],
                          const double fixed_value[3],
                          const double expected[3])
{
    qp_problem_t *problem = make_unconstrained_power_problem(alpha, objective);
    if (!problem)
        return 0;
    for (int slot = 0; slot < 3; ++slot)
    {
        if (fixed[slot] && set_cone_fixed(problem, 0, slot, fixed_value[slot]) != 0)
        {
            qp_problem_free(problem);
            return 0;
        }
    }

    pdhcg_result_t *result = solve_tiny(problem);
    int passed = check_solution(name, result, expected, 3e-5);
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int run_infeasible_fixed_case(void)
{
    static const double objective[] = {0.0, 0.0, 0.0};
    qp_problem_t *problem = make_unconstrained_power_problem(0.3, objective);
    if (!problem)
        return 0;
    int setup_ok = set_cone_fixed(problem, 0, 0, 1.0) == 0 && set_cone_fixed(problem, 0, 1, 1.0) == 0 &&
        set_cone_fixed(problem, 0, 2, 2.0) == 0;
    pdhcg_result_t *result = setup_ok ? solve_tiny(problem) : NULL;
    int passed = setup_ok && result == NULL;
    if (!passed)
        fprintf(stderr, "infeasible fully fixed power cone was not rejected\n");
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int run_diagonal_q_cases(void)
{
    int passed = 1;
    {
        const double center[3] = {1.5, -0.4, -2.2};
        const double weights[3] = {0.3, 2.0, 5.0};
        const double expected[3] = {3.427365210244501, 0.4816579589566281, -1.902365192823554};
        qp_problem_t *problem = make_quadratic_power_problem(0.7, center, weights);
        pdhcg_result_t *result = problem ? solve_tiny(problem) : NULL;
        passed &= check_solution("diagonal Q full power cone", result, expected, 2e-4);
        pdhcg_result_free(result);
        qp_problem_free(problem);
    }
    {
        const double center[3] = {-0.5, 0.2, 1.0};
        const double weights[3] = {2.0, 0.7, 1.0};
        const double expected[3] = {0.3619815616314009, 1.545732644202259, 1.0};
        qp_problem_t *problem = make_quadratic_power_problem(0.3, center, weights);
        if (problem && set_cone_fixed(problem, 0, 2, 1.0) != 0)
        {
            qp_problem_free(problem);
            problem = NULL;
        }
        pdhcg_result_t *result = problem ? solve_tiny(problem) : NULL;
        passed &= check_solution("diagonal Q fixed-z power cone", result, expected, 2e-4);
        pdhcg_result_free(result);
        qp_problem_free(problem);
    }
    return passed;
}

static int run_sharp_fixed_axis_case(void)
{
    const double alpha = 0.97;
    const double center[3] = {0.62147354, -0.01546521, 0.19700471};
    const double weights[3] = {0.131333592, 0.0263303077, 38.3559275};
    const double start[3] = {0.62720941, -0.01546521, 0.19700471};
    qp_problem_t *problem = make_quadratic_power_problem(alpha, center, weights);
    if (!problem || set_cone_fixed(problem, 0, 0, start[0]) != 0)
    {
        qp_problem_free(problem);
        return 0;
    }
    set_start_values(problem, start, NULL);

    pdhcg_result_t *result = solve_tiny(problem);
    int passed = result && result->termination_reason == TERMINATION_REASON_OPTIMAL;
    if (result)
    {
        double x = result->primal_solution[0];
        double y = result->primal_solution[1];
        double z = result->primal_solution[2];
        double bound = x > 0.0 && y > 0.0 ? pow(x, alpha) * pow(y, 1.0 - alpha) : 0.0;
        double violation = fmax(0.0, fmax(-x, fmax(-y, fabs(z) - bound)));
        passed &= x == start[0] && violation <= 1e-12 && result->relative_primal_residual <= 1e-8;
        if (!passed)
            fprintf(stderr,
                    "sharp fixed-axis power cone failed: status=%d x=%.17g y=%.17g z=%.17g violation=%.3e "
                    "primal=%.3e dual=%.3e\n",
                    (int)result->termination_reason,
                    x,
                    y,
                    z,
                    violation,
                    result->relative_primal_residual,
                    result->relative_dual_residual);
    }
    else
    {
        fprintf(stderr, "sharp fixed-axis power cone returned NULL\n");
    }
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

int main(void)
{
    int passed = 1;
    passed &= run_full_cone_case(0.2, NORM_TYPE_L_INF);
    passed &= run_full_cone_case(0.5, NORM_TYPE_L_INF);
    passed &= run_full_cone_case(0.8, NORM_TYPE_L_INF);
    passed &= run_full_cone_case(0.5, NORM_TYPE_L2);

    {
        const double alpha = 0.3;
        const double om = 1.0 - alpha;
        const double lambda = 1.0 / (pow(alpha, alpha) * pow(om, om));
        const double objective[3] = {1.0, 1.0, 0.0};
        const char fixed[3] = {0, 0, 1};
        const double values[3] = {0.0, 0.0, 1.0};
        const double expected[3] = {alpha * lambda, om * lambda, 1.0};
        passed &= run_fixed_case("fixed z", alpha, objective, fixed, values, expected);
    }
    {
        const double alpha = 0.3;
        const double objective[3] = {0.0, 1.0 - alpha, -1.0};
        const char fixed[3] = {1, 0, 0};
        const double values[3] = {1.0, 0.0, 0.0};
        const double expected[3] = {1.0, 1.0, 1.0};
        passed &= run_fixed_case("fixed x", alpha, objective, fixed, values, expected);
    }
    {
        const double alpha = 0.7;
        const double objective[3] = {alpha, 0.0, -1.0};
        const char fixed[3] = {0, 1, 0};
        const double values[3] = {0.0, 1.0, 0.0};
        const double expected[3] = {1.0, 1.0, 1.0};
        passed &= run_fixed_case("fixed y", alpha, objective, fixed, values, expected);
    }
    {
        const double objective[3] = {0.0, 1.0, 0.0};
        const char fixed[3] = {1, 0, 1};
        const double values[3] = {1.0, 0.0, 1.0};
        const double expected[3] = {1.0, 1.0, 1.0};
        passed &= run_fixed_case("fixed x,z", 0.3, objective, fixed, values, expected);
    }
    {
        const double objective[3] = {1.0, 0.0, 0.0};
        const char fixed[3] = {0, 1, 1};
        const double values[3] = {0.0, 1.0, 1.0};
        const double expected[3] = {1.0, 1.0, 1.0};
        passed &= run_fixed_case("fixed y,z", 0.7, objective, fixed, values, expected);
    }
    {
        const double alpha = 0.3;
        const double objective[3] = {0.0, 0.0, -1.0};
        const char fixed[3] = {1, 1, 0};
        const double values[3] = {2.0, 3.0, 0.0};
        const double expected[3] = {2.0, 3.0, pow(2.0, alpha) * pow(3.0, 1.0 - alpha)};
        passed &= run_fixed_case("fixed x,y", alpha, objective, fixed, values, expected);
    }
    {
        const double objective[3] = {1.0, 1.0, 0.0};
        const char fixed[3] = {0, 0, 1};
        const double values[3] = {0.0, 0.0, 0.0};
        const double expected[3] = {0.0, 0.0, 0.0};
        passed &= run_fixed_case("fixed zero z", 0.3, objective, fixed, values, expected);
    }
    {
        const double alpha = 0.86039292839558623;
        const double objective[3] = {0.0, 0.0, 0.0};
        const char fixed[3] = {1, 1, 1};
        const double values[3] = {
            4.4414605580442319e83,
            2.0775280372919238e-95,
            5.6415660092721006e58,
        };
        passed &= run_fixed_case("fully fixed roundoff boundary", alpha, objective, fixed, values, values);
    }
    passed &= run_infeasible_fixed_case();
    passed &= run_diagonal_q_cases();
    passed &= run_sharp_fixed_axis_case();
    return passed ? 0 : 1;
}
