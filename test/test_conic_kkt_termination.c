#include "pdhcg.h"
#include "pdhcg_types.h"

#include <math.h>
#include <stdio.h>

static qp_problem_t *make_empty_cone_problem(cone_type_t type)
{
    static const int row_ptr[] = {0};
    static const double objective[] = {0.0, 0.0, 0.0};
    matrix_desc_t A = {0};
    A.m = 0;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.row_ptr = row_ptr;
    cone_spec_t cone = {
        .type = type,
        .start_idx = 0,
        .v_dim = 1,
        .alpha = type == CONE_POWER ? 0.5 : 0.0,
        .is_fixed = NULL,
    };
    return create_qp_problem(objective, NULL, NULL, NULL, &A, NULL, NULL, NULL, NULL, NULL, 1, &cone);
}

static int rejects_unsupported_fixed_sections(void)
{
    int passed = 1;
    qp_problem_t *problem = make_empty_cone_problem(CONE_STANDARD_SOC);
    if (!problem || set_cone_fixed(problem, 0, 1, 1.0) != 0)
        passed = 0;
    pdhcg_result_t *result = problem ? solve_qp_problem(problem, NULL) : NULL;
    passed &= result == NULL;
    pdhcg_result_free(result);
    qp_problem_free(problem);

    problem = make_empty_cone_problem(CONE_EXPONENTIAL);
    if (!problem || set_cone_fixed(problem, 0, 0, 0.0) != 0)
        passed = 0;
    result = problem ? solve_qp_problem(problem, NULL) : NULL;
    passed &= result == NULL;
    pdhcg_result_free(result);
    qp_problem_free(problem);

    problem = make_empty_cone_problem(CONE_ROTATED_SOC);
    if (!problem || set_cone_fixed(problem, 0, 1, 1.0) != 0)
        passed = 0;
    result = problem ? solve_qp_problem(problem, NULL) : NULL;
    passed &= result == NULL;
    pdhcg_result_free(result);
    qp_problem_free(problem);

    if (!passed)
        fprintf(stderr, "unsupported fixed cone section was not rejected\n");
    return passed;
}

static double initial_soc_dual_residual(double matrix_scale)
{
    static const int row_ptr[] = {0, 1};
    static const int col_ind[] = {3};
    static const double objective[] = {0.0, 0.0, 1.0, 0.0};
    static const double rhs[] = {0.0};
    static const double primal_start[] = {0.0, 0.0, 1.0, 0.0};
    static const double dual_start[] = {0.0};
    static const double var_lb[] = {-INFINITY, -INFINITY, -INFINITY, 0.0};
    static const double var_ub[] = {INFINITY, INFINITY, INFINITY, 0.0};
    const cone_spec_t cone = {
        .type = CONE_STANDARD_SOC,
        .start_idx = 0,
        .v_dim = 1,
        .is_fixed = NULL,
    };
    matrix_desc_t A = {0};
    A.m = 1;
    A.n = 4;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 1;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = &matrix_scale;

    qp_problem_t *problem =
        create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, var_lb, var_ub, NULL, 1, &cone);
    if (!problem)
        return NAN;
    set_start_values(problem, primal_start, dual_start);

    pdhg_parameters_t parameters;
    set_default_parameters(&parameters);
    parameters.verbose = 0;
    parameters.l_inf_ruiz_iterations = 0;
    parameters.has_pock_chambolle_alpha = false;
    parameters.bound_objective_rescaling = false;
    parameters.termination_criteria.iteration_limit = 0;

    pdhcg_result_t *result = solve_qp_problem(problem, &parameters);
    double residual = result ? result->relative_dual_residual : NAN;
    pdhcg_result_free(result);
    qp_problem_free(problem);
    return residual;
}

static int projected_gradient_uses_adaptive_step(void)
{
    double moderate_step_residual = initial_soc_dual_residual(1.0);
    double large_step_residual = initial_soc_dual_residual(1e-3);
    int passed = isfinite(moderate_step_residual) && isfinite(large_step_residual) && moderate_step_residual > 0.1 &&
        large_step_residual < 0.01 * moderate_step_residual;
    if (!passed)
    {
        fprintf(stderr,
                "conic projected-gradient residual did not track the adaptive step: moderate=%.9g large=%.9g\n",
                moderate_step_residual,
                large_step_residual);
    }
    return passed;
}

static int recognizes_soc_with_only_zero_w_fixed_as_optimal(norm_type_t optimality_norm, int v_dim)
{
    static const int row_ptr[] = {0};
    double objective[34] = {0.0};
    double primal_start[34] = {0.0};
    objective[v_dim] = 1.0;

    matrix_desc_t A = {0};
    A.m = 0;
    A.n = v_dim + 2;
    A.fmt = matrix_csr;
    A.data.csr.row_ptr = row_ptr;
    const cone_spec_t cone = {
        .type = CONE_STANDARD_SOC,
        .start_idx = 0,
        .v_dim = v_dim,
        .is_fixed = NULL,
    };

    qp_problem_t *problem = create_qp_problem(objective, NULL, NULL, NULL, &A, NULL, NULL, NULL, NULL, NULL, 1, &cone);
    if (!problem || set_cone_fixed(problem, 0, v_dim, 0.0) != 0)
    {
        qp_problem_free(problem);
        return 0;
    }
    set_start_values(problem, primal_start, NULL);

    pdhg_parameters_t parameters;
    set_default_parameters(&parameters);
    parameters.optimality_norm = optimality_norm;
    parameters.verbose = 0;
    parameters.l_inf_ruiz_iterations = 0;
    parameters.has_pock_chambolle_alpha = false;
    parameters.bound_objective_rescaling = false;
    parameters.termination_evaluation_frequency = 1;
    parameters.termination_criteria.eps_optimal_relative = 1e-8;
    parameters.termination_criteria.eps_feasible_relative = 1e-8;
    parameters.termination_criteria.iteration_limit = 1;

    pdhcg_result_t *result = solve_qp_problem(problem, &parameters);
    int passed = result && result->termination_reason == TERMINATION_REASON_OPTIMAL && result->total_count == 0;
    if (!passed && result)
    {
        fprintf(stderr,
                "SOC with only w=0 fixed was not recognized as optimal: norm=%d v_dim=%d "
                "status=%d iter=%d primal=%.9g dual=%.9g gap=%.9g\n",
                (int)optimality_norm,
                v_dim,
                (int)result->termination_reason,
                result->total_count,
                result->relative_primal_residual,
                result->relative_dual_residual,
                result->relative_objective_gap);
    }

    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int solves_fixed_rsoc_with_large_initial_step(norm_type_t optimality_norm)
{
    /*
     * Fixing s=t=1 reduces this rotated SOC to |x| <= sqrt(2).  The tiny
     * equality coefficient makes the initial primal step very large.  At the
     * feasible interior warm start x=0.5, an adaptive projected-gradient
     * mapping is small but the normal-cone KKT residual must remain nonzero.
     */
    static const int row_ptr[] = {0, 1};
    static const int col_ind[] = {1};
    static const double values[] = {1e-9};
    static const double objective[] = {-1.0, 0.0, 0.0};
    static const double rhs[] = {1e-9};
    static const double primal_start[] = {0.5, 1.0, 1.0};
    static const double dual_start[] = {0.0};
    const cone_spec_t cone = {
        .type = CONE_ROTATED_SOC,
        .start_idx = 0,
        .v_dim = 1,
        .is_fixed = NULL,
    };
    matrix_desc_t A = {0};
    A.m = 1;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 1;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;

    qp_problem_t *problem = create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, NULL, NULL, NULL, 1, &cone);
    if (!problem || set_cone_fixed(problem, 0, 1, 1.0) != 0 || set_cone_fixed(problem, 0, 2, 1.0) != 0)
    {
        qp_problem_free(problem);
        return 0;
    }
    set_start_values(problem, primal_start, dual_start);

    pdhg_parameters_t parameters;
    set_default_parameters(&parameters);
    parameters.optimality_norm = optimality_norm;
    parameters.verbose = 0;
    parameters.l_inf_ruiz_iterations = 0;
    parameters.has_pock_chambolle_alpha = false;
    parameters.bound_objective_rescaling = false;
    parameters.termination_evaluation_frequency = 1;
    parameters.termination_criteria.eps_optimal_relative = 1e-8;
    parameters.termination_criteria.eps_feasible_relative = 1e-8;
    parameters.termination_criteria.iteration_limit = 1;

    pdhcg_result_t *result = solve_qp_problem(problem, &parameters);
    int passed = result && result->termination_reason == TERMINATION_REASON_ITERATION_LIMIT && result->total_count == 1;
    if (!passed && result)
    {
        fprintf(stderr,
                "large-step fixed-RSOC warm start was accepted: status=%d iter=%d x=%.17g "
                "primal=%.9g dual=%.9g gap=%.9g\n",
                (int)result->termination_reason,
                result->total_count,
                result->primal_solution[0],
                result->relative_primal_residual,
                result->relative_dual_residual,
                result->relative_objective_gap);
    }

    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int solves_fixed_soc_with_large_initial_step(norm_type_t optimality_norm)
{
    /* Fix z=1; the free standard-SOC section is v^2 + w^2 <= 1. */
    static const int row_ptr[] = {0, 1};
    static const int col_ind[] = {2};
    static const double values[] = {1e-9};
    static const double objective[] = {-1.0, 0.0, 0.0};
    static const double rhs[] = {1e-9};
    static const double primal_start[] = {0.5, 0.0, 1.0};
    static const double dual_start[] = {0.0};
    const cone_spec_t cone = {
        .type = CONE_STANDARD_SOC,
        .start_idx = 0,
        .v_dim = 1,
        .is_fixed = NULL,
    };
    matrix_desc_t A = {0};
    A.m = 1;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 1;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;

    qp_problem_t *problem = create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, NULL, NULL, NULL, 1, &cone);
    if (!problem || set_cone_fixed(problem, 0, 2, 1.0) != 0)
    {
        qp_problem_free(problem);
        return 0;
    }
    set_start_values(problem, primal_start, dual_start);

    pdhg_parameters_t parameters;
    set_default_parameters(&parameters);
    parameters.optimality_norm = optimality_norm;
    parameters.verbose = 0;
    parameters.l_inf_ruiz_iterations = 0;
    parameters.has_pock_chambolle_alpha = false;
    parameters.bound_objective_rescaling = false;
    parameters.termination_evaluation_frequency = 1;
    parameters.termination_criteria.eps_optimal_relative = 1e-8;
    parameters.termination_criteria.eps_feasible_relative = 1e-8;
    parameters.termination_criteria.iteration_limit = 1;

    pdhcg_result_t *result = solve_qp_problem(problem, &parameters);
    int passed = result && result->termination_reason == TERMINATION_REASON_ITERATION_LIMIT && result->total_count == 1;
    if (!passed && result)
    {
        fprintf(stderr,
                "large-step fixed-SOC warm start was accepted: status=%d iter=%d v=%.17g "
                "primal=%.9g dual=%.9g gap=%.9g\n",
                (int)result->termination_reason,
                result->total_count,
                result->primal_solution[0],
                result->relative_primal_residual,
                result->relative_dual_residual,
                result->relative_objective_gap);
    }

    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int solves_fixed_power_with_large_initial_step(norm_type_t optimality_norm)
{
    /* With x=y=1 and alpha=0.5, the free section is simply |z| <= 1. */
    static const int row_ptr[] = {0, 1};
    static const int col_ind[] = {0};
    static const double values[] = {1e-9};
    static const double objective[] = {0.0, 0.0, -1.0};
    static const double rhs[] = {1e-9};
    static const double primal_start[] = {1.0, 1.0, 0.5};
    static const double dual_start[] = {0.0};
    const cone_spec_t cone = {
        .type = CONE_POWER,
        .start_idx = 0,
        .v_dim = 1,
        .alpha = 0.5,
        .is_fixed = NULL,
    };
    matrix_desc_t A = {0};
    A.m = 1;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 1;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;

    qp_problem_t *problem = create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, NULL, NULL, NULL, 1, &cone);
    if (!problem || set_cone_fixed(problem, 0, 0, 1.0) != 0 || set_cone_fixed(problem, 0, 1, 1.0) != 0)
    {
        qp_problem_free(problem);
        return 0;
    }
    set_start_values(problem, primal_start, dual_start);

    pdhg_parameters_t parameters;
    set_default_parameters(&parameters);
    parameters.optimality_norm = optimality_norm;
    parameters.verbose = 0;
    parameters.l_inf_ruiz_iterations = 0;
    parameters.has_pock_chambolle_alpha = false;
    parameters.bound_objective_rescaling = false;
    parameters.termination_evaluation_frequency = 1;
    parameters.termination_criteria.eps_optimal_relative = 1e-8;
    parameters.termination_criteria.eps_feasible_relative = 1e-8;
    parameters.termination_criteria.iteration_limit = 1;

    pdhcg_result_t *result = solve_qp_problem(problem, &parameters);
    int passed = result && result->termination_reason == TERMINATION_REASON_ITERATION_LIMIT && result->total_count == 1;
    if (!passed && result)
    {
        fprintf(stderr,
                "large-step fixed-power warm start was accepted: status=%d iter=%d z=%.17g "
                "primal=%.9g dual=%.9g gap=%.9g\n",
                (int)result->termination_reason,
                result->total_count,
                result->primal_solution[2],
                result->relative_primal_residual,
                result->relative_dual_residual,
                result->relative_objective_gap);
    }

    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

static int solves_fixed_exp_with_large_initial_step(norm_type_t optimality_norm)
{
    /* Fix y=1 and x=0; minimizing z over z >= exp(x) has solution z=1. */
    static const int row_ptr[] = {0, 1};
    static const int col_ind[] = {0};
    static const double values[] = {1e-10};
    static const double objective[] = {0.0, 0.0, 1.0};
    static const double rhs[] = {0.0};
    static const double primal_start[] = {0.0, 1.0, 2.0};
    static const double dual_start[] = {0.0};
    const cone_spec_t cone = {
        .type = CONE_EXPONENTIAL,
        .start_idx = 0,
        .v_dim = 1,
        .is_fixed = NULL,
    };
    matrix_desc_t A = {0};
    A.m = 1;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 1;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;

    qp_problem_t *problem = create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, NULL, NULL, NULL, 1, &cone);
    if (!problem || set_cone_fixed(problem, 0, 1, 1.0) != 0)
    {
        qp_problem_free(problem);
        return 0;
    }
    set_start_values(problem, primal_start, dual_start);

    pdhg_parameters_t parameters;
    set_default_parameters(&parameters);
    parameters.optimality_norm = optimality_norm;
    parameters.verbose = 0;
    parameters.l_inf_ruiz_iterations = 0;
    parameters.has_pock_chambolle_alpha = false;
    parameters.bound_objective_rescaling = false;
    parameters.termination_evaluation_frequency = 1;
    parameters.termination_criteria.eps_optimal_relative = 1e-8;
    parameters.termination_criteria.eps_feasible_relative = 1e-8;
    parameters.termination_criteria.iteration_limit = 1;

    pdhcg_result_t *result = solve_qp_problem(problem, &parameters);
    int passed = result && result->termination_reason == TERMINATION_REASON_ITERATION_LIMIT && result->total_count == 1;
    if (!passed && result)
    {
        fprintf(stderr,
                "large-step fixed-exp warm start was accepted: status=%d iter=%d z=%.17g "
                "primal=%.9g dual=%.9g gap=%.9g\n",
                (int)result->termination_reason,
                result->total_count,
                result->primal_solution[2],
                result->relative_primal_residual,
                result->relative_dual_residual,
                result->relative_objective_gap);
    }

    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed;
}

int main(void)
{
    /*
     * min -z
     * s.t. t = 2, y = 1 (fixed cone slot), (z, y, t) in K_exp.
     *
     * The warm start (0, 1, 2) is primal feasible and its reduced gradient
     * satisfies the recession-cone sign checks, but it is not stationary.
     * A conic termination test must continue to z = log(2).
     */
    const int row_ptr[] = {0, 1};
    const int col_ind[] = {2};
    const double values[] = {1e-12};
    const double objective[] = {-1.0, 0.0, 0.0};
    const double rhs[] = {2e-12};
    const double primal_start[] = {0.0, 1.0, 2.0};
    const double dual_start[] = {0.0};
    const cone_spec_t cone = {
        .type = CONE_EXPONENTIAL,
        .start_idx = 0,
        .v_dim = 1,
        .is_fixed = NULL,
    };
    matrix_desc_t A = {0};
    A.m = 1;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 1;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;

    qp_problem_t *problem = create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, NULL, NULL, NULL, 1, &cone);
    if (!problem || set_cone_fixed(problem, 0, 1, 1.0) != 0)
    {
        qp_problem_free(problem);
        return 1;
    }
    set_start_values(problem, primal_start, dual_start);

    pdhg_parameters_t parameters;
    set_default_parameters(&parameters);
    parameters.verbose = 0;
    parameters.l_inf_ruiz_iterations = 0;
    parameters.has_pock_chambolle_alpha = false;
    parameters.bound_objective_rescaling = false;
    parameters.termination_criteria.eps_optimal_relative = 1e-8;
    parameters.termination_criteria.eps_feasible_relative = 1e-8;

    pdhcg_result_t *result = solve_qp_problem(problem, &parameters);
    if (!result)
    {
        qp_problem_free(problem);
        return 1;
    }

    double z = result->primal_solution[0];
    int passed = result->termination_reason == TERMINATION_REASON_OPTIMAL && result->total_count > 0 &&
        fabs(z - log(2.0)) <= 1e-7;
    if (!passed)
    {
        fprintf(stderr,
                "conic KKT termination failed: status=%d iter=%d z=%.17g expected=%.17g "
                "primal=%.9g dual=%.9g gap=%.9g\n",
                (int)result->termination_reason,
                result->total_count,
                z,
                log(2.0),
                result->relative_primal_residual,
                result->relative_dual_residual,
                result->relative_objective_gap);
    }
    passed &= projected_gradient_uses_adaptive_step();
    const norm_type_t norms[] = {NORM_TYPE_L_INF, NORM_TYPE_L2};
    for (int norm = 0; norm < 2; ++norm)
    {
        passed &= recognizes_soc_with_only_zero_w_fixed_as_optimal(norms[norm], 1);
        passed &= recognizes_soc_with_only_zero_w_fixed_as_optimal(norms[norm], 32);
        passed &= solves_fixed_rsoc_with_large_initial_step(norms[norm]);
        passed &= solves_fixed_soc_with_large_initial_step(norms[norm]);
        passed &= solves_fixed_power_with_large_initial_step(norms[norm]);
        passed &= solves_fixed_exp_with_large_initial_step(norms[norm]);
    }
    passed &= rejects_unsupported_fixed_sections();

    pdhcg_result_free(result);
    qp_problem_free(problem);
    return passed ? 0 : 1;
}
