/*
Copyright 2025 Haihao Lu
Copyright 2026 Hongpei Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "internal_types.h"
#include "pdhcg.h"
#include "pdhg_core_op.h"
#include "preconditioner.h"
#include "presolve_wrapper.h"
#include "solver.h"
#include "solver_state.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

pdhcg_result_t *optimize(const pdhg_parameters_t *input_params, const qp_problem_t *original_problem)
{
    pdhg_parameters_t copyed_params = *input_params;
    pdhg_parameters_t *params = &copyed_params;

    print_initial_info(input_params, original_problem);

    qp_problem_t *transformed = NULL;
    if (original_problem->num_quadratic_constraints > 0)
    {
        transformed = qcqp_to_socp_qp(original_problem);
        if (!transformed)
        {
            fprintf(stderr, "Error: QCQP -> SOCP transformation failed; cannot solve.\n");
            return NULL;
        }
        if (params->verbose >= 1)
        {
            fprintf(stderr,
                    "[QCQP] %d quadratic constraint(s) reformulated as %d "
                    "rotated SOC block(s); extended problem: %d vars, "
                    "%d rows, %d nnz.\n",
                    original_problem->num_quadratic_constraints,
                    transformed->num_cone_blocks,
                    transformed->num_variables,
                    transformed->num_constraints,
                    transformed->constraint_matrix_num_nonzeros);
        }
        original_problem = transformed;
    }

    pdhcg_presolve_info_t *presolve_info = NULL;
    const qp_problem_t *working_problem = original_problem;
    bool working_problem_needs_free = false;

    if (params->presolve && pdhcg_presolve_available())
    {
        presolve_info = pdhcg_presolve(original_problem, params);
        if (presolve_info)
        {
            if (presolve_info->problem_solved_during_presolve)
            {
                pdhcg_result_t *result = pdhcg_create_result_from_presolve(presolve_info, original_problem);
                if (result)
                {
                    pdhg_final_log(result, params);
                }
                pdhcg_presolve_info_free(presolve_info);
                return result;
            }

            if (presolve_info->reduced_problem)
            {
                working_problem = presolve_info->reduced_problem;
            }
        }
    }

    if (working_problem->num_constraints == 0 || working_problem->constraint_matrix == NULL)
    {
        working_problem = create_problem_with_dummy_constraint(original_problem);
        working_problem_needs_free = true;
    }

    rescale_info_t *rescale_info = rescale_problem(params, working_problem);
    grid_context_t *grid_context = NULL;
    pdhg_solver_state_t *state = initialize_solver_state(params, working_problem, rescale_info, grid_context);

    if (state->quadratic_objective_term->nonconvexity < 0)
    {
        state->inner_solver->iteration_limit = 1;
    }

    rescale_info_free(rescale_info);
    initialize_step_size_and_primal_weight(state, params);
    clock_t start_time = clock();
    bool do_restart = false;

    while (state->total_count < params->termination_criteria.iteration_limit)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_residual(state, params->optimality_norm);
            if (state->is_this_major_iteration && state->total_count < 3 * params->termination_evaluation_frequency)
            {
                compute_infeasibility_information(state);
            }

            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;

            check_termination_criteria(state, &params->termination_criteria);
            display_iteration_stats(state, params->verbose);
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
            {
                break;
            }
        }

        if ((state->is_this_major_iteration || state->total_count == 0))
        {
            do_restart =
                should_do_adaptive_restart(state, &params->restart_params, params->termination_evaluation_frequency);
            if (do_restart)
                perform_restart(state, params);
        }

        state->is_this_major_iteration = ((state->total_count + 1) % params->termination_evaluation_frequency) == 0;

        pdhg_update(state);

        if (state->is_this_major_iteration || do_restart)
        {
            compute_fixed_point_error(state);
            if (do_restart)
            {
                state->initial_fixed_point_error = state->fixed_point_error;
                do_restart = false;
            }
        }
        halpern_update(state, params->reflection_coefficient);

        state->inner_count++;
        state->total_count++;
    }

    if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_residual(state, params->optimality_norm);
        display_iteration_stats(state, params->verbose);
    }

    // if (params->feasibility_polishing &&
    //     state->termination_reason != TERMINATION_REASON_DUAL_INFEASIBLE &&
    //     state->termination_reason != TERMINATION_REASON_PRIMAL_INFEASIBLE) {
    //   feasibility_polish(params, state);
    // }

    pdhcg_result_t *result = create_result_from_state(state, original_problem);

    if (presolve_info && presolve_info->reduced_problem)
    {
        pdhcg_postsolve(presolve_info, result, original_problem);
    }
    if (working_problem_needs_free)
    {
        qp_problem_free((qp_problem_t *)working_problem);
    }

    if (transformed && result && result->primal_solution)
    {
        int n_orig = transformed->num_original_variables;
        int m_ext = transformed->num_constraints;
        int m_orig = m_ext - (transformed->num_variables - n_orig - 2 * transformed->num_cone_blocks);

        if (n_orig > 0 && n_orig < result->num_variables)
        {
            double *new_primal = (double *)safe_malloc((size_t)n_orig * sizeof(double));
            memcpy(new_primal, result->primal_solution, (size_t)n_orig * sizeof(double));
            free(result->primal_solution);
            result->primal_solution = new_primal;
            result->num_variables = n_orig;

            if (result->reduced_cost)
            {
                double *new_rc = (double *)safe_malloc((size_t)n_orig * sizeof(double));
                memcpy(new_rc, result->reduced_cost, (size_t)n_orig * sizeof(double));
                free(result->reduced_cost);
                result->reduced_cost = new_rc;
            }
        }
        if (m_orig > 0 && m_orig < result->num_constraints && result->dual_solution)
        {
            double *new_dual = (double *)safe_malloc((size_t)m_orig * sizeof(double));
            memcpy(new_dual, result->dual_solution, (size_t)m_orig * sizeof(double));
            free(result->dual_solution);
            result->dual_solution = new_dual;
            result->num_constraints = m_orig;
        }
    }

    pdhg_final_log(result, params);
    pdhg_solver_state_free(state);
    pdhcg_presolve_info_free(presolve_info);
    if (transformed)
    {
        qp_problem_free(transformed);
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
