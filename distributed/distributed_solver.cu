/*
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
#include "distributed_interface.h"
#include "distributed_solver.h"
#include "distributed_types.h"
#include "distributed_utils.h"
#include "internal_types.h"
#include "pdhcg.h"
#include "pdhcg_kernels.cuh"
#include "pdhg_core_op.h"
#include "permute.h"
#include "preconditioner.h"
#include "presolve_wrapper.h"
#include "qcqp_transform.h"
#include "solver.h"
#include "solver_state.h"
#include "spmv_backend.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void get_best_grid_dims(int m, int n, int n_procs, int *out_r, int *out_c)
{
    int best_r = 1;
    int best_c = n_procs;
    double best_score = DBL_MAX;

    if (n == 0)
    {
        *out_r = best_r;
        *out_c = best_c;
        return;
    }

    double target_ratio = (double)m / (double)n;

    for (int r = 1; r <= n_procs; ++r)
    {
        if (n_procs % r == 0)
        {
            int c = n_procs / r;
            double grid_ratio = (double)r / (double)c;
            double score = fabs(log(grid_ratio) - log(target_ratio));

            if (score < best_score)
            {
                best_score = score;
                best_r = r;
                best_c = c;
            }
        }
    }

    *out_r = best_r;
    *out_c = best_c;
}

static void select_valid_grid_size(const pdhg_parameters_t *params,
                                   const qp_problem_t *original_problem,
                                   pdhg_parameters_t *sub_params)
{
    int world_size, rank_global;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_global);

    if (params->grid_size.decided)
    {
        int provided_rows = params->grid_size.row_dims;
        int provided_cols = params->grid_size.col_dims;
        int product = provided_rows * provided_cols;

        if (product != world_size)
        {
            if (rank_global == 0)
            {
                fprintf(stderr, "\n[Error] MPI World Size Mismatch!\n");
                fprintf(stderr, "------------------------------------------------\n");
                fprintf(
                    stderr, "User specified grid:  %d x %d = %d processes\n", provided_rows, provided_cols, product);
                fprintf(stderr, "Actual MPI world size: %d processes\n", world_size);
                fprintf(stderr, "Please adjust -n (mpirun) or --n_row_tiles/--n_col_tiles.\n");
                fprintf(stderr, "------------------------------------------------\n");
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        sub_params->grid_size.row_dims = provided_rows;
        sub_params->grid_size.col_dims = provided_cols;
    }
    else
    {
        int dims[2];
        if (rank_global == 0)
        {
            get_best_grid_dims(
                original_problem->num_constraints, original_problem->num_variables, world_size, &dims[0], &dims[1]);

            if (params->verbose)
            {
                printf("[Auto-Grid] Decided grid shape: %d x %d for %d processes.\n", dims[0], dims[1], world_size);
            }
        }
        MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);

        sub_params->grid_size.row_dims = dims[0];
        sub_params->grid_size.col_dims = dims[1];
        sub_params->grid_size.decided = 1;
    }
}

static void allreduce_obj_bound_norm(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        pdhcg_all_reduce_scalar(
            state->grid_context, &state->objective_vector_norm, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);
    }
    else
    {
        double local_sq = state->objective_vector_norm * state->objective_vector_norm;
        pdhcg_all_reduce_scalar(state->grid_context, &local_sq, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        state->objective_vector_norm = sqrt(local_sq);
    }

    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        pdhcg_all_reduce_scalar(
            state->grid_context, &state->constraint_bound_norm, PDHCG_OP_MAX, PDHCG_SCOPE_COL, false);
    }
    else
    {
        double local_sq = state->constraint_bound_norm * state->constraint_bound_norm;
        pdhcg_all_reduce_scalar(state->grid_context, &local_sq, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);
        state->constraint_bound_norm = sqrt(local_sq);
    }
}

pdhcg_result_t *create_result_from_state_distributed(pdhg_solver_state_t *state, const qp_problem_t *original_problem)
{
    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_At,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->pdhg_dual_solution,
                       state->dual_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->dual_product, state->num_variables, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);

    update_obj_product(state, state->pdhg_primal_solution);

    if (state->problem_type == LP)
    {
        compute_and_rescale_reduced_cost_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->dual_slack,
            state->objective_vector,
            state->dual_product,
            state->variable_rescaling,
            state->objective_vector_rescaling,
            state->constraint_bound_rescaling,
            state->num_variables);
    }
    else
    {
        compute_and_rescale_reduced_cost_qp_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->dual_slack,
            state->objective_vector,
            state->quadratic_objective_term->primal_obj_product,
            state->dual_product,
            state->variable_rescaling,
            state->objective_vector_rescaling,
            state->constraint_bound_rescaling,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables);
    }

    rescale_solution(state);

    pdhcg_result_t *results = NULL;
    if (state->grid_context->rank_global == 0)
    {
        results = (pdhcg_result_t *)safe_calloc(1, sizeof(pdhcg_result_t));
    }

    double *global_primal = NULL;
    double *global_dual = NULL;
    double *global_reduced_cost = NULL;

    gather_distributed_vector(state->pdhg_primal_solution,
                              state->num_variables,
                              state->grid_context->comm_col,
                              state->grid_context->comm_row,
                              &global_primal);

    gather_distributed_vector(state->dual_slack,
                              state->num_variables,
                              state->grid_context->comm_col,
                              state->grid_context->comm_row,
                              &global_reduced_cost);

    gather_distributed_vector(state->pdhg_dual_solution,
                              state->num_constraints,
                              state->grid_context->comm_row,
                              state->grid_context->comm_col,
                              &global_dual);

    if (state->grid_context->rank_global == 0)
    {
        if (!global_primal || !global_dual)
        {
            fprintf(stderr, "Error: Failed to gather solution to root.\n");
        }

        results->primal_solution = global_primal;
        results->dual_solution = global_dual;
        results->reduced_cost = global_reduced_cost;

        if (original_problem)
        {
            results->num_variables = original_problem->num_variables;
            results->num_constraints = original_problem->num_constraints;
            results->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;
        }
        results->total_count = state->total_count;
        results->rescaling_time_sec = state->rescaling_time_sec;
        results->cumulative_time_sec = state->cumulative_time_sec;
        results->relative_primal_residual = state->relative_primal_residual;
        results->relative_dual_residual = state->relative_dual_residual;
        results->absolute_primal_residual = state->absolute_primal_residual;
        results->absolute_dual_residual = state->absolute_dual_residual;
        results->primal_objective_value = state->primal_objective_value;
        results->dual_objective_value = state->dual_objective_value;
        results->objective_gap = state->objective_gap;
        results->relative_objective_gap = state->relative_objective_gap;
        results->max_primal_ray_infeasibility = state->max_primal_ray_infeasibility;
        results->max_dual_ray_infeasibility = state->max_dual_ray_infeasibility;
        results->primal_ray_linear_objective = state->primal_ray_linear_objective;
        results->dual_ray_objective = state->dual_ray_objective;
        results->termination_reason = state->termination_reason;
        results->feasibility_polishing_time = state->feasibility_polishing_time;
        results->feasibility_iteration = state->feasibility_iteration;
        results->total_inner_count = state->inner_solver->total_count;
    }

    return results;
}

static pdhcg_result_t *distributed_optimize_core(const pdhg_parameters_t *params,
                                                 const qp_problem_t *working_problem,
                                                 grid_context_t *grid_context)
{
    if (grid_context->rank_global == 0)
    {
        print_initial_info(params, working_problem);
        print_distributed_params(params);
    }

    rescale_info_t *rescale_info = NULL;

    if (grid_context->rank_global == 0)
    {
        rescale_info = rescale_problem(params, working_problem);
    }

    qp_problem_t *local_working_problem = NULL;
    rescale_info_t *local_rescale_info = NULL;

    distribute_data_bcast_then_partition(
        working_problem, rescale_info, grid_context, params, &local_working_problem, &local_rescale_info);

    pdhg_solver_state_t *state =
        initialize_solver_state(params, local_working_problem, local_rescale_info, grid_context);
    qp_problem_free(local_working_problem);

    allreduce_obj_bound_norm(state, params);

    if (local_rescale_info)
        rescale_info_free(local_rescale_info);

    if (state->quadratic_objective_term->nonconvexity < 0)
    {
        state->inner_solver->iteration_limit = 1;
    }

    initialize_step_size_and_primal_weight(state, params);

    compute_residual(state, params->optimality_norm);
    MPI_Barrier(grid_context->comm_global);

    double start_time = MPI_Wtime();
    bool do_restart = false;

    while (state->total_count < params->termination_criteria.iteration_limit)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_residual(state, params->optimality_norm);

            if (!state->has_variable_cones && state->grid_context->global_num_affine_cones == 0 &&
                state->is_this_major_iteration && state->total_count < 3 * params->termination_evaluation_frequency)
            {
                compute_infeasibility_information(state);
            }

            state->cumulative_time_sec = (double)(MPI_Wtime() - start_time);

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
        if (grid_context->rank_global == 0)
            display_iteration_stats(state, params->verbose);
    }

    pdhcg_result_t *result = create_result_from_state_distributed(state, working_problem);

    if (grid_context->rank_global == 0)
    {
        if (rescale_info)
            rescale_info_free(rescale_info);
    }

    pdhg_solver_state_free(state);

    return result;
}

pdhcg_result_t *distributed_optimize(const pdhg_parameters_t *params, const qp_problem_t *original_problem)
{
    pdhg_parameters_t default_params;
    if (!params)
    {
        set_default_parameters(&default_params);
        params = &default_params;
    }
    pdhg_parameters_t sub_params = *params;
    int rank_global = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_global);

    int fixed_section_invalid = 0;
    if (rank_global == 0 && pdhcg_validate_fixed_cone_sections(original_problem) != 0)
        fixed_section_invalid = 1;
    MPI_Bcast(&fixed_section_invalid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (fixed_section_invalid)
        return NULL;

    const qp_problem_t *input_problem = original_problem;
    qp_problem_t *transformed = NULL;
    int transform_failed = 0;
    if (rank_global == 0)
    {
        if (!original_problem)
        {
            fprintf(stderr, "Error: rank 0 did not provide a problem.\n");
            transform_failed = 1;
        }
        else if (original_problem->num_quadratic_constraints > 0)
        {
            transformed = qcqp_to_socp_qp(original_problem, params->default_cone_type);
            if (!transformed)
            {
                fprintf(stderr, "Error: distributed QCQP -> SOCP transformation failed.\n");
                transform_failed = 1;
            }
            else
            {
                original_problem = transformed;
                if (params->verbose >= 1)
                {
                    const char *form_name = params->default_cone_type == CONE_STANDARD_SOC ? "standard" : "rotated";
                    fprintf(stderr,
                            "[QCQP] %d quadratic constraint(s) reformulated as %d %s SOC block(s) before "
                            "distributed partitioning.\n",
                            input_problem->num_quadratic_constraints,
                            transformed->cones.num_cones,
                            form_name);
                }
            }
        }
    }
    MPI_Bcast(&transform_failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (transform_failed)
    {
        if (transformed)
            qp_problem_free(transformed);
        return NULL;
    }

    select_valid_grid_size(params, original_problem, &sub_params);

    grid_context_t grid_context =
        initialize_parallel_context(sub_params.grid_size.row_dims, sub_params.grid_size.col_dims);

    sub_params.verbose = (grid_context.rank_global == 0) ? params->verbose : 0;

    pdhcg_result_t *result = NULL;

    const qp_problem_t *working_problem = original_problem;
    pdhcg_presolve_info_t *presolve_info = NULL;
    qp_problem_t *dummy_problem = NULL;
    qp_problem_t *permuted_problem = NULL;
    int *row_perm = NULL;
    int *col_perm = NULL;
    int is_solved_early = 0;

    if (grid_context.rank_global == 0)
    {
        if (params->presolve && original_problem->affine_cones.num_cones == 0 && pdhcg_presolve_available())
        {
            presolve_info = pdhcg_presolve(original_problem, params);
            if (presolve_info)
            {
                if (presolve_info->problem_solved_during_presolve)
                {
                    is_solved_early = 1;
                }
                else if (presolve_info->reduced_problem)
                {
                    working_problem = presolve_info->reduced_problem;
                }
            }
        }

        if (!is_solved_early)
        {
            if (working_problem->num_constraints == 0 || working_problem->constraint_matrix == NULL)
            {
                dummy_problem = create_problem_with_dummy_constraint(working_problem);
                working_problem = dummy_problem;
            }
        }

        if (!is_solved_early && params->permute_method != NO_PERMUTATION)
        {
            row_perm = (int *)malloc(working_problem->num_constraints * sizeof(int));
            col_perm = (int *)malloc(working_problem->num_variables * sizeof(int));

            generate_cone_aware_permutation(
                working_problem, params->permute_method, params->permute_block_size, col_perm);
            generate_affine_cone_aware_row_permutation(
                working_problem, params->permute_method, params->permute_block_size, row_perm);

            permuted_problem = permute_problem_return_new(working_problem, row_perm, col_perm);
            working_problem = permuted_problem;
        }
    }

    MPI_Bcast(&is_solved_early, 1, MPI_INT, 0, grid_context.comm_global);

    if (is_solved_early)
    {
        if (grid_context.rank_global == 0)
        {
            result = pdhcg_create_result_from_presolve(presolve_info, original_problem);
            restore_qcqp_result_dimensions(result, transformed ? input_problem : NULL);
            if (result)
                pdhg_final_log(result, params);
            if (presolve_info)
                pdhcg_presolve_info_free(presolve_info);
            if (transformed)
                qp_problem_free(transformed);
            destroy_parallel_context(&grid_context);
            return result;
        }
        destroy_parallel_context(&grid_context);
        return NULL;
    }

    result = distributed_optimize_core(&sub_params, working_problem, &grid_context);

    if (grid_context.rank_global == 0 && result != NULL)
    {
        if (params->permute_method != NO_PERMUTATION)
        {
            repermute_solution(result, row_perm, col_perm);
            free(row_perm);
            free(col_perm);
            if (permuted_problem)
                qp_problem_free(permuted_problem);
        }

        if (presolve_info && presolve_info->reduced_problem)
        {
            pdhcg_postsolve(presolve_info, result, original_problem);
        }

        if (dummy_problem)
        {
            qp_problem_free(dummy_problem);
        }

        restore_qcqp_result_dimensions(result, transformed ? input_problem : NULL);
        pdhg_final_log(result, params);
        if (presolve_info)
            pdhcg_presolve_info_free(presolve_info);
        if (transformed)
            qp_problem_free(transformed);
    }
    else if (grid_context.rank_global != 0)
    {
        result = NULL;
    }
    else
    {
        free(row_perm);
        free(col_perm);
        if (permuted_problem)
            qp_problem_free(permuted_problem);
        if (dummy_problem)
            qp_problem_free(dummy_problem);
        if (presolve_info)
            pdhcg_presolve_info_free(presolve_info);
        if (transformed)
            qp_problem_free(transformed);
    }

    destroy_parallel_context(&grid_context);
    return result;
}
