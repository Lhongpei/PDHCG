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

#include "distributed_interface.h"
#include "internal_types.h"
#include "pdhcg.h"
#include "pdhcg_kernels.cuh"
#include "pdhg_core_op.h"
#include "preconditioner.h"
#include "solver.h"
#include "solver_state.h"
#include "spmv_backend.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#ifdef PDHCG_COMPILE_DISTRIBUTED
#include "distributed_types.h"
#endif

static void apply_lowrank_middle(pdhg_solver_state_t *state)
{
    quadratic_objective_term_t *qot = state->quadratic_objective_term;
    int rank = qot->num_rank_lowrank_obj;
    if (qot->lowrank_middle_type == 0 || rank <= 0)
        return;

    if (qot->lowrank_middle_type == 1)
    {
        int nb = (rank + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        element_wise_mul_inplace_kernel<<<nb, THREADS_PER_BLOCK>>>(qot->Rx_product, qot->d_middle_diag, rank);
        return;
    }

    cublasPointerMode_t prev_mode;
    CUBLAS_CHECK(cublasGetPointerMode(state->blas_handle, &prev_mode));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasDsymv(state->blas_handle,
                             CUBLAS_FILL_MODE_LOWER,
                             rank,
                             &HOST_ONE,
                             qot->d_middle_dense,
                             rank,
                             qot->Rx_product,
                             1,
                             &HOST_ZERO,
                             qot->Rx_buffer,
                             1));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, prev_mode));
    CUDA_CHECK(
        cudaMemcpyAsync(qot->Rx_product, qot->Rx_buffer, (size_t)rank * sizeof(double), cudaMemcpyDeviceToDevice));
}

void update_obj_product(pdhg_solver_state_t *state, double *primal_solution)
{
    switch (state->quadratic_objective_term->quad_obj_type)
    {
        case PDHCG_NON_Q:
            return;

        case PDHCG_SPARSE_Q:
            pdhcg_spmv_execute(state->sparse_handle,
                               state->quadratic_objective_term->spmv_ctx_Q,
                               &HOST_ONE,
                               &HOST_ZERO,
                               primal_solution,
                               state->quadratic_objective_term->global_primal_obj_product);

            pdhcg_all_reduce_array(state->grid_context,
                                   state->quadratic_objective_term->global_primal_obj_product,
                                   get_global_n(state),
                                   PDHCG_OP_SUM,
                                   PDHCG_SCOPE_ROW,
                                   0);
            return;

        case PDHCG_DIAG_Q:
            element_wise_mul_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                state->quadratic_objective_term->diagonal_objective_matrix,
                primal_solution,
                state->quadratic_objective_term->primal_obj_product,
                state->num_variables);
            return;

        case PDHCG_LOW_RANK_Q:
            pdhcg_spmv_execute(state->sparse_handle,
                               state->quadratic_objective_term->spmv_ctx_R,
                               &HOST_ONE,
                               &HOST_ZERO,
                               primal_solution,
                               state->quadratic_objective_term->Rx_product);

            pdhcg_all_reduce_array(state->grid_context,
                                   state->quadratic_objective_term->Rx_product,
                                   state->quadratic_objective_term->num_rank_lowrank_obj,
                                   PDHCG_OP_SUM,
                                   PDHCG_SCOPE_ROW,
                                   0);

            apply_lowrank_middle(state);

            pdhcg_spmv_execute(state->sparse_handle,
                               state->quadratic_objective_term->spmv_ctx_Rt,
                               &HOST_ONE,
                               &HOST_ZERO,
                               state->quadratic_objective_term->Rx_product,
                               state->quadratic_objective_term->primal_obj_product);
            return;

        case PDHCG_LOW_RANK_PLUS_SPARSE_Q:
            pdhcg_spmv_execute(state->sparse_handle,
                               state->quadratic_objective_term->spmv_ctx_Q,
                               &HOST_ONE,
                               &HOST_ZERO,
                               primal_solution,
                               state->quadratic_objective_term->global_primal_obj_product);

            pdhcg_all_reduce_array(state->grid_context,
                                   state->quadratic_objective_term->global_primal_obj_product,
                                   get_global_n(state),
                                   PDHCG_OP_SUM,
                                   PDHCG_SCOPE_ROW,
                                   0);

            pdhcg_spmv_execute(state->sparse_handle,
                               state->quadratic_objective_term->spmv_ctx_R,
                               &HOST_ONE,
                               &HOST_ZERO,
                               primal_solution,
                               state->quadratic_objective_term->Rx_product);

            pdhcg_all_reduce_array(state->grid_context,
                                   state->quadratic_objective_term->Rx_product,
                                   state->quadratic_objective_term->num_rank_lowrank_obj,
                                   PDHCG_OP_SUM,
                                   PDHCG_SCOPE_ROW,
                                   0);

            apply_lowrank_middle(state);

            pdhcg_spmv_execute(state->sparse_handle,
                               state->quadratic_objective_term->spmv_ctx_Rt,
                               &HOST_ONE,
                               &HOST_ONE,
                               state->quadratic_objective_term->Rx_product,
                               state->quadratic_objective_term->primal_obj_product);
            return;

        default:
            fprintf(stderr, "Error: Unknown Quadratic Objective Type detected.\n");
            exit(EXIT_FAILURE);
    }
}

double compute_xQx(pdhg_solver_state_t *state, double *primal_sol, double *primal_obj_product)
{
    if (state->quadratic_objective_term->quad_obj_type == PDHCG_NON_Q)
        return 0.0;

    double xQx = 0.0;
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, primal_sol, 1, primal_obj_product, 1, &xQx));
    pdhcg_all_reduce_scalar(state->grid_context, &xQx, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
    return xQx;
}

void lp_primal_update(pdhg_solver_state_t *state, double step_size)
{
    bool force_major_for_cone = state->num_cone_blocks > 0;
    if (state->is_this_major_iteration || force_major_for_cone ||
        ((state->total_count + 2) % get_print_frequency(state->total_count + 2)) == 0)
    {
        compute_lp_next_pdhg_primal_solution_major_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->current_primal_solution,
            state->pdhg_primal_solution,
            state->reflected_primal_solution,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables,
            step_size,
            state->dual_slack);
    }
    else
    {
        compute_lp_next_pdhg_primal_solution_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->current_primal_solution,
            state->reflected_primal_solution,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables,
            step_size);
    }
}

void diag_q_primal_update(pdhg_solver_state_t *state, double step_size)
{
    bool force_major_for_cone = state->num_cone_blocks > 0;
    if (state->is_this_major_iteration ||
        ((state->total_count + 2) % get_print_frequency(state->total_count + 2)) == 0 || force_major_for_cone)
    {
        compute_diagonal_q_next_pdhg_primal_solution_major_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->current_primal_solution,
            state->pdhg_primal_solution,
            state->reflected_primal_solution,
            state->quadratic_objective_term->diagonal_objective_matrix,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables,
            step_size);
    }
    else
    {
        compute_diagonal_q_next_pdhg_primal_solution_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->current_primal_solution,
            state->reflected_primal_solution,
            state->quadratic_objective_term->diagonal_objective_matrix,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables,
            step_size);
    }
}
static __global__ void sqrt_scalar_kernel(double *val)
{
    *val = sqrt(*val);
}

void primal_BB_step_size_update(pdhg_solver_state_t *state, double step_size)
{
    double inv_step_size = 1.0 / step_size;
    int inner_solver_iter = 1;
    double initial_alpha = 1.0 / inv_step_size;

    bb_step_size_t *bb = state->inner_solver->bb_step_size;
    bool precond = bb->precond_enabled;

    double *d_norm_gtg = bb->scalar_buffer;
    double *d_tmp = bb->scalar_buffer + 1;
    double *d_alpha = bb->scalar_buffer + 2;
    double *d_stMs = bb->scalar_buffer + 3;

    if (precond && bb->cached_inv_tau != inv_step_size)
    {
        refresh_inner_precond_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            bb->diag_h_static, inv_step_size, bb->m_diag, bb->m_inv, state->num_variables);
        bb->cached_inv_tau = inv_step_size;

        double sum_m = 0.0;
        CUBLAS_CHECK(cublasDasum(state->blas_handle, state->num_variables, bb->m_diag, 1, &sum_m));
        pdhcg_all_reduce_scalar(state->grid_context, &sum_m, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        int n_global = get_global_n(state);
        if (n_global > 0)
            bb->tol_scale = sqrt(sum_m / (double)n_global);
        else
            bb->tol_scale = 1.0;
    }

    if (precond)
        initial_alpha *= bb->tol_scale * bb->tol_scale;

    update_obj_product(state, state->current_primal_solution);
    if (precond)
    {
        primal_gradient_descent_kernel_bb_init_precond<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->dual_product,
            bb->gradient,
            bb->direction,
            state->current_primal_solution,
            state->pdhg_primal_solution,
            state->objective_vector,
            state->quadratic_objective_term->primal_obj_product,
            state->variable_lower_bound,
            state->variable_upper_bound,
            bb->m_inv,
            initial_alpha,
            state->num_variables);
    }
    else
    {
        primal_gradient_descent_kernel_bb_init<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->dual_product,
            bb->gradient,
            bb->direction,
            state->current_primal_solution,
            state->pdhg_primal_solution,
            state->objective_vector,
            state->quadratic_objective_term->primal_obj_product,
            state->variable_lower_bound,
            state->variable_upper_bound,
            initial_alpha,
            state->num_variables);
    }

    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_DEVICE);

    int check_frequency = 1;
    double h_norm_gtg = 0.0;

    while (inner_solver_iter < state->inner_solver->iteration_limit)
    {
        if (precond)
        {
            element_wise_mul_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                bb->m_diag, bb->direction, bb->Ms_buffer, state->num_variables);
            CUBLAS_CHECK(
                cublasDdot(state->blas_handle, state->num_variables, bb->direction, 1, bb->Ms_buffer, 1, d_stMs));
            pdhcg_all_reduce_scalar(state->grid_context, d_stMs, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, true);
            scalar_sqrt_copy_kernel<<<1, 1>>>(d_stMs, d_norm_gtg);
        }
        else
        {
            CUBLAS_CHECK(
                cublasDdot(state->blas_handle, state->num_variables, bb->direction, 1, bb->direction, 1, d_norm_gtg));
            pdhcg_all_reduce_scalar(state->grid_context, d_norm_gtg, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, true);
            sqrt_scalar_kernel<<<1, 1>>>(d_norm_gtg);
        }

        if (inner_solver_iter == 1 || inner_solver_iter % check_frequency == 0)
        {
            cudaMemcpy(&h_norm_gtg, d_norm_gtg, sizeof(double), cudaMemcpyDeviceToHost);
            if (h_norm_gtg <= state->inner_solver->tol)
                break;
        }

        update_obj_product(state, state->pdhg_primal_solution);
        primal_bb_update_gradient_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->pdhg_primal_solution,
            state->current_primal_solution,
            state->objective_vector,
            state->dual_product,
            state->quadratic_objective_term->primal_obj_product,
            bb->gradient,
            state->inner_solver->primal_buffer,
            inv_step_size,
            state->num_variables);

        CUBLAS_CHECK(cublasDdot(
            state->blas_handle, state->num_variables, bb->direction, 1, state->inner_solver->primal_buffer, 1, d_tmp));

        pdhcg_all_reduce_scalar(state->grid_context, d_tmp, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, true);

        if (precond)
        {
            compute_bb_alpha_M_kernel<<<1, 1>>>(d_stMs, d_tmp, d_alpha);

            primal_bb_update_direction_kernel_precond<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                state->pdhg_primal_solution,
                bb->gradient,
                bb->direction,
                state->variable_lower_bound,
                state->variable_upper_bound,
                bb->m_inv,
                d_alpha,
                state->num_variables);
        }
        else
        {
            compute_bb_alpha_safeguard_kernel<<<1, 1>>>(d_norm_gtg, d_tmp, d_alpha);

            primal_bb_update_direction_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                state->pdhg_primal_solution,
                bb->gradient,
                bb->direction,
                state->variable_lower_bound,
                state->variable_upper_bound,
                d_alpha,
                state->num_variables);
        }
        inner_solver_iter++;
    }

    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST);

    primal_bb_final_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(state->current_primal_solution,
                                                                            state->pdhg_primal_solution,
                                                                            state->reflected_primal_solution,
                                                                            state->num_variables);
    state->inner_solver->total_count += (inner_solver_iter - 1);
}

void primal_gradient_update(pdhg_solver_state_t *state, double step_size)
{
    double inv_step_size = 1.0 / step_size;
    double alpha = 1.0 / (state->quadratic_objective_term->norm + inv_step_size);
    update_obj_product(state, state->current_primal_solution);
    if (state->is_this_major_iteration || ((state->total_count + 2) % get_print_frequency(state->total_count + 2)) == 0)
    {
        primal_gradient_descent_kernel_major<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->dual_product,
            state->current_primal_solution,
            state->reflected_primal_solution,
            state->pdhg_primal_solution,
            state->objective_vector,
            state->quadratic_objective_term->primal_obj_product,
            state->variable_lower_bound,
            state->variable_upper_bound,
            alpha,
            state->num_variables);
    }
    else
    {
        primal_gradient_descent_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->dual_product,
            state->current_primal_solution,
            state->reflected_primal_solution,
            state->objective_vector,
            state->quadratic_objective_term->primal_obj_product,
            state->variable_lower_bound,
            state->variable_upper_bound,
            alpha,
            state->num_variables);
    }
}

void pdhg_update(pdhg_solver_state_t *state)
{
    double primal_step_size = state->step_size / state->primal_weight;
    if (state->quadratic_objective_term->nonconvexity < 0)
    {
        primal_step_size = fmax(primal_step_size, -1.01 * fmin(0.0, state->quadratic_objective_term->nonconvexity));
        primal_step_size /= 100;
    }
    double dual_step_size = state->step_size * state->primal_weight;

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_At,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->current_dual_solution,
                       state->dual_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->dual_product, state->num_variables, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);

    switch (state->quadratic_objective_term->quad_obj_type)
    {
        case PDHCG_NON_Q:
        {
            lp_primal_update(state, primal_step_size);
            break;
        }
        case PDHCG_DIAG_Q:
        {
            diag_q_primal_update(state, primal_step_size);
            break;
        }
        case PDHCG_SPARSE_Q:
        case PDHCG_LOW_RANK_Q:
        case PDHCG_LOW_RANK_PLUS_SPARSE_Q:
        {
            primal_BB_step_size_update(state, primal_step_size);
            break;
        }
        default:
            fprintf(stderr, "Error: Unknown Quadratic Objective Type detected.\n");
            exit(EXIT_FAILURE);
    }

    if (state->num_cone_blocks > 0)
    {
        int threads = THREADS_PER_BLOCK;
        int blocks = (state->num_cone_blocks + threads - 1) / threads;
        project_rotated_soc_kernel<<<blocks, threads>>>(state->pdhg_primal_solution,
                                                        state->cone_block_start_idx_d,
                                                        state->cone_block_v_dim_d,
                                                        state->num_cone_blocks);
        recompute_reflected_at_cone_kernel<<<blocks, threads>>>(state->reflected_primal_solution,
                                                                state->pdhg_primal_solution,
                                                                state->current_primal_solution,
                                                                state->cone_block_start_idx_d,
                                                                state->cone_block_v_dim_d,
                                                                state->num_cone_blocks);
    }

    state->inner_solver->total_count++;

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_A,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->reflected_primal_solution,
                       state->primal_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->primal_product, state->num_constraints, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);

    if (state->is_this_major_iteration || ((state->total_count + 2) % get_print_frequency(state->total_count + 2)) == 0)
    {
        compute_next_pdhg_dual_solution_major_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
            state->current_dual_solution,
            state->pdhg_dual_solution,
            state->reflected_dual_solution,
            state->primal_product,
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            state->num_constraints,
            dual_step_size);
    }
    else
    {
        compute_next_pdhg_dual_solution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
            state->current_dual_solution,
            state->reflected_dual_solution,
            state->primal_product,
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            state->num_constraints,
            dual_step_size);
    }
}

void halpern_update(pdhg_solver_state_t *state, double reflection_coefficient)
{
    double weight = (double)(state->inner_count + 1) / (state->inner_count + 2);
    halpern_update_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(state->initial_primal_solution,
                                                                                state->current_primal_solution,
                                                                                state->reflected_primal_solution,
                                                                                state->initial_dual_solution,
                                                                                state->current_dual_solution,
                                                                                state->reflected_dual_solution,
                                                                                state->num_variables,
                                                                                state->num_constraints,
                                                                                weight,
                                                                                reflection_coefficient);
}

void rescale_solution(pdhg_solver_state_t *state)
{
    rescale_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(state->pdhg_primal_solution,
                                                                                  state->pdhg_dual_solution,
                                                                                  state->variable_rescaling,
                                                                                  state->constraint_rescaling,
                                                                                  state->objective_vector_rescaling,
                                                                                  state->constraint_bound_rescaling,
                                                                                  state->num_variables,
                                                                                  state->num_constraints);
}

void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(state->initial_primal_solution,
                                                                                        state->pdhg_primal_solution,
                                                                                        state->delta_primal_solution,
                                                                                        state->initial_dual_solution,
                                                                                        state->pdhg_dual_solution,
                                                                                        state->delta_dual_solution,
                                                                                        state->num_variables,
                                                                                        state->num_constraints);

    double primal_dist, dual_dist;
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_dist));
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_dist));

    double primal_dist_sq = primal_dist * primal_dist;
    pdhcg_all_reduce_scalar(state->grid_context, &primal_dist_sq, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
    primal_dist = sqrt(primal_dist_sq);

    double dual_dist_sq = dual_dist * dual_dist;
    pdhcg_all_reduce_scalar(state->grid_context, &dual_dist_sq, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);
    dual_dist = sqrt(dual_dist_sq);

    double ratio_infeas = state->relative_dual_residual / state->relative_primal_residual;

    if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 && dual_dist < 1e12 && ratio_infeas > 1e-8 &&
        ratio_infeas < 1e8)
    {
        double error = log(dual_dist) - log(primal_dist) - log(state->primal_weight);
        state->primal_weight_error_sum *= params->restart_params.i_smooth;
        state->primal_weight_error_sum += error;
        double delta_error = error - state->primal_weight_last_error;
        state->primal_weight *=
            exp(params->restart_params.k_p * error + params->restart_params.k_i * state->primal_weight_error_sum +
                params->restart_params.k_d * delta_error);
        state->primal_weight_last_error = error;
    }
    else
    {
        state->primal_weight = state->best_primal_weight;
        state->primal_weight_error_sum = 0.0;
        state->primal_weight_last_error = 0.0;
    }

    double primal_dual_residual_gap = abs(log10(state->relative_dual_residual / state->relative_primal_residual));
    if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
    {
        state->best_primal_dual_residual_gap = primal_dual_residual_gap;
        state->best_primal_weight = state->primal_weight;
    }

    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution,
                          state->pdhg_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution,
                          state->pdhg_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution,
                          state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution,
                          state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    if (state->constraint_matrix->num_nonzeros == 0)
    {
        state->step_size = 1.0;
    }
    else
    {
        double max_sv = estimate_maximum_singular_value(state->sparse_handle,
                                                        state->blas_handle,
                                                        state->constraint_matrix,
                                                        state->constraint_matrix_t,
                                                        params->sv_max_iter,
                                                        params->sv_tol,
                                                        state->grid_context);
        if (max_sv < 1e-12)
        {
            state->step_size = 1.0;
        }
        else
        {
            state->step_size = 0.998 / max_sv;
        }
    }

    if (params->bound_objective_rescaling)
    {
        state->primal_weight = 1.0;
    }
    else
    {
        state->primal_weight = (state->objective_vector_norm + 1.0) / (state->constraint_bound_norm + 1.0);
    }
    state->best_primal_weight = state->primal_weight;
}

void compute_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->current_primal_solution,
        state->reflected_primal_solution,
        state->delta_primal_solution,
        state->current_dual_solution,
        state->reflected_dual_solution,
        state->delta_dual_solution,
        state->num_variables,
        state->num_constraints);

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_At,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->delta_dual_solution,
                       state->dual_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->dual_product, state->num_variables, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);

    double interaction, movement;

    double primal_norm = 0.0;
    double dual_norm = 0.0;
    double cross_term = 0.0;

    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_norm));
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_norm));

    double dual_norm_sq = dual_norm * dual_norm;
    pdhcg_all_reduce_scalar(state->grid_context, &dual_norm_sq, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);
    dual_norm = sqrt(dual_norm_sq);

    double primal_norm_sq = primal_norm * primal_norm;
    pdhcg_all_reduce_scalar(state->grid_context, &primal_norm_sq, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
    primal_norm = sqrt(primal_norm_sq);

    movement = primal_norm * primal_norm * state->primal_weight + dual_norm * dual_norm / state->primal_weight;

    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            state->dual_product,
                            1,
                            state->delta_primal_solution,
                            1,
                            &cross_term));

    pdhcg_all_reduce_scalar(state->grid_context, &cross_term, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);

    interaction = 2 * state->step_size * cross_term;

    state->fixed_point_error = sqrt(movement + interaction);
    if (state->problem_type == CONVEX_QP &&
        (state->quadratic_objective_term->quad_obj_type != PDHCG_NON_Q &&
         state->quadratic_objective_term->quad_obj_type != PDHCG_DIAG_Q))
    {
        state->inner_solver->tol =
            fmin(state->inner_solver->tol,
                 fmax(0.0005 * primal_norm / state->step_size * state->primal_weight, state->inner_solver->min_tol));
    }
}

void compute_residual(pdhg_solver_state_t *state, norm_type_t optimality_norm)
{
    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_A,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->pdhg_primal_solution,
                       state->primal_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->primal_product, state->num_constraints, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);

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
        compute_lp_residual_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
            state->primal_residual,
            state->primal_product,
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            state->pdhg_dual_solution,
            state->dual_residual,
            state->dual_product,
            state->dual_slack,
            state->objective_vector,
            state->constraint_rescaling,
            state->variable_rescaling,
            state->primal_slack,
            state->constraint_lower_bound_finite_val,
            state->constraint_upper_bound_finite_val,
            state->num_constraints,
            state->num_variables);

        if (state->num_cone_blocks > 0)
        {
            int threads = THREADS_PER_BLOCK;
            int blocks = (state->num_cone_blocks + threads - 1) / threads;
            compute_cone_dual_residual_kernel<<<blocks, threads>>>(state->dual_residual,
                                                                   state->objective_vector,
                                                                   state->dual_product,
                                                                   state->variable_rescaling,
                                                                   state->cone_block_start_idx_d,
                                                                   state->cone_block_v_dim_d,
                                                                   state->num_cone_blocks);
        }
    }
    else if (state->problem_type == CONVEX_QP)
    {
        compute_qp_residual_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
            state->primal_residual,
            state->primal_product,
            state->quadratic_objective_term->primal_obj_product,
            state->pdhg_primal_solution,
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->pdhg_dual_solution,
            state->dual_residual,
            state->dual_product,
            state->dual_slack,
            state->objective_vector,
            state->constraint_rescaling,
            state->variable_rescaling,
            state->primal_slack,
            state->constraint_lower_bound_finite_val,
            state->constraint_upper_bound_finite_val,
            state->step_size / state->primal_weight,
            state->num_constraints,
            state->num_variables);

        if (state->num_cone_blocks > 0)
        {
            int threads = THREADS_PER_BLOCK;
            int blocks = (state->num_cone_blocks + threads - 1) / threads;
            compute_cone_dual_residual_kernel<<<blocks, threads>>>(state->dual_residual,
                                                                   state->objective_vector,
                                                                   state->dual_product,
                                                                   state->variable_rescaling,
                                                                   state->cone_block_start_idx_d,
                                                                   state->cone_block_v_dim_d,
                                                                   state->num_cone_blocks);
        }
    }

    if (optimality_norm == NORM_TYPE_L_INF)
    {
        state->absolute_primal_residual =
            get_vector_inf_norm(state->blas_handle, state->num_constraints, state->primal_residual);
        pdhcg_all_reduce_scalar(
            state->grid_context, &state->absolute_primal_residual, PDHCG_OP_MAX, PDHCG_SCOPE_COL, false);
    }
    else
    {
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle, state->num_constraints, state->primal_residual, 1, &state->absolute_primal_residual));
        state->absolute_primal_residual *= state->absolute_primal_residual;
        pdhcg_all_reduce_scalar(
            state->grid_context, &state->absolute_primal_residual, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);
        state->absolute_primal_residual = sqrt(state->absolute_primal_residual);
    }
    state->absolute_primal_residual /= state->constraint_bound_rescaling;

    if (optimality_norm == NORM_TYPE_L_INF)
    {
        state->absolute_dual_residual =
            get_vector_inf_norm(state->blas_handle, state->num_variables, state->dual_residual);
        pdhcg_all_reduce_scalar(
            state->grid_context, &state->absolute_dual_residual, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);
    }
    else
    {
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle, state->num_variables, state->dual_residual, 1, &state->absolute_dual_residual));
        state->absolute_dual_residual *= state->absolute_dual_residual;
        pdhcg_all_reduce_scalar(
            state->grid_context, &state->absolute_dual_residual, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        state->absolute_dual_residual = sqrt(state->absolute_dual_residual);
    }
    state->absolute_dual_residual /= state->objective_vector_rescaling;

    double half_xQx =
        0.5 * compute_xQx(state, state->pdhg_primal_solution, state->quadratic_objective_term->primal_obj_product);

    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            state->objective_vector,
                            1,
                            state->pdhg_primal_solution,
                            1,
                            &state->primal_objective_value));

    pdhcg_all_reduce_scalar(state->grid_context, &state->primal_objective_value, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);

    state->primal_objective_value = (state->primal_objective_value + half_xQx) /
            (state->constraint_bound_rescaling * state->objective_vector_rescaling) +
        state->objective_constant;

    if (state->num_cone_blocks > 0)
    {
        int threads = THREADS_PER_BLOCK;
        int blocks = (state->num_cone_blocks + threads - 1) / threads;
        set_cone_dual_slack_kernel<<<blocks, threads>>>(state->dual_slack,
                                                        state->objective_vector,
                                                        state->dual_product,
                                                        state->cone_block_start_idx_d,
                                                        state->cone_block_v_dim_d,
                                                        state->num_cone_blocks);
    }

    double base_dual_objective;
    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            state->dual_slack,
                            1,
                            state->pdhg_primal_solution,
                            1,
                            &base_dual_objective));

    pdhcg_all_reduce_scalar(state->grid_context, &base_dual_objective, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);

    double dual_slack_sum =
        get_vector_sum(state->blas_handle, state->num_constraints, state->ones_dual_d, state->primal_slack);
    pdhcg_all_reduce_scalar(state->grid_context, &dual_slack_sum, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);

    state->dual_objective_value = (base_dual_objective + dual_slack_sum - half_xQx) /
            (state->constraint_bound_rescaling * state->objective_vector_rescaling) +
        state->objective_constant;

    double relative_primal_dominator = 1.0 + state->constraint_bound_norm;
    state->relative_primal_residual = state->absolute_primal_residual / relative_primal_dominator;

    double relative_dual_dominator;
    if (state->problem_type == LP)
    {
        relative_dual_dominator = 1.0 + state->objective_vector_norm;
    }
    else
    {
        recover_primal_obj_dual_product<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->dual_product,
            state->quadratic_objective_term->primal_obj_product,
            state->variable_rescaling,
            state->num_variables);
        double qx_norm;
        if (optimality_norm == NORM_TYPE_L_INF)
        {
            qx_norm = get_vector_inf_norm(
                state->blas_handle, state->num_variables, state->quadratic_objective_term->primal_obj_product);
            pdhcg_all_reduce_scalar(state->grid_context, &qx_norm, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);
        }
        else
        {
            CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle,
                                           state->num_variables,
                                           state->quadratic_objective_term->primal_obj_product,
                                           1,
                                           &qx_norm));
            qx_norm *= qx_norm;
            pdhcg_all_reduce_scalar(state->grid_context, &qx_norm, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
            qx_norm = sqrt(qx_norm);
        }
        double Ay_norm;
        if (optimality_norm == NORM_TYPE_L_INF)
        {
            Ay_norm = get_vector_inf_norm(state->blas_handle, state->num_variables, state->dual_product);
            pdhcg_all_reduce_scalar(state->grid_context, &Ay_norm, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);
        }
        else
        {
            CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->dual_product, 1, &Ay_norm));
            Ay_norm *= Ay_norm;
            pdhcg_all_reduce_scalar(state->grid_context, &Ay_norm, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
            Ay_norm = sqrt(Ay_norm);
        }
        relative_dual_dominator = 1.0 +
            fmax(state->objective_vector_norm,
                 fmax(qx_norm / state->objective_vector_rescaling, Ay_norm / state->objective_vector_rescaling));
    }
    state->relative_dual_residual = state->absolute_dual_residual / relative_dual_dominator;

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);

    state->relative_objective_gap =
        state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
}

void compute_infeasibility_information(pdhg_solver_state_t *state)
{
    primal_infeasibility_project_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->delta_primal_solution, state->variable_lower_bound, state->variable_upper_bound, state->num_variables);
    dual_infeasibility_project_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(state->delta_dual_solution,
                                                                                     state->constraint_lower_bound,
                                                                                     state->constraint_upper_bound,
                                                                                     state->num_constraints);

    double primal_ray_inf_norm =
        get_vector_inf_norm(state->blas_handle, state->num_variables, state->delta_primal_solution);

    pdhcg_all_reduce_scalar(state->grid_context, &primal_ray_inf_norm, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);

    if (primal_ray_inf_norm > 0.0)
    {
        double scale = 1.0 / primal_ray_inf_norm;
        cublasDscal(state->blas_handle, state->num_variables, &scale, state->delta_primal_solution, 1);
    }

    double dual_ray_inf_norm =
        get_vector_inf_norm(state->blas_handle, state->num_constraints, state->delta_dual_solution);

    pdhcg_all_reduce_scalar(state->grid_context, &dual_ray_inf_norm, PDHCG_OP_MAX, PDHCG_SCOPE_COL, false);

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_A,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->delta_primal_solution,
                       state->primal_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->primal_product, state->num_constraints, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_At,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->delta_dual_solution,
                       state->dual_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->dual_product, state->num_variables, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);

    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            state->objective_vector,
                            1,
                            state->delta_primal_solution,
                            1,
                            &state->primal_ray_linear_objective));

    pdhcg_all_reduce_scalar(
        state->grid_context, &state->primal_ray_linear_objective, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
    state->primal_ray_linear_objective /= (state->constraint_bound_rescaling * state->objective_vector_rescaling);

    dual_solution_dual_objective_contribution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound_finite_val,
        state->constraint_upper_bound_finite_val,
        state->delta_dual_solution,
        state->num_constraints,
        state->primal_slack);

    dual_objective_dual_slack_contribution_array_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->dual_product,
        state->dual_slack,
        state->variable_lower_bound_finite_val,
        state->variable_upper_bound_finite_val,
        state->num_variables);

    double sum_primal_slack =
        get_vector_sum(state->blas_handle, state->num_constraints, state->ones_dual_d, state->primal_slack);

    pdhcg_all_reduce_scalar(state->grid_context, &sum_primal_slack, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);

    double sum_dual_slack =
        get_vector_sum(state->blas_handle, state->num_variables, state->ones_primal_d, state->dual_slack);

    pdhcg_all_reduce_scalar(state->grid_context, &sum_dual_slack, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);

    state->dual_ray_objective =
        (sum_primal_slack + sum_dual_slack) / (state->constraint_bound_rescaling * state->objective_vector_rescaling);

    compute_primal_infeasibility_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(state->primal_product,
                                                                                       state->constraint_lower_bound,
                                                                                       state->constraint_upper_bound,
                                                                                       state->num_constraints,
                                                                                       state->primal_slack,
                                                                                       state->constraint_rescaling);
    compute_dual_infeasibility_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(state->dual_product,
                                                                                       state->variable_lower_bound,
                                                                                       state->variable_upper_bound,
                                                                                       state->num_variables,
                                                                                       state->dual_slack,
                                                                                       state->variable_rescaling);

    state->max_primal_ray_infeasibility =
        get_vector_inf_norm(state->blas_handle, state->num_constraints, state->primal_slack);

    pdhcg_all_reduce_scalar(
        state->grid_context, &state->max_primal_ray_infeasibility, PDHCG_OP_MAX, PDHCG_SCOPE_COL, false);

    if (state->problem_type != LP && state->quadratic_objective_term->quad_obj_type != PDHCG_NON_Q)
    {
        update_obj_product(state, state->delta_primal_solution);
        double q_ray_norm = get_vector_inf_norm(
            state->blas_handle, state->num_variables, state->quadratic_objective_term->primal_obj_product);

        pdhcg_all_reduce_scalar(state->grid_context, &q_ray_norm, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);

        double scaled_q_norm = q_ray_norm / state->objective_vector_rescaling;
        state->max_primal_ray_infeasibility = fmax(state->max_primal_ray_infeasibility, scaled_q_norm);
    }

    double dual_slack_norm = get_vector_inf_norm(state->blas_handle, state->num_variables, state->dual_slack);

    pdhcg_all_reduce_scalar(state->grid_context, &dual_slack_norm, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);

    state->max_dual_ray_infeasibility = dual_slack_norm;

    double scaling_factor = fmax(dual_ray_inf_norm, dual_slack_norm);
    if (scaling_factor > 0.0)
    {
        state->max_dual_ray_infeasibility /= scaling_factor;
        state->dual_ray_objective /= scaling_factor;
    }
    else
    {
        state->max_dual_ray_infeasibility = 0.0;
        state->dual_ray_objective = 0.0;
    }
}

pdhcg_result_t *create_result_from_state(pdhg_solver_state_t *state, const qp_problem_t *original_problem)
{
    pdhcg_result_t *results = (pdhcg_result_t *)safe_calloc(1, sizeof(pdhcg_result_t));

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_At,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->pdhg_dual_solution,
                       state->dual_product);

    update_obj_product(state, state->pdhg_primal_solution);

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

    rescale_solution(state);

    results->primal_solution = (double *)safe_malloc(state->num_variables * sizeof(double));
    results->dual_solution = (double *)safe_malloc(state->num_constraints * sizeof(double));
    results->reduced_cost = (double *)safe_malloc(state->num_variables * sizeof(double));

    CUDA_CHECK(cudaMemcpy(results->primal_solution,
                          state->pdhg_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->dual_solution,
                          state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        results->reduced_cost, state->dual_slack, state->num_variables * sizeof(double), cudaMemcpyDeviceToHost));

    results->num_variables = original_problem->num_variables;
    results->num_constraints = original_problem->num_constraints;
    results->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;
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
    return results;
}

double estimate_maximum_eigenvalue(cusparseHandle_t sparse_handle,
                                   cublasHandle_t blas_handle,
                                   const cu_sparse_matrix_csr_t *A,
                                   int max_iterations,
                                   double tolerance,
                                   struct grid_context_s *ctx)
{
    int n_global = A->num_rows;
    int n_local = A->num_cols;
    int n_start = get_n_start(ctx);
    int row_coord = pdhcg_get_grid_row_coord(ctx);

    int safe_local = n_local > 0 ? n_local : 1;
    int safe_global = n_global > 0 ? n_global : 1;

    double *v_local_d, *Av_global_d;
    CUDA_CHECK(cudaMalloc(&v_local_d, safe_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&Av_global_d, safe_global * sizeof(double)));

    double *v_local_h = (double *)safe_malloc(safe_local * sizeof(double));
    unsigned int seed = 1234 + row_coord;
    for (int i = 0; i < safe_local; ++i)
        v_local_h[i] = (double)rand_r(&seed) / RAND_MAX;

    if (n_local > 0)
        CUDA_CHECK(cudaMemcpy(v_local_d, v_local_h, n_local * sizeof(double), cudaMemcpyHostToDevice));
    free(v_local_h);

    cusparseDnVecDescr_t vecV, vecAv;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecV, n_local, v_local_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecAv, n_global, Av_global_d, CUDA_R_64F));

    pdhcg_spmv_ctx_t *ctx_A = pdhcg_spmv_ctx_create(
        sparse_handle, n_global, n_local, A->num_nonzeros, A->row_ptr, A->col_ind, A->val, vecV, vecAv);

    double lambda = 0.0;

    for (int i = 0; i < max_iterations; ++i)
    {
        double norm = 0.0;
        if (n_local > 0)
            CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, n_local, v_local_d, 1, &norm));

        double norm_sq = norm * norm;
        pdhcg_all_reduce_scalar(ctx, &norm_sq, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        norm = sqrt(norm_sq);

        double inv_norm = 1.0 / norm;
        if (n_local > 0)
            CUBLAS_CHECK(cublasDscal(blas_handle, n_local, &inv_norm, v_local_d, 1));

        pdhcg_spmv_execute(sparse_handle, ctx_A, &HOST_ONE, &HOST_ZERO, v_local_d, Av_global_d);

        pdhcg_all_reduce_array(ctx, Av_global_d, n_global, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);

        double old_lambda = lambda;
        double local_dot = 0.0;

        if (n_local > 0)
        {
            CUBLAS_CHECK(cublasDdot(blas_handle, n_local, v_local_d, 1, Av_global_d + n_start, 1, &local_dot));
        }

        pdhcg_all_reduce_scalar(ctx, &local_dot, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        lambda = local_dot;

        if (i > 0 && fabs(lambda - old_lambda) < tolerance)
            break;

        if (n_local > 0)
            CUDA_CHECK(
                cudaMemcpy(v_local_d, Av_global_d + n_start, n_local * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    pdhcg_spmv_ctx_destroy(ctx_A);
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecV));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecAv));
    CUDA_CHECK(cudaFree(v_local_d));
    CUDA_CHECK(cudaFree(Av_global_d));

    return lambda;
}

double estimate_minimum_eigenvalue(cusparseHandle_t sparse_handle,
                                   cublasHandle_t blas_handle,
                                   const cu_sparse_matrix_csr_t *A,
                                   double lambda_max,
                                   int max_iterations,
                                   double tolerance,
                                   struct grid_context_s *ctx)
{
    int n_global = A->num_rows;
    int n_local = A->num_cols;
    int n_start = get_n_start(ctx);
    int row_coord = pdhcg_get_grid_row_coord(ctx);

    int safe_local = n_local > 0 ? n_local : 1;
    int safe_global = n_global > 0 ? n_global : 1;

    double *v_local_d, *Av_global_d, *shifted_v_local_d;
    CUDA_CHECK(cudaMalloc(&v_local_d, safe_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&Av_global_d, safe_global * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&shifted_v_local_d, safe_local * sizeof(double)));

    double *v_local_h = (double *)safe_malloc(safe_local * sizeof(double));
    unsigned int seed = 1234 + row_coord;
    for (int i = 0; i < safe_local; ++i)
        v_local_h[i] = (double)rand_r(&seed) / RAND_MAX;

    if (n_local > 0)
        CUDA_CHECK(cudaMemcpy(v_local_d, v_local_h, n_local * sizeof(double), cudaMemcpyHostToDevice));
    free(v_local_h);

    cusparseDnVecDescr_t vecV, vecAv;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecV, n_local, v_local_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecAv, n_global, Av_global_d, CUDA_R_64F));

    pdhcg_spmv_ctx_t *ctx_A = pdhcg_spmv_ctx_create(
        sparse_handle, n_global, n_local, A->num_nonzeros, A->row_ptr, A->col_ind, A->val, vecV, vecAv);

    double mu = 0.0;

    for (int i = 0; i < max_iterations; ++i)
    {
        double norm = 0.0;
        if (n_local > 0)
            CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, n_local, v_local_d, 1, &norm));

        double norm_sq = norm * norm;
        pdhcg_all_reduce_scalar(ctx, &norm_sq, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        norm = sqrt(norm_sq);

        double inv_norm = 1.0 / norm;
        if (n_local > 0)
            CUBLAS_CHECK(cublasDscal(blas_handle, n_local, &inv_norm, v_local_d, 1));

        pdhcg_spmv_execute(sparse_handle, ctx_A, &HOST_ONE, &HOST_ZERO, v_local_d, Av_global_d);

        pdhcg_all_reduce_array(ctx, Av_global_d, n_global, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);

        double neg_one = -1.0;
        double old_mu = mu;
        double local_dot = 0.0;

        if (n_local > 0)
        {
            CUDA_CHECK(cudaMemcpy(
                shifted_v_local_d, Av_global_d + n_start, n_local * sizeof(double), cudaMemcpyDeviceToDevice));
            CUBLAS_CHECK(cublasDscal(blas_handle, n_local, &neg_one, shifted_v_local_d, 1));
            CUBLAS_CHECK(cublasDaxpy(blas_handle, n_local, &lambda_max, v_local_d, 1, shifted_v_local_d, 1));
            CUBLAS_CHECK(cublasDdot(blas_handle, n_local, v_local_d, 1, shifted_v_local_d, 1, &local_dot));
        }

        pdhcg_all_reduce_scalar(ctx, &local_dot, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        mu = local_dot;

        if (i > 0 && fabs(mu - old_mu) < tolerance)
            break;

        if (n_local > 0)
            CUDA_CHECK(cudaMemcpy(v_local_d, shifted_v_local_d, n_local * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    double lambda_min = lambda_max - mu;

    pdhcg_spmv_ctx_destroy(ctx_A);
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecV));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecAv));
    CUDA_CHECK(cudaFree(v_local_d));
    CUDA_CHECK(cudaFree(Av_global_d));
    CUDA_CHECK(cudaFree(shifted_v_local_d));

    return lambda_min;
}

double estimate_maximum_singular_value(cusparseHandle_t sparse_handle,
                                       cublasHandle_t blas_handle,
                                       const cu_sparse_matrix_csr_t *A,
                                       const cu_sparse_matrix_csr_t *AT,
                                       int max_iterations,
                                       double tolerance,
                                       struct grid_context_s *ctx)
{
    int m = A->num_rows;
    int n = A->num_cols;

    int row_coord = pdhcg_get_grid_row_coord(ctx);

    int safe_m = m > 0 ? m : 1;
    int safe_n = n > 0 ? n : 1;
    double *eigenvector_d, *next_eigenvector_d, *dual_product_d;

    CUDA_CHECK(cudaMalloc(&eigenvector_d, safe_m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&next_eigenvector_d, safe_m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dual_product_d, safe_n * sizeof(double)));

    double *eigenvector_h = (double *)safe_malloc(safe_m * sizeof(double));
    unsigned int seed = 1234 + row_coord;
    for (int i = 0; i < safe_m; ++i)
    {
        eigenvector_h[i] = ((double)rand_r(&seed) / (double)RAND_MAX) * 2.0 - 1.0;
    }
    if (m > 0)
        CUDA_CHECK(cudaMemcpy(eigenvector_d, eigenvector_h, m * sizeof(double), cudaMemcpyHostToDevice));
    free(eigenvector_h);

    double sigma_max_sq = 1.0;

    cusparseDnVecDescr_t vecEigen, vecNextEigen, vecDual;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecEigen, m, eigenvector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecNextEigen, m, next_eigenvector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDual, n, dual_product_d, CUDA_R_64F));

    pdhcg_spmv_ctx_t *ctx_A = pdhcg_spmv_ctx_create(
        sparse_handle, m, n, A->num_nonzeros, A->row_ptr, A->col_ind, A->val, vecDual, vecNextEigen);
    pdhcg_spmv_ctx_t *ctx_At = pdhcg_spmv_ctx_create(
        sparse_handle, n, m, AT->num_nonzeros, AT->row_ptr, AT->col_ind, AT->val, vecEigen, vecDual);

    double local_norm = 0.0;
    if (m > 0)
        CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, m, eigenvector_d, 1, &local_norm));

    double norm_sq = local_norm * local_norm;
    pdhcg_all_reduce_scalar(ctx, &norm_sq, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);

    double inv_norm = 1.0 / sqrt(norm_sq);
    if (m > 0)
        CUBLAS_CHECK(cublasDscal(blas_handle, m, &inv_norm, eigenvector_d, 1));

    for (int i = 0; i < max_iterations; ++i)
    {
        pdhcg_spmv_execute(sparse_handle, ctx_At, &HOST_ONE, &HOST_ZERO, eigenvector_d, dual_product_d);
        pdhcg_all_reduce_array(ctx, dual_product_d, n, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);

        pdhcg_spmv_execute(sparse_handle, ctx_A, &HOST_ONE, &HOST_ZERO, dual_product_d, next_eigenvector_d);
        pdhcg_all_reduce_array(ctx, next_eigenvector_d, m, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);

        double local_dot = 0.0;
        if (m > 0)
            CUBLAS_CHECK(cublasDdot(blas_handle, m, next_eigenvector_d, 1, eigenvector_d, 1, &local_dot));

        pdhcg_all_reduce_scalar(ctx, &local_dot, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);
        sigma_max_sq = local_dot;

        double neg_sigma_sq = -sigma_max_sq;
        if (m > 0)
            CUBLAS_CHECK(cublasDaxpy(blas_handle, m, &neg_sigma_sq, eigenvector_d, 1, next_eigenvector_d, 1));

        double local_res_norm = 0.0;
        if (m > 0)
            CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, m, next_eigenvector_d, 1, &local_res_norm));

        double res_sq = local_res_norm * local_res_norm;
        pdhcg_all_reduce_scalar(ctx, &res_sq, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);

        if (sqrt(res_sq) < tolerance)
            break;

        if (m > 0)
            CUBLAS_CHECK(cublasDaxpy(blas_handle, m, &sigma_max_sq, eigenvector_d, 1, next_eigenvector_d, 1));

        local_norm = 0.0;
        if (m > 0)
            CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, m, next_eigenvector_d, 1, &local_norm));

        norm_sq = local_norm * local_norm;
        pdhcg_all_reduce_scalar(ctx, &norm_sq, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);

        inv_norm = 1.0 / sqrt(norm_sq);
        if (m > 0)
            CUBLAS_CHECK(cublasDscal(blas_handle, m, &inv_norm, next_eigenvector_d, 1));

        double *tmp = eigenvector_d;
        eigenvector_d = next_eigenvector_d;
        next_eigenvector_d = tmp;

        CUSPARSE_CHECK(cusparseDnVecSetValues(vecEigen, eigenvector_d));
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecNextEigen, next_eigenvector_d));
    }

    pdhcg_spmv_ctx_destroy(ctx_A);
    pdhcg_spmv_ctx_destroy(ctx_At);
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecEigen));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecNextEigen));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecDual));
    CUDA_CHECK(cudaFree(eigenvector_d));
    CUDA_CHECK(cudaFree(next_eigenvector_d));
    CUDA_CHECK(cudaFree(dual_product_d));

    return sqrt(sigma_max_sq);
}
