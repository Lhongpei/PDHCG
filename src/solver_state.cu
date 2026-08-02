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
#include <string.h>
#include <time.h>

#ifdef PDHCG_COMPILE_DISTRIBUTED
#include "distributed_conic.h"
#include "distributed_types.h"
#endif

int get_global_n(pdhg_solver_state_t *state)
{
    int n = state->num_variables;
#ifdef PDHCG_COMPILE_DISTRIBUTED
    if (state->grid_context != NULL && state->grid_context->global_num_variables > 0)
    {
        n = state->grid_context->global_num_variables;
    }
#endif
    (void)state;
    return n;
}

int get_n_start(grid_context_t *ctx)
{
    int start = 0;
#ifdef PDHCG_COMPILE_DISTRIBUTED
    if (ctx != NULL)
    {
        start = ctx->n_start;
    }
#else
    (void)ctx;
#endif
    return start;
}

static void initialize_sparse_component_obj(pdhg_solver_state_t *state, const processed_qp_problem_t *problem)
{
    state->quadratic_objective_term->objective_sparse_matrix =
        (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));

    int q_rows = get_global_n(state);
    int q_cols = problem->num_variables;

    memset(state->quadratic_objective_term->objective_sparse_matrix, 0, sizeof(cu_sparse_matrix_csr_t));
    state->quadratic_objective_term->objective_sparse_matrix->num_rows = q_rows;
    state->quadratic_objective_term->objective_sparse_matrix->num_cols = q_cols;
    state->quadratic_objective_term->objective_sparse_matrix->num_nonzeros =
        problem->objective_sparse_matrix_num_nonzeros;

    ALLOC_AND_COPY_CSR(state->quadratic_objective_term->objective_sparse_matrix,
                       problem->objective_sparse_matrix,
                       q_rows,
                       problem->objective_sparse_matrix_num_nonzeros);

    state->quadratic_objective_term->spmv_ctx_Q =
        pdhcg_spmv_ctx_create(state->sparse_handle,
                              q_rows,
                              q_cols,
                              state->quadratic_objective_term->objective_sparse_matrix->num_nonzeros,
                              state->quadratic_objective_term->objective_sparse_matrix->row_ptr,
                              state->quadratic_objective_term->objective_sparse_matrix->col_ind,
                              state->quadratic_objective_term->objective_sparse_matrix->val,
                              state->vec_primal_sol,
                              state->quadratic_objective_term->vec_global_primal_obj_prod);
}

static void initialize_lowrank_component_obj(pdhg_solver_state_t *state, const processed_qp_problem_t *problem)
{
    state->quadratic_objective_term->num_rank_lowrank_obj = problem->num_rank_lowrank_obj;
    ALLOC_ZERO(state->quadratic_objective_term->Rx_product, problem->num_rank_lowrank_obj * sizeof(double));

    CUSPARSE_CHECK(cusparseCreateDnVec(&state->quadratic_objective_term->vec_Rx_prod,
                                       state->quadratic_objective_term->num_rank_lowrank_obj,
                                       state->quadratic_objective_term->Rx_product,
                                       CUDA_R_64F));

    state->quadratic_objective_term->objective_lowrank_matrix =
        (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));

    memset(state->quadratic_objective_term->objective_lowrank_matrix, 0, sizeof(cu_sparse_matrix_csr_t));
    state->quadratic_objective_term->objective_lowrank_matrix->num_rows = problem->num_rank_lowrank_obj;
    state->quadratic_objective_term->objective_lowrank_matrix->num_cols = problem->num_variables;
    state->quadratic_objective_term->objective_lowrank_matrix->num_nonzeros =
        problem->objective_lowrank_matrix_num_nonzeros;

    ALLOC_AND_COPY_CSR(state->quadratic_objective_term->objective_lowrank_matrix,
                       problem->objective_lowrank_matrix,
                       problem->num_rank_lowrank_obj,
                       problem->objective_lowrank_matrix_num_nonzeros);

    state->quadratic_objective_term->objective_lowrank_matrix_t =
        (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));

    memset(state->quadratic_objective_term->objective_lowrank_matrix_t, 0, sizeof(cu_sparse_matrix_csr_t));
    state->quadratic_objective_term->objective_lowrank_matrix_t->num_rows = problem->num_variables;
    state->quadratic_objective_term->objective_lowrank_matrix_t->num_cols = problem->num_rank_lowrank_obj;
    state->quadratic_objective_term->objective_lowrank_matrix_t->num_nonzeros =
        problem->objective_lowrank_matrix_num_nonzeros;

    CUDA_CHECK(cudaMalloc(&state->quadratic_objective_term->objective_lowrank_matrix_t->row_ptr,
                          (problem->num_variables + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->quadratic_objective_term->objective_lowrank_matrix_t->col_ind,
                          problem->objective_lowrank_matrix_num_nonzeros * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->quadratic_objective_term->objective_lowrank_matrix_t->val,
                          problem->objective_lowrank_matrix_num_nonzeros * sizeof(double)));

    size_t buffer_size = 0;
    void *buffer = nullptr;
    CUSPARSE_CHECK(
        cusparseCsr2cscEx2_bufferSize(state->sparse_handle,
                                      state->quadratic_objective_term->objective_lowrank_matrix->num_rows,
                                      state->quadratic_objective_term->objective_lowrank_matrix->num_cols,
                                      state->quadratic_objective_term->objective_lowrank_matrix->num_nonzeros,
                                      state->quadratic_objective_term->objective_lowrank_matrix->val,
                                      state->quadratic_objective_term->objective_lowrank_matrix->row_ptr,
                                      state->quadratic_objective_term->objective_lowrank_matrix->col_ind,
                                      state->quadratic_objective_term->objective_lowrank_matrix_t->val,
                                      state->quadratic_objective_term->objective_lowrank_matrix_t->row_ptr,
                                      state->quadratic_objective_term->objective_lowrank_matrix_t->col_ind,
                                      CUDA_R_64F,
                                      CUSPARSE_ACTION_NUMERIC,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                      &buffer_size));
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

    CUSPARSE_CHECK(cusparseCsr2cscEx2(state->sparse_handle,
                                      state->quadratic_objective_term->objective_lowrank_matrix->num_rows,
                                      state->quadratic_objective_term->objective_lowrank_matrix->num_cols,
                                      state->quadratic_objective_term->objective_lowrank_matrix->num_nonzeros,
                                      state->quadratic_objective_term->objective_lowrank_matrix->val,
                                      state->quadratic_objective_term->objective_lowrank_matrix->row_ptr,
                                      state->quadratic_objective_term->objective_lowrank_matrix->col_ind,
                                      state->quadratic_objective_term->objective_lowrank_matrix_t->val,
                                      state->quadratic_objective_term->objective_lowrank_matrix_t->row_ptr,
                                      state->quadratic_objective_term->objective_lowrank_matrix_t->col_ind,
                                      CUDA_R_64F,
                                      CUSPARSE_ACTION_NUMERIC,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                      buffer));
    CUDA_CHECK(cudaFree(buffer));

    state->quadratic_objective_term->spmv_ctx_R =
        pdhcg_spmv_ctx_create(state->sparse_handle,
                              state->quadratic_objective_term->num_rank_lowrank_obj,
                              state->num_variables,
                              state->quadratic_objective_term->objective_lowrank_matrix->num_nonzeros,
                              state->quadratic_objective_term->objective_lowrank_matrix->row_ptr,
                              state->quadratic_objective_term->objective_lowrank_matrix->col_ind,
                              state->quadratic_objective_term->objective_lowrank_matrix->val,
                              state->vec_primal_sol,
                              state->quadratic_objective_term->vec_Rx_prod);

    state->quadratic_objective_term->spmv_ctx_Rt =
        pdhcg_spmv_ctx_create(state->sparse_handle,
                              state->num_variables,
                              state->quadratic_objective_term->num_rank_lowrank_obj,
                              state->quadratic_objective_term->objective_lowrank_matrix_t->num_nonzeros,
                              state->quadratic_objective_term->objective_lowrank_matrix_t->row_ptr,
                              state->quadratic_objective_term->objective_lowrank_matrix_t->col_ind,
                              state->quadratic_objective_term->objective_lowrank_matrix_t->val,
                              state->quadratic_objective_term->vec_Rx_prod,
                              state->quadratic_objective_term->vec_primal_obj_prod);

    state->quadratic_objective_term->lowrank_middle_type = (int)problem->objective_lowrank_middle_kind;
    state->quadratic_objective_term->d_middle_diag = NULL;
    state->quadratic_objective_term->d_middle_dense = NULL;
    state->quadratic_objective_term->Rx_buffer = NULL;

    int rank = problem->num_rank_lowrank_obj;
    if (problem->objective_lowrank_middle_kind == PDHCG_D_DIAG && rank > 0)
    {
        ALLOC_AND_COPY(state->quadratic_objective_term->d_middle_diag,
                       problem->objective_lowrank_middle_diag,
                       (size_t)rank * sizeof(double));
    }
    else if (problem->objective_lowrank_middle_kind == PDHCG_D_DENSE && rank > 0)
    {
        size_t bytes = (size_t)rank * (size_t)rank * sizeof(double);
        ALLOC_AND_COPY(state->quadratic_objective_term->d_middle_dense, problem->objective_lowrank_middle_dense, bytes);
        ALLOC_ZERO(state->quadratic_objective_term->Rx_buffer, (size_t)rank * sizeof(double));
    }
}

static void initialize_quadratic_obj_term(pdhg_solver_state_t *state, const processed_qp_problem_t *problem)
{
    state->quadratic_objective_term = (quadratic_objective_term_t *)safe_calloc(1, sizeof(quadratic_objective_term_t));
    state->quadratic_objective_term->quad_obj_type = problem->quad_type;

    if (state->quadratic_objective_term->quad_obj_type == PDHCG_NON_Q)
        return;

    int n_local = problem->num_variables;
    int n_global = get_global_n(state);
    int n_start = get_n_start(state->grid_context);

    size_t alloc_elements = (n_global > n_local) ? n_global : n_local;
    ALLOC_ZERO(state->quadratic_objective_term->global_primal_obj_product, alloc_elements * sizeof(double));

    state->quadratic_objective_term->primal_obj_product =
        state->quadratic_objective_term->global_primal_obj_product + n_start;

    CUSPARSE_CHECK(cusparseCreateDnVec(&state->quadratic_objective_term->vec_primal_obj_prod,
                                       n_local,
                                       state->quadratic_objective_term->primal_obj_product,
                                       CUDA_R_64F));

    if (n_global > n_local)
    {
        CUSPARSE_CHECK(cusparseCreateDnVec(&state->quadratic_objective_term->vec_global_primal_obj_prod,
                                           n_global,
                                           state->quadratic_objective_term->global_primal_obj_product,
                                           CUDA_R_64F));
    }
    else
    {
        state->quadratic_objective_term->vec_global_primal_obj_prod =
            state->quadratic_objective_term->vec_primal_obj_prod;
    }

    switch (state->quadratic_objective_term->quad_obj_type)
    {
        case PDHCG_DIAG_Q:
        {
            ALLOC_AND_COPY(state->quadratic_objective_term->diagonal_objective_matrix,
                           problem->diagonal_quad_objective,
                           n_local * sizeof(double));
            state->quadratic_objective_term->objective_sparse_matrix = NULL;
            state->quadratic_objective_term->objective_lowrank_matrix = NULL;
            break;
        }

        case PDHCG_SPARSE_Q:
        {
            initialize_sparse_component_obj(state, problem);
            CUDA_CHECK(cudaGetLastError());
            state->quadratic_objective_term->diagonal_objective_matrix = NULL;
            break;
        }

        case PDHCG_LOW_RANK_Q:
        {
            initialize_lowrank_component_obj(state, problem);
            CUDA_CHECK(cudaGetLastError());
            state->quadratic_objective_term->diagonal_objective_matrix = NULL;
            break;
        }

        case PDHCG_LOW_RANK_PLUS_SPARSE_Q:
        {
            initialize_sparse_component_obj(state, problem);
            CUDA_CHECK(cudaGetLastError());
            initialize_lowrank_component_obj(state, problem);
            CUDA_CHECK(cudaGetLastError());
            state->quadratic_objective_term->diagonal_objective_matrix = NULL;
            break;
        }

        default:
            fprintf(stderr, "Error: Unknown Quadratic Objective Type detected.\n");
            exit(EXIT_FAILURE);
    }
}

static void initialize_inner_solver(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    state->inner_solver = (inner_solver_t *)safe_calloc(1, sizeof(inner_solver_t));
    state->inner_solver->has_inner_loop = false;

    int iteration_limit = params->inner_solver_parameters.iteration_limit;
    double initial_tol = params->inner_solver_parameters.initial_tolerance;
    double min_tol = params->inner_solver_parameters.min_tolerance;

    if (iteration_limit < 1)
    {
        fprintf(stderr, "Warning: inner_iter_limit (%d) < 1. Resetting to 1.\n", iteration_limit);
        iteration_limit = 1;
    }

    if (initial_tol <= 0.0)
    {
        fprintf(stderr, "Warning: inner_init_tol (%.1e) <= 0. Resetting to default 1e-3.\n", initial_tol);
        initial_tol = 1e-3;
    }

    if (min_tol <= 0.0)
    {
        fprintf(stderr, "Warning: inner_min_tol (%.1e) <= 0. Resetting to default 1e-9.\n", min_tol);
        min_tol = 1e-9;
    }

    if (!(state->quadratic_objective_term->quad_obj_type == PDHCG_NON_Q ||
          state->quadratic_objective_term->quad_obj_type == PDHCG_DIAG_Q))
    {
        ALLOC_ZERO(state->inner_solver->primal_buffer, state->num_variables * sizeof(double));
        ALLOC_ZERO(state->inner_solver->dual_buffer, state->num_constraints * sizeof(double));
    }

    switch (state->quadratic_objective_term->quad_obj_type)
    {
        case PDHCG_NON_Q:
            break;
        case PDHCG_DIAG_Q:
            break;
        case PDHCG_SPARSE_Q:
        case PDHCG_LOW_RANK_Q:
        case PDHCG_LOW_RANK_PLUS_SPARSE_Q:
            state->inner_solver->has_inner_loop = true;
            state->inner_solver->bb_step_size = (bb_step_size_t *)safe_calloc(1, sizeof(bb_step_size_t));
            ALLOC_ZERO(state->inner_solver->bb_step_size->gradient, state->num_variables * sizeof(double));
            ALLOC_ZERO(state->inner_solver->bb_step_size->direction, state->num_variables * sizeof(double));
            ALLOC_ZERO(state->inner_solver->bb_step_size->scalar_buffer, 4 * sizeof(double));

            state->inner_solver->iteration_limit = iteration_limit;
            state->inner_solver->tol = initial_tol;
            state->inner_solver->min_tol = min_tol;

            state->inner_solver->bb_step_size->precond_enabled = params->diag_jacobi_precond;
            if (params->diag_jacobi_precond)
            {
                int n = state->num_variables;
                ALLOC_ZERO(state->inner_solver->bb_step_size->diag_h_static, n * sizeof(double));
                ALLOC_ZERO(state->inner_solver->bb_step_size->m_diag, n * sizeof(double));
                ALLOC_ZERO(state->inner_solver->bb_step_size->m_inv, n * sizeof(double));
                ALLOC_ZERO(state->inner_solver->bb_step_size->Ms_buffer, n * sizeof(double));
                state->inner_solver->bb_step_size->cached_inv_tau = -1.0;
                state->inner_solver->bb_step_size->tol_scale = 1.0;

                if (state->quadratic_objective_term->quad_obj_type == PDHCG_SPARSE_Q ||
                    state->quadratic_objective_term->quad_obj_type == PDHCG_LOW_RANK_PLUS_SPARSE_Q)
                {
                    cu_sparse_matrix_csr_t *Q = state->quadratic_objective_term->objective_sparse_matrix;
                    compute_csr_diag_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                        Q->row_ptr, Q->col_ind, Q->val, state->inner_solver->bb_step_size->diag_h_static, n);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (state->quadratic_objective_term->quad_obj_type == PDHCG_LOW_RANK_Q ||
                    state->quadratic_objective_term->quad_obj_type == PDHCG_LOW_RANK_PLUS_SPARSE_Q)
                {
                    cu_sparse_matrix_csr_t *Rt = state->quadratic_objective_term->objective_lowrank_matrix_t;
                    double *out = state->inner_solver->bb_step_size->Ms_buffer;
                    int mtype = state->quadratic_objective_term->lowrank_middle_type;
                    if (mtype == 1)
                    {
                        compute_csr_row_sq_norm_weighted_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                            Rt->row_ptr, Rt->col_ind, Rt->val, state->quadratic_objective_term->d_middle_diag, out, n);
                    }
                    else if (mtype == 2)
                    {
                        int rank = state->quadratic_objective_term->num_rank_lowrank_obj;
                        compute_csr_row_quad_form_dense_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                            Rt->row_ptr,
                            Rt->col_ind,
                            Rt->val,
                            state->quadratic_objective_term->d_middle_dense,
                            rank,
                            out,
                            n);
                    }
                    else
                    {
                        compute_csr_row_sq_norm_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                            Rt->row_ptr, Rt->val, out, n);
                    }
                    CUDA_CHECK(cudaGetLastError());
                    const double one = 1.0;
                    CUBLAS_CHECK(cublasDaxpy(
                        state->blas_handle, n, &one, out, 1, state->inner_solver->bb_step_size->diag_h_static, 1));
                    CUDA_CHECK(cudaMemset(out, 0, n * sizeof(double)));
                }
            }
            break;
        default:
            fprintf(stderr, "Error: Unknown Quadratic Objective Type detected.\n");
            exit(EXIT_FAILURE);
    }
}

static void decide_problem_type(pdhg_solver_state_t *state)
{
    if (state->quadratic_objective_term->quad_obj_type == PDHCG_NON_Q)
        state->problem_type = LP;
    else
        state->problem_type = CONVEX_QP;
}

void initialize_quadratic_term_information(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    if (state->quadratic_objective_term->quad_obj_type == PDHCG_SPARSE_Q)
    {
        state->quadratic_objective_term->norm =
            estimate_maximum_eigenvalue(state->sparse_handle,
                                        state->blas_handle,
                                        state->quadratic_objective_term->objective_sparse_matrix,
                                        params->sv_max_iter,
                                        params->sv_tol,
                                        state->grid_context);
        state->quadratic_objective_term->nonconvexity =
            estimate_minimum_eigenvalue(state->sparse_handle,
                                        state->blas_handle,
                                        state->quadratic_objective_term->objective_sparse_matrix,
                                        state->quadratic_objective_term->norm,
                                        params->sv_max_iter,
                                        params->sv_tol,
                                        state->grid_context);
        return;
    }
    if (state->quadratic_objective_term->quad_obj_type == PDHCG_DIAG_Q)
    {
        double max_eigen = 0.0;
        double min_eigen = 0.0;
        double *temp_diag_host = (double *)malloc(state->num_variables * sizeof(double));
        cudaMemcpy(temp_diag_host,
                   state->quadratic_objective_term->diagonal_objective_matrix,
                   state->num_variables * sizeof(double),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < state->num_variables; i++)
        {
            double item = temp_diag_host[i];
            max_eigen = fmax(max_eigen, fabs(item));
            min_eigen = fmin(min_eigen, item);
        }
        state->quadratic_objective_term->norm = max_eigen;
        state->quadratic_objective_term->nonconvexity = min_eigen;
        free(temp_diag_host);
    }
}

static cone_proj_method_t pick_cone_proj_method(cone_type_t type, int v_dim)
{
    if (type == CONE_EXPONENTIAL || type == CONE_POWER)
        return PROJ_METHOD_THREAD;
    return (v_dim < 32) ? PROJ_METHOD_THREAD : PROJ_METHOD_WARP;
}

static bool can_use_large_rotated_soc_grid(const cone_blocks_t *cones, int cone, const rescale_info_t *rescale_info)
{
    if (cones->type[cone] != CONE_ROTATED_SOC || cones->v_dim[cone] < PDHCG_LARGE_CONE_MIN_VDIM || !rescale_info ||
        !rescale_info->var_rescale)
    {
        return false;
    }

    int start = cones->start_idx[cone];
    int k = cones->v_dim[cone];
    int s_idx = start + k;
    int t_idx = s_idx + 1;
    if (cones->is_fixed && (cones->is_fixed[s_idx] || cones->is_fixed[t_idx]))
        return false;

    double d_s = rescale_info->var_rescale[s_idx];
    double d_t = rescale_info->var_rescale[t_idx];
    if (!(d_s > 0.0) || !(d_t > 0.0) || !isfinite(d_s) || !isfinite(d_t))
        return false;

    double d_ref = rescale_info->var_rescale[start];
    bool scalar_uniform = (d_s == d_ref && d_t == d_ref);
    for (int i = 1; i < k && scalar_uniform; ++i)
        scalar_uniform = (rescale_info->var_rescale[start + i] == d_ref);
    if (scalar_uniform)
        return true;

    double d_st = sqrt(d_s * d_t);
    for (int i = 0; i < k; ++i)
    {
        if (rescale_info->var_rescale[start + i] != d_st)
            return false;
    }
    return true;
}

static bool can_use_large_standard_soc_grid(const cone_blocks_t *cones, int cone, const rescale_info_t *rescale_info)
{
    if (cones->type[cone] != CONE_STANDARD_SOC || cones->v_dim[cone] < PDHCG_LARGE_CONE_MIN_VDIM || !rescale_info ||
        !rescale_info->var_rescale)
    {
        return false;
    }

    int start = cones->start_idx[cone];
    int k = cones->v_dim[cone];
    int w_idx = start + k;
    int z_idx = w_idx + 1;
    if (cones->is_fixed && (cones->is_fixed[w_idx] || cones->is_fixed[z_idx]))
        return false;

    double d_ref = rescale_info->var_rescale[z_idx];
    if (!(d_ref > 0.0) || !isfinite(d_ref) || rescale_info->var_rescale[w_idx] != d_ref)
    {
        return false;
    }

    for (int i = 0; i < k; ++i)
    {
        if (rescale_info->var_rescale[start + i] != d_ref)
            return false;
    }
    return true;
}

static void initialize_cone_runtime(pdhg_solver_state_t *state,
                                    const qp_problem_t *working_problem,
                                    const rescale_info_t *rescale_info)
{
    memset(&state->cones, 0, sizeof(state->cones));
    state->cones.num_blocks = working_problem->cones.num_cones;

#ifdef PDHCG_COMPILE_DISTRIBUTED
    initialize_split_cones(state, working_problem, rescale_info);
#endif

    bool has_global_cones = state->cones.num_blocks > 0 || state->cones.split != NULL;
#ifdef PDHCG_COMPILE_DISTRIBUTED
    if (state->grid_context && state->grid_context->global_num_cones > 0)
        has_global_cones = true;
#endif
    state->var_set_type = has_global_cones ? VAR_SET_CONTAIN_CONIC : VAR_SET_BOX_ONLY;

    if (working_problem->cones.is_fixed)
    {
        size_t fb = (size_t)state->num_variables * sizeof(char);
        CUDA_CHECK(cudaMalloc(&state->cones.is_fixed, fb));
        CUDA_CHECK(cudaMemcpy(state->cones.is_fixed, working_problem->cones.is_fixed, fb, cudaMemcpyHostToDevice));
    }

    if (state->var_set_type == VAR_SET_CONTAIN_CONIC)
    {
        quad_obj_type_t qt = rescale_info->processed_problem ? rescale_info->processed_problem->quad_type : PDHCG_NON_Q;
        size_t vb = (size_t)state->num_variables * sizeof(double);
        if (qt != PDHCG_NON_Q)
            CUDA_CHECK(cudaMalloc(&state->cones.effective_objective_gradient, vb));
        if (qt == PDHCG_SPARSE_Q || qt == PDHCG_LOW_RANK_Q || qt == PDHCG_LOW_RANK_PLUS_SPARSE_Q)
            CUDA_CHECK(cudaMalloc(&state->cones.bb_primal_snapshot, vb));
    }

    if (state->var_set_type == VAR_SET_BOX_ONLY)
        return;

    if (state->cones.num_blocks == 0)
        return;

    int K = state->cones.num_blocks;
    const cone_blocks_t *cones = &working_problem->cones;

    cone_proj_method_t *methods = (cone_proj_method_t *)safe_malloc(K * sizeof(cone_proj_method_t));
    for (int i = 0; i < K; ++i)
    {
        methods[i] = (can_use_large_rotated_soc_grid(cones, i, rescale_info) ||
                      can_use_large_standard_soc_grid(cones, i, rescale_info))
            ? PROJ_METHOD_GRID
            : pick_cone_proj_method(cones->type[i], cones->v_dim[i]);
    }

    int bucket_count[NUM_CONE_TYPES][NUM_PROJ_METHODS] = {{0}};
    for (int i = 0; i < K; ++i)
        bucket_count[cones->type[i]][methods[i]]++;

    cone_bucket_t buckets_tmp[NUM_CONE_TYPES * NUM_PROJ_METHODS];
    int num_buckets = 0;
    int offset = 0;
    int bucket_offset[NUM_CONE_TYPES][NUM_PROJ_METHODS];
    for (int t = 0; t < NUM_CONE_TYPES; ++t)
    {
        for (int m = 0; m < NUM_PROJ_METHODS; ++m)
        {
            bucket_offset[t][m] = offset;
            if (bucket_count[t][m] > 0)
            {
                buckets_tmp[num_buckets].type = (cone_type_t)t;
                buckets_tmp[num_buckets].method = (cone_proj_method_t)m;
                buckets_tmp[num_buckets].offset = offset;
                buckets_tmp[num_buckets].count = bucket_count[t][m];
                num_buckets++;
            }
            offset += bucket_count[t][m];
        }
    }

    state->cones.num_buckets = num_buckets;
    state->cones.buckets = (cone_bucket_t *)safe_malloc((size_t)num_buckets * sizeof(cone_bucket_t));
    memcpy(state->cones.buckets, buckets_tmp, (size_t)num_buckets * sizeof(cone_bucket_t));

    size_t cb = (size_t)K * sizeof(int);
    int *start_perm = (int *)safe_malloc(cb);
    int *vdim_perm = (int *)safe_malloc(cb);
    double *alpha_perm = NULL;
    if (cones->power_alpha)
        alpha_perm = (double *)safe_malloc((size_t)K * sizeof(double));
    int write_pos[NUM_CONE_TYPES][NUM_PROJ_METHODS];
    memcpy(write_pos, bucket_offset, sizeof(write_pos));
    for (int i = 0; i < K; ++i)
    {
        int t = cones->type[i];
        int m = methods[i];
        int p = write_pos[t][m]++;
        start_perm[p] = cones->start_idx[i];
        vdim_perm[p] = cones->v_dim[i];
        if (alpha_perm)
            alpha_perm[p] = cones->power_alpha[i];
    }

    CUDA_CHECK(cudaMalloc(&state->cones.start_idx, cb));
    CUDA_CHECK(cudaMemcpy(state->cones.start_idx, start_perm, cb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state->cones.v_dim, cb));
    CUDA_CHECK(cudaMemcpy(state->cones.v_dim, vdim_perm, cb, cudaMemcpyHostToDevice));
    if (alpha_perm)
    {
        size_t ab = (size_t)K * sizeof(double);
        CUDA_CHECK(cudaMalloc(&state->cones.power_alpha, ab));
        CUDA_CHECK(cudaMemcpy(state->cones.power_alpha, alpha_perm, ab, cudaMemcpyHostToDevice));
        free(alpha_perm);
    }
    free(start_perm);
    free(vdim_perm);
    free(methods);

    size_t wb = (size_t)K * sizeof(double);
    CUDA_CHECK(cudaMalloc(&state->cones.primal_warm_start, wb));
    CUDA_CHECK(cudaMemset(state->cones.primal_warm_start, 0, wb));
    CUDA_CHECK(cudaMalloc(&state->cones.dual_warm_start, wb));
    CUDA_CHECK(cudaMemset(state->cones.dual_warm_start, 0, wb));
    CUDA_CHECK(cudaMalloc(&state->cones.complementarity_residual, wb));
    CUDA_CHECK(cudaMemset(state->cones.complementarity_residual, 0, wb));

    const double INV_SQRT2 = 0.7071067811865475;
    for (int i = 0; i < K; ++i)
    {
        int aux0 = cones->start_idx[i] + cones->v_dim[i];
        if (cones->type[i] == CONE_STANDARD_SOC)
        {
            int w_idx = aux0;
            int z_idx = aux0 + 1;
            bool w_pinned = (cones->is_fixed && cones->is_fixed[w_idx]);
            bool z_pinned = (cones->is_fixed && cones->is_fixed[z_idx]);
            double w_val = -INV_SQRT2 * rescale_info->con_bound_rescale * rescale_info->var_rescale[w_idx];
            double z_val = INV_SQRT2 * rescale_info->con_bound_rescale * rescale_info->var_rescale[z_idx];
            for (int which = 0; which < 4; ++which)
            {
                double *dst = (which == 0       ? state->initial_primal_solution
                                   : which == 1 ? state->current_primal_solution
                                   : which == 2 ? state->pdhg_primal_solution
                                                : state->reflected_primal_solution);
                if (!w_pinned)
                    CUDA_CHECK(cudaMemcpy(dst + w_idx, &w_val, sizeof(double), cudaMemcpyHostToDevice));
                if (!z_pinned)
                    CUDA_CHECK(cudaMemcpy(dst + z_idx, &z_val, sizeof(double), cudaMemcpyHostToDevice));
            }
        }
        else if (cones->type[i] == CONE_EXPONENTIAL || cones->type[i] == CONE_POWER)
        {
            /* Rely on the ALLOC_ZERO default (0, 0, 0), which is in-cone for both. */
        }
        else
        {
            int t_idx = aux0 + 1;
            bool t_pinned = (cones->is_fixed && cones->is_fixed[t_idx]);
            double t_val = rescale_info->con_bound_rescale * rescale_info->var_rescale[t_idx];
            for (int which = 0; which < 4; ++which)
            {
                double *dst = (which == 0       ? state->initial_primal_solution
                                   : which == 1 ? state->current_primal_solution
                                   : which == 2 ? state->pdhg_primal_solution
                                                : state->reflected_primal_solution);
                if (!t_pinned)
                    CUDA_CHECK(cudaMemcpy(dst + t_idx, &t_val, sizeof(double), cudaMemcpyHostToDevice));
            }
        }
    }
}

pdhg_solver_state_t *initialize_solver_state(const pdhg_parameters_t *params,
                                             const qp_problem_t *working_problem,
                                             const rescale_info_t *rescale_info,
                                             grid_context_t *grid_context)
{
    pdhg_solver_state_t *state = (pdhg_solver_state_t *)safe_calloc(1, sizeof(pdhg_solver_state_t));

    state->grid_context = grid_context;

    int n_vars = rescale_info->scaled_problem->num_variables;
    int n_cons = rescale_info->scaled_problem->num_constraints;
    size_t var_bytes = n_vars * sizeof(double);
    size_t con_bytes = n_cons * sizeof(double);

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->objective_constant = rescale_info->scaled_problem->objective_constant;

    state->constraint_matrix = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix_t = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));

    state->constraint_matrix->num_rows = n_cons;
    state->constraint_matrix->num_cols = n_vars;
    state->constraint_matrix->num_nonzeros = rescale_info->scaled_problem->constraint_matrix_num_nonzeros;

    state->constraint_matrix_t->num_rows = n_vars;
    state->constraint_matrix_t->num_cols = n_cons;
    state->constraint_matrix_t->num_nonzeros = rescale_info->scaled_problem->constraint_matrix_num_nonzeros;

    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;

    state->rescaling_time_sec = rescale_info->rescaling_time_sec;

    ALLOC_AND_COPY_CSR(state->constraint_matrix,
                       rescale_info->scaled_problem->constraint_matrix,
                       rescale_info->scaled_problem->num_constraints,
                       rescale_info->scaled_problem->constraint_matrix_num_nonzeros);

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ptr, (n_vars + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->col_ind,
                          rescale_info->scaled_problem->constraint_matrix_num_nonzeros * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->val,
                          rescale_info->scaled_problem->constraint_matrix_num_nonzeros * sizeof(double)));

    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle));
    CUBLAS_CHECK(cublasCreate(&state->blas_handle));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));
    if (state->constraint_matrix->num_nonzeros > 0)
    {
        size_t buffer_size = 0;
        void *buffer = nullptr;
        CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(state->sparse_handle,
                                                     state->constraint_matrix->num_rows,
                                                     state->constraint_matrix->num_cols,
                                                     state->constraint_matrix->num_nonzeros,
                                                     state->constraint_matrix->val,
                                                     state->constraint_matrix->row_ptr,
                                                     state->constraint_matrix->col_ind,
                                                     state->constraint_matrix_t->val,
                                                     state->constraint_matrix_t->row_ptr,
                                                     state->constraint_matrix_t->col_ind,
                                                     CUDA_R_64F,
                                                     CUSPARSE_ACTION_NUMERIC,
                                                     CUSPARSE_INDEX_BASE_ZERO,
                                                     CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                                     &buffer_size));
        CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

        CUSPARSE_CHECK(cusparseCsr2cscEx2(state->sparse_handle,
                                          state->constraint_matrix->num_rows,
                                          state->constraint_matrix->num_cols,
                                          state->constraint_matrix->num_nonzeros,
                                          state->constraint_matrix->val,
                                          state->constraint_matrix->row_ptr,
                                          state->constraint_matrix->col_ind,
                                          state->constraint_matrix_t->val,
                                          state->constraint_matrix_t->row_ptr,
                                          state->constraint_matrix_t->col_ind,
                                          CUDA_R_64F,
                                          CUSPARSE_ACTION_NUMERIC,
                                          CUSPARSE_INDEX_BASE_ZERO,
                                          CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                          buffer));

        CUDA_CHECK(cudaFree(buffer));
    }
    else
    {
        CUDA_CHECK(cudaMemset(state->constraint_matrix_t->row_ptr, 0, (state->num_variables + 1) * sizeof(int)));
    }
    CUDA_CHECK(cudaGetLastError());
    ALLOC_AND_COPY(state->variable_lower_bound, rescale_info->scaled_problem->variable_lower_bound, var_bytes);
    ALLOC_AND_COPY(state->variable_upper_bound, rescale_info->scaled_problem->variable_upper_bound, var_bytes);
    ALLOC_AND_COPY(state->objective_vector, rescale_info->scaled_problem->objective_vector, var_bytes);
    ALLOC_AND_COPY(state->constraint_lower_bound, rescale_info->scaled_problem->constraint_lower_bound, con_bytes);
    ALLOC_AND_COPY(state->constraint_upper_bound, rescale_info->scaled_problem->constraint_upper_bound, con_bytes);
    ALLOC_AND_COPY(state->constraint_rescaling, rescale_info->con_rescale, con_bytes);
    ALLOC_AND_COPY(state->variable_rescaling, rescale_info->var_rescale, var_bytes);

    state->constraint_bound_rescaling = rescale_info->con_bound_rescale;
    state->objective_vector_rescaling = rescale_info->obj_vec_rescale;

    ALLOC_ZERO(state->initial_primal_solution, var_bytes);
    ALLOC_ZERO(state->current_primal_solution, var_bytes);
    ALLOC_ZERO(state->pdhg_primal_solution, var_bytes);
    ALLOC_ZERO(state->reflected_primal_solution, var_bytes);
    ALLOC_ZERO(state->dual_product, var_bytes);
    ALLOC_ZERO(state->dual_slack, var_bytes);
    ALLOC_ZERO(state->dual_residual, var_bytes);
    ALLOC_ZERO(state->delta_primal_solution, var_bytes);

    ALLOC_ZERO(state->initial_dual_solution, con_bytes);
    ALLOC_ZERO(state->current_dual_solution, con_bytes);
    ALLOC_ZERO(state->pdhg_dual_solution, con_bytes);
    ALLOC_ZERO(state->reflected_dual_solution, con_bytes);
    ALLOC_ZERO(state->primal_product, con_bytes);
    ALLOC_ZERO(state->primal_slack, con_bytes);
    ALLOC_ZERO(state->primal_residual, con_bytes);
    ALLOC_ZERO(state->delta_dual_solution, con_bytes);

    if (working_problem->primal_start)
    {
        double *rescaled = (double *)safe_malloc(var_bytes);
        for (int i = 0; i < n_vars; ++i)
            rescaled[i] =
                working_problem->primal_start[i] * rescale_info->var_rescale[i] * rescale_info->con_bound_rescale;
        CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, rescaled, var_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->current_primal_solution, rescaled, var_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution, rescaled, var_bytes, cudaMemcpyHostToDevice));
        free(rescaled);
    }
    if (working_problem->dual_start)
    {
        double *rescaled = (double *)safe_malloc(con_bytes);
        for (int i = 0; i < n_cons; ++i)
            rescaled[i] = working_problem->dual_start[i] * rescale_info->con_rescale[i] * rescale_info->obj_vec_rescale;
        CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, rescaled, con_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->current_dual_solution, rescaled, con_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution, rescaled, con_bytes, cudaMemcpyHostToDevice));
        free(rescaled);
    }
    CUDA_CHECK(cudaGetLastError());
    double *temp_host = (double *)safe_malloc(fmax(var_bytes, con_bytes));
    for (int i = 0; i < n_cons; ++i)
        temp_host[i] = isfinite(rescale_info->scaled_problem->constraint_lower_bound[i])
            ? rescale_info->scaled_problem->constraint_lower_bound[i]
            : 0.0;
    ALLOC_AND_COPY(state->constraint_lower_bound_finite_val, temp_host, con_bytes);
    for (int i = 0; i < n_cons; ++i)
        temp_host[i] = isfinite(rescale_info->scaled_problem->constraint_upper_bound[i])
            ? rescale_info->scaled_problem->constraint_upper_bound[i]
            : 0.0;
    ALLOC_AND_COPY(state->constraint_upper_bound_finite_val, temp_host, con_bytes);
    for (int i = 0; i < n_vars; ++i)
        temp_host[i] = isfinite(rescale_info->scaled_problem->variable_lower_bound[i])
            ? rescale_info->scaled_problem->variable_lower_bound[i]
            : 0.0;
    ALLOC_AND_COPY(state->variable_lower_bound_finite_val, temp_host, var_bytes);
    for (int i = 0; i < n_vars; ++i)
        temp_host[i] = isfinite(rescale_info->scaled_problem->variable_upper_bound[i])
            ? rescale_info->scaled_problem->variable_upper_bound[i]
            : 0.0;
    ALLOC_AND_COPY(state->variable_upper_bound_finite_val, temp_host, var_bytes);
    free(temp_host);

    double sum_of_squares = 0.0;
    double max_val = 0.0;
    double val = 0.0;

    for (int i = 0; i < n_vars; ++i)
    {
        if (params->optimality_norm == NORM_TYPE_L_INF)
        {
            val = fabs(working_problem->objective_vector[i]);
            if (val > max_val)
                max_val = val;
        }
        else
        {
            sum_of_squares += working_problem->objective_vector[i] * working_problem->objective_vector[i];
        }
    }

    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        state->objective_vector_norm = max_val;
    }
    else
    {
        state->objective_vector_norm = sqrt(sum_of_squares);
    }

    sum_of_squares = 0.0;
    max_val = 0.0;
    val = 0.0;

    for (int i = 0; i < n_cons; ++i)
    {
        double lower = working_problem->constraint_lower_bound[i];
        double upper = working_problem->constraint_upper_bound[i];

        if (params->optimality_norm == NORM_TYPE_L_INF)
        {
            if (isfinite(lower) && (lower != upper))
            {
                val = fabs(lower);
                if (val > max_val)
                    max_val = val;
            }
            if (isfinite(upper))
            {
                val = fabs(upper);
                if (val > max_val)
                    max_val = val;
            }
        }
        else
        {
            if (isfinite(lower) && (lower != upper))
            {
                sum_of_squares += lower * lower;
            }
            if (isfinite(upper))
            {
                sum_of_squares += upper * upper;
            }
        }
    }

    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        state->constraint_bound_norm = max_val;
    }
    else
    {
        state->constraint_bound_norm = sqrt(sum_of_squares);
    }

    state->num_blocks_primal = (state->num_variables + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_dual = (state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_primal_dual =
        (state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    state->best_primal_dual_residual_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size = 0.0;
    state->is_this_major_iteration = false;

    CUSPARSE_CHECK(
        cusparseCreateDnVec(&state->vec_primal_sol, state->num_variables, state->pdhg_primal_solution, CUDA_R_64F));
    CUSPARSE_CHECK(
        cusparseCreateDnVec(&state->vec_dual_sol, state->num_constraints, state->pdhg_dual_solution, CUDA_R_64F));
    CUSPARSE_CHECK(
        cusparseCreateDnVec(&state->vec_primal_prod, state->num_constraints, state->primal_product, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_prod, state->num_variables, state->dual_product, CUDA_R_64F));

    state->spmv_ctx_A = pdhcg_spmv_ctx_create(state->sparse_handle,
                                              state->num_constraints,
                                              state->num_variables,
                                              state->constraint_matrix->num_nonzeros,
                                              state->constraint_matrix->row_ptr,
                                              state->constraint_matrix->col_ind,
                                              state->constraint_matrix->val,
                                              state->vec_primal_sol,
                                              state->vec_primal_prod);

    state->spmv_ctx_At = pdhcg_spmv_ctx_create(state->sparse_handle,
                                               state->num_variables,
                                               state->num_constraints,
                                               state->constraint_matrix_t->num_nonzeros,
                                               state->constraint_matrix_t->row_ptr,
                                               state->constraint_matrix_t->col_ind,
                                               state->constraint_matrix_t->val,
                                               state->vec_dual_sol,
                                               state->vec_dual_prod);

    state->num_original_variables = working_problem->num_original_variables;
    initialize_cone_runtime(state, working_problem, rescale_info);
    if (state->var_set_type == VAR_SET_CONTAIN_CONIC)
    {
        project_primal_onto_cones(state, state->initial_primal_solution);
        CUDA_CHECK(cudaGetLastError());
    }
    if (state->num_variables > 0)
    {
        project_primal_onto_bounds_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->initial_primal_solution,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(
        state->current_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(state->pdhg_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->reflected_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));

    initialize_quadratic_obj_term(state, rescale_info->processed_problem);
    initialize_quadratic_term_information(state, params);
    initialize_inner_solver(state, params);

    CUDA_CHECK(cudaMalloc(&state->ones_primal, state->num_variables * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->ones_dual, state->num_constraints * sizeof(double)));

    double *ones_primal_h = (double *)safe_malloc(state->num_variables * sizeof(double));
    for (int i = 0; i < state->num_variables; ++i)
        ones_primal_h[i] = 1.0;
    CUDA_CHECK(
        cudaMemcpy(state->ones_primal, ones_primal_h, state->num_variables * sizeof(double), cudaMemcpyHostToDevice));
    free(ones_primal_h);

    double *ones_dual_h = (double *)safe_malloc(state->num_constraints * sizeof(double));
    for (int i = 0; i < state->num_constraints; ++i)
        ones_dual_h[i] = 1.0;
    CUDA_CHECK(
        cudaMemcpy(state->ones_dual, ones_dual_h, state->num_constraints * sizeof(double), cudaMemcpyHostToDevice));
    decide_problem_type(state);
    free(ones_dual_h);
    if (params->verbose >= 2)
    {
        printf("-------------------------------------------------------------------"
               "----------"
               "----------------------\n");
        printf("Problem Type: %s\n", problem_type_to_string(state->problem_type));
        printf("Quadratic Objective Matrix Type: %s\n",
               quad_obj_type_to_string(state->quadratic_objective_term->quad_obj_type));
        if (state->quadratic_objective_term->quad_obj_type == PDHCG_SPARSE_Q ||
            state->quadratic_objective_term->quad_obj_type == PDHCG_LOW_RANK_PLUS_SPARSE_Q)
        {
            printf("L2 Norm of Sparse Q: %.3e\n", state->quadratic_objective_term->norm);
            printf("Minimum Eigenvalue of Sparse Q: %.3e\n", state->quadratic_objective_term->nonconvexity);
        }
        printf("-------------------------------------------------------------------"
               "----------"
               "----------------------\n");
        printf("%s | %s | %s | %s \n",
               "        runtime       ",
               "    objective     ",
               "  absolute residuals   ",
               "  relative residuals   ");
        printf("%s %s %s | %s %s | %s %s %s | %s %s %s \n",
               "  iter",
               "  inner",
               "  time ",
               " pr obj ",
               "  du obj ",
               " pr res",
               " du res",
               "  gap  ",
               " pr res",
               " du res",
               "  gap  ");
        printf("-------------------------------------------------------------------"
               "----------"
               "----------------------\n");
    }

    return state;
}

void pdhg_solver_state_free(pdhg_solver_state_t *state)
{
    if (state == NULL)
    {
        return;
    }

    if (state->variable_lower_bound)
        CUDA_CHECK(cudaFree(state->variable_lower_bound));
    if (state->variable_upper_bound)
        CUDA_CHECK(cudaFree(state->variable_upper_bound));
    if (state->objective_vector)
        CUDA_CHECK(cudaFree(state->objective_vector));
    if (state->constraint_matrix->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix->row_ptr));
    if (state->constraint_matrix->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix->col_ind));
    if (state->constraint_matrix->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix->val));
    if (state->constraint_matrix_t->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ptr));
    if (state->constraint_matrix_t->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->col_ind));
    if (state->constraint_matrix_t->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->val));
    if (state->constraint_lower_bound)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound));
    if (state->constraint_upper_bound)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound));
    if (state->constraint_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound_finite_val));
    if (state->constraint_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound_finite_val));
    if (state->variable_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_lower_bound_finite_val));
    if (state->variable_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_upper_bound_finite_val));
    if (state->initial_primal_solution)
        CUDA_CHECK(cudaFree(state->initial_primal_solution));
    if (state->current_primal_solution)
        CUDA_CHECK(cudaFree(state->current_primal_solution));
    if (state->pdhg_primal_solution)
        CUDA_CHECK(cudaFree(state->pdhg_primal_solution));
    if (state->reflected_primal_solution)
        CUDA_CHECK(cudaFree(state->reflected_primal_solution));
    if (state->dual_product)
        CUDA_CHECK(cudaFree(state->dual_product));
    if (state->initial_dual_solution)
        CUDA_CHECK(cudaFree(state->initial_dual_solution));
    if (state->current_dual_solution)
        CUDA_CHECK(cudaFree(state->current_dual_solution));
    if (state->pdhg_dual_solution)
        CUDA_CHECK(cudaFree(state->pdhg_dual_solution));
    if (state->reflected_dual_solution)
        CUDA_CHECK(cudaFree(state->reflected_dual_solution));
    if (state->primal_product)
        CUDA_CHECK(cudaFree(state->primal_product));
    if (state->constraint_rescaling)
        CUDA_CHECK(cudaFree(state->constraint_rescaling));
    if (state->variable_rescaling)
        CUDA_CHECK(cudaFree(state->variable_rescaling));
    if (state->primal_slack)
        CUDA_CHECK(cudaFree(state->primal_slack));
    if (state->dual_slack)
        CUDA_CHECK(cudaFree(state->dual_slack));
    if (state->primal_residual)
        CUDA_CHECK(cudaFree(state->primal_residual));
    if (state->dual_residual)
        CUDA_CHECK(cudaFree(state->dual_residual));
    if (state->delta_primal_solution)
        CUDA_CHECK(cudaFree(state->delta_primal_solution));
    if (state->delta_dual_solution)
        CUDA_CHECK(cudaFree(state->delta_dual_solution));
    if (state->ones_primal)
        CUDA_CHECK(cudaFree(state->ones_primal));
    if (state->ones_dual)
        CUDA_CHECK(cudaFree(state->ones_dual));

    if (state->quadratic_objective_term)
    {
        if (state->quadratic_objective_term->global_primal_obj_product)
            CUDA_CHECK(cudaFree(state->quadratic_objective_term->global_primal_obj_product));

        if (state->quadratic_objective_term->vec_Rx_prod)
            cusparseDestroyDnVec(state->quadratic_objective_term->vec_Rx_prod);
        if (state->quadratic_objective_term->vec_primal_obj_prod &&
            state->quadratic_objective_term->vec_primal_obj_prod !=
                state->quadratic_objective_term->vec_global_primal_obj_prod)
        {
            cusparseDestroyDnVec(state->quadratic_objective_term->vec_primal_obj_prod);
        }
        if (state->quadratic_objective_term->vec_global_primal_obj_prod)
            cusparseDestroyDnVec(state->quadratic_objective_term->vec_global_primal_obj_prod);
        if (state->quadratic_objective_term->vec_primal_obj_prod ==
            state->quadratic_objective_term->vec_global_primal_obj_prod)
        {
            state->quadratic_objective_term->vec_primal_obj_prod = nullptr;
        }
        state->quadratic_objective_term->vec_global_primal_obj_prod = nullptr;
        if (state->quadratic_objective_term->spmv_ctx_Q)
            pdhcg_spmv_ctx_destroy(state->quadratic_objective_term->spmv_ctx_Q);
        if (state->quadratic_objective_term->spmv_ctx_R)
            pdhcg_spmv_ctx_destroy(state->quadratic_objective_term->spmv_ctx_R);
        if (state->quadratic_objective_term->spmv_ctx_Rt)
            pdhcg_spmv_ctx_destroy(state->quadratic_objective_term->spmv_ctx_Rt);

        if (state->quadratic_objective_term->d_middle_diag)
            CUDA_CHECK(cudaFree(state->quadratic_objective_term->d_middle_diag));
        if (state->quadratic_objective_term->d_middle_dense)
            CUDA_CHECK(cudaFree(state->quadratic_objective_term->d_middle_dense));
        if (state->quadratic_objective_term->Rx_buffer)
            CUDA_CHECK(cudaFree(state->quadratic_objective_term->Rx_buffer));

        free(state->quadratic_objective_term);
    }

    if (state->vec_primal_sol)
        cusparseDestroyDnVec(state->vec_primal_sol);
    if (state->vec_dual_sol)
        cusparseDestroyDnVec(state->vec_dual_sol);
    if (state->vec_primal_prod)
        cusparseDestroyDnVec(state->vec_primal_prod);
    if (state->vec_dual_prod)
        cusparseDestroyDnVec(state->vec_dual_prod);

    if (state->spmv_ctx_A)
        pdhcg_spmv_ctx_destroy(state->spmv_ctx_A);
    if (state->spmv_ctx_At)
        pdhcg_spmv_ctx_destroy(state->spmv_ctx_At);

    if (state->blas_handle)
        cublasDestroy(state->blas_handle);
    if (state->sparse_handle)
        cusparseDestroy(state->sparse_handle);

    if (state->inner_solver)
    {
        if (state->inner_solver->bb_step_size)
        {
            bb_step_size_t *bb = state->inner_solver->bb_step_size;
            if (bb->gradient)
                CUDA_CHECK(cudaFree(bb->gradient));
            if (bb->direction)
                CUDA_CHECK(cudaFree(bb->direction));
            if (bb->scalar_buffer)
                CUDA_CHECK(cudaFree(bb->scalar_buffer));
            if (bb->diag_h_static)
                CUDA_CHECK(cudaFree(bb->diag_h_static));
            if (bb->m_diag)
                CUDA_CHECK(cudaFree(bb->m_diag));
            if (bb->m_inv)
                CUDA_CHECK(cudaFree(bb->m_inv));
            if (bb->Ms_buffer)
                CUDA_CHECK(cudaFree(bb->Ms_buffer));
            free(bb);
        }
        if (state->inner_solver->primal_buffer)
            CUDA_CHECK(cudaFree(state->inner_solver->primal_buffer));
        if (state->inner_solver->dual_buffer)
            CUDA_CHECK(cudaFree(state->inner_solver->dual_buffer));
        free(state->inner_solver);
    }

    if (state->cones.start_idx)
        CUDA_CHECK(cudaFree(state->cones.start_idx));
    if (state->cones.v_dim)
        CUDA_CHECK(cudaFree(state->cones.v_dim));
    if (state->cones.power_alpha)
        CUDA_CHECK(cudaFree(state->cones.power_alpha));
    if (state->cones.is_fixed)
        CUDA_CHECK(cudaFree(state->cones.is_fixed));
    if (state->cones.buckets)
        free(state->cones.buckets);
    if (state->cones.primal_warm_start)
        CUDA_CHECK(cudaFree(state->cones.primal_warm_start));
    if (state->cones.effective_objective_gradient)
        CUDA_CHECK(cudaFree(state->cones.effective_objective_gradient));
    if (state->cones.bb_primal_snapshot)
        CUDA_CHECK(cudaFree(state->cones.bb_primal_snapshot));
    if (state->cones.dual_warm_start)
        CUDA_CHECK(cudaFree(state->cones.dual_warm_start));
    if (state->cones.complementarity_residual)
        CUDA_CHECK(cudaFree(state->cones.complementarity_residual));
#ifdef PDHCG_COMPILE_DISTRIBUTED
    free_split_cones(state);
#endif

    free(state);
}

void rescale_info_free(rescale_info_t *info)
{
    if (info == NULL)
    {
        return;
    }
    if (info->processed_problem)
    {
        free_processed_qp_problem(info->processed_problem);
    }
    qp_problem_free(info->scaled_problem);
    free(info->con_rescale);
    free(info->var_rescale);

    free(info);
}
