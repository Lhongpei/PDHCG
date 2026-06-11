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

#pragma once

#include "cusparse_compat.h"
#include "distributed_interface.h"
#include "pdhcg_types.h"
#include "spmv_backend.h"
#include <cublas_v2.h>
#include <cusparse.h>

typedef struct
{
    int num_rows;
    int num_cols;
    int num_nonzeros;
    int *row_ptr;
    int *col_ind;
    double *val;
} cu_sparse_matrix_csr_t;

typedef struct
{
    cu_sparse_matrix_csr_t *objective_sparse_matrix;
    double *diagonal_objective_matrix;
    quad_obj_type_t quad_obj_type;

    pdhcg_spmv_ctx_t *spmv_ctx_Q;

    double norm;
    double nonconvexity;
    double *primal_obj_product;

    cusparseDnVecDescr_t vec_primal_obj_prod;

    // Low rank Component
    cu_sparse_matrix_csr_t *objective_lowrank_matrix;
    cu_sparse_matrix_csr_t *objective_lowrank_matrix_t;
    pdhcg_spmv_ctx_t *spmv_ctx_R;
    pdhcg_spmv_ctx_t *spmv_ctx_Rt;

    double *Rx_product;

    cusparseDnVecDescr_t vec_Rx_prod;
    cusparseDnVecDescr_t vec_RRx_prod;
    int num_rank_lowrank_obj;

    int lowrank_middle_type;
    double *d_middle_diag;
    double *d_middle_dense;
    double *Rx_buffer;

    // Buffer for Distributed Version
    double *global_primal_obj_product;
    cusparseDnVecDescr_t vec_global_primal_obj_prod;
} quadratic_objective_term_t;

typedef struct
{
    double *gradient;
    double *direction;
    double *scalar_buffer;
    bool precond_enabled;
    double *diag_h_static;
    double *m_diag;
    double *m_inv;
    double *Ms_buffer;
    double cached_inv_tau;
    double tol_scale;
} bb_step_size_t;

typedef struct
{
    bb_step_size_t *bb_step_size;
    double *primal_buffer;
    double *dual_buffer;
    double tol;
    double min_tol;
    int iteration_limit;
    int total_count;
    int has_inner_loop;
} inner_solver_t;

typedef enum
{
    LP,
    CONVEX_QP,
    NONCONVEX_QP,
    CONVEX_QCQP
} problem_type_t;

typedef struct
{
    int num_variables;
    int num_constraints;
    double *variable_lower_bound;
    double *variable_upper_bound;
    double *objective_vector;
    double objective_constant;
    quadratic_objective_term_t *quadratic_objective_term;
    cu_sparse_matrix_csr_t *constraint_matrix;
    cu_sparse_matrix_csr_t *constraint_matrix_t;
    double *constraint_lower_bound;
    double *constraint_upper_bound;
    int num_blocks_primal;
    int num_blocks_dual;
    int num_blocks_primal_dual;
    double objective_vector_norm;
    double constraint_bound_norm;
    double *constraint_lower_bound_finite_val;
    double *constraint_upper_bound_finite_val;
    double *variable_lower_bound_finite_val;
    double *variable_upper_bound_finite_val;

    double *initial_primal_solution;
    double *current_primal_solution;
    double *pdhg_primal_solution;
    double *reflected_primal_solution;
    double *dual_product;
    double *initial_dual_solution;
    double *current_dual_solution;
    double *pdhg_dual_solution;
    double *reflected_dual_solution;
    double *primal_product;
    double step_size;
    double primal_weight;
    int total_count;
    bool is_this_major_iteration;
    double primal_weight_error_sum;
    double primal_weight_last_error;
    double best_primal_weight;
    double best_primal_dual_residual_gap;

    double *constraint_rescaling;
    double *variable_rescaling;
    double constraint_bound_rescaling;
    double objective_vector_rescaling;
    double *primal_slack;
    double *dual_slack;
    double rescaling_time_sec;
    double cumulative_time_sec;

    double *primal_residual;
    double absolute_primal_residual;
    double relative_primal_residual;
    double *dual_residual;
    double absolute_dual_residual;
    double relative_dual_residual;
    double primal_objective_value;
    double dual_objective_value;
    double objective_gap;
    double relative_objective_gap;
    double max_primal_ray_infeasibility;
    double max_dual_ray_infeasibility;
    double primal_ray_linear_objective;
    double dual_ray_objective;
    termination_reason_t termination_reason;

    double *delta_primal_solution;
    double *delta_dual_solution;
    double fixed_point_error;
    double initial_fixed_point_error;
    double last_trial_fixed_point_error;
    int inner_count;

    cusparseHandle_t sparse_handle;
    cublasHandle_t blas_handle;

    pdhcg_spmv_ctx_t *spmv_ctx_A;
    pdhcg_spmv_ctx_t *spmv_ctx_At;
    cusparseDnVecDescr_t vec_primal_sol;
    cusparseDnVecDescr_t vec_dual_sol;
    cusparseDnVecDescr_t vec_primal_prod;
    cusparseDnVecDescr_t vec_dual_prod;

    double *ones_primal_d;
    double *ones_dual_d;

    double feasibility_polishing_time;
    int feasibility_iteration;

    problem_type_t problem_type;
    inner_solver_t *inner_solver;
    grid_context_t *grid_context;

    int num_cone_blocks;
    int *cone_block_start_idx_d;
    int *cone_block_v_dim_d;
    int num_original_variables;
} pdhg_solver_state_t;

typedef enum
{
    PDHCG_D_NONE = 0,
    PDHCG_D_DIAG = 1,
    PDHCG_D_DENSE = 2
} lowrank_middle_kind_t;

typedef struct
{
    int num_variables;
    int num_constraints;
    int num_rank_lowrank_obj;
    double *variable_lower_bound;
    double *variable_upper_bound;
    double *objective_vector;
    double objective_constant;

    CsrComponent *constraint_matrix;
    int constraint_matrix_num_nonzeros;

    CsrComponent *objective_sparse_matrix;
    int objective_sparse_matrix_num_nonzeros;

    CsrComponent *objective_lowrank_matrix;
    int objective_lowrank_matrix_num_nonzeros;

    lowrank_middle_kind_t objective_lowrank_middle_kind;
    double *objective_lowrank_middle_diag;
    double *objective_lowrank_middle_dense;

    double *diagonal_quad_objective;

    double *constraint_lower_bound;
    double *constraint_upper_bound;

    double *primal_start;
    double *dual_start;
    quad_obj_type_t quad_type;
} processed_qp_problem_t;

typedef struct
{
    qp_problem_t *scaled_problem;
    processed_qp_problem_t *processed_problem;
    double *con_rescale;
    double *var_rescale;
    double con_bound_rescale;
    double obj_vec_rescale;
    double rescaling_time_sec;
} rescale_info_t;
