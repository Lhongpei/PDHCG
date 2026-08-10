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

#include "internal_types.h"
#include "pdhcg.h"
#include "pdhcg_types.h"
#include "preconditioner.h"
#include "solver.h"
#include "utils.h"

#ifdef __cplusplus
extern "C"
{
#endif
    void update_obj_product(pdhg_solver_state_t *state, double *primal_solution);
    double compute_xQx(pdhg_solver_state_t *state, double *primal_sol, double *primal_obj_product);
    void pdhg_update(pdhg_solver_state_t *state);
    void halpern_update(pdhg_solver_state_t *state, double reflection_coefficient);

    void rescale_solution(pdhg_solver_state_t *state);

    pdhcg_result_t *create_result_from_state(pdhg_solver_state_t *state, const qp_problem_t *original_problem);

    void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params);

    void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params);

    void compute_fixed_point_error(pdhg_solver_state_t *state);

    void compute_residual(pdhg_solver_state_t *state, norm_type_t optimality_norm);

    void compute_infeasibility_information(pdhg_solver_state_t *state);

    double estimate_maximum_singular_value(cusparseHandle_t sparse_handle,
                                           cublasHandle_t blas_handle,
                                           const cu_sparse_matrix_csr_t *A,
                                           const cu_sparse_matrix_csr_t *AT,
                                           int max_iterations,
                                           double tolerance,
                                           struct grid_context_s *ctx);

    double estimate_maximum_eigenvalue(cusparseHandle_t sparse_handle,
                                       cublasHandle_t blas_handle,
                                       const cu_sparse_matrix_csr_t *A,
                                       int max_iterations,
                                       double tolerance,
                                       struct grid_context_s *ctx);

    double estimate_minimum_eigenvalue(cusparseHandle_t sparse_handle,
                                       cublasHandle_t blas_handle,
                                       const cu_sparse_matrix_csr_t *A,
                                       double lambda_max,
                                       int max_iterations,
                                       double tolerance,
                                       struct grid_context_s *ctx);

#ifdef __cplusplus
}
#endif
