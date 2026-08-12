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

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif
    __global__ void project_exp_cone_kernel(double *__restrict__ primal_solution,
                                            const double *__restrict__ variable_rescaling,
                                            double *__restrict__ warm_start,
                                            const int *__restrict__ start_idx,
                                            const int *__restrict__ v_dim,
                                            const char *__restrict__ is_fixed,
                                            int num_blocks);

    __global__ void compute_cone_dual_residual_exp_kernel(double *__restrict__ dual_residual,
                                                          double *__restrict__ complementarity_residual,
                                                          const double *__restrict__ objective_vector,
                                                          const double *__restrict__ dual_product,
                                                          const double *__restrict__ variable_rescaling,
                                                          const double *__restrict__ primal_solution,
                                                          double *__restrict__ warm_start,
                                                          const int *__restrict__ start_idx,
                                                          const int *__restrict__ v_dim,
                                                          const char *__restrict__ is_fixed,
                                                          int num_blocks);

    __global__ void project_exp_cone_diag_q_kernel(double *__restrict__ pdhg_primal,
                                                   double *__restrict__ reflected_primal,
                                                   const double *__restrict__ current_primal,
                                                   const double *__restrict__ variable_rescaling,
                                                   const double *__restrict__ Q_diag,
                                                   double tau,
                                                   double *__restrict__ warm_start,
                                                   const int *__restrict__ start_idx,
                                                   const int *__restrict__ v_dim,
                                                   const char *__restrict__ is_fixed,
                                                   int num_blocks);
#ifdef __cplusplus
}
#endif
