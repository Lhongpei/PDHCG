/*
Copyright 2025-2026 Haihao Lu
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
    __global__ void finish_affine_cone_residuals_kernel(double *primal_residual,
                                                        const double *primal_product,
                                                        const double *affine_cone_offset,
                                                        const double *constraint_rescaling,
                                                        double *dual_membership,
                                                        const double *dual_membership_rescaling,
                                                        int n);

    __global__ void prepare_affine_cone_residuals_kernel(double *projection_point,
                                                         double *complementarity_residual,
                                                         const double *primal_product,
                                                         const double *affine_cone_offset,
                                                         const double *dual_solution,
                                                         const int *start_idx,
                                                         const int *v_dim,
                                                         double constraint_bound_rescaling,
                                                         int num_cones);

    __global__ void prepare_affine_cone_residuals_grid_kernel(double *projection_point,
                                                              double *complementarity_accumulator,
                                                              const double *primal_product,
                                                              const double *affine_cone_offset,
                                                              const double *dual_solution,
                                                              const int *start_idx,
                                                              const int *v_dim,
                                                              int num_cones,
                                                              int blocks_per_cone);

    __global__ void finish_affine_cone_complementarity_kernel(double *complementarity_residual,
                                                              double constraint_bound_rescaling,
                                                              int num_cones);
#ifdef __cplusplus
}
#endif
