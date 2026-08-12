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

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

    __global__ void set_cone_dual_slack_kernel(double *dual_slack,
                                               const double *objective_vector,
                                               const double *dual_product,
                                               const int *start_idx,
                                               const int *v_dim,
                                               int num_blocks);

    __global__ void set_cone_dual_slack_warp_kernel(double *dual_slack,
                                                    const double *objective_vector,
                                                    const double *dual_product,
                                                    const int *start_idx,
                                                    const int *v_dim,
                                                    int num_cones);

    __global__ void set_cone_dual_slack_grid_kernel(double *dual_slack,
                                                    const double *objective_vector,
                                                    const double *dual_product,
                                                    const int *start_idx,
                                                    const int *v_dim,
                                                    int num_cones,
                                                    int blocks_per_cone);

    __global__ void recompute_reflected_at_cone_kernel(double *reflected_primal,
                                                       const double *pdhg_primal,
                                                       const double *current_primal,
                                                       const int *start_idx,
                                                       const int *v_dim,
                                                       int num_blocks);

    __global__ void recompute_reflected_at_cone_warp_kernel(double *reflected_primal,
                                                            const double *pdhg_primal,
                                                            const double *current_primal,
                                                            const int *start_idx,
                                                            const int *v_dim,
                                                            int num_cones);

    __global__ void recompute_reflected_at_cone_block_kernel(double *reflected_primal,
                                                             const double *pdhg_primal,
                                                             const double *current_primal,
                                                             const int *start_idx,
                                                             const int *v_dim,
                                                             int num_cones);

    __global__ void recompute_reflected_at_cone_grid_kernel(double *reflected_primal,
                                                            const double *pdhg_primal,
                                                            const double *current_primal,
                                                            const int *start_idx,
                                                            const int *v_dim,
                                                            int num_cones,
                                                            int blocks_per_cone);

#ifdef __cplusplus
}
#endif
