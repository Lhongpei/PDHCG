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

#include "internal_types.h"

#ifdef __cplusplus
extern "C"
{
#endif
    psd_projection_runtime_t *create_psd_projection_runtime(const int *start_idx,
                                                            const int *matrix_order,
                                                            int num_blocks,
                                                            int complementarity_offset);
    void free_psd_projection_runtime(psd_projection_runtime_t *runtime);

    void project_psd_cones(psd_projection_runtime_t *runtime, double *vector);
    void compute_psd_cone_dual_residual(psd_projection_runtime_t *runtime,
                                        double *dual_residual,
                                        const double *objective_vector,
                                        const double *dual_product,
                                        const double *variable_rescaling);
    void recompute_psd_cone_reflection(psd_projection_runtime_t *runtime,
                                       double *reflected_primal,
                                       const double *pdhg_primal,
                                       const double *current_primal);
    void set_psd_cone_dual_slack(psd_projection_runtime_t *runtime,
                                 double *dual_slack,
                                 const double *objective_vector,
                                 const double *dual_product);
    void prepare_psd_affine_cone_residuals(psd_projection_runtime_t *runtime,
                                           double *projection_point,
                                           double *complementarity_residual,
                                           const double *primal_product,
                                           const double *affine_cone_offset,
                                           const double *dual_solution,
                                           double constraint_bound_rescaling);
#ifdef __cplusplus
}
#endif
