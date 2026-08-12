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

typedef void (*cone_proj_launcher_t)(double *primal,
                                     const double *variable_rescaling,
                                     double *warm_start,
                                     const int *start_idx,
                                     const int *v_dim,
                                     const double *power_alpha,
                                     const char *is_fixed,
                                     int count);

typedef void (*cone_dual_res_launcher_t)(double *dual_residual,
                                         double *complementarity_residual,
                                         const double *objective_vector,
                                         const double *dual_product,
                                         const double *variable_rescaling,
                                         const double *primal_solution,
                                         double *warm_start,
                                         const int *start_idx,
                                         const int *v_dim,
                                         const double *power_alpha,
                                         const char *is_fixed,
                                         int count);

typedef void (*cone_proj_diag_q_launcher_t)(double *pdhg_primal,
                                            double *reflected_primal,
                                            const double *current_primal,
                                            const double *variable_rescaling,
                                            const double *q_diag,
                                            double tau,
                                            double *warm_start,
                                            const int *start_idx,
                                            const int *v_dim,
                                            const double *power_alpha,
                                            const char *is_fixed,
                                            int count);

typedef struct
{
    cone_proj_launcher_t project[NUM_PROJ_METHODS];
    cone_proj_diag_q_launcher_t project_diag_q[NUM_PROJ_METHODS];
    cone_dual_res_launcher_t dual_residual[NUM_PROJ_METHODS];
} cone_kernel_ops_t;

extern const cone_kernel_ops_t pdhcg_rsoc_cone_kernel_ops;
extern const cone_kernel_ops_t pdhcg_soc_cone_kernel_ops;
extern const cone_kernel_ops_t pdhcg_exp_cone_kernel_ops;
extern const cone_kernel_ops_t pdhcg_power_cone_kernel_ops;

void launch_block_projected_mapping_only_dual(double *dual_residual,
                                              double *complementarity_residual,
                                              const double *objective_vector,
                                              const double *dual_product,
                                              const double *variable_rescaling,
                                              const double *primal_solution,
                                              double *warm_start,
                                              const int *start_idx,
                                              const int *v_dim,
                                              const double *power_alpha,
                                              const char *is_fixed,
                                              int count);

void launch_grid_projected_mapping_only_dual(double *dual_residual,
                                             double *complementarity_residual,
                                             const double *objective_vector,
                                             const double *dual_product,
                                             const double *variable_rescaling,
                                             const double *primal_solution,
                                             double *warm_start,
                                             const int *start_idx,
                                             const int *v_dim,
                                             const double *power_alpha,
                                             const char *is_fixed,
                                             int count);

void launch_cone_reflection(cone_proj_method_t method,
                            double *reflected_primal,
                            const double *pdhg_primal,
                            const double *current_primal,
                            const int *start_idx,
                            const int *v_dim,
                            int count);

void launch_cone_dual_slack(cone_proj_method_t method,
                            double *dual_slack,
                            const double *objective_vector,
                            const double *dual_product,
                            const int *start_idx,
                            const int *v_dim,
                            int count);
