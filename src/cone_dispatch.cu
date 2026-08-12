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

#include "cone_dispatch.h"
#include "cone_kernel_ops.h"
#include "distributed_conic.h"
#include "pdhcg_psd_cone.h"
#include "utils.h"

static_assert(CONE_ROTATED_SOC == 0 && CONE_STANDARD_SOC == 1 && CONE_EXPONENTIAL == 2 && CONE_POWER == 3 &&
                  CONE_PSD == 4 && NUM_CONE_TYPES == 5,
              "cone kernel dispatch must match cone_type_t");
static_assert(PROJ_METHOD_THREAD == 0 && PROJ_METHOD_WARP == 1 && PROJ_METHOD_BLOCK == 2 && PROJ_METHOD_GRID == 3 &&
                  PROJ_METHOD_GRID_WEIGHTED == 4 && NUM_PROJ_METHODS == 5,
              "cone kernel dispatch must match cone_proj_method_t");

static const cone_kernel_ops_t *const cone_kernel_ops_by_type[NUM_CONE_TYPES] = {
    &pdhcg_rsoc_cone_kernel_ops,
    &pdhcg_soc_cone_kernel_ops,
    &pdhcg_exp_cone_kernel_ops,
    &pdhcg_power_cone_kernel_ops,
    NULL,
};

void project_cone_runtime(pdhg_solver_state_t *state, cone_runtime_t *runtime, double *vector, double *warm_start)
{
    const double *coordinate_rescaling =
        runtime->axis == CONE_AXIS_VARIABLE ? state->variable_rescaling : runtime->coordinate_rescaling;
    for (int b = 0; b < runtime->num_buckets; ++b)
    {
        const cone_bucket_t *bk = &runtime->buckets[b];
        const double *pa = runtime->power_alpha ? runtime->power_alpha + bk->offset : NULL;
        cone_kernel_ops_by_type[bk->type]->project[bk->method](vector,
                                                               coordinate_rescaling,
                                                               warm_start + PDHCG_CONE_WORKSPACE_STRIDE * bk->offset,
                                                               runtime->start_idx + bk->offset,
                                                               runtime->v_dim + bk->offset,
                                                               pa,
                                                               runtime->is_fixed,
                                                               bk->count);
    }
    project_psd_cones(runtime->psd, vector);
    project_split_cones(state, runtime, vector);
}

void project_cone_runtime_diag_q(pdhg_solver_state_t *state, cone_runtime_t *runtime, double primal_step_size)
{
    const double *Q_diag = state->quadratic_objective_term->diagonal_objective_matrix;
    double *pdhg_primal = state->pdhg_primal_solution;
    double *reflected_primal = state->reflected_primal_solution;
    const double *current_primal = state->current_primal_solution;

    for (int b = 0; b < runtime->num_buckets; ++b)
    {
        const cone_bucket_t *bk = &runtime->buckets[b];
        const double *pa = runtime->power_alpha ? runtime->power_alpha + bk->offset : NULL;
        cone_kernel_ops_by_type[bk->type]->project_diag_q[bk->method](pdhg_primal,
                                                                      reflected_primal,
                                                                      current_primal,
                                                                      state->variable_rescaling,
                                                                      Q_diag,
                                                                      primal_step_size,
                                                                      runtime->projection_warm_start +
                                                                          PDHCG_CONE_WORKSPACE_STRIDE * bk->offset,
                                                                      runtime->start_idx + bk->offset,
                                                                      runtime->v_dim + bk->offset,
                                                                      pa,
                                                                      runtime->is_fixed,
                                                                      bk->count);
    }
    project_split_cones(state, runtime, pdhg_primal);
    recompute_split_cone_reflected(state, reflected_primal, pdhg_primal, current_primal);
}

void compute_cone_dual_residual(pdhg_solver_state_t *state, const double *effective_obj)
{
    if (state->cones.num_blocks > 0)
    {
        CUDA_CHECK(cudaMemsetAsync(
            state->cones.complementarity_residual, 0, (size_t)state->cones.num_blocks * sizeof(double)));
    }
    for (int b = 0; b < state->cones.num_buckets; ++b)
    {
        const cone_bucket_t *bk = &state->cones.buckets[b];
        const double *pa = state->cones.power_alpha ? state->cones.power_alpha + bk->offset : NULL;
        cone_kernel_ops_by_type[bk->type]->dual_residual[bk->method](state->dual_residual,
                                                                     state->cones.complementarity_residual + bk->offset,
                                                                     effective_obj,
                                                                     state->dual_product,
                                                                     state->variable_rescaling,
                                                                     state->pdhg_primal_solution,
                                                                     state->cones.residual_warm_start +
                                                                         PDHCG_CONE_WORKSPACE_STRIDE * bk->offset,
                                                                     state->cones.start_idx + bk->offset,
                                                                     state->cones.v_dim + bk->offset,
                                                                     pa,
                                                                     state->cones.is_fixed,
                                                                     bk->count);
    }
    compute_psd_cone_dual_residual(
        state->cones.psd, state->dual_residual, effective_obj, state->dual_product, state->variable_rescaling);
    compute_split_cone_dual_residual(state, effective_obj);
}

void recompute_cone_reflection(pdhg_solver_state_t *state)
{
    for (int b = 0; b < state->cones.num_buckets; ++b)
    {
        const cone_bucket_t *bk = &state->cones.buckets[b];
        launch_cone_reflection(bk->method,
                               state->reflected_primal_solution,
                               state->pdhg_primal_solution,
                               state->current_primal_solution,
                               state->cones.start_idx + bk->offset,
                               state->cones.v_dim + bk->offset,
                               bk->count);
    }
    recompute_psd_cone_reflection(state->cones.psd,
                                  state->reflected_primal_solution,
                                  state->pdhg_primal_solution,
                                  state->current_primal_solution);
    recompute_split_cone_reflected(
        state, state->reflected_primal_solution, state->pdhg_primal_solution, state->current_primal_solution);
}

void set_cone_dual_slack(pdhg_solver_state_t *state, const double *effective_obj)
{
    for (int b = 0; b < state->cones.num_buckets; ++b)
    {
        const cone_bucket_t *bk = &state->cones.buckets[b];
        launch_cone_dual_slack(bk->method,
                               state->dual_slack,
                               effective_obj,
                               state->dual_product,
                               state->cones.start_idx + bk->offset,
                               state->cones.v_dim + bk->offset,
                               bk->count);
    }
    set_psd_cone_dual_slack(state->cones.psd, state->dual_slack, effective_obj, state->dual_product);
    set_split_cone_dual_slack(state, state->dual_slack, effective_obj, state->dual_product);
}
