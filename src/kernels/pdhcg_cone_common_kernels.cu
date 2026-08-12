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

#include "cone_kernel_ops.h"
#include "pdhcg_cone_common_kernels.h"
#include "utils.h"

#include <cuda_runtime.h>

__global__ void set_cone_dual_slack_kernel(double *__restrict__ dual_slack,
                                           const double *__restrict__ objective_vector,
                                           const double *__restrict__ dual_product,
                                           const int *__restrict__ start_idx,
                                           const int *__restrict__ v_dim,
                                           int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;
    int start = start_idx[blk];
    int k = v_dim[blk];
    for (int m = 0; m < k + 2; ++m)
    {
        int idx = start + m;
        dual_slack[idx] = objective_vector[idx] - dual_product[idx];
    }
}

__global__ void set_cone_dual_slack_grid_kernel(double *__restrict__ dual_slack,
                                                const double *__restrict__ objective_vector,
                                                const double *__restrict__ dual_product,
                                                const int *__restrict__ start_idx,
                                                const int *__restrict__ v_dim,
                                                int num_cones,
                                                int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    if (cone >= num_cones)
        return;

    int part = blockIdx.x - cone * blocks_per_cone;
    int start = start_idx[cone];
    int n = v_dim[cone] + 2;
    for (int m = part * blockDim.x + threadIdx.x; m < n; m += blocks_per_cone * blockDim.x)
    {
        int idx = start + m;
        dual_slack[idx] = objective_vector[idx] - dual_product[idx];
    }
}

__global__ void set_cone_dual_slack_warp_kernel(double *__restrict__ dual_slack,
                                                const double *__restrict__ objective_vector,
                                                const double *__restrict__ dual_product,
                                                const int *__restrict__ start_idx,
                                                const int *__restrict__ v_dim,
                                                int num_cones)
{
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int cone = global_thread >> 5;
    if (cone >= num_cones)
        return;

    int lane = global_thread & 31;
    int start = start_idx[cone];
    int n = v_dim[cone] + 2;
    for (int m = lane; m < n; m += 32)
    {
        int idx = start + m;
        dual_slack[idx] = objective_vector[idx] - dual_product[idx];
    }
}

__global__ void recompute_reflected_at_cone_kernel(double *__restrict__ reflected_primal,
                                                   const double *__restrict__ pdhg_primal,
                                                   const double *__restrict__ current_primal,
                                                   const int *__restrict__ start_idx,
                                                   const int *__restrict__ v_dim,
                                                   int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;
    int start = start_idx[blk];
    int k = v_dim[blk];
    for (int m = 0; m < k + 2; ++m)
    {
        int idx = start + m;
        reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
    }
}

__global__ void recompute_reflected_at_cone_warp_kernel(double *__restrict__ reflected_primal,
                                                        const double *__restrict__ pdhg_primal,
                                                        const double *__restrict__ current_primal,
                                                        const int *__restrict__ start_idx,
                                                        const int *__restrict__ v_dim,
                                                        int num_cones)
{
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int cone = global_thread >> 5;
    if (cone >= num_cones)
        return;

    int lane = global_thread & 31;
    int start = start_idx[cone];
    int n = v_dim[cone] + 2;
    for (int m = lane; m < n; m += 32)
    {
        int idx = start + m;
        reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
    }
}

__global__ void recompute_reflected_at_cone_grid_kernel(double *__restrict__ reflected_primal,
                                                        const double *__restrict__ pdhg_primal,
                                                        const double *__restrict__ current_primal,
                                                        const int *__restrict__ start_idx,
                                                        const int *__restrict__ v_dim,
                                                        int num_cones,
                                                        int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    if (cone >= num_cones)
        return;

    int part = blockIdx.x - cone * blocks_per_cone;
    int start = start_idx[cone];
    int n = v_dim[cone] + 2;
    for (int m = part * blockDim.x + threadIdx.x; m < n; m += blocks_per_cone * blockDim.x)
    {
        int idx = start + m;
        reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
    }
}

__global__ void recompute_reflected_at_cone_block_kernel(double *__restrict__ reflected_primal,
                                                         const double *__restrict__ pdhg_primal,
                                                         const double *__restrict__ current_primal,
                                                         const int *__restrict__ start_idx,
                                                         const int *__restrict__ v_dim,
                                                         int num_cones)
{
    int cone = blockIdx.x;
    if (cone >= num_cones)
        return;
    int start = start_idx[cone];
    int length = v_dim[cone] + 2;
    for (int slot = threadIdx.x; slot < length; slot += blockDim.x)
    {
        int index = start + slot;
        reflected_primal[index] = 2.0 * pdhg_primal[index] - current_primal[index];
    }
}

__global__ void clear_cone_residual_grid_kernel(double *__restrict__ dual_residual,
                                                const int *__restrict__ start_idx,
                                                const int *__restrict__ v_dim,
                                                int num_cones,
                                                int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    if (cone >= num_cones)
        return;
    int part = blockIdx.x - cone * blocks_per_cone;
    int start = start_idx[cone];
    int length = v_dim[cone] + 2;
    for (int slot = part * blockDim.x + threadIdx.x; slot < length; slot += blocks_per_cone * blockDim.x)
        dual_residual[start + slot] = 0.0;
}

static void launch_projected_mapping_only_dual_impl(
    double *dual_residual, const int *start_idx, const int *v_dim, int count, int blocks_per_cone)
{
    clear_cone_residual_grid_kernel<<<count * blocks_per_cone, THREADS_PER_BLOCK>>>(
        dual_residual, start_idx, v_dim, count, blocks_per_cone);
}

void launch_block_projected_mapping_only_dual(double *dr,
                                              double *cr,
                                              const double *obj,
                                              const double *dp,
                                              const double *vr,
                                              const double *ps,
                                              double *ws,
                                              const int *si,
                                              const int *vd,
                                              const double *pa,
                                              const char *isf,
                                              int n)
{
    (void)cr;
    (void)obj;
    (void)dp;
    (void)vr;
    (void)ps;
    (void)ws;
    (void)pa;
    (void)isf;
    launch_projected_mapping_only_dual_impl(dr, si, vd, n, 1);
}

void launch_grid_projected_mapping_only_dual(double *dr,
                                             double *cr,
                                             const double *obj,
                                             const double *dp,
                                             const double *vr,
                                             const double *ps,
                                             double *ws,
                                             const int *si,
                                             const int *vd,
                                             const double *pa,
                                             const char *isf,
                                             int n)
{
    (void)cr;
    (void)obj;
    (void)dp;
    (void)vr;
    (void)ps;
    (void)ws;
    (void)pa;
    (void)isf;
    launch_projected_mapping_only_dual_impl(dr, si, vd, n, PDHCG_LARGE_CONE_BLOCKS_PER_CONE);
}

void launch_cone_reflection(cone_proj_method_t method,
                            double *reflected_primal,
                            const double *pdhg_primal,
                            const double *current_primal,
                            const int *start_idx,
                            const int *v_dim,
                            int count)
{
    int threads = THREADS_PER_BLOCK;
    if (method == PROJ_METHOD_GRID || method == PROJ_METHOD_GRID_WEIGHTED)
    {
        int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
        recompute_reflected_at_cone_grid_kernel<<<count * blocks_per_cone, threads>>>(
            reflected_primal, pdhg_primal, current_primal, start_idx, v_dim, count, blocks_per_cone);
    }
    else if (method == PROJ_METHOD_BLOCK)
    {
        recompute_reflected_at_cone_block_kernel<<<count, threads>>>(
            reflected_primal, pdhg_primal, current_primal, start_idx, v_dim, count);
    }
    else if (method == PROJ_METHOD_WARP)
    {
        int blocks = (count * 32 + threads - 1) / threads;
        recompute_reflected_at_cone_warp_kernel<<<blocks, threads>>>(
            reflected_primal, pdhg_primal, current_primal, start_idx, v_dim, count);
    }
    else
    {
        int blocks = (count + threads - 1) / threads;
        recompute_reflected_at_cone_kernel<<<blocks, threads>>>(
            reflected_primal, pdhg_primal, current_primal, start_idx, v_dim, count);
    }
}

void launch_cone_dual_slack(cone_proj_method_t method,
                            double *dual_slack,
                            const double *objective_vector,
                            const double *dual_product,
                            const int *start_idx,
                            const int *v_dim,
                            int count)
{
    int threads = THREADS_PER_BLOCK;
    if (method == PROJ_METHOD_GRID || method == PROJ_METHOD_GRID_WEIGHTED)
    {
        int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
        set_cone_dual_slack_grid_kernel<<<count * blocks_per_cone, threads>>>(
            dual_slack, objective_vector, dual_product, start_idx, v_dim, count, blocks_per_cone);
    }
    else if (method == PROJ_METHOD_BLOCK)
    {
        set_cone_dual_slack_grid_kernel<<<count, threads>>>(
            dual_slack, objective_vector, dual_product, start_idx, v_dim, count, 1);
    }
    else if (method == PROJ_METHOD_WARP)
    {
        int blocks = (count * 32 + threads - 1) / threads;
        set_cone_dual_slack_warp_kernel<<<blocks, threads>>>(
            dual_slack, objective_vector, dual_product, start_idx, v_dim, count);
    }
    else
    {
        int blocks = (count + threads - 1) / threads;
        set_cone_dual_slack_kernel<<<blocks, threads>>>(
            dual_slack, objective_vector, dual_product, start_idx, v_dim, count);
    }
}
