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
#include "distributed_conic.h"
#include "pdhcg_kernels.cuh"
#include "utils.h"
#include <cuda_runtime.h>

typedef void (*cone_proj_launcher_t)(double *primal,
                                     const double *var_rescale,
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
                                         const double *var_rescale,
                                         const double *primal_solution,
                                         double *warm_start,
                                         const int *start_idx,
                                         const int *v_dim,
                                         const double *power_alpha,
                                         const char *is_fixed,
                                         int count);

static void launch_rotated_thread_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_rotated_soc_kernel<<<b, t>>>(p, vr, ws, si, vd, isf, n);
}
static void launch_rotated_warp_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n * 32 + t - 1) / t;
    project_rotated_soc_warp_kernel<<<b, t>>>(p, vr, ws, si, vd, isf, n);
}
static void launch_rotated_block_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    project_rotated_soc_block_kernel<<<n, THREADS_PER_BLOCK>>>(p, vr, NULL, 0.0, ws, si, vd, isf, n);
}
static void launch_rotated_grid_weighted_impl(double *p,
                                              const double *vr,
                                              const double *qd,
                                              double tau,
                                              double *ws,
                                              const int *si,
                                              const int *vd,
                                              const char *isf,
                                              int n)
{
    int threads = THREADS_PER_BLOCK;
    int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    int blocks = n * blocks_per_cone;
    CUDA_CHECK(cudaMemsetAsync(ws + n, 0, (size_t)5 * n * sizeof(double)));
    initialize_rotated_soc_grid_weighted_kernel<<<blocks, threads>>>(
        p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
    finalize_rotated_soc_grid_weighted_initialization_kernel<<<(n + threads - 1) / threads, threads>>>(
        p, vr, qd, tau, ws, si, vd, isf, n);
    for (int iteration = 0; iteration < PDHCG_CONE_GRID_ROOT_ITERATIONS; ++iteration)
    {
        CUDA_CHECK(cudaMemsetAsync(ws + n, 0, (size_t)2 * n * sizeof(double)));
        reduce_rotated_soc_grid_weighted_root_kernel<<<blocks, threads>>>(
            p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
        finalize_rotated_soc_grid_weighted_root_kernel<<<(n + threads - 1) / threads, threads>>>(
            p, vr, qd, tau, ws, si, vd, isf, n);
    }
    CUDA_CHECK(cudaMemsetAsync(ws + n, 0, (size_t)2 * n * sizeof(double)));
    reduce_rotated_soc_grid_axis_objective_kernel<<<blocks, threads>>>(
        p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
    finalize_rotated_soc_grid_axis_objective_kernel<<<(n + threads - 1) / threads, threads>>>(
        p, vr, qd, tau, ws, si, vd, n);
    apply_rotated_soc_grid_weighted_kernel<<<blocks, threads>>>(p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
}
static void launch_rotated_grid_weighted_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    launch_rotated_grid_weighted_impl(p, vr, NULL, 0.0, ws, si, vd, isf, n);
}
static void launch_rotated_grid_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)vr;
    (void)pa;
    (void)isf;
    int t = THREADS_PER_BLOCK;
    int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    int b = n * blocks_per_cone;
    CUDA_CHECK(cudaMemsetAsync(ws, 0, (size_t)n * sizeof(double)));
    project_rotated_soc_grid_reduce_kernel<<<b, t>>>(p, ws, si, vd, n, blocks_per_cone);
    project_rotated_soc_grid_finalize_kernel<<<(n + t - 1) / t, t>>>(p, ws, si, vd, n);
    project_rotated_soc_grid_apply_kernel<<<b, t>>>(p, ws, si, vd, n, blocks_per_cone);
}
static void launch_standard_thread_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_standard_soc_kernel<<<b, t>>>(p, vr, ws, si, vd, isf, n);
}
static void launch_standard_warp_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n * 32 + t - 1) / t;
    project_standard_soc_warp_kernel<<<b, t>>>(p, vr, ws, si, vd, isf, n);
}
static void launch_standard_block_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    project_standard_soc_block_kernel<<<n, THREADS_PER_BLOCK>>>(p, vr, NULL, 0.0, ws, si, vd, isf, n);
}
static void launch_standard_grid_weighted_impl(double *p,
                                               const double *vr,
                                               const double *qd,
                                               double tau,
                                               double *ws,
                                               const int *si,
                                               const int *vd,
                                               const char *isf,
                                               int n)
{
    int threads = THREADS_PER_BLOCK;
    int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    int blocks = n * blocks_per_cone;
    CUDA_CHECK(cudaMemsetAsync(ws + n, 0, (size_t)5 * n * sizeof(double)));
    initialize_standard_soc_grid_weighted_kernel<<<blocks, threads>>>(
        p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
    finalize_standard_soc_grid_weighted_initialization_kernel<<<(n + threads - 1) / threads, threads>>>(
        p, vr, qd, tau, ws, si, vd, isf, n);
    for (int iteration = 0; iteration < PDHCG_CONE_GRID_ROOT_ITERATIONS; ++iteration)
    {
        CUDA_CHECK(cudaMemsetAsync(ws + n, 0, (size_t)2 * n * sizeof(double)));
        reduce_standard_soc_grid_weighted_root_kernel<<<blocks, threads>>>(
            p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
        finalize_standard_soc_grid_weighted_root_kernel<<<(n + threads - 1) / threads, threads>>>(
            p, vr, qd, tau, ws, si, vd, n);
    }
    apply_standard_soc_grid_weighted_kernel<<<blocks, threads>>>(p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
}
static void launch_standard_grid_weighted_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    launch_standard_grid_weighted_impl(p, vr, NULL, 0.0, ws, si, vd, isf, n);
}
static void launch_standard_grid_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)vr;
    (void)pa;
    (void)isf;
    int t = THREADS_PER_BLOCK;
    int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    int b = n * blocks_per_cone;
    CUDA_CHECK(cudaMemsetAsync(ws, 0, (size_t)n * sizeof(double)));
    project_standard_soc_grid_reduce_kernel<<<b, t>>>(p, ws, si, vd, n, blocks_per_cone);
    project_standard_soc_grid_finalize_kernel<<<(n + t - 1) / t, t>>>(p, ws, si, vd, n);
    project_standard_soc_grid_apply_kernel<<<b, t>>>(p, ws, si, vd, n, blocks_per_cone);
}
static void launch_exp_thread_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_exp_cone_kernel<<<b, t>>>(p, vr, ws, si, vd, isf, n);
}
static void launch_power_thread_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_power_cone_kernel<<<b, t>>>(p, vr, ws, si, vd, pa, isf, n);
}

static const cone_proj_launcher_t proj_launch_table[NUM_CONE_TYPES][NUM_PROJ_METHODS] = {
    [CONE_ROTATED_SOC] =
        {
            [PROJ_METHOD_THREAD] = launch_rotated_thread_proj,
            [PROJ_METHOD_WARP] = launch_rotated_warp_proj,
            [PROJ_METHOD_BLOCK] = launch_rotated_block_proj,
            [PROJ_METHOD_GRID] = launch_rotated_grid_proj,
            [PROJ_METHOD_GRID_WEIGHTED] = launch_rotated_grid_weighted_proj,
        },
    [CONE_STANDARD_SOC] =
        {
            [PROJ_METHOD_THREAD] = launch_standard_thread_proj,
            [PROJ_METHOD_WARP] = launch_standard_warp_proj,
            [PROJ_METHOD_BLOCK] = launch_standard_block_proj,
            [PROJ_METHOD_GRID] = launch_standard_grid_proj,
            [PROJ_METHOD_GRID_WEIGHTED] = launch_standard_grid_weighted_proj,
        },
    [CONE_EXPONENTIAL] =
        {
            [PROJ_METHOD_THREAD] = launch_exp_thread_proj,
            [PROJ_METHOD_WARP] = NULL,
            [PROJ_METHOD_BLOCK] = NULL,
            [PROJ_METHOD_GRID] = NULL,
            [PROJ_METHOD_GRID_WEIGHTED] = NULL,
        },
    [CONE_POWER] =
        {
            [PROJ_METHOD_THREAD] = launch_power_thread_proj,
            [PROJ_METHOD_WARP] = NULL,
            [PROJ_METHOD_BLOCK] = NULL,
            [PROJ_METHOD_GRID] = NULL,
            [PROJ_METHOD_GRID_WEIGHTED] = NULL,
        },
};

static void launch_rotated_thread_dual(double *dr,
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
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    compute_cone_dual_residual_kernel<<<b, t>>>(dr, cr, obj, dp, vr, ps, ws, si, vd, isf, n);
}
static void launch_rotated_warp_dual(double *dr,
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
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n * 32 + t - 1) / t;
    compute_cone_dual_residual_warp_kernel<<<b, t>>>(dr, cr, obj, dp, vr, ps, ws, si, vd, isf, n);
}
static void launch_rotated_grid_dual(double *dr,
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
    (void)ps;
    (void)pa;
    (void)isf;
    int t = THREADS_PER_BLOCK;
    int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    int b = n * blocks_per_cone;
    CUDA_CHECK(cudaMemsetAsync(ws, 0, (size_t)n * sizeof(double)));
    compute_cone_dual_residual_grid_reduce_kernel<<<b, t>>>(obj, dp, ws, si, vd, n, blocks_per_cone);
    compute_cone_dual_residual_grid_finalize_kernel<<<(n + t - 1) / t, t>>>(dr, obj, dp, vr, ws, si, vd, n);
    compute_cone_dual_residual_grid_apply_kernel<<<b, t>>>(dr, obj, dp, vr, ws, si, vd, n, blocks_per_cone);
}
static void launch_standard_thread_dual(double *dr,
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
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    compute_cone_dual_residual_standard_kernel<<<b, t>>>(dr, cr, obj, dp, vr, ps, ws, si, vd, isf, n);
}
static void launch_standard_warp_dual(double *dr,
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
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n * 32 + t - 1) / t;
    compute_cone_dual_residual_standard_warp_kernel<<<b, t>>>(dr, cr, obj, dp, vr, ps, ws, si, vd, isf, n);
}
static void launch_standard_grid_dual(double *dr,
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
    (void)ps;
    (void)pa;
    (void)isf;
    int t = THREADS_PER_BLOCK;
    int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    int b = n * blocks_per_cone;
    CUDA_CHECK(cudaMemsetAsync(ws, 0, (size_t)n * sizeof(double)));
    compute_cone_dual_residual_standard_grid_reduce_kernel<<<b, t>>>(obj, dp, ws, si, vd, n, blocks_per_cone);
    compute_cone_dual_residual_standard_grid_finalize_kernel<<<(n + t - 1) / t, t>>>(dr, obj, dp, vr, ws, si, vd, n);
    compute_cone_dual_residual_standard_grid_apply_kernel<<<b, t>>>(dr, obj, dp, vr, ws, si, vd, n, blocks_per_cone);
}
static void launch_exp_thread_dual(double *dr,
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
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    compute_cone_dual_residual_exp_kernel<<<b, t>>>(dr, cr, obj, dp, vr, ps, ws, si, vd, isf, n);
}
static void launch_power_thread_dual(double *dr,
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
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    compute_cone_dual_residual_power_kernel<<<b, t>>>(dr, cr, obj, dp, vr, ps, ws, si, vd, pa, isf, n);
}

static void launch_projected_mapping_only_dual_impl(
    double *dual_residual, const int *start_idx, const int *v_dim, int count, int blocks_per_cone)
{
    clear_cone_residual_grid_kernel<<<count * blocks_per_cone, THREADS_PER_BLOCK>>>(
        dual_residual, start_idx, v_dim, count, blocks_per_cone);
}

static void launch_block_projected_mapping_only_dual(double *dr,
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

static void launch_grid_projected_mapping_only_dual(double *dr,
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

typedef void (*cone_proj_diag_q_launcher_t)(double *pdhg_primal,
                                            double *reflected_primal,
                                            const double *current_primal,
                                            const double *var_rescale,
                                            const double *Q_diag,
                                            double tau,
                                            double *warm_start,
                                            const int *start_idx,
                                            const int *v_dim,
                                            const double *power_alpha,
                                            const char *is_fixed,
                                            int count);

static void launch_rotated_thread_proj_diag_q(double *pp,
                                              double *rp,
                                              const double *cp,
                                              const double *vr,
                                              const double *qd,
                                              double tau,
                                              double *ws,
                                              const int *si,
                                              const int *vd,
                                              const double *pa,
                                              const char *isf,
                                              int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_rotated_soc_diag_q_kernel<<<b, t>>>(pp, rp, cp, vr, qd, tau, ws, si, vd, isf, n);
}
static void launch_rotated_block_proj_diag_q(double *pp,
                                             double *rp,
                                             const double *cp,
                                             const double *vr,
                                             const double *qd,
                                             double tau,
                                             double *ws,
                                             const int *si,
                                             const int *vd,
                                             const double *pa,
                                             const char *isf,
                                             int n)
{
    (void)pa;
    project_rotated_soc_block_kernel<<<n, THREADS_PER_BLOCK>>>(pp, vr, qd, tau, ws, si, vd, isf, n);
    recompute_reflected_at_cone_block_kernel<<<n, THREADS_PER_BLOCK>>>(rp, pp, cp, si, vd, n);
}
static void launch_rotated_grid_weighted_proj_diag_q(double *pp,
                                                     double *rp,
                                                     const double *cp,
                                                     const double *vr,
                                                     const double *qd,
                                                     double tau,
                                                     double *ws,
                                                     const int *si,
                                                     const int *vd,
                                                     const double *pa,
                                                     const char *isf,
                                                     int n)
{
    (void)pa;
    launch_rotated_grid_weighted_impl(pp, vr, qd, tau, ws, si, vd, isf, n);
    int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    recompute_reflected_at_cone_grid_kernel<<<n * blocks_per_cone, THREADS_PER_BLOCK>>>(
        rp, pp, cp, si, vd, n, blocks_per_cone);
}
static void launch_standard_thread_proj_diag_q(double *pp,
                                               double *rp,
                                               const double *cp,
                                               const double *vr,
                                               const double *qd,
                                               double tau,
                                               double *ws,
                                               const int *si,
                                               const int *vd,
                                               const double *pa,
                                               const char *isf,
                                               int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_standard_soc_diag_q_kernel<<<b, t>>>(pp, rp, cp, vr, qd, tau, ws, si, vd, isf, n);
}
static void launch_standard_block_proj_diag_q(double *pp,
                                              double *rp,
                                              const double *cp,
                                              const double *vr,
                                              const double *qd,
                                              double tau,
                                              double *ws,
                                              const int *si,
                                              const int *vd,
                                              const double *pa,
                                              const char *isf,
                                              int n)
{
    (void)pa;
    project_standard_soc_block_kernel<<<n, THREADS_PER_BLOCK>>>(pp, vr, qd, tau, ws, si, vd, isf, n);
    recompute_reflected_at_cone_block_kernel<<<n, THREADS_PER_BLOCK>>>(rp, pp, cp, si, vd, n);
}
static void launch_standard_grid_weighted_proj_diag_q(double *pp,
                                                      double *rp,
                                                      const double *cp,
                                                      const double *vr,
                                                      const double *qd,
                                                      double tau,
                                                      double *ws,
                                                      const int *si,
                                                      const int *vd,
                                                      const double *pa,
                                                      const char *isf,
                                                      int n)
{
    (void)pa;
    launch_standard_grid_weighted_impl(pp, vr, qd, tau, ws, si, vd, isf, n);
    int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    recompute_reflected_at_cone_grid_kernel<<<n * blocks_per_cone, THREADS_PER_BLOCK>>>(
        rp, pp, cp, si, vd, n, blocks_per_cone);
}
static void launch_exp_thread_proj_diag_q(double *pp,
                                          double *rp,
                                          const double *cp,
                                          const double *vr,
                                          const double *qd,
                                          double tau,
                                          double *ws,
                                          const int *si,
                                          const int *vd,
                                          const double *pa,
                                          const char *isf,
                                          int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_exp_cone_diag_q_kernel<<<b, t>>>(pp, rp, cp, vr, qd, tau, ws, si, vd, isf, n);
}
static void launch_power_thread_proj_diag_q(double *pp,
                                            double *rp,
                                            const double *cp,
                                            const double *vr,
                                            const double *qd,
                                            double tau,
                                            double *ws,
                                            const int *si,
                                            const int *vd,
                                            const double *pa,
                                            const char *isf,
                                            int n)
{
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_power_cone_diag_q_kernel<<<b, t>>>(pp, rp, cp, vr, qd, tau, ws, si, vd, pa, isf, n);
}

static const cone_proj_diag_q_launcher_t proj_diag_q_launch_table[NUM_CONE_TYPES][NUM_PROJ_METHODS] = {
    [CONE_ROTATED_SOC] =
        {
            [PROJ_METHOD_THREAD] = launch_rotated_thread_proj_diag_q,
            [PROJ_METHOD_WARP] = NULL,
            [PROJ_METHOD_BLOCK] = launch_rotated_block_proj_diag_q,
            [PROJ_METHOD_GRID] = NULL,
            [PROJ_METHOD_GRID_WEIGHTED] = launch_rotated_grid_weighted_proj_diag_q,
        },
    [CONE_STANDARD_SOC] =
        {
            [PROJ_METHOD_THREAD] = launch_standard_thread_proj_diag_q,
            [PROJ_METHOD_WARP] = NULL,
            [PROJ_METHOD_BLOCK] = launch_standard_block_proj_diag_q,
            [PROJ_METHOD_GRID] = NULL,
            [PROJ_METHOD_GRID_WEIGHTED] = launch_standard_grid_weighted_proj_diag_q,
        },
    [CONE_EXPONENTIAL] =
        {
            [PROJ_METHOD_THREAD] = launch_exp_thread_proj_diag_q,
            [PROJ_METHOD_WARP] = NULL,
            [PROJ_METHOD_BLOCK] = NULL,
            [PROJ_METHOD_GRID] = NULL,
            [PROJ_METHOD_GRID_WEIGHTED] = NULL,
        },
    [CONE_POWER] =
        {
            [PROJ_METHOD_THREAD] = launch_power_thread_proj_diag_q,
            [PROJ_METHOD_WARP] = NULL,
            [PROJ_METHOD_BLOCK] = NULL,
            [PROJ_METHOD_GRID] = NULL,
            [PROJ_METHOD_GRID_WEIGHTED] = NULL,
        },
};

static const cone_dual_res_launcher_t dual_res_launch_table[NUM_CONE_TYPES][NUM_PROJ_METHODS] = {
    [CONE_ROTATED_SOC] =
        {
            [PROJ_METHOD_THREAD] = launch_rotated_thread_dual,
            [PROJ_METHOD_WARP] = launch_rotated_warp_dual,
            [PROJ_METHOD_BLOCK] = launch_block_projected_mapping_only_dual,
            [PROJ_METHOD_GRID] = launch_rotated_grid_dual,
            [PROJ_METHOD_GRID_WEIGHTED] = launch_grid_projected_mapping_only_dual,
        },
    [CONE_STANDARD_SOC] =
        {
            [PROJ_METHOD_THREAD] = launch_standard_thread_dual,
            [PROJ_METHOD_WARP] = launch_standard_warp_dual,
            [PROJ_METHOD_BLOCK] = launch_block_projected_mapping_only_dual,
            [PROJ_METHOD_GRID] = launch_standard_grid_dual,
            [PROJ_METHOD_GRID_WEIGHTED] = launch_grid_projected_mapping_only_dual,
        },
    [CONE_EXPONENTIAL] =
        {
            [PROJ_METHOD_THREAD] = launch_exp_thread_dual,
            [PROJ_METHOD_WARP] = NULL,
            [PROJ_METHOD_BLOCK] = NULL,
            [PROJ_METHOD_GRID] = NULL,
            [PROJ_METHOD_GRID_WEIGHTED] = NULL,
        },
    [CONE_POWER] =
        {
            [PROJ_METHOD_THREAD] = launch_power_thread_dual,
            [PROJ_METHOD_WARP] = NULL,
            [PROJ_METHOD_BLOCK] = NULL,
            [PROJ_METHOD_GRID] = NULL,
            [PROJ_METHOD_GRID_WEIGHTED] = NULL,
        },
};

void project_cone_runtime(pdhg_solver_state_t *state, cone_runtime_t *runtime, double *vector, double *warm_start)
{
    const double *coordinate_rescaling =
        runtime->axis == CONE_AXIS_VARIABLE ? state->variable_rescaling : runtime->coordinate_rescaling;
    for (int b = 0; b < runtime->num_buckets; ++b)
    {
        const cone_bucket_t *bk = &runtime->buckets[b];
        const double *pa = runtime->power_alpha ? runtime->power_alpha + bk->offset : NULL;
        proj_launch_table[bk->type][bk->method](vector,
                                                coordinate_rescaling,
                                                warm_start + PDHCG_CONE_WORKSPACE_STRIDE * bk->offset,
                                                runtime->start_idx + bk->offset,
                                                runtime->v_dim + bk->offset,
                                                pa,
                                                runtime->is_fixed,
                                                bk->count);
    }
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
        cone_proj_method_t method = PROJ_METHOD_THREAD;
        if (bk->type == CONE_STANDARD_SOC && bk->method != PROJ_METHOD_THREAD)
            method = bk->method == PROJ_METHOD_GRID || bk->method == PROJ_METHOD_GRID_WEIGHTED
                ? PROJ_METHOD_GRID_WEIGHTED
                : PROJ_METHOD_BLOCK;
        else if (bk->type == CONE_ROTATED_SOC && bk->method != PROJ_METHOD_THREAD)
            method = bk->method == PROJ_METHOD_GRID || bk->method == PROJ_METHOD_GRID_WEIGHTED
                ? PROJ_METHOD_GRID_WEIGHTED
                : PROJ_METHOD_BLOCK;
        proj_diag_q_launch_table[bk->type][method](pdhg_primal,
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
        dual_res_launch_table[bk->type][bk->method](state->dual_residual,
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
    compute_split_cone_dual_residual(state, effective_obj);
}

void recompute_cone_reflection(pdhg_solver_state_t *state)
{
    int threads = THREADS_PER_BLOCK;
    for (int b = 0; b < state->cones.num_buckets; ++b)
    {
        const cone_bucket_t *bk = &state->cones.buckets[b];
        if (bk->method == PROJ_METHOD_GRID || bk->method == PROJ_METHOD_GRID_WEIGHTED)
        {
            int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
            recompute_reflected_at_cone_grid_kernel<<<bk->count * blocks_per_cone, threads>>>(
                state->reflected_primal_solution,
                state->pdhg_primal_solution,
                state->current_primal_solution,
                state->cones.start_idx + bk->offset,
                state->cones.v_dim + bk->offset,
                bk->count,
                blocks_per_cone);
        }
        else if (bk->method == PROJ_METHOD_BLOCK)
        {
            recompute_reflected_at_cone_block_kernel<<<bk->count, threads>>>(state->reflected_primal_solution,
                                                                             state->pdhg_primal_solution,
                                                                             state->current_primal_solution,
                                                                             state->cones.start_idx + bk->offset,
                                                                             state->cones.v_dim + bk->offset,
                                                                             bk->count);
        }
        else if (bk->method == PROJ_METHOD_WARP)
        {
            int blocks = (bk->count * 32 + threads - 1) / threads;
            recompute_reflected_at_cone_warp_kernel<<<blocks, threads>>>(state->reflected_primal_solution,
                                                                         state->pdhg_primal_solution,
                                                                         state->current_primal_solution,
                                                                         state->cones.start_idx + bk->offset,
                                                                         state->cones.v_dim + bk->offset,
                                                                         bk->count);
        }
        else
        {
            int blocks = (bk->count + threads - 1) / threads;
            recompute_reflected_at_cone_kernel<<<blocks, threads>>>(state->reflected_primal_solution,
                                                                    state->pdhg_primal_solution,
                                                                    state->current_primal_solution,
                                                                    state->cones.start_idx + bk->offset,
                                                                    state->cones.v_dim + bk->offset,
                                                                    bk->count);
        }
    }
    recompute_split_cone_reflected(
        state, state->reflected_primal_solution, state->pdhg_primal_solution, state->current_primal_solution);
}

void set_cone_dual_slack(pdhg_solver_state_t *state, const double *effective_obj)
{
    int threads = THREADS_PER_BLOCK;
    for (int b = 0; b < state->cones.num_buckets; ++b)
    {
        const cone_bucket_t *bk = &state->cones.buckets[b];
        if (bk->method == PROJ_METHOD_GRID || bk->method == PROJ_METHOD_GRID_WEIGHTED)
        {
            int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
            set_cone_dual_slack_grid_kernel<<<bk->count * blocks_per_cone, threads>>>(state->dual_slack,
                                                                                      effective_obj,
                                                                                      state->dual_product,
                                                                                      state->cones.start_idx +
                                                                                          bk->offset,
                                                                                      state->cones.v_dim + bk->offset,
                                                                                      bk->count,
                                                                                      blocks_per_cone);
        }
        else if (bk->method == PROJ_METHOD_BLOCK)
        {
            set_cone_dual_slack_grid_kernel<<<bk->count, threads>>>(state->dual_slack,
                                                                    effective_obj,
                                                                    state->dual_product,
                                                                    state->cones.start_idx + bk->offset,
                                                                    state->cones.v_dim + bk->offset,
                                                                    bk->count,
                                                                    1);
        }
        else if (bk->method == PROJ_METHOD_WARP)
        {
            int blocks = (bk->count * 32 + threads - 1) / threads;
            set_cone_dual_slack_warp_kernel<<<blocks, threads>>>(state->dual_slack,
                                                                 effective_obj,
                                                                 state->dual_product,
                                                                 state->cones.start_idx + bk->offset,
                                                                 state->cones.v_dim + bk->offset,
                                                                 bk->count);
        }
        else
        {
            int blocks = (bk->count + threads - 1) / threads;
            set_cone_dual_slack_kernel<<<blocks, threads>>>(state->dual_slack,
                                                            effective_obj,
                                                            state->dual_product,
                                                            state->cones.start_idx + bk->offset,
                                                            state->cones.v_dim + bk->offset,
                                                            bk->count);
        }
    }
    set_split_cone_dual_slack(state, state->dual_slack, effective_obj, state->dual_product);
}
