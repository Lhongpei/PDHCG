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

#include "internal_types.h"
#include "pdhcg_cone_common_kernels.h"
#include "pdhcg_rsoc_cone_kernels.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double max_abs_difference(const double *a, const double *b, int n)
{
    double error = 0.0;
    for (int i = 0; i < n; ++i)
        error = fmax(error, fabs(a[i] - b[i]));
    return error;
}

static int compare_primal_projection(int k, double s, double t)
{
    const int n = k + 2;
    const int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    double *input = (double *)malloc((size_t)n * sizeof(double));
    double *warp_result = (double *)malloc((size_t)n * sizeof(double));
    double *grid_result = (double *)malloc((size_t)n * sizeof(double));
    double *scaling = (double *)malloc((size_t)n * sizeof(double));
    if (!input || !warp_result || !grid_result || !scaling)
        return 0;

    for (int i = 0; i < k; ++i)
        input[i] = (double)((i % 17) - 8) * 1.0e-3;
    input[k] = s;
    input[k + 1] = t;
    for (int i = 0; i < n; ++i)
        scaling[i] = 1.0;

    double *d_warp = NULL;
    double *d_grid = NULL;
    double *d_scaling = NULL;
    double *d_warp_workspace = NULL;
    double *d_grid_workspace = NULL;
    int *d_start = NULL;
    int *d_vdim = NULL;
    int start = 0;

    CUDA_CHECK(cudaMalloc(&d_warp, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grid, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scaling, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_warp_workspace, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grid_workspace, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_start, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vdim, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_warp, input, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid, input, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scaling, scaling, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_start, &start, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vdim, &k, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_warp_workspace, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grid_workspace, 0, sizeof(double)));

    project_rotated_soc_warp_kernel<<<1, THREADS_PER_BLOCK>>>(
        d_warp, d_scaling, d_warp_workspace, d_start, d_vdim, NULL, 1);
    project_rotated_soc_grid_reduce_kernel<<<blocks_per_cone, THREADS_PER_BLOCK>>>(
        d_grid, d_grid_workspace, d_start, d_vdim, 1, blocks_per_cone);
    project_rotated_soc_grid_finalize_kernel<<<1, THREADS_PER_BLOCK>>>(d_grid, d_grid_workspace, d_start, d_vdim, 1);
    project_rotated_soc_grid_apply_kernel<<<blocks_per_cone, THREADS_PER_BLOCK>>>(
        d_grid, d_grid_workspace, d_start, d_vdim, 1, blocks_per_cone);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(warp_result, d_warp, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_result, d_grid, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    double error = max_abs_difference(warp_result, grid_result, n);
    int pass = error <= 1.0e-10;
    printf("large RSOC primal s=%g t=%g max_error=%.3e: %s\n", s, t, error, pass ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_warp));
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_scaling));
    CUDA_CHECK(cudaFree(d_warp_workspace));
    CUDA_CHECK(cudaFree(d_grid_workspace));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_vdim));
    free(input);
    free(warp_result);
    free(grid_result);
    free(scaling);
    return pass;
}

static int compare_dual_residual(int k)
{
    const int n = k + 2;
    const int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    double *objective = (double *)malloc((size_t)n * sizeof(double));
    double *dual_product = (double *)malloc((size_t)n * sizeof(double));
    double *scaling = (double *)malloc((size_t)n * sizeof(double));
    double *warp_result = (double *)malloc((size_t)n * sizeof(double));
    double *grid_result = (double *)malloc((size_t)n * sizeof(double));
    if (!objective || !dual_product || !scaling || !warp_result || !grid_result)
        return 0;

    for (int i = 0; i < k; ++i)
    {
        objective[i] = (double)((i % 13) - 6) * 2.0e-3;
        dual_product[i] = (double)((i % 7) - 3) * 5.0e-4;
        scaling[i] = 1.0;
    }
    objective[k] = 0.3;
    objective[k + 1] = -0.1;
    dual_product[k] = -0.2;
    dual_product[k + 1] = 0.05;
    scaling[k] = 1.0;
    scaling[k + 1] = 1.0;

    double *d_objective = NULL;
    double *d_dual_product = NULL;
    double *d_scaling = NULL;
    double *d_primal = NULL;
    double *d_warp_result = NULL;
    double *d_complementarity = NULL;
    double *d_grid_result = NULL;
    double *d_warp_workspace = NULL;
    double *d_grid_workspace = NULL;
    int *d_start = NULL;
    int *d_vdim = NULL;
    int start = 0;

    CUDA_CHECK(cudaMalloc(&d_objective, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dual_product, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scaling, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_primal, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_warp_result, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_complementarity, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grid_result, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_warp_workspace, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grid_workspace, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_start, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vdim, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_objective, objective, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dual_product, dual_product, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scaling, scaling, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_primal, 0, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_warp_result, 0, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_complementarity, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grid_result, 0, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_warp_workspace, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grid_workspace, 0, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_start, &start, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vdim, &k, sizeof(int), cudaMemcpyHostToDevice));

    compute_cone_dual_residual_warp_kernel<<<1, THREADS_PER_BLOCK>>>(d_warp_result,
                                                                     d_complementarity,
                                                                     d_objective,
                                                                     d_dual_product,
                                                                     d_scaling,
                                                                     d_primal,
                                                                     d_warp_workspace,
                                                                     d_start,
                                                                     d_vdim,
                                                                     NULL,
                                                                     1);
    compute_cone_dual_residual_grid_reduce_kernel<<<blocks_per_cone, THREADS_PER_BLOCK>>>(
        d_objective, d_dual_product, d_grid_workspace, d_start, d_vdim, 1, blocks_per_cone);
    compute_cone_dual_residual_grid_finalize_kernel<<<1, THREADS_PER_BLOCK>>>(
        d_grid_result, d_objective, d_dual_product, d_scaling, d_grid_workspace, d_start, d_vdim, 1);
    compute_cone_dual_residual_grid_apply_kernel<<<blocks_per_cone, THREADS_PER_BLOCK>>>(
        d_grid_result, d_objective, d_dual_product, d_scaling, d_grid_workspace, d_start, d_vdim, 1, blocks_per_cone);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(warp_result, d_warp_result, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_result, d_grid_result, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    double error = max_abs_difference(warp_result, grid_result, n);
    int pass = error <= 1.0e-10;
    printf("large RSOC dual residual max_error=%.3e: %s\n", error, pass ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_objective));
    CUDA_CHECK(cudaFree(d_dual_product));
    CUDA_CHECK(cudaFree(d_scaling));
    CUDA_CHECK(cudaFree(d_primal));
    CUDA_CHECK(cudaFree(d_warp_result));
    CUDA_CHECK(cudaFree(d_complementarity));
    CUDA_CHECK(cudaFree(d_grid_result));
    CUDA_CHECK(cudaFree(d_warp_workspace));
    CUDA_CHECK(cudaFree(d_grid_workspace));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_vdim));
    free(objective);
    free(dual_product);
    free(scaling);
    free(warp_result);
    free(grid_result);
    return pass;
}

static int compare_vector_updates(int k)
{
    const int n = k + 2;
    const int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    const size_t bytes = (size_t)n * sizeof(double);
    double *current = (double *)malloc(bytes);
    double *pdhg = (double *)malloc(bytes);
    double *objective = (double *)malloc(bytes);
    double *dual_product = (double *)malloc(bytes);
    double *thread_result = (double *)malloc(bytes);
    double *grid_result = (double *)malloc(bytes);
    if (!current || !pdhg || !objective || !dual_product || !thread_result || !grid_result)
        return 0;

    for (int i = 0; i < n; ++i)
    {
        current[i] = (double)((i % 19) - 9) * 0.02;
        pdhg[i] = (double)((i % 23) - 11) * 0.03;
        objective[i] = (double)((i % 29) - 14) * 0.04;
        dual_product[i] = (double)((i % 31) - 15) * 0.01;
    }

    double *d_current = NULL;
    double *d_pdhg = NULL;
    double *d_objective = NULL;
    double *d_dual_product = NULL;
    double *d_thread_result = NULL;
    double *d_grid_result = NULL;
    int *d_start = NULL;
    int *d_vdim = NULL;
    int start = 0;

    CUDA_CHECK(cudaMalloc(&d_current, bytes));
    CUDA_CHECK(cudaMalloc(&d_pdhg, bytes));
    CUDA_CHECK(cudaMalloc(&d_objective, bytes));
    CUDA_CHECK(cudaMalloc(&d_dual_product, bytes));
    CUDA_CHECK(cudaMalloc(&d_thread_result, bytes));
    CUDA_CHECK(cudaMalloc(&d_grid_result, bytes));
    CUDA_CHECK(cudaMalloc(&d_start, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vdim, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_current, current, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pdhg, pdhg, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_objective, objective, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dual_product, dual_product, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_start, &start, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vdim, &k, sizeof(int), cudaMemcpyHostToDevice));

    recompute_reflected_at_cone_kernel<<<1, THREADS_PER_BLOCK>>>(
        d_thread_result, d_pdhg, d_current, d_start, d_vdim, 1);
    recompute_reflected_at_cone_grid_kernel<<<blocks_per_cone, THREADS_PER_BLOCK>>>(
        d_grid_result, d_pdhg, d_current, d_start, d_vdim, 1, blocks_per_cone);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(thread_result, d_thread_result, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_result, d_grid_result, bytes, cudaMemcpyDeviceToHost));
    double reflected_error = max_abs_difference(thread_result, grid_result, n);

    recompute_reflected_at_cone_warp_kernel<<<1, THREADS_PER_BLOCK>>>(
        d_grid_result, d_pdhg, d_current, d_start, d_vdim, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(grid_result, d_grid_result, bytes, cudaMemcpyDeviceToHost));
    double reflected_warp_error = max_abs_difference(thread_result, grid_result, n);

    set_cone_dual_slack_kernel<<<1, THREADS_PER_BLOCK>>>(
        d_thread_result, d_objective, d_dual_product, d_start, d_vdim, 1);
    set_cone_dual_slack_grid_kernel<<<blocks_per_cone, THREADS_PER_BLOCK>>>(
        d_grid_result, d_objective, d_dual_product, d_start, d_vdim, 1, blocks_per_cone);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(thread_result, d_thread_result, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_result, d_grid_result, bytes, cudaMemcpyDeviceToHost));
    double slack_error = max_abs_difference(thread_result, grid_result, n);

    set_cone_dual_slack_warp_kernel<<<1, THREADS_PER_BLOCK>>>(
        d_grid_result, d_objective, d_dual_product, d_start, d_vdim, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(grid_result, d_grid_result, bytes, cudaMemcpyDeviceToHost));
    double slack_warp_error = max_abs_difference(thread_result, grid_result, n);

    int pass = reflected_error == 0.0 && reflected_warp_error == 0.0 && slack_error == 0.0 && slack_warp_error == 0.0;
    printf("large RSOC vector updates grid=(%.3e, %.3e) warp=(%.3e, %.3e): %s\n",
           reflected_error,
           slack_error,
           reflected_warp_error,
           slack_warp_error,
           pass ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_pdhg));
    CUDA_CHECK(cudaFree(d_objective));
    CUDA_CHECK(cudaFree(d_dual_product));
    CUDA_CHECK(cudaFree(d_thread_result));
    CUDA_CHECK(cudaFree(d_grid_result));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_vdim));
    free(current);
    free(pdhg);
    free(objective);
    free(dual_product);
    free(thread_result);
    free(grid_result);
    return pass;
}

int main(void)
{
    const int k = 65536;
    int pass = 1;
    pass &= compare_primal_projection(k, 2.0, 2.0);
    pass &= compare_primal_projection(k, 0.2, 0.8);
    pass &= compare_primal_projection(k, -2.0, -2.0);
    pass &= compare_dual_residual(k);
    pass &= compare_vector_updates(k);
    return pass ? 0 : 1;
}
