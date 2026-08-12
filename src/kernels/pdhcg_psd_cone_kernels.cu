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

#include "pdhcg_psd_cone.h"
#include "utils.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PDHCG_PSD_BATCHED_MAX_ORDER 32
#define CUSOLVER_CHECK(call)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        cusolverStatus_t status = call;                                                                                \
        if (status != CUSOLVER_STATUS_SUCCESS)                                                                         \
        {                                                                                                              \
            fprintf(stderr, "cuSOLVER Error at %s:%d: status %d\n", __FILE__, __LINE__, (int)status);                  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

typedef struct
{
    int matrix_order;
    int packed_length;
    int count;
    int *start_idx;
    int *block_idx;
    int *packed_row;
    int *packed_col;
    double *matrices;
    double *eigenvalues;
    double *workspace;
    int workspace_size;
    int *info;
    bool use_batched_jacobi;
} psd_order_bucket_t;

struct psd_projection_runtime_s
{
    int complementarity_offset;
    int num_buckets;
    psd_order_bucket_t *buckets;
    cusolverDnHandle_t solver_handle;
    syevjInfo_t jacobi_params;
};

__global__ static void gather_svec_matrices_kernel(double *matrices,
                                                   const double *vector,
                                                   const int *start_idx,
                                                   const int *packed_row,
                                                   const int *packed_col,
                                                   int matrix_order,
                                                   int packed_length,
                                                   int count)
{
    int matrix = blockIdx.x;
    if (matrix >= count)
        return;

    const double inv_sqrt_two = 0.70710678118654752440;
    size_t matrix_offset = (size_t)matrix * (size_t)matrix_order * (size_t)matrix_order;
    int vector_offset = start_idx[matrix];
    for (int slot = threadIdx.x; slot < packed_length; slot += blockDim.x)
    {
        int row = packed_row[slot];
        int col = packed_col[slot];
        double value = vector[vector_offset + slot];
        if (row != col)
            value *= inv_sqrt_two;
        matrices[matrix_offset + row + col * matrix_order] = value;
        matrices[matrix_offset + col + row * matrix_order] = value;
    }
}

__global__ static void scatter_positive_eigenspace_kernel(double *vector,
                                                          const double *eigenvectors,
                                                          const double *eigenvalues,
                                                          const int *info,
                                                          const int *start_idx,
                                                          const int *packed_row,
                                                          const int *packed_col,
                                                          int matrix_order,
                                                          int packed_length,
                                                          int count)
{
    int matrix = blockIdx.x;
    if (matrix >= count)
        return;

    const double sqrt_two = 1.41421356237309504880;
    int vector_offset = start_idx[matrix];
    size_t matrix_offset = (size_t)matrix * (size_t)matrix_order * (size_t)matrix_order;
    size_t eigenvalue_offset = (size_t)matrix * (size_t)matrix_order;
    bool valid = info[matrix] == 0;
    for (int slot = threadIdx.x; slot < packed_length; slot += blockDim.x)
    {
        double value = 0.0;
        int row = packed_row[slot];
        int col = packed_col[slot];
        if (valid)
        {
            for (int eigenvector = 0; eigenvector < matrix_order; ++eigenvector)
            {
                double eigenvalue = fmax(eigenvalues[eigenvalue_offset + eigenvector], 0.0);
                double u_row = eigenvectors[matrix_offset + row + eigenvector * matrix_order];
                double u_col = eigenvectors[matrix_offset + col + eigenvector * matrix_order];
                value += eigenvalue * u_row * u_col;
            }
            if (row != col)
                value *= sqrt_two;
        }
        else
        {
            value = NAN;
        }
        vector[vector_offset + slot] = value;
    }
}

__global__ static void project_psd_scalars_kernel(double *vector, const int *start_idx, int count)
{
    int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block < count)
    {
        int index = start_idx[block];
        vector[index] = fmax(vector[index], 0.0);
    }
}

__global__ static void prepare_psd_dual_residual_kernel(double *dual_residual,
                                                        const double *objective_vector,
                                                        const double *dual_product,
                                                        const int *start_idx,
                                                        int packed_length,
                                                        int count)
{
    int block = blockIdx.x;
    if (block >= count)
        return;
    int start = start_idx[block];
    for (int slot = threadIdx.x; slot < packed_length; slot += blockDim.x)
    {
        int index = start + slot;
        dual_residual[index] = objective_vector[index] - dual_product[index];
    }
}

__global__ static void finish_psd_dual_residual_kernel(double *dual_residual,
                                                       const double *objective_vector,
                                                       const double *dual_product,
                                                       const double *variable_rescaling,
                                                       const int *start_idx,
                                                       int packed_length,
                                                       int count)
{
    int block = blockIdx.x;
    if (block >= count)
        return;
    int start = start_idx[block];
    for (int slot = threadIdx.x; slot < packed_length; slot += blockDim.x)
    {
        int index = start + slot;
        double residual = objective_vector[index] - dual_product[index];
        dual_residual[index] = (residual - dual_residual[index]) * variable_rescaling[index];
    }
}

__global__ static void recompute_psd_reflection_kernel(double *reflected_primal,
                                                       const double *pdhg_primal,
                                                       const double *current_primal,
                                                       const int *start_idx,
                                                       int packed_length,
                                                       int count)
{
    int block = blockIdx.x;
    if (block >= count)
        return;
    int start = start_idx[block];
    for (int slot = threadIdx.x; slot < packed_length; slot += blockDim.x)
    {
        int index = start + slot;
        reflected_primal[index] = 2.0 * pdhg_primal[index] - current_primal[index];
    }
}

__global__ static void set_psd_dual_slack_kernel(double *dual_slack,
                                                 const double *objective_vector,
                                                 const double *dual_product,
                                                 const int *start_idx,
                                                 int packed_length,
                                                 int count)
{
    int block = blockIdx.x;
    if (block >= count)
        return;
    int start = start_idx[block];
    for (int slot = threadIdx.x; slot < packed_length; slot += blockDim.x)
    {
        int index = start + slot;
        dual_slack[index] = objective_vector[index] - dual_product[index];
    }
}

__global__ static void prepare_psd_affine_residuals_kernel(double *projection_point,
                                                           double *complementarity_residual,
                                                           const double *primal_product,
                                                           const double *affine_cone_offset,
                                                           const double *dual_solution,
                                                           const int *start_idx,
                                                           const int *block_idx,
                                                           int packed_length,
                                                           int complementarity_offset,
                                                           double constraint_bound_rescaling,
                                                           int count)
{
    int block = blockIdx.x;
    if (block >= count)
        return;
    int start = start_idx[block];
    double dot = 0.0;
    for (int slot = threadIdx.x; slot < packed_length; slot += blockDim.x)
    {
        int index = start + slot;
        double dual = dual_solution[index];
        projection_point[index] = -dual;
        dot += dual * (primal_product[index] + affine_cone_offset[index]);
    }

    extern __shared__ double partial_sum[];
    partial_sum[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        int output = complementarity_offset + block_idx[block];
        complementarity_residual[output] = fabs(partial_sum[0]) / constraint_bound_rescaling;
    }
}

static void initialize_bucket(psd_projection_runtime_t *runtime,
                              psd_order_bucket_t *bucket,
                              int matrix_order,
                              int count,
                              const int *start_idx,
                              const int *block_idx)
{
    bucket->matrix_order = matrix_order;
    bucket->packed_length = (int)((long long)matrix_order * (matrix_order + 1LL) / 2LL);
    bucket->count = count;

    size_t block_bytes = (size_t)count * sizeof(int);
    ALLOC_AND_COPY(bucket->start_idx, start_idx, block_bytes);
    ALLOC_AND_COPY(bucket->block_idx, block_idx, block_bytes);

    if (matrix_order == 1)
        return;

    int *packed_row = (int *)safe_malloc((size_t)bucket->packed_length * sizeof(int));
    int *packed_col = (int *)safe_malloc((size_t)bucket->packed_length * sizeof(int));
    int slot = 0;
    for (int col = 0; col < matrix_order; ++col)
    {
        for (int row = col; row < matrix_order; ++row)
        {
            packed_row[slot] = row;
            packed_col[slot] = col;
            ++slot;
        }
    }
    size_t packed_index_bytes = (size_t)bucket->packed_length * sizeof(int);
    ALLOC_AND_COPY(bucket->packed_row, packed_row, packed_index_bytes);
    ALLOC_AND_COPY(bucket->packed_col, packed_col, packed_index_bytes);
    free(packed_row);
    free(packed_col);

    size_t matrix_entries = (size_t)count * (size_t)matrix_order * (size_t)matrix_order;
    size_t eigenvalue_entries = (size_t)count * (size_t)matrix_order;
    CUDA_CHECK(cudaMalloc(&bucket->matrices, matrix_entries * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bucket->eigenvalues, eigenvalue_entries * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bucket->info, block_bytes));

    bucket->use_batched_jacobi = matrix_order <= PDHCG_PSD_BATCHED_MAX_ORDER;
    if (bucket->use_batched_jacobi)
    {
        CUSOLVER_CHECK(cusolverDnDsyevjBatched_bufferSize(runtime->solver_handle,
                                                          CUSOLVER_EIG_MODE_VECTOR,
                                                          CUBLAS_FILL_MODE_LOWER,
                                                          matrix_order,
                                                          bucket->matrices,
                                                          matrix_order,
                                                          bucket->eigenvalues,
                                                          &bucket->workspace_size,
                                                          runtime->jacobi_params,
                                                          count));
    }
    else
    {
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(runtime->solver_handle,
                                                   CUSOLVER_EIG_MODE_VECTOR,
                                                   CUBLAS_FILL_MODE_LOWER,
                                                   matrix_order,
                                                   bucket->matrices,
                                                   matrix_order,
                                                   bucket->eigenvalues,
                                                   &bucket->workspace_size));
    }
    if (bucket->workspace_size > 0)
        CUDA_CHECK(cudaMalloc(&bucket->workspace, (size_t)bucket->workspace_size * sizeof(double)));
}

psd_projection_runtime_t *
create_psd_projection_runtime(const int *start_idx, const int *matrix_order, int num_blocks, int complementarity_offset)
{
    if (num_blocks <= 0)
        return NULL;

    psd_projection_runtime_t *runtime = (psd_projection_runtime_t *)safe_calloc(1, sizeof(psd_projection_runtime_t));
    runtime->complementarity_offset = complementarity_offset;
    CUSOLVER_CHECK(cusolverDnCreate(&runtime->solver_handle));
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&runtime->jacobi_params));
    CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(runtime->jacobi_params, 1e-12));
    CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(runtime->jacobi_params, 100));
    CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(runtime->jacobi_params, 1));

    int *unique_orders = (int *)safe_malloc((size_t)num_blocks * sizeof(int));
    int *bucket_counts = (int *)safe_calloc((size_t)num_blocks, sizeof(int));
    for (int block = 0; block < num_blocks; ++block)
    {
        int bucket = 0;
        while (bucket < runtime->num_buckets && unique_orders[bucket] != matrix_order[block])
            ++bucket;
        if (bucket == runtime->num_buckets)
            unique_orders[runtime->num_buckets++] = matrix_order[block];
        ++bucket_counts[bucket];
    }

    runtime->buckets = (psd_order_bucket_t *)safe_calloc((size_t)runtime->num_buckets, sizeof(psd_order_bucket_t));
    for (int bucket = 0; bucket < runtime->num_buckets; ++bucket)
    {
        int count = bucket_counts[bucket];
        int *starts = (int *)safe_malloc((size_t)count * sizeof(int));
        int *indices = (int *)safe_malloc((size_t)count * sizeof(int));
        int output = 0;
        for (int block = 0; block < num_blocks; ++block)
        {
            if (matrix_order[block] == unique_orders[bucket])
            {
                starts[output] = start_idx[block];
                indices[output] = block;
                ++output;
            }
        }
        initialize_bucket(runtime, &runtime->buckets[bucket], unique_orders[bucket], count, starts, indices);
        free(starts);
        free(indices);
    }
    free(unique_orders);
    free(bucket_counts);
    return runtime;
}

void free_psd_projection_runtime(psd_projection_runtime_t *runtime)
{
    if (!runtime)
        return;
    for (int bucket_idx = 0; bucket_idx < runtime->num_buckets; ++bucket_idx)
    {
        psd_order_bucket_t *bucket = &runtime->buckets[bucket_idx];
        if (bucket->start_idx)
            CUDA_CHECK(cudaFree(bucket->start_idx));
        if (bucket->block_idx)
            CUDA_CHECK(cudaFree(bucket->block_idx));
        if (bucket->packed_row)
            CUDA_CHECK(cudaFree(bucket->packed_row));
        if (bucket->packed_col)
            CUDA_CHECK(cudaFree(bucket->packed_col));
        if (bucket->matrices)
            CUDA_CHECK(cudaFree(bucket->matrices));
        if (bucket->eigenvalues)
            CUDA_CHECK(cudaFree(bucket->eigenvalues));
        if (bucket->workspace)
            CUDA_CHECK(cudaFree(bucket->workspace));
        if (bucket->info)
            CUDA_CHECK(cudaFree(bucket->info));
    }
    free(runtime->buckets);
    CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(runtime->jacobi_params));
    CUSOLVER_CHECK(cusolverDnDestroy(runtime->solver_handle));
    free(runtime);
}

static void project_psd_bucket(psd_projection_runtime_t *runtime, psd_order_bucket_t *bucket, double *vector)
{
    int threads = THREADS_PER_BLOCK;
    if (bucket->matrix_order == 1)
    {
        int blocks = (bucket->count + threads - 1) / threads;
        project_psd_scalars_kernel<<<blocks, threads>>>(vector, bucket->start_idx, bucket->count);
        return;
    }

    gather_svec_matrices_kernel<<<bucket->count, threads>>>(bucket->matrices,
                                                            vector,
                                                            bucket->start_idx,
                                                            bucket->packed_row,
                                                            bucket->packed_col,
                                                            bucket->matrix_order,
                                                            bucket->packed_length,
                                                            bucket->count);
    if (bucket->use_batched_jacobi)
    {
        CUSOLVER_CHECK(cusolverDnDsyevjBatched(runtime->solver_handle,
                                               CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_LOWER,
                                               bucket->matrix_order,
                                               bucket->matrices,
                                               bucket->matrix_order,
                                               bucket->eigenvalues,
                                               bucket->workspace,
                                               bucket->workspace_size,
                                               bucket->info,
                                               runtime->jacobi_params,
                                               bucket->count));
    }
    else
    {
        size_t matrix_stride = (size_t)bucket->matrix_order * (size_t)bucket->matrix_order;
        for (int matrix = 0; matrix < bucket->count; ++matrix)
        {
            CUSOLVER_CHECK(cusolverDnDsyevd(runtime->solver_handle,
                                            CUSOLVER_EIG_MODE_VECTOR,
                                            CUBLAS_FILL_MODE_LOWER,
                                            bucket->matrix_order,
                                            bucket->matrices + matrix * matrix_stride,
                                            bucket->matrix_order,
                                            bucket->eigenvalues + (size_t)matrix * (size_t)bucket->matrix_order,
                                            bucket->workspace,
                                            bucket->workspace_size,
                                            bucket->info + matrix));
        }
    }
    scatter_positive_eigenspace_kernel<<<bucket->count, threads>>>(vector,
                                                                   bucket->matrices,
                                                                   bucket->eigenvalues,
                                                                   bucket->info,
                                                                   bucket->start_idx,
                                                                   bucket->packed_row,
                                                                   bucket->packed_col,
                                                                   bucket->matrix_order,
                                                                   bucket->packed_length,
                                                                   bucket->count);
}

void project_psd_cones(psd_projection_runtime_t *runtime, double *vector)
{
    if (!runtime)
        return;
    for (int bucket = 0; bucket < runtime->num_buckets; ++bucket)
        project_psd_bucket(runtime, &runtime->buckets[bucket], vector);
    CUDA_CHECK(cudaGetLastError());
}

void compute_psd_cone_dual_residual(psd_projection_runtime_t *runtime,
                                    double *dual_residual,
                                    const double *objective_vector,
                                    const double *dual_product,
                                    const double *variable_rescaling)
{
    if (!runtime)
        return;
    for (int bucket_idx = 0; bucket_idx < runtime->num_buckets; ++bucket_idx)
    {
        psd_order_bucket_t *bucket = &runtime->buckets[bucket_idx];
        prepare_psd_dual_residual_kernel<<<bucket->count, THREADS_PER_BLOCK>>>(
            dual_residual, objective_vector, dual_product, bucket->start_idx, bucket->packed_length, bucket->count);
    }
    project_psd_cones(runtime, dual_residual);
    for (int bucket_idx = 0; bucket_idx < runtime->num_buckets; ++bucket_idx)
    {
        psd_order_bucket_t *bucket = &runtime->buckets[bucket_idx];
        finish_psd_dual_residual_kernel<<<bucket->count, THREADS_PER_BLOCK>>>(dual_residual,
                                                                              objective_vector,
                                                                              dual_product,
                                                                              variable_rescaling,
                                                                              bucket->start_idx,
                                                                              bucket->packed_length,
                                                                              bucket->count);
    }
    CUDA_CHECK(cudaGetLastError());
}

void recompute_psd_cone_reflection(psd_projection_runtime_t *runtime,
                                   double *reflected_primal,
                                   const double *pdhg_primal,
                                   const double *current_primal)
{
    if (!runtime)
        return;
    for (int bucket_idx = 0; bucket_idx < runtime->num_buckets; ++bucket_idx)
    {
        psd_order_bucket_t *bucket = &runtime->buckets[bucket_idx];
        recompute_psd_reflection_kernel<<<bucket->count, THREADS_PER_BLOCK>>>(
            reflected_primal, pdhg_primal, current_primal, bucket->start_idx, bucket->packed_length, bucket->count);
    }
    CUDA_CHECK(cudaGetLastError());
}

void set_psd_cone_dual_slack(psd_projection_runtime_t *runtime,
                             double *dual_slack,
                             const double *objective_vector,
                             const double *dual_product)
{
    if (!runtime)
        return;
    for (int bucket_idx = 0; bucket_idx < runtime->num_buckets; ++bucket_idx)
    {
        psd_order_bucket_t *bucket = &runtime->buckets[bucket_idx];
        set_psd_dual_slack_kernel<<<bucket->count, THREADS_PER_BLOCK>>>(
            dual_slack, objective_vector, dual_product, bucket->start_idx, bucket->packed_length, bucket->count);
    }
    CUDA_CHECK(cudaGetLastError());
}

void prepare_psd_affine_cone_residuals(psd_projection_runtime_t *runtime,
                                       double *projection_point,
                                       double *complementarity_residual,
                                       const double *primal_product,
                                       const double *affine_cone_offset,
                                       const double *dual_solution,
                                       double constraint_bound_rescaling)
{
    if (!runtime)
        return;
    for (int bucket_idx = 0; bucket_idx < runtime->num_buckets; ++bucket_idx)
    {
        psd_order_bucket_t *bucket = &runtime->buckets[bucket_idx];
        prepare_psd_affine_residuals_kernel<<<bucket->count, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(
            projection_point,
            complementarity_residual,
            primal_product,
            affine_cone_offset,
            dual_solution,
            bucket->start_idx,
            bucket->block_idx,
            bucket->packed_length,
            runtime->complementarity_offset,
            constraint_bound_rescaling,
            bucket->count);
    }
    CUDA_CHECK(cudaGetLastError());
}
