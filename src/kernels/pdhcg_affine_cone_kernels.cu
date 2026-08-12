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

#include "pdhcg_affine_cone_kernels.h"

#include <cuda_runtime.h>
#include <math.h>

__global__ void finish_affine_cone_residuals_kernel(double *primal_residual,
                                                    const double *primal_product,
                                                    const double *affine_cone_offset,
                                                    const double *constraint_rescaling,
                                                    double *dual_membership,
                                                    const double *dual_membership_rescaling,
                                                    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double value = primal_product[i] + affine_cone_offset[i];
        primal_residual[i] = (value - primal_residual[i]) * constraint_rescaling[i];
        dual_membership[i] *= dual_membership_rescaling[i];
    }
}

__global__ void prepare_affine_cone_residuals_kernel(double *projection_point,
                                                     double *complementarity_residual,
                                                     const double *primal_product,
                                                     const double *affine_cone_offset,
                                                     const double *dual_solution,
                                                     const int *start_idx,
                                                     const int *v_dim,
                                                     double constraint_bound_rescaling,
                                                     int num_cones)
{
    int cone = blockIdx.x;
    if (cone >= num_cones)
        return;
    int start = start_idx[cone];
    int length = v_dim[cone] + 2;
    double dot = 0.0;
    for (int slot = threadIdx.x; slot < length; slot += blockDim.x)
    {
        int i = start + slot;
        double dual = dual_solution[i];
        projection_point[i] = -dual;
        dot += dual * (primal_product[i] + affine_cone_offset[i]);
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
        complementarity_residual[cone] = fabs(partial_sum[0]) / constraint_bound_rescaling;
}

__global__ void prepare_affine_cone_residuals_grid_kernel(double *projection_point,
                                                          double *complementarity_accumulator,
                                                          const double *primal_product,
                                                          const double *affine_cone_offset,
                                                          const double *dual_solution,
                                                          const int *start_idx,
                                                          const int *v_dim,
                                                          int num_cones,
                                                          int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    if (cone >= num_cones)
        return;
    int part = blockIdx.x - cone * blocks_per_cone;
    int start = start_idx[cone];
    int length = v_dim[cone] + 2;
    double dot = 0.0;
    for (int slot = part * blockDim.x + threadIdx.x; slot < length; slot += blocks_per_cone * blockDim.x)
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
        atomicAdd(complementarity_accumulator + cone, partial_sum[0]);
}

__global__ void finish_affine_cone_complementarity_kernel(double *complementarity_residual,
                                                          double constraint_bound_rescaling,
                                                          int num_cones)
{
    int cone = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone < num_cones)
        complementarity_residual[cone] = fabs(complementarity_residual[cone]) / constraint_bound_rescaling;
}
