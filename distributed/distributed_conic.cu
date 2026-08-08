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

#include "distributed_conic.h"
#include "distributed_interface.h"
#include "distributed_types.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define DIST_CONE_THREADS 256
#define PROJECTION_STATS 3
#define RESIDUAL_STATS 10

enum
{
    RES_RV2 = 0,
    RES_R0 = 1,
    RES_R1 = 2,
    RES_DOT = 3,
    RES_N2 = 4,
    RES_XN2 = 5,
    RES_X0 = 6,
    RES_D0 = 7,
    RES_X1 = 8,
    RES_D1 = 9
};

enum
{
    RESIDUAL_MODE_ZERO = 0,
    RESIDUAL_MODE_FIXED_VECTOR = 1,
    RESIDUAL_MODE_FIXED_BALL = 2,
    RESIDUAL_MODE_FREE_CONE = 3
};

struct distributed_cone_split_s
{
    int num_cones;
    int blocks_per_cone;
    int *local_start;
    int *local_first;
    int *local_count;
    int *v_dim;
    cone_type_t *type;
    unsigned char *fixed_mask;
    double *stats;
    double *complementarity_residual;
};

static __global__ void collect_projection_stats_kernel(const double *__restrict__ primal,
                                                       const int *__restrict__ local_start,
                                                       const int *__restrict__ local_first,
                                                       const int *__restrict__ local_count,
                                                       const int *__restrict__ v_dim,
                                                       double *__restrict__ stats,
                                                       int num_cones)
{
    int cone = blockIdx.x;
    if (cone >= num_cones)
        return;

    __shared__ double partial[DIST_CONE_THREADS];
    int first = local_first[cone];
    int count = local_count[cone];
    int start = local_start[cone];
    int k = v_dim[cone];
    double sum = 0.0;
    int first_offset = (int)blockIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.y;

    for (int offset = first_offset; offset < count; offset += stride)
    {
        int relative = first + offset;
        double value = primal[start + offset];
        if (relative < k)
            sum += value * value;
        else if (relative == k)
            stats[cone * PROJECTION_STATS + 1] = value;
        else if (relative == k + 1)
            stats[cone * PROJECTION_STATS + 2] = value;
    }

    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(&stats[cone * PROJECTION_STATS], partial[0]);
}

static __global__ void apply_projection_kernel(double *__restrict__ primal,
                                               const int *__restrict__ local_start,
                                               const int *__restrict__ local_first,
                                               const int *__restrict__ local_count,
                                               const int *__restrict__ v_dim,
                                               const cone_type_t *__restrict__ type,
                                               const unsigned char *__restrict__ fixed_mask,
                                               const double *__restrict__ stats,
                                               int num_cones)
{
    int cone = blockIdx.x;
    if (cone >= num_cones)
        return;

    const double INV_SQRT2 = 0.70710678118654752440;
    int first = local_first[cone];
    int count = local_count[cone];
    int start = local_start[cone];
    int k = v_dim[cone];
    unsigned char fixed = fixed_mask[cone];
    double sum_v2 = stats[cone * PROJECTION_STATS];
    double aux0 = stats[cone * PROJECTION_STATS + 1];
    double aux1 = stats[cone * PROJECTION_STATS + 2];

    double vector_factor = 1.0;
    double new_aux0 = aux0;
    double new_aux1 = aux1;
    bool update_aux0 = false;
    bool update_aux1 = false;
    int first_offset = (int)blockIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.y;

    if (type[cone] == CONE_STANDARD_SOC)
    {
        bool aux0_fixed = (fixed & PDHCG_DIST_CONE_FIXED_AUX0) != 0;
        bool aux1_fixed = (fixed & PDHCG_DIST_CONE_FIXED_AUX1) != 0;
        if (aux0_fixed && aux1_fixed)
        {
            double radius2 = aux1 * aux1 - aux0 * aux0;
            if (!(radius2 > 0.0))
                vector_factor = 0.0;
            else if (sum_v2 > radius2)
                vector_factor = sqrt(radius2 / sum_v2);
        }
        else if (!aux0_fixed && aux1_fixed)
        {
            double norm2 = sum_v2 + aux0 * aux0;
            double radius2 = aux1 * aux1;
            if (!(radius2 > 0.0))
                vector_factor = 0.0;
            else if (norm2 > radius2)
                vector_factor = sqrt(radius2 / norm2);
            new_aux0 = aux0 * vector_factor;
            update_aux0 = true;
        }
        else if (aux0_fixed)
        {
            /* Validation permits this section only for aux0 == 0. */
            double norm = sqrt(sum_v2);
            update_aux1 = true;
            if (norm <= aux1)
            {
                vector_factor = 1.0;
            }
            else if (norm <= -aux1)
            {
                vector_factor = 0.0;
                new_aux1 = 0.0;
            }
            else
            {
                double scale = (norm + aux1) / (2.0 * norm);
                vector_factor = scale;
                new_aux1 = scale * norm;
            }
        }
        else
        {
            double norm = sqrt(sum_v2 + aux0 * aux0);
            update_aux0 = true;
            update_aux1 = true;
            if (norm <= aux1)
            {
                vector_factor = 1.0;
            }
            else if (norm <= -aux1)
            {
                vector_factor = 0.0;
                new_aux0 = 0.0;
                new_aux1 = 0.0;
            }
            else
            {
                double scale = (norm + aux1) / (2.0 * norm);
                vector_factor = scale;
                new_aux0 = scale * aux0;
                new_aux1 = scale * norm;
            }
        }
    }
    else
    {
        bool both_fixed = fixed == (PDHCG_DIST_CONE_FIXED_AUX0 | PDHCG_DIST_CONE_FIXED_AUX1);
        if (both_fixed)
        {
            double radius2 = 2.0 * aux0 * aux1;
            if (!(radius2 > 0.0))
                vector_factor = 0.0;
            else if (sum_v2 > radius2)
                vector_factor = sqrt(radius2 / sum_v2);
        }
        else
        {
            double w = (aux0 - aux1) * INV_SQRT2;
            double z = (aux0 + aux1) * INV_SQRT2;
            double norm = sqrt(sum_v2 + w * w);
            update_aux0 = true;
            update_aux1 = true;
            if (norm <= z)
            {
                vector_factor = 1.0;
            }
            else if (norm <= -z)
            {
                vector_factor = 0.0;
                new_aux0 = 0.0;
                new_aux1 = 0.0;
            }
            else
            {
                double scale = (norm + z) / (2.0 * norm);
                double new_w = scale * w;
                double new_z = scale * norm;
                vector_factor = scale;
                new_aux0 = (new_z + new_w) * INV_SQRT2;
                new_aux1 = (new_z - new_w) * INV_SQRT2;
            }
        }
    }

    for (int offset = first_offset; offset < count; offset += stride)
    {
        int relative = first + offset;
        if (relative < k)
            primal[start + offset] *= vector_factor;
        else if (relative == k && update_aux0)
            primal[start + offset] = new_aux0;
        else if (relative == k + 1 && update_aux1)
            primal[start + offset] = new_aux1;
    }
}

static __global__ void collect_residual_stats_kernel(const double *__restrict__ effective_objective,
                                                     const double *__restrict__ dual_product,
                                                     const double *__restrict__ primal,
                                                     const double *__restrict__ rescaling,
                                                     const int *__restrict__ local_start,
                                                     const int *__restrict__ local_first,
                                                     const int *__restrict__ local_count,
                                                     const int *__restrict__ v_dim,
                                                     double *__restrict__ stats,
                                                     int num_cones)
{
    int cone = blockIdx.x;
    if (cone >= num_cones)
        return;

    __shared__ double partial_r2[DIST_CONE_THREADS];
    __shared__ double partial_dot[DIST_CONE_THREADS];
    __shared__ double partial_n2[DIST_CONE_THREADS];
    __shared__ double partial_xn2[DIST_CONE_THREADS];
    int first = local_first[cone];
    int count = local_count[cone];
    int start = local_start[cone];
    int k = v_dim[cone];
    double r2 = 0.0;
    double dot = 0.0;
    double n2 = 0.0;
    double xn2 = 0.0;
    int first_offset = (int)blockIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.y;

    for (int offset = first_offset; offset < count; offset += stride)
    {
        int relative = first + offset;
        int index = start + offset;
        double r = effective_objective[index] - dual_product[index];
        double x = primal[index];
        double d = rescaling[index];
        if (relative < k)
        {
            double normal = x / (d * d);
            r2 += r * r;
            dot += r * normal;
            n2 += normal * normal;
            xn2 += (x / d) * (x / d);
        }
        else if (relative == k)
        {
            stats[cone * RESIDUAL_STATS + RES_R0] = r;
            stats[cone * RESIDUAL_STATS + RES_X0] = x;
            stats[cone * RESIDUAL_STATS + RES_D0] = d;
        }
        else if (relative == k + 1)
        {
            stats[cone * RESIDUAL_STATS + RES_R1] = r;
            stats[cone * RESIDUAL_STATS + RES_X1] = x;
            stats[cone * RESIDUAL_STATS + RES_D1] = d;
        }
    }

    partial_r2[threadIdx.x] = r2;
    partial_dot[threadIdx.x] = dot;
    partial_n2[threadIdx.x] = n2;
    partial_xn2[threadIdx.x] = xn2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            partial_r2[threadIdx.x] += partial_r2[threadIdx.x + stride];
            partial_dot[threadIdx.x] += partial_dot[threadIdx.x + stride];
            partial_n2[threadIdx.x] += partial_n2[threadIdx.x + stride];
            partial_xn2[threadIdx.x] += partial_xn2[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(&stats[cone * RESIDUAL_STATS + RES_RV2], partial_r2[0]);
        atomicAdd(&stats[cone * RESIDUAL_STATS + RES_DOT], partial_dot[0]);
        atomicAdd(&stats[cone * RESIDUAL_STATS + RES_N2], partial_n2[0]);
        atomicAdd(&stats[cone * RESIDUAL_STATS + RES_XN2], partial_xn2[0]);
    }
}

static __global__ void apply_residual_kernel(double *__restrict__ dual_residual,
                                             double *__restrict__ complementarity_residual,
                                             const double *__restrict__ effective_objective,
                                             const double *__restrict__ dual_product,
                                             const double *__restrict__ primal,
                                             const double *__restrict__ rescaling,
                                             const int *__restrict__ local_start,
                                             const int *__restrict__ local_first,
                                             const int *__restrict__ local_count,
                                             const int *__restrict__ v_dim,
                                             const cone_type_t *__restrict__ type,
                                             const unsigned char *__restrict__ fixed_mask,
                                             const double *__restrict__ stats,
                                             int num_cones)
{
    int cone = blockIdx.x;
    if (cone >= num_cones)
        return;

    const double INV_SQRT2 = 0.70710678118654752440;
    int first = local_first[cone];
    int count = local_count[cone];
    int start = local_start[cone];
    int k = v_dim[cone];
    unsigned char fixed = fixed_mask[cone];
    const double *cone_stats = stats + cone * RESIDUAL_STATS;
    double rv2 = cone_stats[RES_RV2];
    double r0 = cone_stats[RES_R0];
    double r1 = cone_stats[RES_R1];
    double x0 = cone_stats[RES_X0];
    double d0 = cone_stats[RES_D0];
    double x1 = cone_stats[RES_X1];
    double d1 = cone_stats[RES_D1];

    double vector_factor = 0.0;
    double endpoint_residual0 = 0.0;
    double endpoint_residual1 = 0.0;
    double lambda = 0.0;
    double complementarity = 0.0;
    int mode = RESIDUAL_MODE_ZERO;
    int first_offset = (int)blockIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.y;

    if (type[cone] == CONE_STANDARD_SOC)
    {
        bool aux0_fixed = (fixed & PDHCG_DIST_CONE_FIXED_AUX0) != 0;
        bool aux1_fixed = (fixed & PDHCG_DIST_CONE_FIXED_AUX1) != 0;
        if (aux0_fixed && aux1_fixed)
        {
            double w = x0 / d0;
            double z = x1 / d1;
            double radius2 = z * z - w * w;
            if (radius2 > 0.0)
            {
                double dot = cone_stats[RES_DOT];
                double n2 = cone_stats[RES_N2];
                lambda = (dot < 0.0 && n2 > 0.0) ? -dot / n2 : 0.0;
                complementarity = lambda * fmax(radius2 - cone_stats[RES_XN2], 0.0) / (2.0 * sqrt(radius2));
                mode = RESIDUAL_MODE_FIXED_VECTOR;
            }
        }
        else if (!aux0_fixed && aux1_fixed)
        {
            double normal0 = x0 / (d0 * d0);
            double dot = cone_stats[RES_DOT] + r0 * normal0;
            double n2 = cone_stats[RES_N2] + normal0 * normal0;
            double xnorm2 = cone_stats[RES_XN2] + (x0 / d0) * (x0 / d0);
            double radius = x1 / d1;
            double radius2 = radius * radius;
            if (radius2 > 0.0)
            {
                lambda = (dot < 0.0 && n2 > 0.0) ? -dot / n2 : 0.0;
                complementarity = lambda * fmax(radius2 - xnorm2, 0.0) / (2.0 * sqrt(radius2));
                mode = RESIDUAL_MODE_FIXED_BALL;
            }
        }
        else if (aux0_fixed)
        {
            /* The projected-gradient mapping handles this reduced SOC section. */
            mode = RESIDUAL_MODE_ZERO;
        }
        else
        {
            double norm = sqrt(rv2 + r0 * r0);
            double projected0;
            double projected1;
            if (norm <= r1)
            {
                vector_factor = 0.0;
                projected0 = r0;
                projected1 = r1;
            }
            else if (norm <= -r1)
            {
                vector_factor = 1.0;
                projected0 = 0.0;
                projected1 = 0.0;
            }
            else
            {
                double scale = (r1 + norm) / (2.0 * norm);
                vector_factor = 1.0 - scale;
                projected0 = scale * r0;
                projected1 = scale * norm;
            }
            endpoint_residual0 = (r0 - projected0) * d0;
            endpoint_residual1 = (r1 - projected1) * d1;
            mode = RESIDUAL_MODE_FREE_CONE;
        }
    }
    else
    {
        bool both_fixed = fixed == (PDHCG_DIST_CONE_FIXED_AUX0 | PDHCG_DIST_CONE_FIXED_AUX1);
        if (both_fixed)
        {
            double s = x0 / d0;
            double t = x1 / d1;
            double radius2 = 2.0 * s * t;
            if (radius2 > 0.0)
            {
                double dot = cone_stats[RES_DOT];
                double n2 = cone_stats[RES_N2];
                lambda = (dot < 0.0 && n2 > 0.0) ? -dot / n2 : 0.0;
                complementarity = lambda * fmax(radius2 - cone_stats[RES_XN2], 0.0) / (2.0 * sqrt(radius2));
                mode = RESIDUAL_MODE_FIXED_VECTOR;
            }
        }
        else
        {
            double rw = (r0 - r1) * INV_SQRT2;
            double rz = (r0 + r1) * INV_SQRT2;
            double norm = sqrt(rv2 + rw * rw);
            double projected_s;
            double projected_t;
            if (norm <= rz)
            {
                vector_factor = 0.0;
                projected_s = r0;
                projected_t = r1;
            }
            else if (norm <= -rz)
            {
                vector_factor = 1.0;
                projected_s = 0.0;
                projected_t = 0.0;
            }
            else
            {
                double scale = (rz + norm) / (2.0 * norm);
                double projected_w = scale * rw;
                double projected_z = scale * norm;
                vector_factor = 1.0 - scale;
                projected_s = (projected_z + projected_w) * INV_SQRT2;
                projected_t = (projected_z - projected_w) * INV_SQRT2;
            }
            endpoint_residual0 = (r0 - projected_s) * d0;
            endpoint_residual1 = (r1 - projected_t) * d1;
            mode = RESIDUAL_MODE_FREE_CONE;
        }
    }

    if (first_offset == 0)
        complementarity_residual[cone] = complementarity;

    for (int offset = first_offset; offset < count; offset += stride)
    {
        int relative = first + offset;
        int index = start + offset;
        double r = effective_objective[index] - dual_product[index];
        double d = rescaling[index];
        if (relative < k)
        {
            if (mode == RESIDUAL_MODE_FIXED_VECTOR || mode == RESIDUAL_MODE_FIXED_BALL)
            {
                double normal = primal[index] / (d * d);
                double residual = (r + lambda * normal) * d;
                dual_residual[index] = residual;
            }
            else if (mode == RESIDUAL_MODE_ZERO)
                dual_residual[index] = 0.0;
            else
                dual_residual[index] = r * vector_factor * d;
        }
        else if (relative == k)
        {
            if (mode == RESIDUAL_MODE_FIXED_BALL)
            {
                double normal = primal[index] / (d * d);
                dual_residual[index] = (r + lambda * normal) * d;
            }
            else if (mode == RESIDUAL_MODE_FREE_CONE)
                dual_residual[index] = endpoint_residual0;
            else
                dual_residual[index] = 0.0;
        }
        else if (relative == k + 1)
        {
            dual_residual[index] = (mode == RESIDUAL_MODE_FREE_CONE) ? endpoint_residual1 : 0.0;
        }
    }
}

static __global__ void recompute_reflected_kernel(double *__restrict__ reflected,
                                                  const double *__restrict__ primal,
                                                  const double *__restrict__ current,
                                                  const int *__restrict__ local_start,
                                                  const int *__restrict__ local_count,
                                                  int num_cones)
{
    int cone = blockIdx.x;
    if (cone >= num_cones)
        return;
    int start = local_start[cone];
    int count = local_count[cone];
    int first_offset = (int)blockIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.y;
    for (int offset = first_offset; offset < count; offset += stride)
        reflected[start + offset] = 2.0 * primal[start + offset] - current[start + offset];
}

static __global__ void set_dual_slack_kernel(double *__restrict__ dual_slack,
                                             const double *__restrict__ effective_objective,
                                             const double *__restrict__ dual_product,
                                             const int *__restrict__ local_start,
                                             const int *__restrict__ local_count,
                                             int num_cones)
{
    int cone = blockIdx.x;
    if (cone >= num_cones)
        return;
    int start = local_start[cone];
    int count = local_count[cone];
    int first_offset = (int)blockIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.y;
    for (int offset = first_offset; offset < count; offset += stride)
        dual_slack[start + offset] = effective_objective[start + offset] - dual_product[start + offset];
}

static __global__ void prepare_affine_residuals_kernel(double *__restrict__ projection_point,
                                                       const double *__restrict__ primal_product,
                                                       const double *__restrict__ affine_cone_offset,
                                                       const double *__restrict__ dual_solution,
                                                       const int *__restrict__ local_start,
                                                       const int *__restrict__ local_count,
                                                       double *__restrict__ dot_products,
                                                       int num_cones)
{
    int cone = blockIdx.x;
    if (cone >= num_cones)
        return;

    __shared__ double partial[DIST_CONE_THREADS];
    int start = local_start[cone];
    int count = local_count[cone];
    double dot = 0.0;
    int first_offset = (int)blockIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.y;
    for (int offset = first_offset; offset < count; offset += stride)
    {
        int index = start + offset;
        double dual = dual_solution[index];
        projection_point[index] = -dual;
        dot += dual * (primal_product[index] + affine_cone_offset[index]);
    }
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (int reduction_stride = blockDim.x / 2; reduction_stride > 0; reduction_stride >>= 1)
    {
        if (threadIdx.x < reduction_stride)
            partial[threadIdx.x] += partial[threadIdx.x + reduction_stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(dot_products + cone, partial[0]);
}

static __global__ void finalize_affine_complementarity_kernel(double *__restrict__ complementarity_residual,
                                                              double constraint_bound_rescaling,
                                                              int num_cones)
{
    int cone = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone < num_cones)
        complementarity_residual[cone] = fabs(complementarity_residual[cone]) / constraint_bound_rescaling;
}

static void copy_int_array(int **device, const int *host, int count)
{
    CUDA_CHECK(cudaMalloc(device, (size_t)count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(*device, host, (size_t)count * sizeof(int), cudaMemcpyHostToDevice));
}

static distributed_cone_split_t *allocate_split_runtime(pdhg_solver_state_t *state,
                                                        const distributed_cone_partition_t *partition,
                                                        const double *coordinate_rescaling,
                                                        MPI_Comm communicator,
                                                        const char *axis_name)
{
    if (!partition || partition->num_cones <= 0)
        return NULL;

    int K = partition->num_cones;
    distributed_cone_split_t *split = (distributed_cone_split_t *)safe_calloc(1, sizeof(distributed_cone_split_t));
    split->num_cones = K;
    int max_local_count = 0;
    for (int cone = 0; cone < K; ++cone)
        if (partition->local_count[cone] > max_local_count)
            max_local_count = partition->local_count[cone];

    int device = 0;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int desired_blocks = (max_local_count + DIST_CONE_THREADS - 1) / DIST_CONE_THREADS;
    int block_budget = (4 * properties.multiProcessorCount) / K;
    desired_blocks = desired_blocks > 0 ? desired_blocks : 1;
    block_budget = block_budget > 0 ? block_budget : 1;
    block_budget = block_budget < 64 ? block_budget : 64;
    split->blocks_per_cone = desired_blocks < block_budget ? desired_blocks : block_budget;

    copy_int_array(&split->local_start, partition->local_start, K);
    copy_int_array(&split->local_first, partition->local_first, K);
    copy_int_array(&split->local_count, partition->local_count, K);
    copy_int_array(&split->v_dim, partition->v_dim, K);
    CUDA_CHECK(cudaMalloc(&split->type, (size_t)K * sizeof(cone_type_t)));
    CUDA_CHECK(cudaMemcpy(split->type, partition->type, (size_t)K * sizeof(cone_type_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&split->fixed_mask, (size_t)K * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(
        split->fixed_mask, partition->fixed_mask, (size_t)K * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&split->stats, (size_t)K * RESIDUAL_STATS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&split->complementarity_residual, (size_t)K * sizeof(double)));
    CUDA_CHECK(cudaMemset(split->complementarity_residual, 0, (size_t)K * sizeof(double)));

    double *local_min = (double *)malloc((size_t)K * sizeof(double));
    double *local_max = (double *)malloc((size_t)K * sizeof(double));
    double *global_min = (double *)malloc((size_t)K * sizeof(double));
    double *global_max = (double *)malloc((size_t)K * sizeof(double));
    for (int cone = 0; cone < K; ++cone)
    {
        local_min[cone] = DBL_MAX;
        local_max[cone] = 0.0;
        int start = partition->local_start[cone];
        for (int slot = 0; slot < partition->local_count[cone]; ++slot)
        {
            double d = coordinate_rescaling[start + slot];
            local_min[cone] = fmin(local_min[cone], d);
            local_max[cone] = fmax(local_max[cone], d);
        }
    }
    MPI_Allreduce(local_min, global_min, K, MPI_DOUBLE, MPI_MIN, communicator);
    MPI_Allreduce(local_max, global_max, K, MPI_DOUBLE, MPI_MAX, communicator);
    for (int cone = 0; cone < K; ++cone)
    {
        if (!(global_min[cone] > 0.0) || fabs(global_max[cone] - global_min[cone]) > 1e-12 * (1.0 + global_max[cone]))
        {
            fprintf(stderr,
                    "Error: split %s cone %d has heterogeneous coordinate scaling; keep it on one GPU or disable "
                    "the split.\n",
                    axis_name,
                    cone);
            MPI_Abort(state->grid_context->comm_global, EXIT_FAILURE);
        }
    }
    free(local_min);
    free(local_max);
    free(global_min);
    free(global_max);
    return split;
}

void initialize_split_cones(pdhg_solver_state_t *state, const rescale_info_t *rescale_info)
{
    state->cones.split = NULL;
    state->affine_cones.split = NULL;
    if (!state->grid_context)
        return;

    const distributed_cone_partition_t *partition = &state->grid_context->split_cones;
    int K = partition->num_cones;
    distributed_cone_split_t *split =
        allocate_split_runtime(state, partition, rescale_info->var_rescale, state->grid_context->comm_row, "variable");
    state->cones.split = split;

    if (split && rescale_info->processed_problem && rescale_info->processed_problem->quad_type == PDHCG_DIAG_Q)
    {
        double local_max_q = 0.0;
        const double *diag = rescale_info->processed_problem->diagonal_quad_objective;
        for (int cone = 0; cone < K; ++cone)
        {
            int start = partition->local_start[cone];
            for (int slot = 0; slot < partition->local_count[cone]; ++slot)
                local_max_q = fmax(local_max_q, fabs(diag[start + slot]));
        }
        double global_max_q = 0.0;
        MPI_Allreduce(&local_max_q, &global_max_q, 1, MPI_DOUBLE, MPI_MAX, state->grid_context->comm_row);
        if (global_max_q != 0.0)
        {
            fprintf(stderr, "Error: a split cone has a nonzero diagonal quadratic objective coefficient.\n");
            MPI_Abort(state->grid_context->comm_global, EXIT_FAILURE);
        }
    }

    const double INV_SQRT2 = 0.70710678118654752440;
    for (int cone = 0; split && cone < K; ++cone)
    {
        int first = partition->local_first[cone];
        int count = partition->local_count[cone];
        int local_start = partition->local_start[cone];
        int k = partition->v_dim[cone];
        unsigned char fixed = partition->fixed_mask[cone];
        for (int endpoint = 0; endpoint < 2; ++endpoint)
        {
            int relative = k + endpoint;
            if (relative < first || relative >= first + count)
                continue;
            bool pinned =
                endpoint == 0 ? (fixed & PDHCG_DIST_CONE_FIXED_AUX0) != 0 : (fixed & PDHCG_DIST_CONE_FIXED_AUX1) != 0;
            if (pinned)
                continue;
            int index = local_start + relative - first;
            double value = 0.0;
            if (partition->type[cone] == CONE_STANDARD_SOC)
            {
                value = (endpoint == 0 ? -INV_SQRT2 : INV_SQRT2) * rescale_info->con_bound_rescale *
                    rescale_info->var_rescale[index];
            }
            else if (endpoint == 1)
            {
                value = rescale_info->con_bound_rescale * rescale_info->var_rescale[index];
            }
            double *destinations[] = {state->initial_primal_solution,
                                      state->current_primal_solution,
                                      state->pdhg_primal_solution,
                                      state->reflected_primal_solution};
            for (double *destination : destinations)
                CUDA_CHECK(cudaMemcpy(destination + index, &value, sizeof(double), cudaMemcpyHostToDevice));
        }
    }

    const distributed_cone_partition_t *affine_partition = &state->grid_context->split_affine_cones;
    state->affine_cones.split = allocate_split_runtime(
        state, affine_partition, rescale_info->con_rescale, state->grid_context->comm_col, "affine");
}

static void free_split_runtime(distributed_cone_split_t *split)
{
    if (!split)
        return;
    CUDA_CHECK(cudaFree(split->local_start));
    CUDA_CHECK(cudaFree(split->local_first));
    CUDA_CHECK(cudaFree(split->local_count));
    CUDA_CHECK(cudaFree(split->v_dim));
    CUDA_CHECK(cudaFree(split->type));
    CUDA_CHECK(cudaFree(split->fixed_mask));
    CUDA_CHECK(cudaFree(split->stats));
    CUDA_CHECK(cudaFree(split->complementarity_residual));
    free(split);
}

void free_split_cones(pdhg_solver_state_t *state)
{
    if (!state)
        return;
    free_split_runtime(state->cones.split);
    free_split_runtime(state->affine_cones.split);
    state->cones.split = NULL;
    state->affine_cones.split = NULL;
}

void project_split_cones(pdhg_solver_state_t *state, cone_runtime_t *runtime, double *vector)
{
    distributed_cone_split_t *split = runtime ? runtime->split : NULL;
    if (!split || split->num_cones <= 0)
        return;
    int K = split->num_cones;
    dim3 grid((unsigned int)K, (unsigned int)split->blocks_per_cone);
    CUDA_CHECK(cudaMemset(split->stats, 0, (size_t)K * PROJECTION_STATS * sizeof(double)));
    collect_projection_stats_kernel<<<grid, DIST_CONE_THREADS>>>(
        vector, split->local_start, split->local_first, split->local_count, split->v_dim, split->stats, K);
    CUDA_CHECK(cudaGetLastError());
    pdhcg_comm_scope_t scope = runtime->axis == CONE_AXIS_VARIABLE ? PDHCG_SCOPE_ROW : PDHCG_SCOPE_COL;
    pdhcg_all_reduce_array(state->grid_context, split->stats, K * PROJECTION_STATS, PDHCG_OP_SUM, scope, 0);
    apply_projection_kernel<<<grid, DIST_CONE_THREADS>>>(vector,
                                                         split->local_start,
                                                         split->local_first,
                                                         split->local_count,
                                                         split->v_dim,
                                                         split->type,
                                                         split->fixed_mask,
                                                         split->stats,
                                                         K);
    CUDA_CHECK(cudaGetLastError());
}

void recompute_split_cone_reflected(pdhg_solver_state_t *state,
                                    double *reflected_primal,
                                    const double *pdhg_primal,
                                    const double *current_primal)
{
    distributed_cone_split_t *split = state->cones.split;
    if (!split || split->num_cones <= 0)
        return;
    dim3 grid((unsigned int)split->num_cones, (unsigned int)split->blocks_per_cone);
    recompute_reflected_kernel<<<grid, DIST_CONE_THREADS>>>(
        reflected_primal, pdhg_primal, current_primal, split->local_start, split->local_count, split->num_cones);
    CUDA_CHECK(cudaGetLastError());
}

void compute_split_cone_dual_residual(pdhg_solver_state_t *state, const double *effective_objective)
{
    distributed_cone_split_t *split = state->cones.split;
    if (!split || split->num_cones <= 0)
        return;
    int K = split->num_cones;
    dim3 grid((unsigned int)K, (unsigned int)split->blocks_per_cone);
    CUDA_CHECK(cudaMemset(split->stats, 0, (size_t)K * RESIDUAL_STATS * sizeof(double)));
    CUDA_CHECK(cudaMemset(split->complementarity_residual, 0, (size_t)K * sizeof(double)));
    collect_residual_stats_kernel<<<grid, DIST_CONE_THREADS>>>(effective_objective,
                                                               state->dual_product,
                                                               state->pdhg_primal_solution,
                                                               state->variable_rescaling,
                                                               split->local_start,
                                                               split->local_first,
                                                               split->local_count,
                                                               split->v_dim,
                                                               split->stats,
                                                               K);
    CUDA_CHECK(cudaGetLastError());
    pdhcg_all_reduce_array(state->grid_context, split->stats, K * RESIDUAL_STATS, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);
    apply_residual_kernel<<<grid, DIST_CONE_THREADS>>>(state->dual_residual,
                                                       split->complementarity_residual,
                                                       effective_objective,
                                                       state->dual_product,
                                                       state->pdhg_primal_solution,
                                                       state->variable_rescaling,
                                                       split->local_start,
                                                       split->local_first,
                                                       split->local_count,
                                                       split->v_dim,
                                                       split->type,
                                                       split->fixed_mask,
                                                       split->stats,
                                                       K);
    CUDA_CHECK(cudaGetLastError());
}

double get_split_cone_complementarity_norm(pdhg_solver_state_t *state, norm_type_t norm)
{
    distributed_cone_split_t *split = state->cones.split;
    if (!split || split->num_cones <= 0 || !state->grid_context || state->grid_context->coords[1] != 0)
        return 0.0;

    if (norm == NORM_TYPE_L_INF)
        return get_vector_inf_norm(state->blas_handle, split->num_cones, split->complementarity_residual);

    double residual_norm = 0.0;
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, split->num_cones, split->complementarity_residual, 1, &residual_norm));
    return residual_norm;
}

void finalize_split_affine_cone_complementarity(pdhg_solver_state_t *state)
{
    distributed_cone_split_t *split = state->affine_cones.split;
    if (!split || split->num_cones <= 0)
        return;

    int K = split->num_cones;
    pdhcg_all_reduce_array(state->grid_context, split->complementarity_residual, K, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);
    int blocks = (K + DIST_CONE_THREADS - 1) / DIST_CONE_THREADS;
    finalize_affine_complementarity_kernel<<<blocks, DIST_CONE_THREADS>>>(
        split->complementarity_residual, state->constraint_bound_rescaling, K);
    CUDA_CHECK(cudaGetLastError());
}

void prepare_split_affine_cone_residuals(pdhg_solver_state_t *state,
                                         double *projection_point,
                                         const double *primal_product,
                                         const double *affine_cone_offset,
                                         const double *dual_solution)
{
    distributed_cone_split_t *split = state->affine_cones.split;
    if (!split || split->num_cones <= 0)
        return;
    int K = split->num_cones;
    dim3 grid((unsigned int)K, (unsigned int)split->blocks_per_cone);
    /* Keep the dot products separate from stats, which split projection reuses. */
    CUDA_CHECK(cudaMemset(split->complementarity_residual, 0, (size_t)K * sizeof(double)));
    prepare_affine_residuals_kernel<<<grid, DIST_CONE_THREADS>>>(projection_point,
                                                                 primal_product,
                                                                 affine_cone_offset,
                                                                 dual_solution,
                                                                 split->local_start,
                                                                 split->local_count,
                                                                 split->complementarity_residual,
                                                                 K);
    CUDA_CHECK(cudaGetLastError());
}

double get_split_affine_cone_complementarity_norm(pdhg_solver_state_t *state, norm_type_t norm)
{
    distributed_cone_split_t *split = state->affine_cones.split;
    if (!split || split->num_cones <= 0 || !state->grid_context || state->grid_context->coords[0] != 0)
        return 0.0;

    if (norm == NORM_TYPE_L_INF)
        return get_vector_inf_norm(state->blas_handle, split->num_cones, split->complementarity_residual);

    double residual_norm = 0.0;
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, split->num_cones, split->complementarity_residual, 1, &residual_norm));
    return residual_norm;
}

void set_split_cone_dual_slack(pdhg_solver_state_t *state,
                               double *dual_slack,
                               const double *effective_objective,
                               const double *dual_product)
{
    distributed_cone_split_t *split = state->cones.split;
    if (!split || split->num_cones <= 0)
        return;
    dim3 grid((unsigned int)split->num_cones, (unsigned int)split->blocks_per_cone);
    set_dual_slack_kernel<<<grid, DIST_CONE_THREADS>>>(
        dual_slack, effective_objective, dual_product, split->local_start, split->local_count, split->num_cones);
    CUDA_CHECK(cudaGetLastError());
}
