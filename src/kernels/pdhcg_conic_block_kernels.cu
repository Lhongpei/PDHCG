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

#include "cone_section_projection.cuh"
#include "pdhcg_kernels.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__device__ static inline void cone_block_sum3(double *first, double *second, double *third, double scratch[96])
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    unsigned mask = __activemask();
    double a = *first;
    double b = *second;
    double c = *third;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        a += __shfl_down_sync(mask, a, offset);
        b += __shfl_down_sync(mask, b, offset);
        c += __shfl_down_sync(mask, c, offset);
    }
    if (lane == 0)
    {
        scratch[3 * warp + 0] = a;
        scratch[3 * warp + 1] = b;
        scratch[3 * warp + 2] = c;
    }
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    if (warp == 0)
    {
        a = lane < num_warps ? scratch[3 * lane + 0] : 0.0;
        b = lane < num_warps ? scratch[3 * lane + 1] : 0.0;
        c = lane < num_warps ? scratch[3 * lane + 2] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            a += __shfl_down_sync(0xffffffffu, a, offset);
            b += __shfl_down_sync(0xffffffffu, b, offset);
            c += __shfl_down_sync(0xffffffffu, c, offset);
        }
        if (lane == 0)
        {
            scratch[0] = a;
            scratch[1] = b;
            scratch[2] = c;
        }
    }
    __syncthreads();
    *first = scratch[0];
    *second = scratch[1];
    *third = scratch[2];
    __syncthreads();
}

__device__ static inline double cone_block_max(double value, double scratch[96])
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    unsigned mask = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmax(value, __shfl_down_sync(mask, value, offset));
    if (lane == 0)
        scratch[warp] = value;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    if (warp == 0)
    {
        value = lane < num_warps ? scratch[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
            value = fmax(value, __shfl_down_sync(0xffffffffu, value, offset));
        if (lane == 0)
            scratch[0] = value;
    }
    __syncthreads();
    value = scratch[0];
    __syncthreads();
    return value;
}

__device__ static inline void cone_atomic_max_positive(double *address, double value)
{
    atomicMax(reinterpret_cast<unsigned long long *>(address),
              static_cast<unsigned long long>(__double_as_longlong(value)));
}

enum standard_soc_block_mode
{
    SOC_BLOCK_IDENTITY = 0,
    SOC_BLOCK_ZERO_FREE = 1,
    SOC_BLOCK_APEX = 2,
    SOC_BLOCK_SCALAR_Z = 3,
    SOC_BLOCK_FIXED_Z_ROOT = 4,
    SOC_BLOCK_FREE_Z_ROOT = 5,
    SOC_BLOCK_ZERO_Z_ROOT = 6
};

__global__ void project_standard_soc_block_kernel(double *__restrict__ point,
                                                  const double *__restrict__ rescaling,
                                                  const double *__restrict__ q_diag,
                                                  double tau,
                                                  double *__restrict__ warm_start,
                                                  const int *__restrict__ start_idx,
                                                  const int *__restrict__ v_dim,
                                                  const char *__restrict__ is_fixed,
                                                  int num_cones)
{
    int cone = blockIdx.x;
    if (cone >= num_cones)
        return;

    __shared__ double scratch[96];
    __shared__ double fixed_norm2;
    __shared__ double radius2;
    __shared__ double lambda;
    __shared__ double lo;
    __shared__ double hi;
    __shared__ double z_input;
    __shared__ double omega_z;
    __shared__ int mode;
    __shared__ int lower_branch;
    __shared__ int done;

    int start = start_idx[cone];
    int k = v_dim[cone];
    int u_length = k + 1;
    int z_index = start + u_length;
    bool fixed_z = is_fixed && is_fixed[z_index];

    double local_fixed_norm2 = 0.0;
    double local_free_norm2 = 0.0;
    double local_polar_norm2 = 0.0;
    double local_max_omega = 0.0;
    for (int slot = threadIdx.x; slot < u_length; slot += blockDim.x)
    {
        int index = start + slot;
        double value = point[index] / rescaling[index];
        if (is_fixed && is_fixed[index])
            local_fixed_norm2 += value * value;
        else
        {
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            local_free_norm2 += value * value;
            local_polar_norm2 += (omega * value) * (omega * value);
            local_max_omega = fmax(local_max_omega, omega);
        }
    }
    cone_block_sum3(&local_fixed_norm2, &local_free_norm2, &local_polar_norm2, scratch);
    local_max_omega = cone_block_max(local_max_omega, scratch);

    if (threadIdx.x == 0)
    {
        fixed_norm2 = local_fixed_norm2;
        z_input = point[z_index] / rescaling[z_index];
        omega_z = cone_section_weight(rescaling, q_diag, tau, z_index);
        int free_count = 0;
        for (int slot = 0; slot < u_length; ++slot)
            free_count += !(is_fixed && is_fixed[start + slot]);

        if (fixed_z)
        {
            radius2 = fmax(0.0, z_input * z_input - fixed_norm2);
            if (free_count == 0 || local_free_norm2 <= radius2)
                mode = SOC_BLOCK_IDENTITY;
            else if (!(radius2 > 0.0))
                mode = SOC_BLOCK_ZERO_FREE;
            else
            {
                mode = SOC_BLOCK_FIXED_Z_ROOT;
                hi = sqrt(local_polar_norm2) / sqrt(radius2) * (1.0 + 64.0 * DBL_EPSILON);
            }
        }
        else if (z_input >= 0.0 && fixed_norm2 + local_free_norm2 <= z_input * z_input)
        {
            mode = SOC_BLOCK_IDENTITY;
        }
        else if (free_count == 0)
        {
            mode = SOC_BLOCK_SCALAR_Z;
        }
        else if (fixed_norm2 == 0.0 && -omega_z * z_input >= sqrt(local_polar_norm2))
        {
            mode = SOC_BLOCK_APEX;
        }
        else if (z_input == 0.0)
        {
            mode = SOC_BLOCK_ZERO_Z_ROOT;
            lambda = omega_z;
        }
        else
        {
            mode = SOC_BLOCK_FREE_Z_ROOT;
            lower_branch = z_input > 0.0;
            lo = lower_branch ? 0.0 : omega_z * (1.0 + 1e-14);
            hi = lower_branch ? omega_z * (1.0 - 1e-14)
                              : cone_section_negative_soc_upper(
                                    omega_z, -omega_z * z_input, fixed_norm2, local_polar_norm2, local_max_omega);
        }
    }
    __syncthreads();

    if (mode == SOC_BLOCK_IDENTITY)
        return;
    if (mode == SOC_BLOCK_ZERO_FREE || mode == SOC_BLOCK_APEX)
    {
        for (int slot = threadIdx.x; slot < u_length; slot += blockDim.x)
            if (!(is_fixed && is_fixed[start + slot]))
                point[start + slot] = 0.0;
        if (mode == SOC_BLOCK_APEX && threadIdx.x == 0)
            point[z_index] = 0.0;
        return;
    }
    if (mode == SOC_BLOCK_SCALAR_Z)
    {
        if (threadIdx.x == 0)
            point[z_index] = fmax(z_input, sqrt(fixed_norm2)) * rescaling[z_index];
        return;
    }

    if (mode == SOC_BLOCK_FIXED_Z_ROOT)
    {
        if (threadIdx.x == 0)
        {
            lo = 0.0;
            done = hi > 0.0 && isfinite(hi);
            if (!done)
                hi = warm_start && warm_start[cone] > 0.0 && isfinite(warm_start[cone]) ? warm_start[cone] : 1.0;
        }
        __syncthreads();
        for (int expansion = 0; expansion < 80; ++expansion)
        {
            if (done)
                break;
            double norm2 = 0.0;
            double unused = 0.0;
            double unused2 = 0.0;
            for (int slot = threadIdx.x; slot < u_length; slot += blockDim.x)
            {
                int index = start + slot;
                if (is_fixed && is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = (point[index] / rescaling[index]) * omega / (omega + hi);
                norm2 += value * value;
            }
            cone_block_sum3(&norm2, &unused, &unused2, scratch);
            if (threadIdx.x == 0)
            {
                done = norm2 <= radius2;
                if (!done)
                    hi *= 2.0;
            }
            __syncthreads();
            if (done)
                break;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            double warm = warm_start ? warm_start[cone] : 0.0;
            lambda = warm > lo && warm < hi && isfinite(warm) ? warm : 0.5 * (lo + hi);
            done = 0;
        }
        __syncthreads();
        for (int iteration = 0; iteration < 30; ++iteration)
        {
            double norm2 = 0.0;
            double derivative = 0.0;
            double unused = 0.0;
            for (int slot = threadIdx.x; slot < u_length; slot += blockDim.x)
            {
                int index = start + slot;
                if (is_fixed && is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = (point[index] / rescaling[index]) * omega / (omega + lambda);
                norm2 += value * value;
                derivative -= 2.0 * value * value / (omega + lambda);
            }
            cone_block_sum3(&norm2, &derivative, &unused, scratch);
            if (threadIdx.x == 0)
            {
                double f = norm2 - radius2;
                if (f > 0.0)
                    lo = lambda;
                else
                    hi = lambda;
                double next = lambda - f / derivative;
                if (!isfinite(next) || !(next > lo && next < hi))
                    next = 0.5 * (lo + hi);
                done = fabs(f) <= 1e-13 * (1.0 + radius2) || hi - lo <= 1e-13 * (1.0 + hi + lo);
                if (!done)
                    lambda = next;
            }
            __syncthreads();
            if (done)
                break;
        }
    }
    else if (mode == SOC_BLOCK_ZERO_Z_ROOT)
    {
        double norm2 = 0.0;
        double unused = 0.0;
        double unused2 = 0.0;
        for (int slot = threadIdx.x; slot < u_length; slot += blockDim.x)
        {
            int index = start + slot;
            if (is_fixed && is_fixed[index])
                continue;
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            double value = (point[index] / rescaling[index]) * omega / (omega + lambda);
            norm2 += value * value;
        }
        cone_block_sum3(&norm2, &unused, &unused2, scratch);
        if (threadIdx.x == 0)
            point[z_index] = sqrt(fixed_norm2 + norm2) * rescaling[z_index];
    }
    else
    {
        if (!lower_branch)
        {
            if (threadIdx.x == 0)
            {
                done = hi > lo && isfinite(hi);
                if (!done)
                    hi = 2.0 * omega_z;
            }
            __syncthreads();
            for (int expansion = 0; expansion < 80; ++expansion)
            {
                if (done)
                    break;
                double norm2 = 0.0;
                double unused = 0.0;
                double unused2 = 0.0;
                for (int slot = threadIdx.x; slot < u_length; slot += blockDim.x)
                {
                    int index = start + slot;
                    if (is_fixed && is_fixed[index])
                        continue;
                    double omega = cone_section_weight(rescaling, q_diag, tau, index);
                    double value = (point[index] / rescaling[index]) * omega / (omega + hi);
                    norm2 += value * value;
                }
                cone_block_sum3(&norm2, &unused, &unused2, scratch);
                if (threadIdx.x == 0)
                {
                    double z = omega_z * z_input / (omega_z - hi);
                    done = fixed_norm2 + norm2 >= z * z;
                    if (!done)
                        hi *= 2.0;
                }
                __syncthreads();
                if (done)
                    break;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            double warm = warm_start ? warm_start[cone] : 0.0;
            lambda = warm > lo && warm < hi && isfinite(warm) ? warm : 0.5 * (lo + hi);
            done = 0;
        }
        __syncthreads();
        for (int iteration = 0; iteration < 35; ++iteration)
        {
            double norm2 = 0.0;
            double derivative = 0.0;
            double unused = 0.0;
            for (int slot = threadIdx.x; slot < u_length; slot += blockDim.x)
            {
                int index = start + slot;
                if (is_fixed && is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = (point[index] / rescaling[index]) * omega / (omega + lambda);
                norm2 += value * value;
                derivative -= 2.0 * value * value / (omega + lambda);
            }
            cone_block_sum3(&norm2, &derivative, &unused, scratch);
            if (threadIdx.x == 0)
            {
                double z = omega_z * z_input / (omega_z - lambda);
                double f = fixed_norm2 + norm2 - z * z;
                derivative -= 2.0 * z * z / (omega_z - lambda);
                if ((lower_branch && f > 0.0) || (!lower_branch && f < 0.0))
                    lo = lambda;
                else
                    hi = lambda;
                double next = lambda - f / derivative;
                if (!isfinite(next) || !(next > lo && next < hi))
                    next = 0.5 * (lo + hi);
                done = fabs(f) <= 1e-13 * (1.0 + fixed_norm2 + norm2 + z * z) || hi - lo <= 1e-13 * (1.0 + hi + lo);
                if (!done)
                    lambda = next;
            }
            __syncthreads();
            if (done)
                break;
        }
        if (threadIdx.x == 0)
            point[z_index] *= omega_z / (omega_z - lambda);
    }

    if (warm_start && threadIdx.x == 0)
        warm_start[cone] = lambda;
    for (int slot = threadIdx.x; slot < u_length; slot += blockDim.x)
    {
        int index = start + slot;
        if (!(is_fixed && is_fixed[index]))
        {
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            point[index] *= omega / (omega + lambda);
        }
    }
}

enum standard_soc_grid_weighted_mode
{
    SOC_GRID_IDENTITY = 0,
    SOC_GRID_ZERO_FREE = 1,
    SOC_GRID_APEX = 2,
    SOC_GRID_SCALAR_Z = 3,
    SOC_GRID_FIXED_EXPAND = 4,
    SOC_GRID_FIXED_ROOT = 5,
    SOC_GRID_FREE_EXPAND = 6,
    SOC_GRID_FREE_ROOT = 7,
    SOC_GRID_ZERO_Z_EVAL = 8,
    SOC_GRID_FIXED_APPLY = 9,
    SOC_GRID_FREE_APPLY = 10,
    SOC_GRID_ZERO_Z_APPLY = 11
};

__global__ void initialize_standard_soc_grid_weighted_kernel(const double *__restrict__ point,
                                                             const double *__restrict__ rescaling,
                                                             const double *__restrict__ q_diag,
                                                             double tau,
                                                             double *__restrict__ workspace,
                                                             const int *__restrict__ start_idx,
                                                             const int *__restrict__ v_dim,
                                                             const char *__restrict__ is_fixed,
                                                             int num_cones,
                                                             int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    if (cone >= num_cones)
        return;
    int part = blockIdx.x - cone * blocks_per_cone;
    int start = start_idx[cone];
    int u_length = v_dim[cone] + 1;
    double fixed_norm2 = 0.0;
    double free_norm2 = 0.0;
    double polar_norm2 = 0.0;
    double free_count = 0.0;
    double max_omega = 0.0;
    for (int slot = part * blockDim.x + threadIdx.x; slot < u_length; slot += blocks_per_cone * blockDim.x)
    {
        int index = start + slot;
        double value = point[index] / rescaling[index];
        if (is_fixed && is_fixed[index])
            fixed_norm2 += value * value;
        else
        {
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            free_norm2 += value * value;
            polar_norm2 += (omega * value) * (omega * value);
            free_count += 1.0;
            max_omega = fmax(max_omega, omega);
        }
    }
    __shared__ double scratch[96];
    cone_block_sum3(&fixed_norm2, &free_norm2, &polar_norm2, scratch);
    double unused = 0.0;
    double unused2 = 0.0;
    cone_block_sum3(&free_count, &unused, &unused2, scratch);
    max_omega = cone_block_max(max_omega, scratch);
    if (threadIdx.x == 0)
    {
        atomicAdd(workspace + num_cones + cone, fixed_norm2);
        atomicAdd(workspace + 2 * num_cones + cone, free_norm2);
        atomicAdd(workspace + 3 * num_cones + cone, polar_norm2);
        atomicAdd(workspace + 4 * num_cones + cone, free_count);
        cone_atomic_max_positive(workspace + 5 * num_cones + cone, max_omega);
    }
}

__global__ void finalize_standard_soc_grid_weighted_initialization_kernel(const double *__restrict__ point,
                                                                          const double *__restrict__ rescaling,
                                                                          const double *__restrict__ q_diag,
                                                                          double tau,
                                                                          double *__restrict__ workspace,
                                                                          const int *__restrict__ start_idx,
                                                                          const int *__restrict__ v_dim,
                                                                          const char *__restrict__ is_fixed,
                                                                          int num_cones)
{
    int cone = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone >= num_cones)
        return;
    int start = start_idx[cone];
    int z_index = start + v_dim[cone] + 1;
    double warm = workspace[cone];
    double fixed_norm2 = workspace[num_cones + cone];
    double free_norm2 = workspace[2 * num_cones + cone];
    double polar_norm2 = workspace[3 * num_cones + cone];
    int free_count = (int)workspace[4 * num_cones + cone];
    double max_omega = workspace[5 * num_cones + cone];
    double z = point[z_index] / rescaling[z_index];
    double omega_z_value = cone_section_weight(rescaling, q_diag, tau, z_index);
    bool fixed_z = is_fixed && is_fixed[z_index];
    int selected_mode;
    double constant = fixed_norm2;
    double lower = 0.0;
    double upper = 0.0;
    double trial = warm;

    if (fixed_z)
    {
        constant = fmax(0.0, z * z - fixed_norm2);
        if (free_count == 0 || free_norm2 <= constant)
            selected_mode = SOC_GRID_IDENTITY;
        else if (!(constant > 0.0))
            selected_mode = SOC_GRID_ZERO_FREE;
        else
        {
            lower = 0.0;
            upper = sqrt(polar_norm2) / sqrt(constant) * (1.0 + 64.0 * DBL_EPSILON);
            if (upper > 0.0 && isfinite(upper))
            {
                selected_mode = SOC_GRID_FIXED_ROOT;
                trial = warm > lower && warm < upper && isfinite(warm) ? warm : 0.5 * upper;
            }
            else
            {
                selected_mode = SOC_GRID_FIXED_EXPAND;
                trial = warm > 0.0 && isfinite(warm) ? warm : 1.0;
                upper = trial;
            }
        }
    }
    else if (z >= 0.0 && fixed_norm2 + free_norm2 <= z * z)
    {
        selected_mode = SOC_GRID_IDENTITY;
    }
    else if (free_count == 0)
    {
        selected_mode = SOC_GRID_SCALAR_Z;
    }
    else if (fixed_norm2 == 0.0 && -omega_z_value * z >= sqrt(polar_norm2))
    {
        selected_mode = SOC_GRID_APEX;
    }
    else if (z == 0.0)
    {
        selected_mode = SOC_GRID_ZERO_Z_EVAL;
        trial = omega_z_value;
    }
    else if (z > 0.0)
    {
        selected_mode = SOC_GRID_FREE_ROOT;
        lower = 0.0;
        upper = omega_z_value * (1.0 - 1e-14);
        trial = warm > lower && warm < upper && isfinite(warm) ? warm : 0.5 * (lower + upper);
    }
    else
    {
        lower = omega_z_value * (1.0 + 1e-14);
        double endpoint_polar = -omega_z_value * z;
        upper = cone_section_negative_soc_upper(omega_z_value, endpoint_polar, fixed_norm2, polar_norm2, max_omega);
        if (upper > lower && isfinite(upper))
        {
            selected_mode = SOC_GRID_FREE_ROOT;
            trial = warm > lower && warm < upper && isfinite(warm) ? warm : 0.5 * (lower + upper);
        }
        else
        {
            selected_mode = SOC_GRID_FREE_EXPAND;
            trial = warm > lower && isfinite(warm) ? warm : 2.0 * omega_z_value;
            upper = trial;
        }
    }

    workspace[cone] = trial;
    workspace[4 * num_cones + cone] = (double)selected_mode;
    workspace[5 * num_cones + cone] = constant;
    workspace[6 * num_cones + cone] = lower;
    workspace[7 * num_cones + cone] = upper;
}

__global__ void reduce_standard_soc_grid_weighted_root_kernel(const double *__restrict__ point,
                                                              const double *__restrict__ rescaling,
                                                              const double *__restrict__ q_diag,
                                                              double tau,
                                                              double *__restrict__ workspace,
                                                              const int *__restrict__ start_idx,
                                                              const int *__restrict__ v_dim,
                                                              const char *__restrict__ is_fixed,
                                                              int num_cones,
                                                              int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    if (cone >= num_cones)
        return;
    int selected_mode = (int)workspace[4 * num_cones + cone];
    if (selected_mode < SOC_GRID_FIXED_EXPAND || selected_mode > SOC_GRID_ZERO_Z_EVAL)
        return;
    int part = blockIdx.x - cone * blocks_per_cone;
    int start = start_idx[cone];
    int u_length = v_dim[cone] + 1;
    double lambda_value = workspace[cone];
    double norm2 = 0.0;
    double derivative = 0.0;
    for (int slot = part * blockDim.x + threadIdx.x; slot < u_length; slot += blocks_per_cone * blockDim.x)
    {
        int index = start + slot;
        if (is_fixed && is_fixed[index])
            continue;
        double omega = cone_section_weight(rescaling, q_diag, tau, index);
        double value = (point[index] / rescaling[index]) * omega / (omega + lambda_value);
        norm2 += value * value;
        derivative -= 2.0 * value * value / (omega + lambda_value);
    }
    __shared__ double scratch[96];
    double unused = 0.0;
    cone_block_sum3(&norm2, &derivative, &unused, scratch);
    if (threadIdx.x == 0)
    {
        atomicAdd(workspace + num_cones + cone, norm2);
        atomicAdd(workspace + 2 * num_cones + cone, derivative);
    }
}

__global__ void finalize_standard_soc_grid_weighted_root_kernel(double *__restrict__ point,
                                                                const double *__restrict__ rescaling,
                                                                const double *__restrict__ q_diag,
                                                                double tau,
                                                                double *__restrict__ workspace,
                                                                const int *__restrict__ start_idx,
                                                                const int *__restrict__ v_dim,
                                                                int num_cones)
{
    int cone = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone >= num_cones)
        return;
    int selected_mode = (int)workspace[4 * num_cones + cone];
    if (selected_mode < SOC_GRID_FIXED_EXPAND || selected_mode > SOC_GRID_ZERO_Z_EVAL)
        return;
    int start = start_idx[cone];
    int z_index = start + v_dim[cone] + 1;
    double lambda_value = workspace[cone];
    double sum = workspace[num_cones + cone];
    double derivative = workspace[2 * num_cones + cone];
    double constant = workspace[5 * num_cones + cone];
    double lower = workspace[6 * num_cones + cone];
    double upper = workspace[7 * num_cones + cone];

    if (selected_mode == SOC_GRID_ZERO_Z_EVAL)
    {
        point[z_index] = sqrt(constant + sum) * rescaling[z_index];
        workspace[4 * num_cones + cone] = (double)SOC_GRID_ZERO_Z_APPLY;
        return;
    }

    double f;
    if (selected_mode == SOC_GRID_FIXED_EXPAND || selected_mode == SOC_GRID_FIXED_ROOT)
    {
        f = sum - constant;
        if (selected_mode == SOC_GRID_FIXED_EXPAND)
        {
            if (f > 0.0)
            {
                lower = lambda_value;
                lambda_value *= 2.0;
            }
            else
            {
                upper = lambda_value;
                selected_mode = SOC_GRID_FIXED_ROOT;
                lambda_value = 0.5 * (lower + upper);
            }
        }
        else
        {
            if (f > 0.0)
                lower = lambda_value;
            else
                upper = lambda_value;
            bool converged = fabs(f) <= 1e-13 * (1.0 + constant) || upper - lower <= 1e-13 * (1.0 + upper + lower);
            if (converged)
                selected_mode = SOC_GRID_FIXED_APPLY;
            else
            {
                double next = lambda_value - f / derivative;
                lambda_value = isfinite(next) && next > lower && next < upper ? next : 0.5 * (lower + upper);
            }
        }
    }
    else
    {
        double z_input = point[z_index] / rescaling[z_index];
        double omega_z_value = cone_section_weight(rescaling, q_diag, tau, z_index);
        double z = omega_z_value * z_input / (omega_z_value - lambda_value);
        f = constant + sum - z * z;
        derivative -= 2.0 * z * z / (omega_z_value - lambda_value);
        if (selected_mode == SOC_GRID_FREE_EXPAND)
        {
            if (f < 0.0)
            {
                lower = lambda_value;
                lambda_value *= 2.0;
            }
            else
            {
                upper = lambda_value;
                selected_mode = SOC_GRID_FREE_ROOT;
                lambda_value = 0.5 * (lower + upper);
            }
        }
        else
        {
            bool lower_branch_value = z_input > 0.0;
            if ((lower_branch_value && f > 0.0) || (!lower_branch_value && f < 0.0))
                lower = lambda_value;
            else
                upper = lambda_value;
            bool converged =
                fabs(f) <= 1e-13 * (1.0 + constant + sum + z * z) || upper - lower <= 1e-13 * (1.0 + upper + lower);
            if (converged)
                selected_mode = SOC_GRID_FREE_APPLY;
            else
            {
                double next = lambda_value - f / derivative;
                lambda_value = isfinite(next) && next > lower && next < upper ? next : 0.5 * (lower + upper);
            }
        }
    }
    workspace[cone] = lambda_value;
    workspace[4 * num_cones + cone] = (double)selected_mode;
    workspace[6 * num_cones + cone] = lower;
    workspace[7 * num_cones + cone] = upper;
}

__global__ void apply_standard_soc_grid_weighted_kernel(double *__restrict__ point,
                                                        const double *__restrict__ rescaling,
                                                        const double *__restrict__ q_diag,
                                                        double tau,
                                                        const double *__restrict__ workspace,
                                                        const int *__restrict__ start_idx,
                                                        const int *__restrict__ v_dim,
                                                        const char *__restrict__ is_fixed,
                                                        int num_cones,
                                                        int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    if (cone >= num_cones)
        return;
    int selected_mode = (int)workspace[4 * num_cones + cone];
    if (selected_mode == SOC_GRID_IDENTITY || selected_mode == SOC_GRID_SCALAR_Z)
    {
        if (selected_mode == SOC_GRID_SCALAR_Z && blockIdx.x % blocks_per_cone == 0 && threadIdx.x == 0)
        {
            int z_index = start_idx[cone] + v_dim[cone] + 1;
            double z = point[z_index] / rescaling[z_index];
            point[z_index] = fmax(z, sqrt(workspace[5 * num_cones + cone])) * rescaling[z_index];
        }
        return;
    }
    int part = blockIdx.x - cone * blocks_per_cone;
    int start = start_idx[cone];
    int u_length = v_dim[cone] + 1;
    if (selected_mode == SOC_GRID_ZERO_FREE || selected_mode == SOC_GRID_APEX)
    {
        for (int slot = part * blockDim.x + threadIdx.x; slot < u_length; slot += blocks_per_cone * blockDim.x)
            if (!(is_fixed && is_fixed[start + slot]))
                point[start + slot] = 0.0;
        if (selected_mode == SOC_GRID_APEX && part == 0 && threadIdx.x == 0)
            point[start + u_length] = 0.0;
        return;
    }

    double lambda_value = workspace[cone];
    for (int slot = part * blockDim.x + threadIdx.x; slot < u_length; slot += blocks_per_cone * blockDim.x)
    {
        int index = start + slot;
        if (!(is_fixed && is_fixed[index]))
        {
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            point[index] *= omega / (omega + lambda_value);
        }
    }
    bool free_z_mode = selected_mode == SOC_GRID_FREE_EXPAND || selected_mode == SOC_GRID_FREE_ROOT ||
        selected_mode == SOC_GRID_FREE_APPLY;
    if (free_z_mode && part == 0 && threadIdx.x == 0)
    {
        int z_index = start + u_length;
        double omega_z_value = cone_section_weight(rescaling, q_diag, tau, z_index);
        point[z_index] *= omega_z_value / (omega_z_value - lambda_value);
    }
}

enum rotated_soc_block_mode
{
    RSOC_BLOCK_IDENTITY = 0,
    RSOC_BLOCK_ZERO_FREE = 1,
    RSOC_BLOCK_FIXED_ENDPOINTS_ROOT = 2,
    RSOC_BLOCK_ONE_ENDPOINT_ZERO = 3,
    RSOC_BLOCK_ONE_ENDPOINT_SCALAR = 4,
    RSOC_BLOCK_ONE_ENDPOINT_ROOT = 5,
    RSOC_BLOCK_APEX = 6,
    RSOC_BLOCK_BALANCED = 7,
    RSOC_BLOCK_FREE_ROOT = 8,
    RSOC_BLOCK_AXIS = 9
};

__global__ void project_rotated_soc_block_kernel(double *__restrict__ point,
                                                 const double *__restrict__ rescaling,
                                                 const double *__restrict__ q_diag,
                                                 double tau,
                                                 double *__restrict__ warm_start,
                                                 const int *__restrict__ start_idx,
                                                 const int *__restrict__ v_dim,
                                                 const char *__restrict__ is_fixed,
                                                 int num_cones)
{
    int cone = blockIdx.x;
    if (cone >= num_cones)
        return;

    __shared__ double scratch[96];
    __shared__ double fixed_norm2;
    __shared__ double radius2;
    __shared__ double lambda;
    __shared__ double lo;
    __shared__ double hi;
    __shared__ double s_input;
    __shared__ double t_input;
    __shared__ double omega_s;
    __shared__ double omega_t;
    __shared__ double projected_s;
    __shared__ double projected_t;
    __shared__ double free_objective;
    __shared__ int mode;
    __shared__ int lower_branch;
    __shared__ int done;

    int start = start_idx[cone];
    int k = v_dim[cone];
    int s_index = start + k;
    int t_index = s_index + 1;
    bool fixed_s = is_fixed && is_fixed[s_index];
    bool fixed_t = is_fixed && is_fixed[t_index];

    double local_fixed_norm2 = 0.0;
    double local_free_norm2 = 0.0;
    double local_polar_norm2 = 0.0;
    double local_free_objective = 0.0;
    double local_max_omega = 0.0;
    for (int slot = threadIdx.x; slot < k; slot += blockDim.x)
    {
        int index = start + slot;
        double value = point[index] / rescaling[index];
        if (is_fixed && is_fixed[index])
            local_fixed_norm2 += value * value;
        else
        {
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            local_free_norm2 += value * value;
            local_polar_norm2 += (omega * value) * (omega * value);
            local_free_objective += omega * value * value;
            local_max_omega = fmax(local_max_omega, omega);
        }
    }
    cone_block_sum3(&local_fixed_norm2, &local_free_norm2, &local_polar_norm2, scratch);
    double unused = 0.0;
    double unused2 = 0.0;
    cone_block_sum3(&local_free_objective, &unused, &unused2, scratch);
    local_max_omega = cone_block_max(local_max_omega, scratch);

    if (threadIdx.x == 0)
    {
        fixed_norm2 = local_fixed_norm2;
        free_objective = local_free_objective;
        s_input = point[s_index] / rescaling[s_index];
        t_input = point[t_index] / rescaling[t_index];
        omega_s = cone_section_weight(rescaling, q_diag, tau, s_index);
        omega_t = cone_section_weight(rescaling, q_diag, tau, t_index);
        int free_count = 0;
        for (int slot = 0; slot < k; ++slot)
            free_count += !(is_fixed && is_fixed[start + slot]);

        if (fixed_s && fixed_t)
        {
            radius2 = fmax(0.0, 2.0 * s_input * t_input - fixed_norm2);
            if (free_count == 0 || local_free_norm2 <= radius2)
                mode = RSOC_BLOCK_IDENTITY;
            else if (!(radius2 > 0.0))
                mode = RSOC_BLOCK_ZERO_FREE;
            else
            {
                mode = RSOC_BLOCK_FIXED_ENDPOINTS_ROOT;
                hi = sqrt(local_polar_norm2) / sqrt(radius2) * (1.0 + 64.0 * DBL_EPSILON);
            }
        }
        else if (fixed_s || fixed_t)
        {
            double fixed_endpoint = fixed_s ? s_input : t_input;
            double free_endpoint = fixed_s ? t_input : s_input;
            if (!(fixed_endpoint > 0.0))
                mode = RSOC_BLOCK_ONE_ENDPOINT_ZERO;
            else if (free_endpoint >= 0.0 && fixed_norm2 + local_free_norm2 <= 2.0 * fixed_endpoint * free_endpoint)
                mode = RSOC_BLOCK_IDENTITY;
            else if (free_count == 0)
                mode = RSOC_BLOCK_ONE_ENDPOINT_SCALAR;
            else
            {
                mode = RSOC_BLOCK_ONE_ENDPOINT_ROOT;
                double metric = fixed_s ? omega_t : omega_s;
                double violation = fixed_norm2 + local_free_norm2 - 2.0 * fixed_endpoint * free_endpoint;
                hi = metric * violation / (2.0 * fixed_endpoint * fixed_endpoint);
                hi *= 1.0 + 64.0 * DBL_EPSILON;
            }
        }
        else if (s_input >= 0.0 && t_input >= 0.0 && fixed_norm2 + local_free_norm2 <= 2.0 * s_input * t_input)
        {
            mode = RSOC_BLOCK_IDENTITY;
        }
        else
        {
            double bs = omega_s * s_input;
            double bt = omega_t * t_input;
            if (fixed_norm2 == 0.0 && bs <= 0.0 && bt <= 0.0 && local_polar_norm2 <= 2.0 * bs * bt)
            {
                mode = RSOC_BLOCK_APEX;
            }
            else
            {
                double root_metric = sqrt(omega_s) * sqrt(omega_t);
                double balance = sqrt(omega_s) * s_input + sqrt(omega_t) * t_input;
                double balance_scale = 1.0 + fabs(sqrt(omega_s) * s_input) + fabs(sqrt(omega_t) * t_input);
                lambda = root_metric;
                if (fabs(balance) <= 64.0 * DBL_EPSILON * balance_scale)
                    mode = RSOC_BLOCK_BALANCED;
                else
                {
                    mode = RSOC_BLOCK_FREE_ROOT;
                    lower_branch = balance > 0.0;
                    lo = lower_branch ? 0.0 : root_metric * (1.0 + 1e-14);
                    if (lower_branch)
                    {
                        hi = root_metric * (1.0 - 1e-14);
                    }
                    else
                    {
                        hi = cone_section_negative_rsoc_upper(
                            omega_s, omega_t, s_input, t_input, fixed_norm2, local_polar_norm2, local_max_omega);
                    }
                }
            }
        }
    }
    __syncthreads();

    if (mode == RSOC_BLOCK_IDENTITY)
        return;
    if (mode == RSOC_BLOCK_ZERO_FREE || mode == RSOC_BLOCK_APEX)
    {
        for (int slot = threadIdx.x; slot < k; slot += blockDim.x)
            if (!(is_fixed && is_fixed[start + slot]))
                point[start + slot] = 0.0;
        if (mode == RSOC_BLOCK_APEX && threadIdx.x == 0)
        {
            point[s_index] = 0.0;
            point[t_index] = 0.0;
        }
        return;
    }
    if (mode == RSOC_BLOCK_ONE_ENDPOINT_ZERO)
    {
        for (int slot = threadIdx.x; slot < k; slot += blockDim.x)
            if (!(is_fixed && is_fixed[start + slot]))
                point[start + slot] = 0.0;
        if (threadIdx.x == 0)
        {
            if (fixed_s)
                point[t_index] = fmax(t_input, 0.0) * rescaling[t_index];
            else
                point[s_index] = fmax(s_input, 0.0) * rescaling[s_index];
        }
        return;
    }
    if (mode == RSOC_BLOCK_ONE_ENDPOINT_SCALAR)
    {
        if (threadIdx.x == 0)
        {
            double fixed_endpoint = fixed_s ? s_input : t_input;
            double free_endpoint = fixed_s ? t_input : s_input;
            double projected = fmax(free_endpoint, fixed_norm2 / (2.0 * fixed_endpoint));
            if (fixed_s)
                point[t_index] = projected * rescaling[t_index];
            else
                point[s_index] = projected * rescaling[s_index];
        }
        return;
    }

    if (mode == RSOC_BLOCK_FIXED_ENDPOINTS_ROOT || mode == RSOC_BLOCK_ONE_ENDPOINT_ROOT)
    {
        if (threadIdx.x == 0)
        {
            lo = 0.0;
            double metric = mode == RSOC_BLOCK_ONE_ENDPOINT_ROOT ? (fixed_s ? omega_t : omega_s) : 1.0;
            done = hi > 0.0 && isfinite(hi);
            if (!done)
                hi = warm_start && warm_start[cone] > 0.0 && isfinite(warm_start[cone]) ? warm_start[cone] : metric;
        }
        __syncthreads();
        for (int expansion = 0; expansion < 80; ++expansion)
        {
            if (done)
                break;
            double norm2 = 0.0;
            double dummy = 0.0;
            double dummy2 = 0.0;
            for (int slot = threadIdx.x; slot < k; slot += blockDim.x)
            {
                int index = start + slot;
                if (is_fixed && is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = (point[index] / rescaling[index]) * omega / (omega + hi);
                norm2 += value * value;
            }
            cone_block_sum3(&norm2, &dummy, &dummy2, scratch);
            if (threadIdx.x == 0)
            {
                if (mode == RSOC_BLOCK_FIXED_ENDPOINTS_ROOT)
                    done = norm2 <= radius2;
                else
                {
                    double fixed_endpoint = fixed_s ? s_input : t_input;
                    double free_endpoint = fixed_s ? t_input : s_input;
                    double omega_endpoint = fixed_s ? omega_t : omega_s;
                    double endpoint = free_endpoint + hi * fixed_endpoint / omega_endpoint;
                    done = fixed_norm2 + norm2 <= 2.0 * fixed_endpoint * endpoint;
                }
                if (!done)
                    hi *= 2.0;
            }
            __syncthreads();
            if (done)
                break;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            double warm = warm_start ? warm_start[cone] : 0.0;
            lambda = warm > lo && warm < hi && isfinite(warm) ? warm : 0.5 * (lo + hi);
            done = 0;
        }
        __syncthreads();
        for (int iteration = 0; iteration < 30; ++iteration)
        {
            double norm2 = 0.0;
            double derivative = 0.0;
            double dummy = 0.0;
            for (int slot = threadIdx.x; slot < k; slot += blockDim.x)
            {
                int index = start + slot;
                if (is_fixed && is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = (point[index] / rescaling[index]) * omega / (omega + lambda);
                norm2 += value * value;
                derivative -= 2.0 * value * value / (omega + lambda);
            }
            cone_block_sum3(&norm2, &derivative, &dummy, scratch);
            if (threadIdx.x == 0)
            {
                double target;
                double f;
                if (mode == RSOC_BLOCK_FIXED_ENDPOINTS_ROOT)
                {
                    target = radius2;
                    f = norm2 - target;
                }
                else
                {
                    double fixed_endpoint = fixed_s ? s_input : t_input;
                    double free_endpoint = fixed_s ? t_input : s_input;
                    double omega_endpoint = fixed_s ? omega_t : omega_s;
                    double endpoint = free_endpoint + lambda * fixed_endpoint / omega_endpoint;
                    target = 2.0 * fixed_endpoint * endpoint;
                    f = fixed_norm2 + norm2 - target;
                    derivative -= 2.0 * fixed_endpoint * fixed_endpoint / omega_endpoint;
                }
                if (f > 0.0)
                    lo = lambda;
                else
                    hi = lambda;
                double next = lambda - f / derivative;
                if (!isfinite(next) || !(next > lo && next < hi))
                    next = 0.5 * (lo + hi);
                done = fabs(f) <= 1e-13 * (1.0 + target) || hi - lo <= 1e-13 * (1.0 + hi + lo);
                if (!done)
                    lambda = next;
            }
            __syncthreads();
            if (done)
                break;
        }

        if (threadIdx.x == 0 && mode == RSOC_BLOCK_ONE_ENDPOINT_ROOT)
        {
            double fixed_endpoint = fixed_s ? s_input : t_input;
            double free_endpoint = fixed_s ? t_input : s_input;
            double omega_endpoint = fixed_s ? omega_t : omega_s;
            double projected = free_endpoint + lambda * fixed_endpoint / omega_endpoint;
            if (fixed_s)
                point[t_index] = projected * rescaling[t_index];
            else
                point[s_index] = projected * rescaling[s_index];
        }
    }
    else if (mode == RSOC_BLOCK_BALANCED)
    {
        double norm2 = 0.0;
        double dummy = 0.0;
        double dummy2 = 0.0;
        for (int slot = threadIdx.x; slot < k; slot += blockDim.x)
        {
            int index = start + slot;
            if (is_fixed && is_fixed[index])
                continue;
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            double value = (point[index] / rescaling[index]) * omega / (omega + lambda);
            norm2 += value * value;
        }
        cone_block_sum3(&norm2, &dummy, &dummy2, scratch);
        if (threadIdx.x == 0)
        {
            double root_metric = sqrt(omega_s) * sqrt(omega_t);
            double product = 0.5 * root_metric * (fixed_norm2 + norm2);
            double delta = sqrt(omega_s) * s_input;
            double scaled_t = 0.5 * (-delta + sqrt(fmax(0.0, delta * delta + 4.0 * product)));
            double scaled_s = scaled_t + delta;
            projected_s = scaled_s / sqrt(omega_s);
            projected_t = scaled_t / sqrt(omega_t);
        }
        __syncthreads();
    }
    else
    {
        if (!lower_branch)
        {
            if (threadIdx.x == 0)
            {
                done = hi > lo && isfinite(hi);
                if (!done)
                    hi = 2.0 * sqrt(omega_s) * sqrt(omega_t);
            }
            __syncthreads();
            for (int expansion = 0; expansion < 80; ++expansion)
            {
                if (done)
                    break;
                double norm2 = 0.0;
                double dummy = 0.0;
                double dummy2 = 0.0;
                for (int slot = threadIdx.x; slot < k; slot += blockDim.x)
                {
                    int index = start + slot;
                    if (is_fixed && is_fixed[index])
                        continue;
                    double omega = cone_section_weight(rescaling, q_diag, tau, index);
                    double value = (point[index] / rescaling[index]) * omega / (omega + hi);
                    norm2 += value * value;
                }
                cone_block_sum3(&norm2, &dummy, &dummy2, scratch);
                if (threadIdx.x == 0)
                {
                    double determinant = omega_s * omega_t - hi * hi;
                    double s = omega_t * (omega_s * s_input + hi * t_input) / determinant;
                    double t = omega_s * (omega_t * t_input + hi * s_input) / determinant;
                    double f = (s >= 0.0 && t >= 0.0) ? fixed_norm2 + norm2 - 2.0 * s * t : INFINITY;
                    done = f >= 0.0;
                    if (!done)
                        hi *= 2.0;
                }
                __syncthreads();
                if (done)
                    break;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            double warm = warm_start ? warm_start[cone] : 0.0;
            lambda = warm > lo && warm < hi && isfinite(warm) ? warm : 0.5 * (lo + hi);
            done = 0;
        }
        __syncthreads();
        for (int iteration = 0; iteration < 40; ++iteration)
        {
            double norm2 = 0.0;
            double derivative = 0.0;
            double dummy = 0.0;
            for (int slot = threadIdx.x; slot < k; slot += blockDim.x)
            {
                int index = start + slot;
                if (is_fixed && is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = (point[index] / rescaling[index]) * omega / (omega + lambda);
                norm2 += value * value;
                derivative -= 2.0 * value * value / (omega + lambda);
            }
            cone_block_sum3(&norm2, &derivative, &dummy, scratch);
            if (threadIdx.x == 0)
            {
                double determinant = omega_s * omega_t - lambda * lambda;
                double s = omega_t * (omega_s * s_input + lambda * t_input) / determinant;
                double t = omega_s * (omega_t * t_input + lambda * s_input) / determinant;
                double f = INFINITY;
                if (s >= 0.0 && t >= 0.0)
                {
                    f = fixed_norm2 + norm2 - 2.0 * s * t;
                    double ds = (omega_t * t + lambda * s) / determinant;
                    double dt = (lambda * t + omega_s * s) / determinant;
                    derivative -= 2.0 * (ds * t + s * dt);
                }
                if ((lower_branch && f > 0.0) || (!lower_branch && f < 0.0))
                    lo = lambda;
                else
                    hi = lambda;
                double next = lambda - f / derivative;
                if (!isfinite(next) || !(next > lo && next < hi))
                    next = 0.5 * (lo + hi);
                done = isfinite(f) &&
                    (fabs(f) <= 1e-13 * (1.0 + fixed_norm2 + norm2 + 2.0 * s * t) ||
                     hi - lo <= 1e-13 * (1.0 + hi + lo));
                if (!done)
                    lambda = next;
            }
            __syncthreads();
            if (done)
                break;
        }
        if (threadIdx.x == 0)
        {
            double determinant = omega_s * omega_t - lambda * lambda;
            projected_s = omega_t * (omega_s * s_input + lambda * t_input) / determinant;
            projected_t = omega_s * (omega_t * t_input + lambda * s_input) / determinant;
        }
        __syncthreads();
    }

    if (mode == RSOC_BLOCK_BALANCED || mode == RSOC_BLOCK_FREE_ROOT)
    {
        double smooth_vector_objective = 0.0;
        double dummy = 0.0;
        double dummy2 = 0.0;
        for (int slot = threadIdx.x; slot < k; slot += blockDim.x)
        {
            int index = start + slot;
            if (is_fixed && is_fixed[index])
                continue;
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            double input = point[index] / rescaling[index];
            double value = input * omega / (omega + lambda);
            double delta = value - input;
            smooth_vector_objective += omega * delta * delta;
        }
        cone_block_sum3(&smooth_vector_objective, &dummy, &dummy2, scratch);
        if (threadIdx.x == 0 && fixed_norm2 == 0.0)
        {
            double smooth_objective = smooth_vector_objective +
                omega_s * (projected_s - s_input) * (projected_s - s_input) +
                omega_t * (projected_t - t_input) * (projected_t - t_input);
            double s_axis = fmax(s_input, 0.0);
            double s_axis_objective =
                free_objective + omega_s * (s_axis - s_input) * (s_axis - s_input) + omega_t * t_input * t_input;
            double t_axis = fmax(t_input, 0.0);
            double t_axis_objective =
                free_objective + omega_s * s_input * s_input + omega_t * (t_axis - t_input) * (t_axis - t_input);
            if (s_axis_objective < smooth_objective && s_axis_objective <= t_axis_objective)
            {
                projected_s = s_axis;
                projected_t = 0.0;
                mode = RSOC_BLOCK_AXIS;
            }
            else if (t_axis_objective < smooth_objective)
            {
                projected_s = 0.0;
                projected_t = t_axis;
                mode = RSOC_BLOCK_AXIS;
            }
        }
        __syncthreads();
    }

    if (mode == RSOC_BLOCK_AXIS)
    {
        for (int slot = threadIdx.x; slot < k; slot += blockDim.x)
            if (!(is_fixed && is_fixed[start + slot]))
                point[start + slot] = 0.0;
    }
    else
    {
        for (int slot = threadIdx.x; slot < k; slot += blockDim.x)
        {
            int index = start + slot;
            if (!(is_fixed && is_fixed[index]))
            {
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                point[index] *= omega / (omega + lambda);
            }
        }
        if (warm_start && threadIdx.x == 0)
            warm_start[cone] = lambda;
    }
    if (threadIdx.x == 0 && (mode == RSOC_BLOCK_BALANCED || mode == RSOC_BLOCK_FREE_ROOT || mode == RSOC_BLOCK_AXIS))
    {
        point[s_index] = projected_s * rescaling[s_index];
        point[t_index] = projected_t * rescaling[t_index];
    }
}

enum rotated_soc_grid_weighted_mode
{
    RSOC_GRID_IDENTITY = 0,
    RSOC_GRID_ZERO_FREE = 1,
    RSOC_GRID_ONE_ENDPOINT_ZERO = 2,
    RSOC_GRID_ONE_ENDPOINT_SCALAR = 3,
    RSOC_GRID_APEX = 4,
    RSOC_GRID_FIXED_EXPAND = 5,
    RSOC_GRID_FIXED_ROOT = 6,
    RSOC_GRID_FIXED_APPLY = 7,
    RSOC_GRID_ONE_EXPAND = 8,
    RSOC_GRID_ONE_ROOT = 9,
    RSOC_GRID_ONE_APPLY = 10,
    RSOC_GRID_BALANCED_EVAL = 11,
    RSOC_GRID_FREE_EXPAND = 12,
    RSOC_GRID_FREE_ROOT = 13,
    RSOC_GRID_FREE_APPLY = 14,
    RSOC_GRID_BALANCED_APPLY = 15,
    RSOC_GRID_AXIS = 16
};

__device__ static inline void rotated_soc_grid_free_endpoints(const double *point,
                                                              const double *rescaling,
                                                              const double *q_diag,
                                                              double tau,
                                                              int s_index,
                                                              int t_index,
                                                              double lambda,
                                                              double *projected_s,
                                                              double *projected_t)
{
    double s = point[s_index] / rescaling[s_index];
    double t = point[t_index] / rescaling[t_index];
    double omega_s = cone_section_weight(rescaling, q_diag, tau, s_index);
    double omega_t = cone_section_weight(rescaling, q_diag, tau, t_index);
    double determinant = omega_s * omega_t - lambda * lambda;
    *projected_s = omega_t * (omega_s * s + lambda * t) / determinant;
    *projected_t = omega_s * (omega_t * t + lambda * s) / determinant;
}

__global__ void initialize_rotated_soc_grid_weighted_kernel(const double *__restrict__ point,
                                                            const double *__restrict__ rescaling,
                                                            const double *__restrict__ q_diag,
                                                            double tau,
                                                            double *__restrict__ workspace,
                                                            const int *__restrict__ start_idx,
                                                            const int *__restrict__ v_dim,
                                                            const char *__restrict__ is_fixed,
                                                            int num_cones,
                                                            int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    if (cone >= num_cones)
        return;
    int part = blockIdx.x - cone * blocks_per_cone;
    int start = start_idx[cone];
    int k = v_dim[cone];
    double fixed_norm2 = 0.0;
    double free_norm2 = 0.0;
    double polar_norm2 = 0.0;
    double free_count = 0.0;
    double max_omega = 0.0;
    for (int slot = part * blockDim.x + threadIdx.x; slot < k; slot += blocks_per_cone * blockDim.x)
    {
        int index = start + slot;
        double value = point[index] / rescaling[index];
        if (is_fixed && is_fixed[index])
            fixed_norm2 += value * value;
        else
        {
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            free_norm2 += value * value;
            polar_norm2 += (omega * value) * (omega * value);
            free_count += 1.0;
            max_omega = fmax(max_omega, omega);
        }
    }
    __shared__ double scratch[96];
    cone_block_sum3(&fixed_norm2, &free_norm2, &polar_norm2, scratch);
    double unused = 0.0;
    double unused2 = 0.0;
    cone_block_sum3(&free_count, &unused, &unused2, scratch);
    max_omega = cone_block_max(max_omega, scratch);
    if (threadIdx.x == 0)
    {
        atomicAdd(workspace + num_cones + cone, fixed_norm2);
        atomicAdd(workspace + 2 * num_cones + cone, free_norm2);
        atomicAdd(workspace + 3 * num_cones + cone, polar_norm2);
        atomicAdd(workspace + 4 * num_cones + cone, free_count);
        cone_atomic_max_positive(workspace + 5 * num_cones + cone, max_omega);
    }
}

__global__ void finalize_rotated_soc_grid_weighted_initialization_kernel(const double *__restrict__ point,
                                                                         const double *__restrict__ rescaling,
                                                                         const double *__restrict__ q_diag,
                                                                         double tau,
                                                                         double *__restrict__ workspace,
                                                                         const int *__restrict__ start_idx,
                                                                         const int *__restrict__ v_dim,
                                                                         const char *__restrict__ is_fixed,
                                                                         int num_cones)
{
    int cone = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone >= num_cones)
        return;
    int start = start_idx[cone];
    int k = v_dim[cone];
    int s_index = start + k;
    int t_index = s_index + 1;
    bool fixed_s = is_fixed && is_fixed[s_index];
    bool fixed_t = is_fixed && is_fixed[t_index];
    double warm = workspace[cone];
    double fixed_norm2 = workspace[num_cones + cone];
    double free_norm2 = workspace[2 * num_cones + cone];
    double polar_norm2 = workspace[3 * num_cones + cone];
    int free_count = (int)workspace[4 * num_cones + cone];
    double max_omega = workspace[5 * num_cones + cone];
    double s = point[s_index] / rescaling[s_index];
    double t = point[t_index] / rescaling[t_index];
    double omega_s_value = cone_section_weight(rescaling, q_diag, tau, s_index);
    double omega_t_value = cone_section_weight(rescaling, q_diag, tau, t_index);
    int selected_mode;
    double constant = fixed_norm2;
    double lower = 0.0;
    double upper = 0.0;
    double trial = warm;

    if (fixed_s && fixed_t)
    {
        constant = fmax(0.0, 2.0 * s * t - fixed_norm2);
        if (free_count == 0 || free_norm2 <= constant)
            selected_mode = RSOC_GRID_IDENTITY;
        else if (!(constant > 0.0))
            selected_mode = RSOC_GRID_ZERO_FREE;
        else
        {
            upper = sqrt(polar_norm2) / sqrt(constant) * (1.0 + 64.0 * DBL_EPSILON);
            if (upper > 0.0 && isfinite(upper))
            {
                selected_mode = RSOC_GRID_FIXED_ROOT;
                trial = warm > 0.0 && warm < upper && isfinite(warm) ? warm : 0.5 * upper;
            }
            else
            {
                selected_mode = RSOC_GRID_FIXED_EXPAND;
                trial = warm > 0.0 && isfinite(warm) ? warm : 1.0;
                upper = trial;
            }
        }
    }
    else if (fixed_s || fixed_t)
    {
        double fixed_endpoint = fixed_s ? s : t;
        double free_endpoint = fixed_s ? t : s;
        if (!(fixed_endpoint > 0.0))
            selected_mode = RSOC_GRID_ONE_ENDPOINT_ZERO;
        else if (free_endpoint >= 0.0 && fixed_norm2 + free_norm2 <= 2.0 * fixed_endpoint * free_endpoint)
            selected_mode = RSOC_GRID_IDENTITY;
        else if (free_count == 0)
            selected_mode = RSOC_GRID_ONE_ENDPOINT_SCALAR;
        else
        {
            double metric = fixed_s ? omega_t_value : omega_s_value;
            double violation = fixed_norm2 + free_norm2 - 2.0 * fixed_endpoint * free_endpoint;
            upper = metric * violation / (2.0 * fixed_endpoint * fixed_endpoint);
            upper *= 1.0 + 64.0 * DBL_EPSILON;
            if (upper > 0.0 && isfinite(upper))
            {
                selected_mode = RSOC_GRID_ONE_ROOT;
                trial = warm > 0.0 && warm < upper && isfinite(warm) ? warm : 0.5 * upper;
            }
            else
            {
                selected_mode = RSOC_GRID_ONE_EXPAND;
                trial = warm > 0.0 && isfinite(warm) ? warm : metric;
                upper = trial;
            }
        }
    }
    else if (s >= 0.0 && t >= 0.0 && fixed_norm2 + free_norm2 <= 2.0 * s * t)
    {
        selected_mode = RSOC_GRID_IDENTITY;
    }
    else
    {
        double bs = omega_s_value * s;
        double bt = omega_t_value * t;
        if (fixed_norm2 == 0.0 && bs <= 0.0 && bt <= 0.0 && polar_norm2 <= 2.0 * bs * bt)
        {
            selected_mode = RSOC_GRID_APEX;
        }
        else
        {
            double sqrt_omega_s = sqrt(omega_s_value);
            double sqrt_omega_t = sqrt(omega_t_value);
            double root_metric = sqrt_omega_s * sqrt_omega_t;
            double scaled_s = sqrt_omega_s * s;
            double scaled_t = sqrt_omega_t * t;
            double balance = scaled_s + scaled_t;
            double balance_scale = 1.0 + fabs(scaled_s) + fabs(scaled_t);
            if (fabs(balance) <= 64.0 * DBL_EPSILON * balance_scale)
            {
                selected_mode = RSOC_GRID_BALANCED_EVAL;
                trial = root_metric;
            }
            else if (balance > 0.0)
            {
                selected_mode = RSOC_GRID_FREE_ROOT;
                lower = 0.0;
                upper = root_metric * (1.0 - 1e-14);
                trial = warm > lower && warm < upper && isfinite(warm) ? warm : 0.5 * (lower + upper);
            }
            else
            {
                lower = root_metric * (1.0 + 1e-14);
                upper = cone_section_negative_rsoc_upper(
                    omega_s_value, omega_t_value, s, t, fixed_norm2, polar_norm2, max_omega);
                if (upper > lower && isfinite(upper))
                {
                    selected_mode = RSOC_GRID_FREE_ROOT;
                    trial = warm > lower && warm < upper && isfinite(warm) ? warm : 0.5 * (lower + upper);
                }
                else
                {
                    selected_mode = RSOC_GRID_FREE_EXPAND;
                    trial = warm > lower && isfinite(warm) ? warm : 2.0 * root_metric;
                    upper = trial;
                }
            }
        }
    }

    workspace[cone] = trial;
    workspace[4 * num_cones + cone] = (double)selected_mode;
    workspace[5 * num_cones + cone] = constant;
    workspace[6 * num_cones + cone] = lower;
    workspace[7 * num_cones + cone] = upper;
}

__global__ void reduce_rotated_soc_grid_weighted_root_kernel(const double *__restrict__ point,
                                                             const double *__restrict__ rescaling,
                                                             const double *__restrict__ q_diag,
                                                             double tau,
                                                             double *__restrict__ workspace,
                                                             const int *__restrict__ start_idx,
                                                             const int *__restrict__ v_dim,
                                                             const char *__restrict__ is_fixed,
                                                             int num_cones,
                                                             int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    if (cone >= num_cones)
        return;
    int selected_mode = (int)workspace[4 * num_cones + cone];
    bool active = selected_mode == RSOC_GRID_FIXED_EXPAND || selected_mode == RSOC_GRID_FIXED_ROOT ||
        selected_mode == RSOC_GRID_ONE_EXPAND || selected_mode == RSOC_GRID_ONE_ROOT ||
        selected_mode == RSOC_GRID_BALANCED_EVAL || selected_mode == RSOC_GRID_FREE_EXPAND ||
        selected_mode == RSOC_GRID_FREE_ROOT;
    if (!active)
        return;
    int part = blockIdx.x - cone * blocks_per_cone;
    int start = start_idx[cone];
    int k = v_dim[cone];
    double lambda_value = workspace[cone];
    double norm2 = 0.0;
    double derivative = 0.0;
    for (int slot = part * blockDim.x + threadIdx.x; slot < k; slot += blocks_per_cone * blockDim.x)
    {
        int index = start + slot;
        if (is_fixed && is_fixed[index])
            continue;
        double omega = cone_section_weight(rescaling, q_diag, tau, index);
        double value = (point[index] / rescaling[index]) * omega / (omega + lambda_value);
        norm2 += value * value;
        derivative -= 2.0 * value * value / (omega + lambda_value);
    }
    __shared__ double scratch[96];
    double unused = 0.0;
    cone_block_sum3(&norm2, &derivative, &unused, scratch);
    if (threadIdx.x == 0)
    {
        atomicAdd(workspace + num_cones + cone, norm2);
        atomicAdd(workspace + 2 * num_cones + cone, derivative);
    }
}

__global__ void finalize_rotated_soc_grid_weighted_root_kernel(const double *__restrict__ point,
                                                               const double *__restrict__ rescaling,
                                                               const double *__restrict__ q_diag,
                                                               double tau,
                                                               double *__restrict__ workspace,
                                                               const int *__restrict__ start_idx,
                                                               const int *__restrict__ v_dim,
                                                               const char *__restrict__ is_fixed,
                                                               int num_cones)
{
    int cone = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone >= num_cones)
        return;
    int selected_mode = (int)workspace[4 * num_cones + cone];
    bool active = selected_mode == RSOC_GRID_FIXED_EXPAND || selected_mode == RSOC_GRID_FIXED_ROOT ||
        selected_mode == RSOC_GRID_ONE_EXPAND || selected_mode == RSOC_GRID_ONE_ROOT ||
        selected_mode == RSOC_GRID_BALANCED_EVAL || selected_mode == RSOC_GRID_FREE_EXPAND ||
        selected_mode == RSOC_GRID_FREE_ROOT;
    if (!active)
        return;
    int start = start_idx[cone];
    int k = v_dim[cone];
    int s_index = start + k;
    int t_index = s_index + 1;
    bool fixed_s = is_fixed && is_fixed[s_index];
    double s_input_value = point[s_index] / rescaling[s_index];
    double t_input_value = point[t_index] / rescaling[t_index];
    double omega_s_value = cone_section_weight(rescaling, q_diag, tau, s_index);
    double omega_t_value = cone_section_weight(rescaling, q_diag, tau, t_index);
    double lambda_value = workspace[cone];
    double sum = workspace[num_cones + cone];
    double derivative = workspace[2 * num_cones + cone];
    double constant = workspace[5 * num_cones + cone];
    double lower = workspace[6 * num_cones + cone];
    double upper = workspace[7 * num_cones + cone];
    double f = 0.0;

    if (selected_mode == RSOC_GRID_FIXED_EXPAND || selected_mode == RSOC_GRID_FIXED_ROOT)
    {
        f = sum - constant;
        if (selected_mode == RSOC_GRID_FIXED_EXPAND)
        {
            if (f > 0.0)
            {
                lower = lambda_value;
                lambda_value *= 2.0;
            }
            else
            {
                upper = lambda_value;
                selected_mode = RSOC_GRID_FIXED_ROOT;
                lambda_value = 0.5 * (lower + upper);
            }
        }
        else
        {
            if (f > 0.0)
                lower = lambda_value;
            else
                upper = lambda_value;
            bool converged = fabs(f) <= 1e-13 * (1.0 + constant) || upper - lower <= 1e-13 * (1.0 + upper + lower);
            if (converged)
                selected_mode = RSOC_GRID_FIXED_APPLY;
            else
            {
                double next = lambda_value - f / derivative;
                lambda_value = isfinite(next) && next > lower && next < upper ? next : 0.5 * (lower + upper);
            }
        }
    }
    else if (selected_mode == RSOC_GRID_ONE_EXPAND || selected_mode == RSOC_GRID_ONE_ROOT)
    {
        double fixed_endpoint = fixed_s ? s_input_value : t_input_value;
        double free_endpoint = fixed_s ? t_input_value : s_input_value;
        double omega_endpoint = fixed_s ? omega_t_value : omega_s_value;
        double projected_endpoint = free_endpoint + lambda_value * fixed_endpoint / omega_endpoint;
        f = constant + sum - 2.0 * fixed_endpoint * projected_endpoint;
        derivative -= 2.0 * fixed_endpoint * fixed_endpoint / omega_endpoint;
        if (selected_mode == RSOC_GRID_ONE_EXPAND)
        {
            if (f > 0.0)
            {
                lower = lambda_value;
                lambda_value *= 2.0;
            }
            else
            {
                upper = lambda_value;
                selected_mode = RSOC_GRID_ONE_ROOT;
                lambda_value = 0.5 * (lower + upper);
            }
        }
        else
        {
            if (f > 0.0)
                lower = lambda_value;
            else
                upper = lambda_value;
            bool converged =
                fabs(f) <= 1e-13 * (1.0 + constant + sum) || upper - lower <= 1e-13 * (1.0 + upper + lower);
            if (converged)
                selected_mode = RSOC_GRID_ONE_APPLY;
            else
            {
                double next = lambda_value - f / derivative;
                lambda_value = isfinite(next) && next > lower && next < upper ? next : 0.5 * (lower + upper);
            }
        }
    }
    else if (selected_mode == RSOC_GRID_BALANCED_EVAL)
    {
        double root_metric = sqrt(omega_s_value) * sqrt(omega_t_value);
        double product = 0.5 * root_metric * (constant + sum);
        double delta = sqrt(omega_s_value) * s_input_value;
        double scaled_t = 0.5 * (-delta + sqrt(fmax(0.0, delta * delta + 4.0 * product)));
        workspace[6 * num_cones + cone] = (scaled_t + delta) / sqrt(omega_s_value);
        workspace[7 * num_cones + cone] = scaled_t / sqrt(omega_t_value);
        selected_mode = RSOC_GRID_BALANCED_APPLY;
    }
    else
    {
        double determinant = omega_s_value * omega_t_value - lambda_value * lambda_value;
        double projected_s_value;
        double projected_t_value;
        rotated_soc_grid_free_endpoints(
            point, rescaling, q_diag, tau, s_index, t_index, lambda_value, &projected_s_value, &projected_t_value);
        f = projected_s_value >= 0.0 && projected_t_value >= 0.0
            ? constant + sum - 2.0 * projected_s_value * projected_t_value
            : INFINITY;
        if (isfinite(f))
        {
            double ds = (omega_t_value * projected_t_value + lambda_value * projected_s_value) / determinant;
            double dt = (lambda_value * projected_t_value + omega_s_value * projected_s_value) / determinant;
            derivative -= 2.0 * (ds * projected_t_value + projected_s_value * dt);
        }
        if (selected_mode == RSOC_GRID_FREE_EXPAND)
        {
            if (f < 0.0)
            {
                lower = lambda_value;
                lambda_value *= 2.0;
            }
            else
            {
                upper = lambda_value;
                selected_mode = RSOC_GRID_FREE_ROOT;
                lambda_value = 0.5 * (lower + upper);
            }
        }
        else
        {
            bool lower_branch_value = sqrt(omega_s_value) * s_input_value + sqrt(omega_t_value) * t_input_value > 0.0;
            if ((lower_branch_value && f > 0.0) || (!lower_branch_value && f < 0.0))
                lower = lambda_value;
            else
                upper = lambda_value;
            bool converged = isfinite(f) &&
                (fabs(f) <= 1e-13 * (1.0 + constant + sum + 2.0 * projected_s_value * projected_t_value) ||
                 upper - lower <= 1e-13 * (1.0 + upper + lower));
            if (converged)
                selected_mode = RSOC_GRID_FREE_APPLY;
            else
            {
                double next = lambda_value - f / derivative;
                lambda_value = isfinite(next) && next > lower && next < upper ? next : 0.5 * (lower + upper);
            }
        }
    }

    workspace[cone] = lambda_value;
    workspace[4 * num_cones + cone] = (double)selected_mode;
    if (selected_mode != RSOC_GRID_BALANCED_APPLY)
    {
        workspace[6 * num_cones + cone] = lower;
        workspace[7 * num_cones + cone] = upper;
    }
}

__global__ void reduce_rotated_soc_grid_axis_objective_kernel(const double *__restrict__ point,
                                                              const double *__restrict__ rescaling,
                                                              const double *__restrict__ q_diag,
                                                              double tau,
                                                              double *__restrict__ workspace,
                                                              const int *__restrict__ start_idx,
                                                              const int *__restrict__ v_dim,
                                                              const char *__restrict__ is_fixed,
                                                              int num_cones,
                                                              int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    if (cone >= num_cones || workspace[5 * num_cones + cone] != 0.0)
        return;
    int selected_mode = (int)workspace[4 * num_cones + cone];
    if (selected_mode != RSOC_GRID_FREE_EXPAND && selected_mode != RSOC_GRID_FREE_ROOT &&
        selected_mode != RSOC_GRID_FREE_APPLY && selected_mode != RSOC_GRID_BALANCED_APPLY)
        return;
    int part = blockIdx.x - cone * blocks_per_cone;
    int start = start_idx[cone];
    int k = v_dim[cone];
    double lambda_value = workspace[cone];
    double smooth_objective = 0.0;
    double axis_objective = 0.0;
    for (int slot = part * blockDim.x + threadIdx.x; slot < k; slot += blocks_per_cone * blockDim.x)
    {
        int index = start + slot;
        if (is_fixed && is_fixed[index])
            continue;
        double omega = cone_section_weight(rescaling, q_diag, tau, index);
        double input = point[index] / rescaling[index];
        double projected = input * omega / (omega + lambda_value);
        double delta = projected - input;
        smooth_objective += omega * delta * delta;
        axis_objective += omega * input * input;
    }
    __shared__ double scratch[96];
    double unused = 0.0;
    cone_block_sum3(&smooth_objective, &axis_objective, &unused, scratch);
    if (threadIdx.x == 0)
    {
        atomicAdd(workspace + num_cones + cone, smooth_objective);
        atomicAdd(workspace + 2 * num_cones + cone, axis_objective);
    }
}

__global__ void finalize_rotated_soc_grid_axis_objective_kernel(const double *__restrict__ point,
                                                                const double *__restrict__ rescaling,
                                                                const double *__restrict__ q_diag,
                                                                double tau,
                                                                double *__restrict__ workspace,
                                                                const int *__restrict__ start_idx,
                                                                const int *__restrict__ v_dim,
                                                                int num_cones)
{
    int cone = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone >= num_cones || workspace[5 * num_cones + cone] != 0.0)
        return;
    int selected_mode = (int)workspace[4 * num_cones + cone];
    if (selected_mode != RSOC_GRID_FREE_EXPAND && selected_mode != RSOC_GRID_FREE_ROOT &&
        selected_mode != RSOC_GRID_FREE_APPLY && selected_mode != RSOC_GRID_BALANCED_APPLY)
        return;
    int start = start_idx[cone];
    int k = v_dim[cone];
    int s_index = start + k;
    int t_index = s_index + 1;
    double s_input_value = point[s_index] / rescaling[s_index];
    double t_input_value = point[t_index] / rescaling[t_index];
    double omega_s_value = cone_section_weight(rescaling, q_diag, tau, s_index);
    double omega_t_value = cone_section_weight(rescaling, q_diag, tau, t_index);
    double projected_s_value;
    double projected_t_value;
    if (selected_mode == RSOC_GRID_BALANCED_APPLY)
    {
        projected_s_value = workspace[6 * num_cones + cone];
        projected_t_value = workspace[7 * num_cones + cone];
    }
    else
    {
        rotated_soc_grid_free_endpoints(
            point, rescaling, q_diag, tau, s_index, t_index, workspace[cone], &projected_s_value, &projected_t_value);
    }
    double smooth_objective = workspace[num_cones + cone] +
        omega_s_value * (projected_s_value - s_input_value) * (projected_s_value - s_input_value) +
        omega_t_value * (projected_t_value - t_input_value) * (projected_t_value - t_input_value);
    double vector_axis_objective = workspace[2 * num_cones + cone];
    double s_axis = fmax(s_input_value, 0.0);
    double s_axis_objective = vector_axis_objective +
        omega_s_value * (s_axis - s_input_value) * (s_axis - s_input_value) +
        omega_t_value * t_input_value * t_input_value;
    double t_axis = fmax(t_input_value, 0.0);
    double t_axis_objective = vector_axis_objective + omega_s_value * s_input_value * s_input_value +
        omega_t_value * (t_axis - t_input_value) * (t_axis - t_input_value);
    if (s_axis_objective < smooth_objective && s_axis_objective <= t_axis_objective)
    {
        workspace[6 * num_cones + cone] = s_axis;
        workspace[7 * num_cones + cone] = 0.0;
        workspace[4 * num_cones + cone] = (double)RSOC_GRID_AXIS;
    }
    else if (t_axis_objective < smooth_objective)
    {
        workspace[6 * num_cones + cone] = 0.0;
        workspace[7 * num_cones + cone] = t_axis;
        workspace[4 * num_cones + cone] = (double)RSOC_GRID_AXIS;
    }
}

__global__ void apply_rotated_soc_grid_weighted_kernel(double *__restrict__ point,
                                                       const double *__restrict__ rescaling,
                                                       const double *__restrict__ q_diag,
                                                       double tau,
                                                       const double *__restrict__ workspace,
                                                       const int *__restrict__ start_idx,
                                                       const int *__restrict__ v_dim,
                                                       const char *__restrict__ is_fixed,
                                                       int num_cones,
                                                       int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    if (cone >= num_cones)
        return;
    int selected_mode = (int)workspace[4 * num_cones + cone];
    if (selected_mode == RSOC_GRID_IDENTITY)
        return;
    int part = blockIdx.x - cone * blocks_per_cone;
    int start = start_idx[cone];
    int k = v_dim[cone];
    int s_index = start + k;
    int t_index = s_index + 1;
    bool fixed_s = is_fixed && is_fixed[s_index];
    bool fixed_t = is_fixed && is_fixed[t_index];

    bool zero_vector = selected_mode == RSOC_GRID_ZERO_FREE || selected_mode == RSOC_GRID_ONE_ENDPOINT_ZERO ||
        selected_mode == RSOC_GRID_APEX || selected_mode == RSOC_GRID_AXIS;
    if (zero_vector)
    {
        for (int slot = part * blockDim.x + threadIdx.x; slot < k; slot += blocks_per_cone * blockDim.x)
            if (!(is_fixed && is_fixed[start + slot]))
                point[start + slot] = 0.0;
    }
    else if (selected_mode != RSOC_GRID_ONE_ENDPOINT_SCALAR)
    {
        double lambda_value = workspace[cone];
        for (int slot = part * blockDim.x + threadIdx.x; slot < k; slot += blocks_per_cone * blockDim.x)
        {
            int index = start + slot;
            if (!(is_fixed && is_fixed[index]))
            {
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                point[index] *= omega / (omega + lambda_value);
            }
        }
    }

    if (part == 0 && threadIdx.x == 0)
    {
        if (selected_mode == RSOC_GRID_ONE_ENDPOINT_ZERO)
        {
            if (fixed_s)
                point[t_index] = fmax(point[t_index] / rescaling[t_index], 0.0) * rescaling[t_index];
            else
                point[s_index] = fmax(point[s_index] / rescaling[s_index], 0.0) * rescaling[s_index];
        }
        else if (selected_mode == RSOC_GRID_ONE_ENDPOINT_SCALAR)
        {
            double fixed_endpoint = fixed_s ? point[s_index] / rescaling[s_index] : point[t_index] / rescaling[t_index];
            int free_index = fixed_s ? t_index : s_index;
            double input = point[free_index] / rescaling[free_index];
            point[free_index] =
                fmax(input, workspace[5 * num_cones + cone] / (2.0 * fixed_endpoint)) * rescaling[free_index];
        }
        else if (selected_mode == RSOC_GRID_APEX)
        {
            point[s_index] = 0.0;
            point[t_index] = 0.0;
        }
        else if (selected_mode == RSOC_GRID_ONE_EXPAND || selected_mode == RSOC_GRID_ONE_ROOT ||
                 selected_mode == RSOC_GRID_ONE_APPLY)
        {
            double lambda_value = workspace[cone];
            double fixed_endpoint = fixed_s ? point[s_index] / rescaling[s_index] : point[t_index] / rescaling[t_index];
            int free_index = fixed_s ? t_index : s_index;
            double input = point[free_index] / rescaling[free_index];
            double omega = cone_section_weight(rescaling, q_diag, tau, free_index);
            point[free_index] = (input + lambda_value * fixed_endpoint / omega) * rescaling[free_index];
        }
        else if (selected_mode == RSOC_GRID_BALANCED_APPLY || selected_mode == RSOC_GRID_AXIS)
        {
            if (!fixed_s)
                point[s_index] = workspace[6 * num_cones + cone] * rescaling[s_index];
            if (!fixed_t)
                point[t_index] = workspace[7 * num_cones + cone] * rescaling[t_index];
        }
        else if (selected_mode == RSOC_GRID_FREE_EXPAND || selected_mode == RSOC_GRID_FREE_ROOT ||
                 selected_mode == RSOC_GRID_FREE_APPLY)
        {
            double projected_s;
            double projected_t;
            rotated_soc_grid_free_endpoints(
                point, rescaling, q_diag, tau, s_index, t_index, workspace[cone], &projected_s, &projected_t);
            point[s_index] = projected_s * rescaling[s_index];
            point[t_index] = projected_t * rescaling[t_index];
        }
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
