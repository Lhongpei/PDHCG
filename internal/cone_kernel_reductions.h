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

#include <cuda_runtime.h>
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

static __device__ __forceinline__ double large_cone_block_sum(double value)
{
    __shared__ double warp_sums[32];
    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(mask, value, offset);
    if (lane == 0)
        warp_sums[warp] = value;
    __syncthreads();

    value = (warp == 0 && lane < num_warps) ? warp_sums[lane] : 0.0;
    if (warp == 0)
    {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}
