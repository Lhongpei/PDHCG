#include "fisher_func.h"
#include <math.h>  
template <typename real>
__device__ real max(real a, real b) {
    return (a > b) ? a : b;
}

template <typename real>
__device__ __forceinline__ real warpReduce(real val){
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ******************************************
// This Kernel is used to reduce the values in a block. Note that each block will have the duty of reducing 2 * blockSize values.
template <int blockSize, typename real>
__global__ void blockReduce(real* d_A, const int N){
    __shared__ real shm[blockSize];
    int tid = threadIdx.x;
    int i = 2 * blockSize * blockIdx.x + threadIdx.x;
    real sum = i < N ? d_A[i] : 0.0f;
    if (i + blockSize < N) {
        sum += d_A[i + blockSize];
    }
    shm[tid] = sum;
    __syncthreads();
    if (blockSize >= 1024 && tid < 512) { shm[tid] = sum = shm[tid] + shm[tid + 512]; __syncthreads(); }
    if (blockSize >= 512 && tid < 256) { shm[tid] = sum = shm[tid] + shm[tid + 256]; __syncthreads(); }
    if (blockSize >= 256 && tid < 128) { shm[tid] = sum = shm[tid] + shm[tid + 128]; __syncthreads(); }
    if (blockSize >= 128 && tid < 64) { shm[tid] = sum = shm[tid] + shm[tid + 64]; __syncthreads(); }
    if (blockSize >= 64 && tid < 32) { shm[tid] = sum = shm[tid] + shm[tid + 32]; __syncthreads(); }
    if (tid < 32) {
        sum = warpReduce<real>(sum);
    }
    if (tid == 0) {
        d_A[blockIdx.x] = sum;  // 将结果存回全局内存
    }
}

template <int blockSize, typename real>
__global__ void gradient_csr_kernel(
    const int m,
    const real* __restrict__ x_val,
    const real* __restrict__ u_val,
    const real* __restrict__ w,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indice,
    const real power,
    const real* __restrict__ p,
    const real pho,
    const real* __restrict__ x_old_val,
    const real* __restrict__ utility_no_power,
    real* __restrict__ gradient)
{
    int row = blockIdx.x;
    if (row >= m) return;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    int tid = threadIdx.x;

    real utility_denom = fmaxf(utility_no_power[row], 1e-12f);
    real w_row = w[row];
    real inv_pho = 1.0f / pho;

    #pragma unroll 4
    for (int j = start + tid; j < end; j += blockDim.x) {
        real x = __ldg(&x_val[j]);
        real u = __ldg(&u_val[j]);
        real x_old = __ldg(&x_old_val[j]);
        real p_col = __ldg(&p[col_indice[j]]);

        real x_safe = fmaxf(x, 1e-12f);
        real x_pow = (power == 2) ? x_safe : powf(x_safe, power - 1); // 特化 power=2 的情况

        gradient[j] = -w_row * u * x_pow / utility_denom + p_col + inv_pho * (x - x_old);
    }
}

template <typename real>
void launch_gradient_csr(
    int m,
    const real* d_x_val,
    const real* d_u_val,
    const real* d_w,
    const int* d_row_ptr,
    const int* d_col_indice,
    const real power,
    const real* d_p,
    const real pho,
    const real* d_x_old_val,
    real* d_utility_no_power,
    real* d_gradient)
{
    const int blockSize = 256;  
    const int gridSize = m;  // 每行一个 block
    launch_utility_csr<real>(m, d_x_val, d_u_val, d_row_ptr, d_utility_no_power, power);
    cudaDeviceSynchronize();
    gradient_csr_kernel<blockSize, real><<<gridSize, blockSize, 0>>>(
        m, d_x_val, d_u_val, d_w, d_row_ptr, d_col_indice, power,
        d_p, pho, d_x_old_val, d_utility_no_power, d_gradient);
    cudaDeviceSynchronize();
}
template void launch_gradient_csr<double>(
    int, const double*, const double*, const double*, const int*, const int*, const double, const double*, const double, const double*, double*, double*);
template void launch_gradient_csr<float>(
    int, const float*,  const float*,  const float*, const int*, const int*, const float, const float*, const float, const float*, float*, float*);

template <int blockSize, typename real>
__global__ void objective_csr_kernel(
    const int m,
    const real* __restrict__ x_val,
    const real* __restrict__ u_val,
    const real* __restrict__ w,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indice,
    const real power,
    real* __restrict__ objective,
    const real* __restrict__ p,
    const real pho,
    const real* __restrict__ x_old_val)
{
    int row = blockIdx.x;
    if (row >= m) return;

    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    int tid = threadIdx.x;

    __shared__ real shm_sum[blockSize];
    __shared__ real shm_ptot[blockSize];
    __shared__ real shm_delta[blockSize];

    real sum = 0.0;
    real ptot = 0.0;
    real x_delta = 0.0;

    for (int j = start + tid; j < end; j += blockDim.x) {
        real xi = __ldg(&x_val[j]);
        real x_old = __ldg(&x_old_val[j]);
        real ui = __ldg(&u_val[j]);
        int col = __ldg(&col_indice[j]);
        real pj = __ldg(&p[col]);

        real x_pow;
        if (power == 2) x_pow = xi * xi;
        else if (power == 3) x_pow = xi * xi * xi;
        else x_pow = powf(xi, power);

        sum += x_pow * ui;
        ptot += xi * pj;
        x_delta += (xi - x_old) * (xi - x_old);
    }

    shm_sum[tid] = sum;
    shm_ptot[tid] = ptot;
    shm_delta[tid] = x_delta;
    __syncthreads();

    for (int offset = blockSize >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            shm_sum[tid] = sum = shm_sum[tid] + shm_sum[tid + offset];
            shm_ptot[tid] = ptot = shm_ptot[tid] + shm_ptot[tid + offset];
            shm_delta[tid] = x_delta = shm_delta[tid] + shm_delta[tid + offset];
        }
        __syncthreads();
    }
    if (tid < 32) {
        sum = warpReduce<real>(sum);
        ptot = warpReduce<real>(ptot);
        x_delta = warpReduce<real>(x_delta);
    }
    if (tid == 0) {
        real log_sum = logf(fmaxf(sum, 1e-12f));
        objective[row] = -w[row] * log_sum + ptot + (0.5f / pho) * x_delta;
    }
}

template <typename real>
void launch_objective_csr(
    int m,
    const real* d_x_val,
    const real* d_u_val,
    const real* d_w,
    const int* d_row_ptr,
    const int* d_col_indice,
    const real power,
    real*       d_objective,
    const real* d_p,
    const real pho,
    const real* d_x_old_val)
{
    const int blockSize = 256;                // 每 block 最多 256 线程
    int gridSize  = m;                  // 每行一个 block
    objective_csr_kernel<256, real><<<gridSize, blockSize, 0>>>(
        m, d_x_val, d_u_val, d_w, d_row_ptr, d_col_indice, power,
        d_objective, d_p, pho, d_x_old_val);
    // 使用 blockReduce 来归约每个 block 的结果
    int numBlocks = m;
    // blockReduce<blockSize, real><<<numBlocks, blockSize, 0, stream>>>(d_objective, m);
    for (numBlocks = m; numBlocks > 1; numBlocks = (numBlocks + blockSize - 1) / blockSize) {
        int gridSize = (numBlocks + blockSize - 1) / blockSize;
        blockReduce<blockSize, real><<<gridSize, blockSize, 0>>>(d_objective, numBlocks);
        cudaDeviceSynchronize();
    }
    // 注意：这里的 d_objective 需要
    // 是一个足够大的数组来存储每个 block 的结果。
    // 最终的 objective 值将存储在 d_objective[0] 中
    cudaDeviceSynchronize();  // 确保 kernel 执行完毕
}
// 显式实例化常用类型
template void launch_objective_csr<double>(
    int, const double*, const double*, const double*, const int*, const int*, const double, double*, const double*, const double, const double*);
template void launch_objective_csr<float>(
    int, const float*,  const float*,  const float*, const int*, const int*, const float, float*,  const float*, const float, const float*);

template <int blockSize, typename real>
__global__ void utility_csr_kernel(
    int m,
    const real* __restrict__ x_val,
    const real* __restrict__ u_val,
    const int* __restrict__ row_ptr,
    real* __restrict__ utility,
    const real power)
{
    int row = blockIdx.x;
    if (row >= m) return;

    int start = row_ptr[row];
    int end   = row_ptr[row + 1];

    __shared__ real shm[blockSize];
    int tid = threadIdx.x;
    real sum = 0.0;

    for (int j = start + tid; j < end; j += blockDim.x) {
        real xi = x_val[j];
        real ui = u_val[j];
        real tmp = 1.0;
        // for (int k = 0; k < power; ++k) tmp *= xi;
        sum += pow(xi, power) * ui;  // 使用 pow 函数计算 xi 的 power 次方
    }
    shm[tid] = sum;
    __syncthreads();

    for (int offset = blockSize >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            shm[tid] = sum = shm[tid] + shm[tid + offset];
        }
        __syncthreads();
    }
    if (tid < 32) {
        sum = warpReduce<real>(sum);
    }
    if (tid == 0) utility[row] = sum;
}

template <typename real>
void launch_utility_csr(
    int m,
    const real* d_x_val,
    const real* d_u_val,
    const int* d_row_ptr,
    real*       d_utility,
    const real power
)
{
    int blockSize = 256;                // 每 block 最多 256 线程
    int gridSize  = m;                  // 每行一个 block
    utility_csr_kernel<256, real><<<gridSize, blockSize, 0>>>(
        m, d_x_val, d_u_val, d_row_ptr, d_utility, power);
    cudaDeviceSynchronize();
}

// 显式实例化常用类型
template void launch_utility_csr<double>(
    int, const double*, const double*, const int*, double*, const double);
template void launch_utility_csr<float>(
    int, const float*,  const float*,  const int*, float*,  const float);