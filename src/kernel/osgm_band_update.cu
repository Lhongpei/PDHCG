#include <cuda_runtime.h>
// #include <cuda/std/cmath>
extern "C" __global__ void osgm_band_kernel(
    const double* __restrict__ prev_grad,
    const double* __restrict__ grad,
    double* __restrict__ G_band,
    double* __restrict__ Q_band,
    double* __restrict__ x,
    int n,
    double norm_sq,
    double eps,
    double lr,
    int bandwidth
) {
    extern __shared__ double shared[];
    double* shared_prev = shared;
    double* shared_grad = shared + blockDim.x;

    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int g_idx = block_start + tid;

    // --- Finalized Shared Memory Loading ---

    // 1. Load main chunk of prev_grad and grad
    if (g_idx < n) {
        shared_prev[tid] = prev_grad[g_idx];
        shared_grad[bandwidth + tid] = grad[g_idx];
    }

    // 2. First 'bandwidth' threads load left AND right halos
    if (tid < bandwidth) {
        int left_halo_idx = block_start - bandwidth + tid;
        if (left_halo_idx >= 0) {
            shared_grad[tid] = grad[left_halo_idx];
        } else {
            shared_grad[tid] = 0.0f;
        }
    }
    
    if (tid >= blockDim.x - bandwidth && tid < blockDim.x) {
        int right_halo_idx = block_start + blockDim.x + (tid - (blockDim.x - bandwidth));
        if (right_halo_idx < n) {
            shared_grad[bandwidth + blockDim.x + (tid - (blockDim.x - bandwidth))] = grad[right_halo_idx];
        } else {
            shared_grad[bandwidth + blockDim.x + (tid - (blockDim.x - bandwidth))] = 0.0f;
        }
    }
    __syncthreads();

      // --- Optimized Computation Loop ---

    int i = block_start + tid;
    if (i < n) {
        double qg_dot = 0.0f;
        const double gi = shared_prev[tid];
        
        #pragma unroll
        for (int offset = -bandwidth; offset <= bandwidth; ++offset) {
            int j = i + offset;
            if (j >= 0 && j < n) {
                const double gj = shared_grad[tid + bandwidth + offset];
                const double gr = -__fdividef(gi * gj, norm_sq + 1e-20f);
                
                // OPTIMIZED #1: New index calculation for coalesced memory access.
                // This requires G_band and Q_band to have the shape (2*bandwidth+1, n).
                const int band_offset = offset + bandwidth;
                const int band_idx = band_offset * n + i;

                // OPTIMIZED #3: Latency Hiding. Read both old values first.
                const double gval_old = G_band[band_idx];
                const double qval_old = Q_band[band_idx];

                // Perform computations using local variables.
                const double gval_new = fmaf(gr, gr, gval_old);
                
                // OPTIMIZED #2: Use faster reciprocal square root.
                const double rsqrt_g = rsqrtf(gval_new + eps);
                const double qval_new = fmaf(-lr * gr, rsqrt_g, qval_old);
                
                // Accumulate dot product.
                qg_dot += qval_new * gj;
                
                // OPTIMIZED #3: Write both new values back at the end.
                G_band[band_idx] = gval_new;
                Q_band[band_idx] = qval_new;
            }
        }
        x[i] -= qg_dot;
    }
}
