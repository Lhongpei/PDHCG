#include "pdhcg_psd_cone.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int test_batched_projection(void)
{
    const double sqrt_two = 1.41421356237309504880;
    const int order[] = {2, 2, 3, 1};
    const int start[] = {0, 3, 6, 12};
    double host[] = {
        1.0,
        2.0 * sqrt_two,
        1.0,
        2.0,
        -sqrt_two,
        2.0,
        1.0,
        2.0 * sqrt_two,
        3.0 * sqrt_two,
        4.0,
        6.0 * sqrt_two,
        9.0,
        -3.0,
    };
    const double expected[] = {
        1.5,
        1.5 * sqrt_two,
        1.5,
        2.0,
        -sqrt_two,
        2.0,
        1.0,
        2.0 * sqrt_two,
        3.0 * sqrt_two,
        4.0,
        6.0 * sqrt_two,
        9.0,
        0.0,
    };
    double *device = NULL;
    cudaMalloc(&device, sizeof(host));
    cudaMemcpy(device, host, sizeof(host), cudaMemcpyHostToDevice);

    psd_projection_runtime_t *runtime = create_psd_projection_runtime(start, order, 4, 0);
    project_psd_cones(runtime, device);
    cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost);

    int passed = 1;
    for (int slot = 0; slot < 13; ++slot)
        if (!isfinite(host[slot]) || fabs(host[slot] - expected[slot]) > 1e-10)
            passed = 0;

    free_psd_projection_runtime(runtime);
    cudaFree(device);
    return passed;
}

static int test_large_diagonal_projection(void)
{
    const int order = 33;
    const int packed_length = order * (order + 1) / 2;
    double *host = (double *)calloc((size_t)packed_length, sizeof(double));
    double *device = NULL;
    if (!host)
        return 0;

    int slot = 0;
    for (int col = 0; col < order; ++col)
    {
        for (int row = col; row < order; ++row)
        {
            if (row == col)
                host[slot] = col % 2 == 0 ? -(double)(col + 1) : (double)(col + 1);
            ++slot;
        }
    }

    cudaMalloc(&device, (size_t)packed_length * sizeof(double));
    cudaMemcpy(device, host, (size_t)packed_length * sizeof(double), cudaMemcpyHostToDevice);
    const int start = 0;
    psd_projection_runtime_t *runtime = create_psd_projection_runtime(&start, &order, 1, 0);
    project_psd_cones(runtime, device);
    cudaMemcpy(host, device, (size_t)packed_length * sizeof(double), cudaMemcpyDeviceToHost);

    int passed = 1;
    slot = 0;
    for (int col = 0; col < order; ++col)
    {
        for (int row = col; row < order; ++row)
        {
            double expected = row == col && col % 2 == 1 ? (double)(col + 1) : 0.0;
            if (!isfinite(host[slot]) || fabs(host[slot] - expected) > 1e-9 * (1.0 + expected))
                passed = 0;
            ++slot;
        }
    }

    free_psd_projection_runtime(runtime);
    cudaFree(device);
    free(host);
    return passed;
}

int main(void)
{
    int batched = test_batched_projection();
    int large = test_large_diagonal_projection();
    printf("PSD batched spectral projection:  %s\n", batched ? "PASS" : "FAIL");
    printf("PSD order-33 spectral projection: %s\n", large ? "PASS" : "FAIL");
    return batched && large ? 0 : 1;
}
