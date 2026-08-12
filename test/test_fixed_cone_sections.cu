#include "internal_types.h"
#include "pdhcg_affine_cone_kernels.h"
#include "pdhcg_cone_common_kernels.h"
#include "pdhcg_exp_cone_kernels.h"
#include "pdhcg_kernels.h"
#include "pdhcg_rsoc_cone_kernels.h"
#include "pdhcg_soc_cone_kernels.h"
#include "pdhcg_types.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static int cuda_ok(cudaError_t error, const char *label)
{
    if (error == cudaSuccess)
        return 1;
    std::fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(error));
    return 0;
}

static double unit_sample(int sample, int slot)
{
    return 0.5 + 0.5 * std::sin(1.61803398875 * (sample + 1) * (slot + 2));
}

static int cone_feasible(cone_type_t type, const double point[3])
{
    if (type == CONE_STANDARD_SOC)
        return std::hypot(point[0], point[1]) <= point[2] + 2e-9;
    if (type == CONE_ROTATED_SOC)
        return point[1] >= -2e-9 && point[2] >= -2e-9 && point[0] * point[0] <= 2.0 * point[1] * point[2] + 2e-9;
    if (point[1] > 0.0 && point[2] > 0.0)
        return std::log(point[1]) + point[0] / point[1] <= std::log(point[2]) + 2e-9;
    return std::fabs(point[1]) <= 2e-9 && point[0] <= 2e-9 && point[2] >= -2e-9;
}

static void make_soc_candidate(int mask, int sample, const double fixed[3], double candidate[3])
{
    candidate[0] = (mask & 1) ? fixed[0] : 1.8 * (2.0 * unit_sample(sample, 0) - 1.0);
    candidate[1] = (mask & 2) ? fixed[1] : 1.8 * (2.0 * unit_sample(sample, 1) - 1.0);
    if (mask & 4)
    {
        candidate[2] = fixed[2];
        double fixed_norm2 =
            ((mask & 1) ? candidate[0] * candidate[0] : 0.0) + ((mask & 2) ? candidate[1] * candidate[1] : 0.0);
        double free_norm2 =
            ((mask & 1) ? 0.0 : candidate[0] * candidate[0]) + ((mask & 2) ? 0.0 : candidate[1] * candidate[1]);
        double radius2 = std::fmax(0.0, candidate[2] * candidate[2] - fixed_norm2);
        if (free_norm2 > 0.8 * radius2 && free_norm2 > 0.0)
        {
            double scale = std::sqrt(0.8 * radius2 / free_norm2);
            if (!(mask & 1))
                candidate[0] *= scale;
            if (!(mask & 2))
                candidate[1] *= scale;
        }
    }
    else
    {
        candidate[2] = std::hypot(candidate[0], candidate[1]) + 0.1 + unit_sample(sample, 2);
    }
}

static void make_rsoc_candidate(int mask, int sample, const double fixed[3], double candidate[3])
{
    candidate[0] = (mask & 1) ? fixed[0] : 1.8 * (2.0 * unit_sample(sample, 0) - 1.0);
    bool fixed_s = (mask & 2) != 0;
    bool fixed_t = (mask & 4) != 0;
    candidate[1] = fixed_s ? fixed[1] : 0.0;
    candidate[2] = fixed_t ? fixed[2] : 0.0;
    if (fixed_s && fixed_t)
    {
        if (!(mask & 1))
        {
            double radius = std::sqrt(2.0 * candidate[1] * candidate[2]);
            candidate[0] = (2.0 * unit_sample(sample, 0) - 1.0) * 0.8 * radius;
        }
    }
    else if (fixed_s)
    {
        candidate[2] = candidate[0] * candidate[0] / (2.0 * candidate[1]) + 0.1 + unit_sample(sample, 2);
    }
    else if (fixed_t)
    {
        candidate[1] = candidate[0] * candidate[0] / (2.0 * candidate[2]) + 0.1 + unit_sample(sample, 1);
    }
    else
    {
        candidate[1] = 0.2 + 1.5 * unit_sample(sample, 1);
        candidate[2] = candidate[0] * candidate[0] / (2.0 * candidate[1]) + 0.1 + unit_sample(sample, 2);
    }
}

static void make_exp_candidate(int mask, int sample, const double fixed[3], double candidate[3])
{
    bool fixed_x = (mask & 1) != 0;
    bool fixed_y = (mask & 2) != 0;
    bool fixed_z = (mask & 4) != 0;
    candidate[0] = fixed_x ? fixed[0] : 0.0;
    candidate[1] = fixed_y ? fixed[1] : 0.2 + 1.2 * unit_sample(sample, 1);
    candidate[2] = fixed_z ? fixed[2] : 0.0;

    if (fixed_z)
    {
        if (!fixed_x)
        {
            double upper = candidate[1] * (std::log(candidate[2]) - std::log(candidate[1]));
            candidate[0] = upper - 0.05 - unit_sample(sample, 0);
        }
    }
    else
    {
        if (!fixed_x)
            candidate[0] = 1.2 * (2.0 * unit_sample(sample, 0) - 1.0);
        double boundary = candidate[1] * std::exp(candidate[0] / candidate[1]);
        candidate[2] = boundary + 0.05 + unit_sample(sample, 2);
    }
}

static int run_mask(cone_type_t type, int mask, int diagonal_q, int use_block)
{
    const double scale[3] = {0.7, 1.6, 2.3};
    const double q_diag[3] = {0.5, 2.0, 4.0};
    const double tau = 0.7;
    const double soc_fixed[3] = {0.25, -0.35, 1.4};
    const double rsoc_fixed[3] = {0.2, 1.2, 1.1};
    const double exp_fixed[3] = {0.1, 1.0, 2.0};
    const double *fixed = type == CONE_STANDARD_SOC ? soc_fixed : type == CONE_ROTATED_SOC ? rsoc_fixed : exp_fixed;
    const double raw_input[3] = {2.0, -1.5, -0.6};
    double input[3];
    char fixed_mask[3];
    for (int slot = 0; slot < 3; ++slot)
    {
        fixed_mask[slot] = (mask >> slot) & 1;
        double actual = fixed_mask[slot] ? fixed[slot] : raw_input[slot];
        input[slot] = scale[slot] * actual;
    }

    double *d_point = nullptr;
    double *d_reflected = nullptr;
    double *d_current = nullptr;
    double *d_scale = nullptr;
    double *d_q = nullptr;
    double *d_warm = nullptr;
    int *d_start = nullptr;
    int *d_dim = nullptr;
    char *d_fixed = nullptr;
    int start = 0;
    int dim = 1;
    int ok = cuda_ok(cudaMalloc(&d_point, 3 * sizeof(double)), "cudaMalloc(point)") &&
        cuda_ok(cudaMalloc(&d_scale, 3 * sizeof(double)), "cudaMalloc(scale)") &&
        cuda_ok(cudaMalloc(&d_warm, sizeof(double)), "cudaMalloc(warm)") &&
        cuda_ok(cudaMalloc(&d_start, sizeof(int)), "cudaMalloc(start)") &&
        cuda_ok(cudaMalloc(&d_dim, sizeof(int)), "cudaMalloc(dim)") &&
        cuda_ok(cudaMalloc(&d_fixed, 3 * sizeof(char)), "cudaMalloc(fixed)");
    if (diagonal_q)
        ok &= cuda_ok(cudaMalloc(&d_reflected, 3 * sizeof(double)), "cudaMalloc(reflected)") &&
            cuda_ok(cudaMalloc(&d_current, 3 * sizeof(double)), "cudaMalloc(current)") &&
            cuda_ok(cudaMalloc(&d_q, 3 * sizeof(double)), "cudaMalloc(q)");
    if (!ok)
        goto cleanup;

    ok &= cuda_ok(cudaMemcpy(d_point, input, sizeof(input), cudaMemcpyHostToDevice), "copy point") &&
        cuda_ok(cudaMemcpy(d_scale, scale, sizeof(scale), cudaMemcpyHostToDevice), "copy scale") &&
        cuda_ok(cudaMemcpy(d_start, &start, sizeof(start), cudaMemcpyHostToDevice), "copy start") &&
        cuda_ok(cudaMemcpy(d_dim, &dim, sizeof(dim), cudaMemcpyHostToDevice), "copy dim") &&
        cuda_ok(cudaMemcpy(d_fixed, fixed_mask, sizeof(fixed_mask), cudaMemcpyHostToDevice), "copy fixed") &&
        cuda_ok(cudaMemset(d_warm, 0, sizeof(double)), "clear warm");
    if (diagonal_q)
        ok &= cuda_ok(cudaMemcpy(d_current, input, sizeof(input), cudaMemcpyHostToDevice), "copy current") &&
            cuda_ok(cudaMemcpy(d_q, q_diag, sizeof(q_diag), cudaMemcpyHostToDevice), "copy q");
    if (!ok)
        goto cleanup;

    if (!diagonal_q)
    {
        if (type == CONE_STANDARD_SOC)
        {
            if (use_block)
                project_standard_soc_block_kernel<<<1, 256>>>(
                    d_point, d_scale, NULL, 0.0, d_warm, d_start, d_dim, d_fixed, 1);
            else
                project_standard_soc_kernel<<<1, 1>>>(d_point, d_scale, d_warm, d_start, d_dim, d_fixed, 1);
        }
        else if (type == CONE_ROTATED_SOC)
        {
            if (use_block)
                project_rotated_soc_block_kernel<<<1, 256>>>(
                    d_point, d_scale, NULL, 0.0, d_warm, d_start, d_dim, d_fixed, 1);
            else
                project_rotated_soc_kernel<<<1, 1>>>(d_point, d_scale, d_warm, d_start, d_dim, d_fixed, 1);
        }
        else
            project_exp_cone_kernel<<<1, 1>>>(d_point, d_scale, d_warm, d_start, d_dim, d_fixed, 1);
    }
    else if (type == CONE_STANDARD_SOC)
    {
        if (use_block)
        {
            project_standard_soc_block_kernel<<<1, 256>>>(
                d_point, d_scale, d_q, tau, d_warm, d_start, d_dim, d_fixed, 1);
            recompute_reflected_at_cone_block_kernel<<<1, 256>>>(d_reflected, d_point, d_current, d_start, d_dim, 1);
        }
        else
            project_standard_soc_diag_q_kernel<<<1, 1>>>(
                d_point, d_reflected, d_current, d_scale, d_q, tau, d_warm, d_start, d_dim, d_fixed, 1);
    }
    else if (type == CONE_ROTATED_SOC)
    {
        if (use_block)
        {
            project_rotated_soc_block_kernel<<<1, 256>>>(
                d_point, d_scale, d_q, tau, d_warm, d_start, d_dim, d_fixed, 1);
            recompute_reflected_at_cone_block_kernel<<<1, 256>>>(d_reflected, d_point, d_current, d_start, d_dim, 1);
        }
        else
            project_rotated_soc_diag_q_kernel<<<1, 1>>>(
                d_point, d_reflected, d_current, d_scale, d_q, tau, d_warm, d_start, d_dim, d_fixed, 1);
    }
    else
        project_exp_cone_diag_q_kernel<<<1, 1>>>(
            d_point, d_reflected, d_current, d_scale, d_q, tau, d_warm, d_start, d_dim, d_fixed, 1);
    ok &= cuda_ok(cudaGetLastError(), "projection launch") && cuda_ok(cudaDeviceSynchronize(), "projection sync");

    double projected_scaled[3];
    ok &=
        cuda_ok(cudaMemcpy(projected_scaled, d_point, sizeof(projected_scaled), cudaMemcpyDeviceToHost), "copy result");
    double projected[3];
    for (int slot = 0; slot < 3; ++slot)
    {
        projected[slot] = projected_scaled[slot] / scale[slot];
        if (!std::isfinite(projected[slot]))
            ok = 0;
        if (fixed_mask[slot] && projected_scaled[slot] != input[slot])
        {
            std::fprintf(stderr,
                         "type=%d mask=%d diag=%d block=%d changed fixed slot %d: %.17g -> %.17g\n",
                         (int)type,
                         mask,
                         diagonal_q,
                         use_block,
                         slot,
                         input[slot],
                         projected_scaled[slot]);
            ok = 0;
        }
    }
    if (!cone_feasible(type, projected))
    {
        std::fprintf(stderr,
                     "type=%d mask=%d diag=%d block=%d infeasible projection: (%.17g, %.17g, %.17g)\n",
                     (int)type,
                     mask,
                     diagonal_q,
                     use_block,
                     projected[0],
                     projected[1],
                     projected[2]);
        ok = 0;
    }

    for (int sample = 0; sample < 200 && ok; ++sample)
    {
        double candidate[3];
        if (type == CONE_STANDARD_SOC)
            make_soc_candidate(mask, sample, fixed, candidate);
        else if (type == CONE_ROTATED_SOC)
            make_rsoc_candidate(mask, sample, fixed, candidate);
        else
            make_exp_candidate(mask, sample, fixed, candidate);
        if (!cone_feasible(type, candidate))
        {
            std::fprintf(
                stderr, "test generated an infeasible candidate: type=%d mask=%d sample=%d\n", (int)type, mask, sample);
            ok = 0;
            break;
        }
        double dot = 0.0;
        double scale_norm = 1.0;
        for (int slot = 0; slot < 3; ++slot)
        {
            double candidate_scaled = scale[slot] * candidate[slot];
            double metric = diagonal_q ? 1.0 + tau * q_diag[slot] : 1.0;
            double gradient = metric * (projected_scaled[slot] - input[slot]);
            double direction = candidate_scaled - projected_scaled[slot];
            dot += gradient * direction;
            scale_norm += std::fabs(gradient) * (1.0 + std::fabs(direction));
        }
        if (dot < -2e-6 * scale_norm)
        {
            std::fprintf(stderr,
                         "type=%d mask=%d diag=%d block=%d violates projection VI: "
                         "dot=%.3e scale=%.3e sample=%d\n",
                         (int)type,
                         mask,
                         diagonal_q,
                         use_block,
                         dot,
                         scale_norm,
                         sample);
            ok = 0;
        }
    }

cleanup:
    cudaFree(d_point);
    cudaFree(d_reflected);
    cudaFree(d_current);
    cudaFree(d_scale);
    cudaFree(d_q);
    cudaFree(d_warm);
    cudaFree(d_start);
    cudaFree(d_dim);
    cudaFree(d_fixed);
    return ok;
}

enum large_section_pattern
{
    LARGE_SECTION_FREE = 0,
    LARGE_SECTION_FIXED_ENDPOINT = 1,
    LARGE_SECTION_FIXED_VECTOR = 2,
    LARGE_SECTION_FIXED_BOTH_ENDPOINTS = 3,
    LARGE_SECTION_EXTREME_FIXED_BALL = 4,
    LARGE_SECTION_NEAR_POLAR = 5,
    LARGE_SECTION_FIXED_OTHER_ENDPOINT = 6,
    LARGE_SECTION_FIXED_ALL_VECTOR = 7,
};

static int compare_parallel_projection(cone_type_t type, int diagonal_q, int section_pattern, int use_grid)
{
    const int k = use_grid ? PDHCG_LARGE_CONE_MIN_VDIM : 769;
    const int length = k + 2;
    const double tau = 0.6;
    std::vector<double> input(length);
    std::vector<double> scale(length);
    std::vector<double> q_diag(length);
    std::vector<char> fixed(length, 0);
    for (int slot = 0; slot < k; ++slot)
    {
        scale[slot] = 0.4 + 0.03 * (slot % 37);
        q_diag[slot] = 0.05 * (slot % 19);
        double actual = 1.7 * std::sin(0.013 * (slot + 1));
        if (section_pattern == LARGE_SECTION_FIXED_ALL_VECTOR ||
            ((section_pattern == LARGE_SECTION_FIXED_ENDPOINT || section_pattern == LARGE_SECTION_FIXED_VECTOR ||
              section_pattern == LARGE_SECTION_FIXED_BOTH_ENDPOINTS) &&
             slot % 8191 == 0))
        {
            fixed[slot] = 1;
            actual = 0.02;
        }
        input[slot] = scale[slot] * actual;
    }
    scale[k] = 1.3;
    scale[k + 1] = 2.1;
    q_diag[k] = 0.7;
    q_diag[k + 1] = 1.4;
    if (type == CONE_STANDARD_SOC)
    {
        input[k] = scale[k] * -0.8;
        input[k + 1] = scale[k + 1] * -0.4;
        if (section_pattern == LARGE_SECTION_FIXED_ENDPOINT)
        {
            fixed[k + 1] = 1;
            input[k + 1] = scale[k + 1] * 5.0;
        }
        else if (section_pattern == LARGE_SECTION_FIXED_OTHER_ENDPOINT)
        {
            fixed[k] = 1;
            input[k] = scale[k] * 0.5;
        }
        else if (section_pattern == LARGE_SECTION_FIXED_ALL_VECTOR)
        {
            fixed[k] = 1;
            input[k] = scale[k] * 0.5;
        }
        else if (section_pattern == LARGE_SECTION_FIXED_BOTH_ENDPOINTS)
        {
            fixed[k] = 1;
            fixed[k + 1] = 1;
            input[k] = scale[k] * 0.5;
            input[k + 1] = scale[k + 1] * 5.0;
        }
        else if (section_pattern == LARGE_SECTION_EXTREME_FIXED_BALL)
        {
            std::fill(input.begin(), input.begin() + k, 0.0);
            input[0] = scale[0] * std::ldexp(1.0, 120);
            input[k] = 0.0;
            fixed[k + 1] = 1;
            input[k + 1] = scale[k + 1];
        }
    }
    else
    {
        input[k] = scale[k] * -0.3;
        input[k + 1] = scale[k + 1] * 0.2;
        if (section_pattern == LARGE_SECTION_FIXED_ENDPOINT)
        {
            fixed[k] = 1;
            input[k] = scale[k] * 1.2;
        }
        else if (section_pattern == LARGE_SECTION_FIXED_OTHER_ENDPOINT)
        {
            fixed[k + 1] = 1;
            input[k + 1] = scale[k + 1] * 1.1;
        }
        else if (section_pattern == LARGE_SECTION_FIXED_BOTH_ENDPOINTS)
        {
            fixed[k] = 1;
            fixed[k + 1] = 1;
            input[k] = scale[k] * 1.2;
            input[k + 1] = scale[k + 1] * 1.1;
        }
        else if (section_pattern == LARGE_SECTION_EXTREME_FIXED_BALL)
        {
            std::fill(input.begin(), input.begin() + k, 0.0);
            input[0] = scale[0] * std::ldexp(1.0, 120);
            fixed[k] = 1;
            fixed[k + 1] = 1;
            input[k] = scale[k];
            input[k + 1] = scale[k + 1];
        }
    }

    if (section_pattern == LARGE_SECTION_NEAR_POLAR)
    {
        std::fill(input.begin(), input.end(), 0.0);
        std::fill(scale.begin(), scale.end(), 1.0);
        std::fill(q_diag.begin(), q_diag.end(), 0.0);
        input[0] = 1.0 + std::ldexp(1.0, -45);
        if (type == CONE_STANDARD_SOC)
        {
            input[k] = 0.0;
            input[k + 1] = -1.0;
        }
        else
        {
            input[k] = -0.70710678118654752440;
            input[k + 1] = -0.70710678118654752440;
        }
    }

    const int has_fixed_section = std::any_of(fixed.begin(), fixed.end(), [](char value) { return value != 0; });

    double *d_block = nullptr;
    double *d_serial = nullptr;
    double *d_scale = nullptr;
    double *d_q = nullptr;
    double *d_warm_block = nullptr;
    double *d_warm_serial = nullptr;
    double *d_reflected = nullptr;
    double *d_current = nullptr;
    int *d_start = nullptr;
    int *d_dim = nullptr;
    char *d_fixed = nullptr;
    int start = 0;
    int ok = cuda_ok(cudaMalloc(&d_block, (size_t)length * sizeof(double)), "large cudaMalloc(block)") &&
        cuda_ok(cudaMalloc(&d_serial, (size_t)length * sizeof(double)), "large cudaMalloc(serial)") &&
        cuda_ok(cudaMalloc(&d_scale, (size_t)length * sizeof(double)), "large cudaMalloc(scale)") &&
        cuda_ok(cudaMalloc(&d_q, (size_t)length * sizeof(double)), "large cudaMalloc(q)") &&
        cuda_ok(cudaMalloc(&d_warm_block, PDHCG_CONE_WORKSPACE_STRIDE * sizeof(double)),
                "large cudaMalloc(warm block)") &&
        cuda_ok(cudaMalloc(&d_warm_serial, sizeof(double)), "large cudaMalloc(warm serial)") &&
        cuda_ok(cudaMalloc(&d_reflected, (size_t)length * sizeof(double)), "large cudaMalloc(reflected)") &&
        cuda_ok(cudaMalloc(&d_current, (size_t)length * sizeof(double)), "large cudaMalloc(current)") &&
        cuda_ok(cudaMalloc(&d_start, sizeof(int)), "large cudaMalloc(start)") &&
        cuda_ok(cudaMalloc(&d_dim, sizeof(int)), "large cudaMalloc(dim)");
    if (has_fixed_section)
        ok &= cuda_ok(cudaMalloc(&d_fixed, (size_t)length * sizeof(char)), "large cudaMalloc(fixed)");
    if (!ok)
        goto cleanup;

    ok &= cuda_ok(cudaMemcpy(d_block, input.data(), (size_t)length * sizeof(double), cudaMemcpyHostToDevice),
                  "large copy block") &&
        cuda_ok(cudaMemcpy(d_serial, input.data(), (size_t)length * sizeof(double), cudaMemcpyHostToDevice),
                "large copy serial") &&
        cuda_ok(cudaMemcpy(d_current, input.data(), (size_t)length * sizeof(double), cudaMemcpyHostToDevice),
                "large copy current") &&
        cuda_ok(cudaMemcpy(d_scale, scale.data(), (size_t)length * sizeof(double), cudaMemcpyHostToDevice),
                "large copy scale") &&
        cuda_ok(cudaMemcpy(d_q, q_diag.data(), (size_t)length * sizeof(double), cudaMemcpyHostToDevice),
                "large copy q") &&
        cuda_ok(cudaMemcpy(d_start, &start, sizeof(start), cudaMemcpyHostToDevice), "large copy start") &&
        cuda_ok(cudaMemcpy(d_dim, &k, sizeof(k), cudaMemcpyHostToDevice), "large copy dim") &&
        cuda_ok(cudaMemset(d_warm_block, 0, PDHCG_CONE_WORKSPACE_STRIDE * sizeof(double)), "large clear warm block") &&
        cuda_ok(cudaMemset(d_warm_serial, 0, sizeof(double)), "large clear warm serial");
    if (has_fixed_section)
        ok &= cuda_ok(cudaMemcpy(d_fixed, fixed.data(), (size_t)length * sizeof(char), cudaMemcpyHostToDevice),
                      "large copy fixed");
    if (!ok)
        goto cleanup;

    if (type == CONE_STANDARD_SOC)
    {
        if (use_grid)
        {
            const int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
            cudaMemset(d_warm_block + 1, 0, 5 * sizeof(double));
            initialize_standard_soc_grid_weighted_kernel<<<blocks_per_cone, 256>>>(d_block,
                                                                                   d_scale,
                                                                                   diagonal_q ? d_q : NULL,
                                                                                   diagonal_q ? tau : 0.0,
                                                                                   d_warm_block,
                                                                                   d_start,
                                                                                   d_dim,
                                                                                   d_fixed,
                                                                                   1,
                                                                                   blocks_per_cone);
            finalize_standard_soc_grid_weighted_initialization_kernel<<<1, 256>>>(d_block,
                                                                                  d_scale,
                                                                                  diagonal_q ? d_q : NULL,
                                                                                  diagonal_q ? tau : 0.0,
                                                                                  d_warm_block,
                                                                                  d_start,
                                                                                  d_dim,
                                                                                  d_fixed,
                                                                                  1);
            for (int iteration = 0; iteration < PDHCG_CONE_GRID_ROOT_ITERATIONS; ++iteration)
            {
                cudaMemset(d_warm_block + 1, 0, 2 * sizeof(double));
                reduce_standard_soc_grid_weighted_root_kernel<<<blocks_per_cone, 256>>>(d_block,
                                                                                        d_scale,
                                                                                        diagonal_q ? d_q : NULL,
                                                                                        diagonal_q ? tau : 0.0,
                                                                                        d_warm_block,
                                                                                        d_start,
                                                                                        d_dim,
                                                                                        d_fixed,
                                                                                        1,
                                                                                        blocks_per_cone);
                finalize_standard_soc_grid_weighted_root_kernel<<<1, 256>>>(
                    d_block, d_scale, diagonal_q ? d_q : NULL, diagonal_q ? tau : 0.0, d_warm_block, d_start, d_dim, 1);
            }
            apply_standard_soc_grid_weighted_kernel<<<blocks_per_cone, 256>>>(d_block,
                                                                              d_scale,
                                                                              diagonal_q ? d_q : NULL,
                                                                              diagonal_q ? tau : 0.0,
                                                                              d_warm_block,
                                                                              d_start,
                                                                              d_dim,
                                                                              d_fixed,
                                                                              1,
                                                                              blocks_per_cone);
        }
        else
            project_standard_soc_block_kernel<<<1, 256>>>(d_block,
                                                          d_scale,
                                                          diagonal_q ? d_q : NULL,
                                                          diagonal_q ? tau : 0.0,
                                                          d_warm_block,
                                                          d_start,
                                                          d_dim,
                                                          d_fixed,
                                                          1);
        if (diagonal_q)
            project_standard_soc_diag_q_kernel<<<1, 1>>>(
                d_serial, d_reflected, d_current, d_scale, d_q, tau, d_warm_serial, d_start, d_dim, d_fixed, 1);
        else
            project_standard_soc_kernel<<<1, 1>>>(d_serial, d_scale, d_warm_serial, d_start, d_dim, d_fixed, 1);
    }
    else
    {
        if (use_grid)
        {
            const int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
            cudaMemset(d_warm_block + 1, 0, 5 * sizeof(double));
            initialize_rotated_soc_grid_weighted_kernel<<<blocks_per_cone, 256>>>(d_block,
                                                                                  d_scale,
                                                                                  diagonal_q ? d_q : NULL,
                                                                                  diagonal_q ? tau : 0.0,
                                                                                  d_warm_block,
                                                                                  d_start,
                                                                                  d_dim,
                                                                                  d_fixed,
                                                                                  1,
                                                                                  blocks_per_cone);
            finalize_rotated_soc_grid_weighted_initialization_kernel<<<1, 256>>>(d_block,
                                                                                 d_scale,
                                                                                 diagonal_q ? d_q : NULL,
                                                                                 diagonal_q ? tau : 0.0,
                                                                                 d_warm_block,
                                                                                 d_start,
                                                                                 d_dim,
                                                                                 d_fixed,
                                                                                 1);
            for (int iteration = 0; iteration < PDHCG_CONE_GRID_ROOT_ITERATIONS; ++iteration)
            {
                cudaMemset(d_warm_block + 1, 0, 2 * sizeof(double));
                reduce_rotated_soc_grid_weighted_root_kernel<<<blocks_per_cone, 256>>>(d_block,
                                                                                       d_scale,
                                                                                       diagonal_q ? d_q : NULL,
                                                                                       diagonal_q ? tau : 0.0,
                                                                                       d_warm_block,
                                                                                       d_start,
                                                                                       d_dim,
                                                                                       d_fixed,
                                                                                       1,
                                                                                       blocks_per_cone);
                finalize_rotated_soc_grid_weighted_root_kernel<<<1, 256>>>(d_block,
                                                                           d_scale,
                                                                           diagonal_q ? d_q : NULL,
                                                                           diagonal_q ? tau : 0.0,
                                                                           d_warm_block,
                                                                           d_start,
                                                                           d_dim,
                                                                           d_fixed,
                                                                           1);
            }
            cudaMemset(d_warm_block + 1, 0, 2 * sizeof(double));
            reduce_rotated_soc_grid_axis_objective_kernel<<<blocks_per_cone, 256>>>(d_block,
                                                                                    d_scale,
                                                                                    diagonal_q ? d_q : NULL,
                                                                                    diagonal_q ? tau : 0.0,
                                                                                    d_warm_block,
                                                                                    d_start,
                                                                                    d_dim,
                                                                                    d_fixed,
                                                                                    1,
                                                                                    blocks_per_cone);
            finalize_rotated_soc_grid_axis_objective_kernel<<<1, 256>>>(
                d_block, d_scale, diagonal_q ? d_q : NULL, diagonal_q ? tau : 0.0, d_warm_block, d_start, d_dim, 1);
            apply_rotated_soc_grid_weighted_kernel<<<blocks_per_cone, 256>>>(d_block,
                                                                             d_scale,
                                                                             diagonal_q ? d_q : NULL,
                                                                             diagonal_q ? tau : 0.0,
                                                                             d_warm_block,
                                                                             d_start,
                                                                             d_dim,
                                                                             d_fixed,
                                                                             1,
                                                                             blocks_per_cone);
        }
        else
            project_rotated_soc_block_kernel<<<1, 256>>>(d_block,
                                                         d_scale,
                                                         diagonal_q ? d_q : NULL,
                                                         diagonal_q ? tau : 0.0,
                                                         d_warm_block,
                                                         d_start,
                                                         d_dim,
                                                         d_fixed,
                                                         1);
        if (diagonal_q)
            project_rotated_soc_diag_q_kernel<<<1, 1>>>(
                d_serial, d_reflected, d_current, d_scale, d_q, tau, d_warm_serial, d_start, d_dim, d_fixed, 1);
        else
            project_rotated_soc_kernel<<<1, 1>>>(d_serial, d_scale, d_warm_serial, d_start, d_dim, d_fixed, 1);
    }
    ok &= cuda_ok(cudaGetLastError(), "large projection launch") &&
        cuda_ok(cudaDeviceSynchronize(), "large projection sync");
    if (!ok)
        goto cleanup;

    {
        std::vector<double> block(length);
        std::vector<double> serial(length);
        ok &= cuda_ok(cudaMemcpy(block.data(), d_block, (size_t)length * sizeof(double), cudaMemcpyDeviceToHost),
                      "large copy block result") &&
            cuda_ok(cudaMemcpy(serial.data(), d_serial, (size_t)length * sizeof(double), cudaMemcpyDeviceToHost),
                    "large copy serial result");
        double max_error = 0.0;
        double norm2 = 0.0;
        for (int slot = 0; slot < length; ++slot)
        {
            max_error = std::fmax(max_error, std::fabs(block[slot] - serial[slot]) / (1.0 + std::fabs(serial[slot])));
            if (has_fixed_section && fixed[slot] && block[slot] != input[slot])
                ok = 0;
            if (slot < k)
            {
                double actual = block[slot] / scale[slot];
                norm2 += actual * actual;
            }
        }
        double endpoint0 = block[k] / scale[k];
        double endpoint1 = block[k + 1] / scale[k + 1];
        double violation = type == CONE_STANDARD_SOC ? norm2 + endpoint0 * endpoint0 - endpoint1 * endpoint1
                                                     : norm2 - 2.0 * endpoint0 * endpoint1;
        double error_tolerance = section_pattern == LARGE_SECTION_NEAR_POLAR ? 1e-13 : 2e-8;
        if (max_error > error_tolerance || violation > 2e-7 * (1.0 + norm2))
        {
            std::fprintf(stderr,
                         "%s projection mismatch: type=%d diag=%d pattern=%d error=%.3e violation=%.3e\n",
                         use_grid ? "grid" : "block",
                         (int)type,
                         diagonal_q,
                         section_pattern,
                         max_error,
                         violation);
            ok = 0;
        }
    }

cleanup:
    cudaFree(d_block);
    cudaFree(d_serial);
    cudaFree(d_scale);
    cudaFree(d_q);
    cudaFree(d_warm_block);
    cudaFree(d_warm_serial);
    cudaFree(d_reflected);
    cudaFree(d_current);
    cudaFree(d_start);
    cudaFree(d_dim);
    cudaFree(d_fixed);
    return ok;
}

static int compare_large_affine_complementarity(void)
{
    const int k = PDHCG_LARGE_CONE_MIN_VDIM;
    const int length = k + 2;
    const int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    const double bound_rescaling = 3.7;
    const size_t bytes = (size_t)length * sizeof(double);
    std::vector<double> primal_product(length);
    std::vector<double> offset(length);
    std::vector<double> dual(length);
    std::vector<double> block_point(length);
    std::vector<double> grid_point(length);
    double expected_dot = 0.0;
    for (int slot = 0; slot < length; ++slot)
    {
        primal_product[slot] = 0.2 + 1e-5 * (slot % 97);
        offset[slot] = 0.1 + 2e-5 * (slot % 53);
        dual[slot] = 0.3 + 3e-5 * (slot % 71);
        expected_dot += dual[slot] * (primal_product[slot] + offset[slot]);
    }

    double *d_primal_product = nullptr;
    double *d_offset = nullptr;
    double *d_dual = nullptr;
    double *d_block_point = nullptr;
    double *d_grid_point = nullptr;
    double *d_block_complementarity = nullptr;
    double *d_grid_complementarity = nullptr;
    int *d_start = nullptr;
    int *d_dim = nullptr;
    int start = 0;
    int ok = cuda_ok(cudaMalloc(&d_primal_product, bytes), "affine cudaMalloc(primal product)") &&
        cuda_ok(cudaMalloc(&d_offset, bytes), "affine cudaMalloc(offset)") &&
        cuda_ok(cudaMalloc(&d_dual, bytes), "affine cudaMalloc(dual)") &&
        cuda_ok(cudaMalloc(&d_block_point, bytes), "affine cudaMalloc(block point)") &&
        cuda_ok(cudaMalloc(&d_grid_point, bytes), "affine cudaMalloc(grid point)") &&
        cuda_ok(cudaMalloc(&d_block_complementarity, sizeof(double)), "affine cudaMalloc(block complementarity)") &&
        cuda_ok(cudaMalloc(&d_grid_complementarity, sizeof(double)), "affine cudaMalloc(grid complementarity)") &&
        cuda_ok(cudaMalloc(&d_start, sizeof(int)), "affine cudaMalloc(start)") &&
        cuda_ok(cudaMalloc(&d_dim, sizeof(int)), "affine cudaMalloc(dim)");
    if (!ok)
        goto cleanup;

    ok &= cuda_ok(cudaMemcpy(d_primal_product, primal_product.data(), bytes, cudaMemcpyHostToDevice),
                  "affine copy primal product") &&
        cuda_ok(cudaMemcpy(d_offset, offset.data(), bytes, cudaMemcpyHostToDevice), "affine copy offset") &&
        cuda_ok(cudaMemcpy(d_dual, dual.data(), bytes, cudaMemcpyHostToDevice), "affine copy dual") &&
        cuda_ok(cudaMemcpy(d_start, &start, sizeof(int), cudaMemcpyHostToDevice), "affine copy start") &&
        cuda_ok(cudaMemcpy(d_dim, &k, sizeof(int), cudaMemcpyHostToDevice), "affine copy dim") &&
        cuda_ok(cudaMemset(d_grid_complementarity, 0, sizeof(double)), "affine clear grid complementarity");
    if (!ok)
        goto cleanup;

    prepare_affine_cone_residuals_kernel<<<1, 256, 256 * sizeof(double)>>>(
        d_block_point, d_block_complementarity, d_primal_product, d_offset, d_dual, d_start, d_dim, bound_rescaling, 1);
    prepare_affine_cone_residuals_grid_kernel<<<blocks_per_cone, 256, 256 * sizeof(double)>>>(
        d_grid_point, d_grid_complementarity, d_primal_product, d_offset, d_dual, d_start, d_dim, 1, blocks_per_cone);
    finish_affine_cone_complementarity_kernel<<<1, 1>>>(d_grid_complementarity, bound_rescaling, 1);
    ok &= cuda_ok(cudaGetLastError(), "affine residual launch") &&
        cuda_ok(cudaDeviceSynchronize(), "affine residual sync") &&
        cuda_ok(cudaMemcpy(block_point.data(), d_block_point, bytes, cudaMemcpyDeviceToHost),
                "affine copy block point") &&
        cuda_ok(cudaMemcpy(grid_point.data(), d_grid_point, bytes, cudaMemcpyDeviceToHost), "affine copy grid point");

    if (ok)
    {
        double block_complementarity = 0.0;
        double grid_complementarity = 0.0;
        ok &=
            cuda_ok(cudaMemcpy(&block_complementarity, d_block_complementarity, sizeof(double), cudaMemcpyDeviceToHost),
                    "affine copy block complementarity") &&
            cuda_ok(cudaMemcpy(&grid_complementarity, d_grid_complementarity, sizeof(double), cudaMemcpyDeviceToHost),
                    "affine copy grid complementarity");
        double max_point_error = 0.0;
        for (int slot = 0; slot < length; ++slot)
        {
            max_point_error = std::fmax(max_point_error, std::fabs(block_point[slot] - grid_point[slot]));
            max_point_error = std::fmax(max_point_error, std::fabs(grid_point[slot] + dual[slot]));
        }
        double expected = std::fabs(expected_dot) / bound_rescaling;
        double complementarity_error =
            std::fmax(std::fabs(block_complementarity - expected), std::fabs(grid_complementarity - expected)) /
            (1.0 + expected);
        if (max_point_error != 0.0 || complementarity_error > 2e-13)
        {
            std::fprintf(stderr,
                         "large affine residual mismatch: point=%.3e complementarity=%.3e "
                         "block=%.17g grid=%.17g expected=%.17g\n",
                         max_point_error,
                         complementarity_error,
                         block_complementarity,
                         grid_complementarity,
                         expected);
            ok = 0;
        }
    }

cleanup:
    cudaFree(d_primal_product);
    cudaFree(d_offset);
    cudaFree(d_dual);
    cudaFree(d_block_point);
    cudaFree(d_grid_point);
    cudaFree(d_block_complementarity);
    cudaFree(d_grid_complementarity);
    cudaFree(d_start);
    cudaFree(d_dim);
    return ok;
}

int main(void)
{
    int passed = 1;
    const cone_type_t types[] = {CONE_STANDARD_SOC, CONE_ROTATED_SOC, CONE_EXPONENTIAL};
    for (cone_type_t type : types)
        for (int diagonal_q = 0; diagonal_q <= 1; ++diagonal_q)
            for (int mask = 0; mask < 8; ++mask)
            {
                passed &= run_mask(type, mask, diagonal_q, 0);
                if (type == CONE_STANDARD_SOC || type == CONE_ROTATED_SOC)
                    passed &= run_mask(type, mask, diagonal_q, 1);
            }
    for (cone_type_t type : {CONE_STANDARD_SOC, CONE_ROTATED_SOC})
        for (int diagonal_q = 0; diagonal_q <= 1; ++diagonal_q)
            for (int section_pattern = LARGE_SECTION_FREE; section_pattern <= LARGE_SECTION_FIXED_ALL_VECTOR;
                 ++section_pattern)
                passed &= compare_parallel_projection(type, diagonal_q, section_pattern, 1);
    for (cone_type_t type : {CONE_STANDARD_SOC, CONE_ROTATED_SOC})
        for (int diagonal_q = 0; diagonal_q <= 1; ++diagonal_q)
            for (int section_pattern = LARGE_SECTION_FREE; section_pattern <= LARGE_SECTION_FIXED_ALL_VECTOR;
                 ++section_pattern)
                passed &= compare_parallel_projection(type, diagonal_q, section_pattern, 0);
    passed &= compare_large_affine_complementarity();
    std::printf("fixed cone section projections: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
