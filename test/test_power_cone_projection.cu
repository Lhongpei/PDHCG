#include "pdhcg_power_cone_kernels.h"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int cuda_ok(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
        return 1;
    fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return 0;
}

static double power_bound(double x, double y, double alpha)
{
    if (x <= 0.0 || y <= 0.0)
        return 0.0;
    double log_bound = alpha * log(x) + (1.0 - alpha) * log(y);
    if (log_bound >= log(DBL_MAX))
        return INFINITY;
    if (log_bound <= log(DBL_MIN))
        return 0.0;
    return exp(log_bound);
}

static int check_free_projection_kkt(
    const char *name, double alpha, const double input[3], const double weights[3], const double projected[3])
{
    double point_scale = 0.0;
    for (int i = 0; i < 3; ++i)
        point_scale = fmax(point_scale, fmax(fabs(input[i]), fabs(projected[i])));
    if (point_scale == 0.0)
        return 1;

    double normalized_input_x = input[0] / point_scale;
    double normalized_input_y = input[1] / point_scale;
    double normalized_input_z = input[2] / point_scale;
    double normalized_input_bound = power_bound(normalized_input_x, normalized_input_y, alpha);
    if (normalized_input_x >= 0.0 && normalized_input_y >= 0.0 &&
        fabs(normalized_input_z) <= normalized_input_bound * (1.0 + 64.0 * DBL_EPSILON))
        return 1;

    double polar[3];
    double polar_scale = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        polar[i] = weights[i] * (input[i] / point_scale - projected[i] / point_scale);
        polar_scale = fmax(polar_scale, fabs(polar[i]));
    }
    if (polar_scale == 0.0)
        return 1;

    /* When an active-coordinate correction is below one ULP of the input,
       the returned point no longer contains enough information to reconstruct
       that component of the normal vector.  Explicit extreme-scale cases are
       checked against reference projections separately. */
    if (projected[0] != 0.0 && projected[1] != 0.0 && projected[2] != 0.0 &&
        (fabs(polar[0]) <= 64.0 * DBL_EPSILON * polar_scale || fabs(polar[1]) <= 64.0 * DBL_EPSILON * polar_scale ||
         fabs(polar[2]) <= 64.0 * DBL_EPSILON * polar_scale))
        return 1;
    for (int i = 0; i < 3; ++i)
        polar[i] /= polar_scale;

    const double tolerance = 1e-5;
    int ok = 1;
    if (polar[0] > tolerance || polar[1] > tolerance)
        ok = 0;

    double abs_w = fabs(polar[2]);
    if (abs_w > tolerance)
    {
        double u = -polar[0] / alpha;
        double v = -polar[1] / (1.0 - alpha);
        if (u <= 0.0 || v <= 0.0 || alpha * log(u) + (1.0 - alpha) * log(v) + tolerance < log(abs_w))
            ok = 0;
    }

    double complementarity = 0.0;
    double complementarity_scale = 1.0;
    for (int i = 0; i < 3; ++i)
    {
        double term = (projected[i] / point_scale) * polar[i];
        complementarity += term;
        complementarity_scale += fabs(term);
    }
    if (fabs(complementarity) > tolerance * complementarity_scale)
        ok = 0;

    if (!ok)
    {
        fprintf(stderr,
                "%s: projection KKT failed: alpha=%.17g, input=(%.17g, %.17g, %.17g), "
                "weights=(%.17g, %.17g, %.17g), point=(%.17g, %.17g, %.17g), "
                "polar=(%.3e, %.3e, %.3e), complementarity=%.3e\n",
                name,
                alpha,
                input[0],
                input[1],
                input[2],
                weights[0],
                weights[1],
                weights[2],
                projected[0],
                projected[1],
                projected[2],
                polar[0],
                polar[1],
                polar[2],
                complementarity);
    }
    return ok;
}

static int run_case(const char *name,
                    double alpha,
                    const double input[3],
                    const double weights[3],
                    const char fixed[3],
                    const double expected[3],
                    double tolerance)
{
    double scaled_input[3];
    double scale[3];
    double actual[3] = {0.0, 0.0, 0.0};
    double actual_violation = 0.0;
    double input_bound = 0.0;
    double expected_violation = 0.0;
    double projected_bound = 0.0;
    double projected_violation = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        scale[i] = sqrt(weights[i]);
        scaled_input[i] = input[i] * scale[i];
    }

    double *d_point = NULL;
    double *d_scale = NULL;
    double *d_warm = NULL;
    double *d_alpha = NULL;
    double *d_violation = NULL;
    double *d_relative_violation = NULL;
    int *d_start = NULL;
    int *d_dim = NULL;
    char *d_fixed = NULL;
    int start = 0;
    int dim = 1;
    double warm = 0.0;
    int ok = cuda_ok(cudaMalloc(&d_point, 3 * sizeof(double)), "cudaMalloc(point)") &&
        cuda_ok(cudaMalloc(&d_scale, 3 * sizeof(double)), "cudaMalloc(scale)") &&
        cuda_ok(cudaMalloc(&d_warm, sizeof(double)), "cudaMalloc(warm)") &&
        cuda_ok(cudaMalloc(&d_alpha, sizeof(double)), "cudaMalloc(alpha)") &&
        cuda_ok(cudaMalloc(&d_violation, sizeof(double)), "cudaMalloc(violation)") &&
        cuda_ok(cudaMalloc(&d_relative_violation, sizeof(double)), "cudaMalloc(relative violation)") &&
        cuda_ok(cudaMalloc(&d_start, sizeof(int)), "cudaMalloc(start)") &&
        cuda_ok(cudaMalloc(&d_dim, sizeof(int)), "cudaMalloc(dim)") &&
        cuda_ok(cudaMalloc(&d_fixed, 3 * sizeof(char)), "cudaMalloc(fixed)");
    if (!ok)
        goto cleanup;

    ok = cuda_ok(cudaMemcpy(d_point, scaled_input, 3 * sizeof(double), cudaMemcpyHostToDevice), "copy point") &&
        cuda_ok(cudaMemcpy(d_scale, scale, 3 * sizeof(double), cudaMemcpyHostToDevice), "copy scale") &&
        cuda_ok(cudaMemcpy(d_warm, &warm, sizeof(double), cudaMemcpyHostToDevice), "copy warm") &&
        cuda_ok(cudaMemcpy(d_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice), "copy alpha") &&
        cuda_ok(cudaMemcpy(d_start, &start, sizeof(int), cudaMemcpyHostToDevice), "copy start") &&
        cuda_ok(cudaMemcpy(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice), "copy dim") &&
        cuda_ok(cudaMemcpy(d_fixed, fixed, 3 * sizeof(char), cudaMemcpyHostToDevice), "copy fixed");
    if (!ok)
        goto cleanup;

    compute_power_cone_primal_violation_kernel<<<1, 1>>>(
        d_violation, d_relative_violation, d_point, d_scale, d_start, d_alpha, 1.0, 1);
    ok = cuda_ok(cudaGetLastError(), "power violation kernel launch") &&
        cuda_ok(cudaDeviceSynchronize(), "power violation kernel sync") &&
        cuda_ok(cudaMemcpy(&actual_violation, d_violation, sizeof(double), cudaMemcpyDeviceToHost), "copy violation");
    input_bound = power_bound(input[0], input[1], alpha);
    expected_violation = fmax(0.0, fmax(-input[0], fmax(-input[1], fabs(input[2]) - input_bound)));
    if (!ok || fabs(actual_violation - expected_violation) > 1e-12 * (1.0 + expected_violation))
    {
        fprintf(
            stderr, "%s: membership violation is %.17g, expected %.17g\n", name, actual_violation, expected_violation);
        ok = 0;
        goto cleanup;
    }

    project_power_cone_kernel<<<1, 1>>>(d_point, d_scale, d_warm, d_start, d_dim, d_alpha, d_fixed, 1);
    ok = cuda_ok(cudaGetLastError(), "project_power_cone_kernel launch") &&
        cuda_ok(cudaDeviceSynchronize(), "project_power_cone_kernel sync") &&
        cuda_ok(cudaMemcpy(scaled_input, d_point, 3 * sizeof(double), cudaMemcpyDeviceToHost), "copy result");
    if (!ok)
        goto cleanup;

    for (int i = 0; i < 3; ++i)
    {
        actual[i] = scaled_input[i] / scale[i];
        double error = expected ? fabs(actual[i] - expected[i]) : 0.0;
        if (expected && error > tolerance * (1.0 + fabs(expected[i])))
        {
            fprintf(stderr,
                    "%s: coordinate %d is %.17g, expected %.17g (error %.3e)\n",
                    name,
                    i,
                    actual[i],
                    expected[i],
                    error);
            ok = 0;
        }
        if (fixed[i] && actual[i] != input[i])
        {
            fprintf(stderr, "%s: fixed coordinate %d changed from %.17g to %.17g\n", name, i, input[i], actual[i]);
            ok = 0;
        }
    }
    projected_bound = power_bound(actual[0], actual[1], alpha);
    projected_violation = fmax(0.0, fmax(-actual[0], fmax(-actual[1], fabs(actual[2]) - projected_bound)));
    if (projected_violation > 1e-10 * (1.0 + fabs(actual[2])))
    {
        fprintf(stderr, "%s: projected point violates the power cone by %.3e\n", name, projected_violation);
        ok = 0;
    }
    if (!expected && !fixed[0] && !fixed[1] && !fixed[2])
        ok &= check_free_projection_kkt(name, alpha, input, weights, actual);

cleanup:
    cudaFree(d_point);
    cudaFree(d_scale);
    cudaFree(d_warm);
    cudaFree(d_alpha);
    cudaFree(d_violation);
    cudaFree(d_relative_violation);
    cudaFree(d_start);
    cudaFree(d_dim);
    cudaFree(d_fixed);
    return ok;
}

static uint64_t random_state = UINT64_C(0x8d12e93a5bc7416f);

static double random_unit(void)
{
    random_state = random_state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return (double)(random_state >> 11) * 0x1.0p-53;
}

static int run_random_free_projection_cases(void)
{
    static const char free_slots[3] = {0, 0, 0};
    int passed = 1;
    for (int case_idx = 0; case_idx < 96; ++case_idx)
    {
        double alpha = 0.1 + 0.8 * random_unit();
        double common_scale = pow(10.0, -120.0 + 240.0 * random_unit());
        double input[3] = {
            common_scale * (6.0 * random_unit() - 3.0),
            common_scale * (6.0 * random_unit() - 3.0),
            common_scale * (6.0 * random_unit() - 3.0),
        };
        double weights[3] = {
            pow(10.0, -2.0 + 4.0 * random_unit()),
            pow(10.0, -2.0 + 4.0 * random_unit()),
            pow(10.0, -2.0 + 4.0 * random_unit()),
        };
        char name[64];
        snprintf(name, sizeof(name), "random-free-%d", case_idx);
        passed &= run_case(name, alpha, input, weights, free_slots, NULL, 0.0);
    }
    return passed;
}

int main(void)
{
    static const char free_slots[3] = {0, 0, 0};
    static const char fixed_z[3] = {0, 0, 1};
    static const char fixed_x[3] = {1, 0, 0};
    static const char fixed_y[3] = {0, 1, 0};
    static const char fixed_xz[3] = {1, 0, 1};
    static const char fixed_yz[3] = {0, 1, 1};
    static const char fixed_xy[3] = {1, 1, 0};
    static const char fixed_all[3] = {1, 1, 1};
    static const double unit_weights[3] = {1.0, 1.0, 1.0};

    int passed = 1;
    {
        const double input[3] = {-0.7, 1.2, 2.0};
        const double expected[3] = {0.2999138114091835, 1.629436048818783, 0.9806749092638409};
        passed &= run_case("full-unweighted", 0.3, input, unit_weights, free_slots, expected, 3e-6);
    }
    {
        const double input[3] = {-0.7e200, 1.2e200, 2.0e200};
        const double expected[3] = {0.2999138114091835e200, 1.629436048818783e200, 0.9806749092638409e200};
        passed &= run_case("full-huge-scale", 0.3, input, unit_weights, free_slots, expected, 3e-6);
    }
    {
        const double input[3] = {1.5, -0.4, -2.2};
        const double weights[3] = {0.3, 2.0, 5.0};
        const double expected[3] = {3.427365210244501, 0.4816579589566281, -1.902365192823554};
        passed &= run_case("full-weighted", 0.7, input, weights, free_slots, expected, 3e-6);
    }
    {
        const double input[3] = {-1e16, 1e16, 5e15};
        const double expected[3] = {2.8874860407696615e-15, 1e16, 5.833305132868003e-15};
        passed &= run_case("full-sharp-root", 0.99, input, unit_weights, free_slots, expected, 1e-12);
    }
    {
        const double input[3] = {-1e200, 1e200, 1e100};
        const double expected[3] = {1.0 / 9.0, 1e200, 1e100 / 3.0};
        passed &= run_case("full-wide-dynamic-range", 0.5, input, unit_weights, free_slots, expected, 1e-12);
    }
    {
        const double input[3] = {2.0, -3.0, 0.0};
        const double weights[3] = {2.0, 7.0, 0.25};
        const double expected[3] = {2.0, 0.0, 0.0};
        passed &= run_case("full-zero-z", 0.5, input, weights, free_slots, expected, 1e-12);
    }
    {
        const double input[3] = {-1.0, -1.0, 0.2};
        const double expected[3] = {0.0, 0.0, 0.0};
        passed &= run_case("full-opposite-cone", 0.5, input, unit_weights, free_slots, expected, 1e-12);
    }
    {
        const double input[3] = {-0.5, 0.2, 1.0};
        const double weights[3] = {2.0, 0.7, 1.0};
        const double expected[3] = {0.3619815616314009, 1.545732644202259, 1.0};
        passed &= run_case("fixed-z", 0.3, input, weights, fixed_z, expected, 3e-5);
    }
    {
        const double input[3] = {-0.5e200, 0.2e200, 1.0e200};
        const double weights[3] = {2.0, 0.7, 1.0};
        const double expected[3] = {0.3619815616314009e200, 1.545732644202259e200, 1.0e200};
        passed &= run_case("fixed-z-huge-scale", 0.3, input, weights, fixed_z, expected, 3e-5);
    }
    {
        const double input[3] = {0.0, 0.0, 1.0};
        const double weights[3] = {1e200, 1e200, 1.0};
        const double expected[3] = {1.0, 1.0, 1.0};
        passed &= run_case("fixed-z-huge-weight", 0.5, input, weights, fixed_z, expected, 1e-10);
    }
    {
        const double input[3] = {0.0, 0.0, 1.0};
        const double weights[3] = {1e-200, 1e-200, 1.0};
        const double expected[3] = {1.0, 1.0, 1.0};
        passed &= run_case("fixed-z-tiny-weight", 0.5, input, weights, fixed_z, expected, 1e-10);
    }
    {
        const double input[3] = {1.0, -0.4, 2.0};
        const double weights[3] = {1.0, 2.0, 0.5};
        const double expected[3] = {1.0, 0.1445424296046515, 0.2582240838862325};
        passed &= run_case("fixed-x", 0.3, input, weights, fixed_x, expected, 3e-5);
    }
    {
        const double alpha = 0.97;
        const double input[3] = {0.62720941, -0.01546521, 0.19700471};
        const double expected_y = pow(fabs(input[2]) / pow(input[0], alpha), 1.0 / (1.0 - alpha));
        const double expected[3] = {input[0], expected_y, input[2]};
        passed &= run_case("fixed-x-sharp", alpha, input, unit_weights, fixed_x, expected, 1e-10);
    }
    {
        const double alpha = 0.999;
        const double input[3] = {1.0, -0.1, 0.5};
        const double expected_y = pow(0.5, 1.0 / (1.0 - alpha));
        const double expected[3] = {input[0], expected_y, input[2]};
        passed &= run_case("fixed-x-ultrasharp", alpha, input, unit_weights, fixed_x, expected, 1e-10);
    }
    {
        const double input[3] = {0.00093260334688321987, 0.0, 1.0};
        const double expected[3] = {
            input[0],
            0.0030706556671616642,
            0.00094378334963825893,
        };
        passed &= run_case("fixed-x-huge-feasible-bound", 0.99, input, unit_weights, fixed_x, expected, 1e-9);
    }
    {
        const double input[3] = {-0.3, 1.0, -1.8};
        const double weights[3] = {3.0, 1.0, 0.8};
        const double expected[3] = {0.1744696096520296, 1.0, -0.2945804034203198};
        passed &= run_case("fixed-y", 0.7, input, weights, fixed_y, expected, 3e-5);
    }
    {
        const double input[3] = {1.0, -0.4, 1.0};
        const double weights[3] = {1.0, 2.0, 0.5};
        const double expected[3] = {1.0, 1.0, 1.0};
        passed &= run_case("fixed-xz", 0.3, input, weights, fixed_xz, expected, 1e-12);
    }
    {
        const double input[3] = {-0.3, 1.0, -1.0};
        const double weights[3] = {3.0, 1.0, 0.8};
        const double expected[3] = {1.0, 1.0, -1.0};
        passed &= run_case("fixed-yz", 0.7, input, weights, fixed_yz, expected, 1e-12);
    }
    {
        const double input[3] = {2.0, 3.0, 4.0};
        const double weights[3] = {2.0, 7.0, 0.25};
        const double expected[3] = {2.0, 3.0, 2.656402479886323};
        passed &= run_case("fixed-xy", 0.3, input, weights, fixed_xy, expected, 1e-12);
    }
    {
        const double input[3] = {1.0, 1.0, 0.5};
        const double expected[3] = {1.0, 1.0, 0.5};
        passed &= run_case("fixed-all", 0.3, input, unit_weights, fixed_all, expected, 1e-12);
    }
    passed &= run_random_free_projection_cases();

    return passed ? 0 : 1;
}
