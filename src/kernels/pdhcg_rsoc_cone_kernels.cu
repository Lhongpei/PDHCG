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

#include "cone_kernel_ops.h"
#include "cone_kernel_reductions.h"
#include "cone_projection_utils.h"
#include "pdhcg_rsoc_cone_kernels.h"
/* Map a weighted rotated SOC to a weighted standard SOC in sum/difference
   endpoint coordinates, then reuse its negative-branch bracket. */
__device__ static inline double cone_section_negative_rsoc_upper(double omega_s,
                                                                 double omega_t,
                                                                 double s,
                                                                 double t,
                                                                 double fixed_norm2,
                                                                 double polar_norm2,
                                                                 double max_vector_metric)
{
    const double inv_sqrt2 = 0.70710678118654752440;
    double sqrt_omega_s = sqrt(omega_s);
    double sqrt_omega_t = sqrt(omega_t);
    double root_metric = sqrt_omega_s * sqrt_omega_t;
    double scaled_s = sqrt_omega_s * s;
    double scaled_t = sqrt_omega_t * t;
    double transformed_w = (scaled_s - scaled_t) * inv_sqrt2;
    double endpoint_polar = -(scaled_s + scaled_t) * inv_sqrt2;
    double transformed_fixed_norm2 = root_metric * fixed_norm2;
    double transformed_polar_norm2 = polar_norm2 / root_metric + transformed_w * transformed_w;
    double transformed_max_metric = fmax(1.0, max_vector_metric / root_metric);
    double transformed_upper = cone_section_negative_soc_upper(
        1.0, endpoint_polar, transformed_fixed_norm2, transformed_polar_norm2, transformed_max_metric);
    return root_metric * transformed_upper;
}
__device__ static inline double rotated_soc_smooth_objective(const double *point,
                                                             const double *rescaling,
                                                             const double *q_diag,
                                                             double tau,
                                                             int start,
                                                             int k,
                                                             const char *is_fixed,
                                                             double lambda,
                                                             double s,
                                                             double t)
{
    double objective = 0.0;
    for (int slot = 0; slot < k; ++slot)
    {
        int index = start + slot;
        if (is_fixed[index])
            continue;
        double omega = cone_section_weight(rescaling, q_diag, tau, index);
        double input = cone_section_actual(point, rescaling, index);
        double value = input * omega / (omega + lambda);
        double delta = value - input;
        objective += omega * delta * delta;
    }
    int s_index = start + k;
    int t_index = s_index + 1;
    double omega_s = cone_section_weight(rescaling, q_diag, tau, s_index);
    double omega_t = cone_section_weight(rescaling, q_diag, tau, t_index);
    double ds = s - cone_section_actual(point, rescaling, s_index);
    double dt = t - cone_section_actual(point, rescaling, t_index);
    return objective + omega_s * ds * ds + omega_t * dt * dt;
}

/* Weighted projection onto an arbitrary nonempty fixed section of
   { (v,s,t) : ||v||_2^2 <= 2 s t, s >= 0, t >= 0 }. */
__device__ static inline void project_rotated_soc_section_serial(double *point,
                                                                 const double *rescaling,
                                                                 const double *q_diag,
                                                                 double tau,
                                                                 double *warm_start,
                                                                 int start,
                                                                 int k,
                                                                 const char *is_fixed)
{
    int s_index = start + k;
    int t_index = s_index + 1;
    bool fixed_s = is_fixed[s_index] != 0;
    bool fixed_t = is_fixed[t_index] != 0;
    double s_input = cone_section_actual(point, rescaling, s_index);
    double t_input = cone_section_actual(point, rescaling, t_index);
    double fixed_norm2 = 0.0;
    double free_norm2 = 0.0;
    double polar_norm2 = 0.0;
    double max_omega = 0.0;
    int free_count = 0;
    for (int slot = 0; slot < k; ++slot)
    {
        int index = start + slot;
        double value = cone_section_actual(point, rescaling, index);
        if (is_fixed[index])
            fixed_norm2 += value * value;
        else
        {
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            free_norm2 += value * value;
            polar_norm2 += (omega * value) * (omega * value);
            max_omega = fmax(max_omega, omega);
            ++free_count;
        }
    }

    if (fixed_s && fixed_t)
    {
        double radius2 = fmax(0.0, 2.0 * s_input * t_input - fixed_norm2);
        if (free_count == 0 || free_norm2 <= radius2)
            return;
        if (!(radius2 > 0.0))
        {
            for (int slot = 0; slot < k; ++slot)
                if (!is_fixed[start + slot])
                    point[start + slot] = 0.0;
            return;
        }

        double lo = 0.0;
        double hi = sqrt(polar_norm2) / sqrt(radius2) * (1.0 + 64.0 * DBL_EPSILON);
        if (!(hi > 0.0) || !isfinite(hi))
        {
            hi = warm_start && *warm_start > 0.0 && isfinite(*warm_start) ? *warm_start : 1.0;
            for (int expansion = 0; expansion < 100; ++expansion)
            {
                double norm2 = 0.0;
                for (int slot = 0; slot < k; ++slot)
                {
                    int index = start + slot;
                    if (is_fixed[index])
                        continue;
                    double omega = cone_section_weight(rescaling, q_diag, tau, index);
                    double value = cone_section_actual(point, rescaling, index) * omega / (omega + hi);
                    norm2 += value * value;
                }
                if (norm2 <= radius2)
                    break;
                hi *= 2.0;
            }
        }
        for (int iteration = 0; iteration < 80; ++iteration)
        {
            double lambda = 0.5 * (lo + hi);
            double norm2 = 0.0;
            for (int slot = 0; slot < k; ++slot)
            {
                int index = start + slot;
                if (is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = cone_section_actual(point, rescaling, index) * omega / (omega + lambda);
                norm2 += value * value;
            }
            if (norm2 > radius2)
                lo = lambda;
            else
                hi = lambda;
            if ((hi - lo) <= 1e-13 * (1.0 + hi + lo))
                break;
        }
        double lambda = 0.5 * (lo + hi);
        if (warm_start)
            *warm_start = lambda;
        for (int slot = 0; slot < k; ++slot)
        {
            int index = start + slot;
            if (!is_fixed[index])
            {
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                point[index] *= omega / (omega + lambda);
            }
        }
        return;
    }

    if (fixed_s || fixed_t)
    {
        int free_endpoint_index = fixed_s ? t_index : s_index;
        double fixed_endpoint = fixed_s ? s_input : t_input;
        double free_endpoint_input = fixed_s ? t_input : s_input;
        double omega_endpoint = cone_section_weight(rescaling, q_diag, tau, free_endpoint_index);
        if (!(fixed_endpoint > 0.0))
        {
            for (int slot = 0; slot < k; ++slot)
                if (!is_fixed[start + slot])
                    point[start + slot] = 0.0;
            point[free_endpoint_index] = fmax(free_endpoint_input, 0.0) * rescaling[free_endpoint_index];
            return;
        }
        if (free_endpoint_input >= 0.0 && fixed_norm2 + free_norm2 <= 2.0 * fixed_endpoint * free_endpoint_input)
            return;
        if (free_count == 0)
        {
            double lower_bound = fixed_norm2 / (2.0 * fixed_endpoint);
            point[free_endpoint_index] = fmax(free_endpoint_input, lower_bound) * rescaling[free_endpoint_index];
            return;
        }

        double lo = 0.0;
        double violation = fixed_norm2 + free_norm2 - 2.0 * fixed_endpoint * free_endpoint_input;
        double hi = omega_endpoint * violation / (2.0 * fixed_endpoint * fixed_endpoint);
        hi *= 1.0 + 64.0 * DBL_EPSILON;
        if (!(hi > 0.0) || !isfinite(hi))
        {
            hi = warm_start && *warm_start > 0.0 && isfinite(*warm_start) ? *warm_start : omega_endpoint;
            for (int expansion = 0; expansion < 100; ++expansion)
            {
                double norm2 = fixed_norm2;
                for (int slot = 0; slot < k; ++slot)
                {
                    int index = start + slot;
                    if (is_fixed[index])
                        continue;
                    double omega = cone_section_weight(rescaling, q_diag, tau, index);
                    double value = cone_section_actual(point, rescaling, index) * omega / (omega + hi);
                    norm2 += value * value;
                }
                double endpoint = free_endpoint_input + hi * fixed_endpoint / omega_endpoint;
                if (norm2 <= 2.0 * fixed_endpoint * endpoint)
                    break;
                hi *= 2.0;
            }
        }
        for (int iteration = 0; iteration < 80; ++iteration)
        {
            double lambda = 0.5 * (lo + hi);
            double norm2 = fixed_norm2;
            for (int slot = 0; slot < k; ++slot)
            {
                int index = start + slot;
                if (is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = cone_section_actual(point, rescaling, index) * omega / (omega + lambda);
                norm2 += value * value;
            }
            double endpoint = free_endpoint_input + lambda * fixed_endpoint / omega_endpoint;
            if (norm2 > 2.0 * fixed_endpoint * endpoint)
                lo = lambda;
            else
                hi = lambda;
            if ((hi - lo) <= 1e-13 * (1.0 + hi + lo))
                break;
        }
        double lambda = 0.5 * (lo + hi);
        if (warm_start)
            *warm_start = lambda;
        for (int slot = 0; slot < k; ++slot)
        {
            int index = start + slot;
            if (!is_fixed[index])
            {
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                point[index] *= omega / (omega + lambda);
            }
        }
        point[free_endpoint_index] =
            (free_endpoint_input + lambda * fixed_endpoint / omega_endpoint) * rescaling[free_endpoint_index];
        return;
    }

    double total_norm2 = fixed_norm2 + free_norm2;
    if (s_input >= 0.0 && t_input >= 0.0 && total_norm2 <= 2.0 * s_input * t_input)
        return;

    double omega_s = cone_section_weight(rescaling, q_diag, tau, s_index);
    double omega_t = cone_section_weight(rescaling, q_diag, tau, t_index);
    if (fixed_norm2 == 0.0)
    {
        double bs = omega_s * s_input;
        double bt = omega_t * t_input;
        if (bs <= 0.0 && bt <= 0.0 && polar_norm2 <= 2.0 * bs * bt)
        {
            for (int slot = 0; slot < k; ++slot)
                if (!is_fixed[start + slot])
                    point[start + slot] = 0.0;
            point[s_index] = 0.0;
            point[t_index] = 0.0;
            return;
        }
    }

    double root_metric = sqrt(omega_s) * sqrt(omega_t);
    double balance = sqrt(omega_s) * s_input + sqrt(omega_t) * t_input;
    double balance_scale = 1.0 + fabs(sqrt(omega_s) * s_input) + fabs(sqrt(omega_t) * t_input);
    double lambda = root_metric;
    double projected_s = 0.0;
    double projected_t = 0.0;
    bool smooth_valid = true;

    if (fabs(balance) <= 64.0 * DBL_EPSILON * balance_scale)
    {
        double norm2 = fixed_norm2;
        for (int slot = 0; slot < k; ++slot)
        {
            int index = start + slot;
            if (is_fixed[index])
                continue;
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            double value = cone_section_actual(point, rescaling, index) * omega / (omega + lambda);
            norm2 += value * value;
        }
        double product = 0.5 * root_metric * norm2;
        double delta = sqrt(omega_s) * s_input;
        double scaled_t = 0.5 * (-delta + sqrt(fmax(0.0, delta * delta + 4.0 * product)));
        double scaled_s = scaled_t + delta;
        projected_s = scaled_s / sqrt(omega_s);
        projected_t = scaled_t / sqrt(omega_t);
        smooth_valid = projected_s >= 0.0 && projected_t >= 0.0;
    }
    else
    {
        bool lower_branch = balance > 0.0;
        double lo = lower_branch ? 0.0 : root_metric * (1.0 + 1e-14);
        double hi = lower_branch ? root_metric * (1.0 - 1e-14) : 2.0 * root_metric;

        if (!lower_branch)
        {
            hi = cone_section_negative_rsoc_upper(
                omega_s, omega_t, s_input, t_input, fixed_norm2, polar_norm2, max_omega);
            if (!(hi > lo) || !isfinite(hi))
            {
                hi = 2.0 * root_metric;
                for (int expansion = 0; expansion < 100; ++expansion)
                {
                    double determinant = omega_s * omega_t - hi * hi;
                    double s = omega_t * (omega_s * s_input + hi * t_input) / determinant;
                    double t = omega_s * (omega_t * t_input + hi * s_input) / determinant;
                    double f = INFINITY;
                    if (s >= 0.0 && t >= 0.0)
                    {
                        double norm2 = fixed_norm2;
                        for (int slot = 0; slot < k; ++slot)
                        {
                            int index = start + slot;
                            if (is_fixed[index])
                                continue;
                            double omega = cone_section_weight(rescaling, q_diag, tau, index);
                            double value = cone_section_actual(point, rescaling, index) * omega / (omega + hi);
                            norm2 += value * value;
                        }
                        f = norm2 - 2.0 * s * t;
                    }
                    if (f >= 0.0)
                        break;
                    hi *= 2.0;
                }
            }
        }

        for (int iteration = 0; iteration < 90; ++iteration)
        {
            double trial = 0.5 * (lo + hi);
            double determinant = omega_s * omega_t - trial * trial;
            double s = omega_t * (omega_s * s_input + trial * t_input) / determinant;
            double t = omega_s * (omega_t * t_input + trial * s_input) / determinant;
            double f = INFINITY;
            if (s >= 0.0 && t >= 0.0)
            {
                double norm2 = fixed_norm2;
                for (int slot = 0; slot < k; ++slot)
                {
                    int index = start + slot;
                    if (is_fixed[index])
                        continue;
                    double omega = cone_section_weight(rescaling, q_diag, tau, index);
                    double value = cone_section_actual(point, rescaling, index) * omega / (omega + trial);
                    norm2 += value * value;
                }
                f = norm2 - 2.0 * s * t;
            }
            if ((lower_branch && f > 0.0) || (!lower_branch && f < 0.0))
                lo = trial;
            else
                hi = trial;
            if ((hi - lo) <= 1e-13 * (1.0 + hi + lo))
                break;
        }
        lambda = 0.5 * (lo + hi);
        double determinant = omega_s * omega_t - lambda * lambda;
        projected_s = omega_t * (omega_s * s_input + lambda * t_input) / determinant;
        projected_t = omega_s * (omega_t * t_input + lambda * s_input) / determinant;
        smooth_valid = isfinite(projected_s) && isfinite(projected_t) && projected_s >= 0.0 && projected_t >= 0.0;
    }

    double best_objective = smooth_valid
        ? rotated_soc_smooth_objective(
              point, rescaling, q_diag, tau, start, k, is_fixed, lambda, projected_s, projected_t)
        : INFINITY;
    int mode = smooth_valid ? 0 : 1;
    if (fixed_norm2 == 0.0)
    {
        double vector_objective = 0.0;
        for (int slot = 0; slot < k; ++slot)
        {
            int index = start + slot;
            if (!is_fixed[index])
            {
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = cone_section_actual(point, rescaling, index);
                vector_objective += omega * value * value;
            }
        }
        double s_axis = fmax(s_input, 0.0);
        double s_axis_objective =
            vector_objective + omega_s * (s_axis - s_input) * (s_axis - s_input) + omega_t * t_input * t_input;
        if (s_axis_objective < best_objective)
        {
            best_objective = s_axis_objective;
            projected_s = s_axis;
            projected_t = 0.0;
            mode = 1;
        }
        double t_axis = fmax(t_input, 0.0);
        double t_axis_objective =
            vector_objective + omega_s * s_input * s_input + omega_t * (t_axis - t_input) * (t_axis - t_input);
        if (t_axis_objective < best_objective)
        {
            projected_s = 0.0;
            projected_t = t_axis;
            mode = 1;
        }
    }

    if (mode == 0)
    {
        for (int slot = 0; slot < k; ++slot)
        {
            int index = start + slot;
            if (!is_fixed[index])
            {
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                point[index] *= omega / (omega + lambda);
            }
        }
        if (warm_start)
            *warm_start = lambda;
    }
    else
    {
        for (int slot = 0; slot < k; ++slot)
            if (!is_fixed[start + slot])
                point[start + slot] = 0.0;
        if (warm_start)
            *warm_start = 0.0;
    }
    point[s_index] = projected_s * rescaling[s_index];
    point[t_index] = projected_t * rescaling[t_index];
}
#include "utils.h"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void project_rotated_soc_kernel(double *__restrict__ primal_solution,
                                           const double *__restrict__ variable_rescaling,
                                           double *__restrict__ warm_start,
                                           const int *__restrict__ start_idx,
                                           const int *__restrict__ v_dim,
                                           const char *__restrict__ is_fixed,
                                           int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    const double INV_SQRT2 = 0.7071067811865475;

    int start = start_idx[blk];
    int k = v_dim[blk];
    if (cone_section_has_fixed(is_fixed, start, k + 2))
    {
        project_rotated_soc_section_serial(
            primal_solution, variable_rescaling, NULL, 0.0, warm_start + blk, start, k, is_fixed);
        return;
    }
    double *v = primal_solution + start;
    double *sptr = primal_solution + start + k;
    double *tptr = primal_solution + start + k + 1;

    double s = *sptr;
    double t = *tptr;

    double w = (s - t) * INV_SQRT2;
    double z = (s + t) * INV_SQRT2;

    double d_s = variable_rescaling[start + k];
    double d_t = variable_rescaling[start + k + 1];
    double d_st = sqrt(d_s * d_t);

    bool diag_uniform = true;
    for (int m = 0; m < k && diag_uniform; ++m)
    {
        if (variable_rescaling[start + m] != d_st)
            diag_uniform = false;
    }
    if (diag_uniform)
    {
        double sumsq = w * w;
        for (int m = 0; m < k; ++m)
            sumsq += v[m] * v[m];
        double r = sqrt(sumsq);
        if (r <= z)
            return;
        if (r <= -z)
        {
            for (int m = 0; m < k; ++m)
                v[m] = 0.0;
            *sptr = 0.0;
            *tptr = 0.0;
            return;
        }
        double scale = (z + r) / (2.0 * r);
        for (int m = 0; m < k; ++m)
            v[m] *= scale;
        double w_new = scale * w;
        double z_new = scale * r;
        *sptr = (z_new + w_new) * INV_SQRT2;
        *tptr = (z_new - w_new) * INV_SQRT2;
        return;
    }

    double r_inv_sq = w * w;
    double r_pos_sq = w * w;
    for (int m = 0; m < k; ++m)
    {
        double dh = variable_rescaling[start + m] / d_st;
        double v_m = v[m];
        r_inv_sq += (v_m / dh) * (v_m / dh);
        r_pos_sq += (v_m * dh) * (v_m * dh);
    }
    double r_inv = sqrt(r_inv_sq);
    if (r_inv <= z)
        return;
    double r_pos = sqrt(r_pos_sq);
    if (r_pos <= -z)
    {
        for (int m = 0; m < k; ++m)
            v[m] = 0.0;
        *sptr = 0.0;
        *tptr = 0.0;
        return;
    }

    double lo, hi;
    bool z_pos = (z > 0.0);
    if (z_pos)
    {
        lo = 0.0;
        hi = 0.5 - 1e-14;
    }
    else
    {
        lo = 0.5 + 1e-14;
        hi = 1.0;
        for (int doubling = 0; doubling < 60; ++doubling)
        {
            double sum_hi = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double dh = variable_rescaling[start + m] / d_st;
                double dh2 = dh * dh;
                double tt = v[m] * dh / (dh2 + 2.0 * hi);
                sum_hi += tt * tt;
            }
            double tw_hi = w / (1.0 + 2.0 * hi);
            sum_hi += tw_hi * tw_hi;
            double zt_hi = z / (1.0 - 2.0 * hi);
            double f_hi = sum_hi - zt_hi * zt_hi;
            if (f_hi > 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_lam = warm_start[blk];
    if (warm_lam > lo && warm_lam < hi)
    {
        double sum_w = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double dh = variable_rescaling[start + m] / d_st;
            double dh2 = dh * dh;
            double tt = v[m] * dh / (dh2 + 2.0 * warm_lam);
            sum_w += tt * tt;
        }
        double tw = w / (1.0 + 2.0 * warm_lam);
        sum_w += tw * tw;
        double zt = z / (1.0 - 2.0 * warm_lam);
        double f = sum_w - zt * zt;
        if (fabs(f) < 1e-12)
        {
            double w_new = w / (1.0 + 2.0 * warm_lam);
            double z_new = z / (1.0 - 2.0 * warm_lam);
            for (int m = 0; m < k; ++m)
            {
                double dh = variable_rescaling[start + m] / d_st;
                double dh2 = dh * dh;
                v[m] = v[m] * dh2 / (dh2 + 2.0 * warm_lam);
            }
            *sptr = (z_new + w_new) * INV_SQRT2;
            *tptr = (z_new - w_new) * INV_SQRT2;
            return;
        }
        if (z_pos)
        {
            if (f > 0.0)
                lo = warm_lam;
            else
                hi = warm_lam;
        }
        else
        {
            if (f > 0.0)
                hi = warm_lam;
            else
                lo = warm_lam;
        }
    }

    for (int it = 0; it < 60; ++it)
    {
        double lam = 0.5 * (lo + hi);
        double sum = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double dh = variable_rescaling[start + m] / d_st;
            double dh2 = dh * dh;
            double tt = v[m] * dh / (dh2 + 2.0 * lam);
            sum += tt * tt;
        }
        double tw = w / (1.0 + 2.0 * lam);
        sum += tw * tw;
        double zt = z / (1.0 - 2.0 * lam);
        double f = sum - zt * zt;
        if (z_pos)
        {
            if (f > 0.0)
                lo = lam;
            else
                hi = lam;
        }
        else
        {
            if (f > 0.0)
                hi = lam;
            else
                lo = lam;
        }
        if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
            break;
    }
    double lam = 0.5 * (lo + hi);
    warm_start[blk] = lam;

    double w_new = w / (1.0 + 2.0 * lam);
    double z_new = z / (1.0 - 2.0 * lam);
    for (int m = 0; m < k; ++m)
    {
        double dh = variable_rescaling[start + m] / d_st;
        double dh2 = dh * dh;
        v[m] = v[m] * dh2 / (dh2 + 2.0 * lam);
    }
    *sptr = (z_new + w_new) * INV_SQRT2;
    *tptr = (z_new - w_new) * INV_SQRT2;
}

__global__ void compute_cone_dual_residual_kernel(double *__restrict__ dual_residual,
                                                  double *__restrict__ complementarity_residual,
                                                  const double *__restrict__ objective_vector,
                                                  const double *__restrict__ dual_product,
                                                  const double *__restrict__ variable_rescaling,
                                                  const double *__restrict__ primal_solution,
                                                  double *__restrict__ warm_start,
                                                  const int *__restrict__ start_idx,
                                                  const int *__restrict__ v_dim,
                                                  const char *__restrict__ is_fixed,
                                                  int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    const double INV_SQRT2 = 0.7071067811865475;
    int start = start_idx[blk];
    int k = v_dim[blk];

    if (cone_section_has_fixed(is_fixed, start, k + 2))
    {
        for (int slot = 0; slot < k + 2; ++slot)
        {
            int index = start + slot;
            double residual = objective_vector[index] - dual_product[index];
            dual_residual[index] = is_fixed[index] ? primal_solution[index] : primal_solution[index] - residual;
        }
        project_rotated_soc_section_serial(
            dual_residual, variable_rescaling, NULL, 0.0, warm_start + blk, start, k, is_fixed);
        for (int slot = 0; slot < k + 2; ++slot)
        {
            int index = start + slot;
            dual_residual[index] =
                is_fixed[index] ? 0.0 : (primal_solution[index] - dual_residual[index]) * variable_rescaling[index];
        }
        complementarity_residual[blk] = 0.0;
        return;
    }

    double r_s = objective_vector[start + k] - dual_product[start + k];
    double r_t = objective_vector[start + k + 1] - dual_product[start + k + 1];
    double r_w = (r_s - r_t) * INV_SQRT2;
    double r_z = (r_s + r_t) * INV_SQRT2;

    double d_s = variable_rescaling[start + k];
    double d_t = variable_rescaling[start + k + 1];
    double d_st = sqrt(d_s * d_t);

    bool diag_uniform = true;
    for (int m = 0; m < k && diag_uniform; ++m)
    {
        if (variable_rescaling[start + m] != d_st)
            diag_uniform = false;
    }

    if (diag_uniform)
    {
        double sumsq = r_w * r_w;
        for (int m = 0; m < k; ++m)
        {
            double v_m = objective_vector[start + m] - dual_product[start + m];
            sumsq += v_m * v_m;
        }
        double r_norm = sqrt(sumsq);

        double v_factor, p_s, p_t;
        if (r_norm <= r_z)
        {
            v_factor = 0.0;
            p_s = r_s;
            p_t = r_t;
        }
        else if (r_norm <= -r_z)
        {
            v_factor = 1.0;
            p_s = 0.0;
            p_t = 0.0;
        }
        else
        {
            double scale = (r_z + r_norm) / (2.0 * r_norm);
            v_factor = 1.0 - scale;
            double w_new = scale * r_w;
            double z_new = scale * r_norm;
            p_s = (z_new + w_new) * INV_SQRT2;
            p_t = (z_new - w_new) * INV_SQRT2;
        }
        for (int m = 0; m < k; ++m)
        {
            double v_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = v_m * v_factor * variable_rescaling[start + m];
        }
        dual_residual[start + k] = (r_s - p_s) * variable_rescaling[start + k];
        dual_residual[start + k + 1] = (r_t - p_t) * variable_rescaling[start + k + 1];
        return;
    }

    double r_inv_sq = r_w * r_w;
    double r_pos_sq = r_w * r_w;
    for (int m = 0; m < k; ++m)
    {
        double e_m = d_st / variable_rescaling[start + m];
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        r_inv_sq += (rc_m / e_m) * (rc_m / e_m);
        r_pos_sq += (rc_m * e_m) * (rc_m * e_m);
    }
    double r_inv = sqrt(r_inv_sq);
    double r_pos = sqrt(r_pos_sq);

    if (r_inv <= r_z)
    {
        for (int m = 0; m < k; ++m)
            dual_residual[start + m] = 0.0;
        dual_residual[start + k] = 0.0;
        dual_residual[start + k + 1] = 0.0;
        return;
    }
    if (r_pos <= -r_z)
    {
        for (int m = 0; m < k; ++m)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = rc_m * variable_rescaling[start + m];
        }
        dual_residual[start + k] = r_s * variable_rescaling[start + k];
        dual_residual[start + k + 1] = r_t * variable_rescaling[start + k + 1];
        return;
    }

    double lo, hi;
    bool z_pos = (r_z > 0.0);
    if (z_pos)
    {
        lo = 0.0;
        hi = 0.5 - 1e-14;
    }
    else
    {
        lo = 0.5 + 1e-14;
        hi = 1.0;
        for (int doubling = 0; doubling < 60; ++doubling)
        {
            double sum_hi = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double e_m = d_st / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double tt = rc_m * e_m / (e_m2 + 2.0 * hi);
                sum_hi += tt * tt;
            }
            double tw_hi = r_w / (1.0 + 2.0 * hi);
            sum_hi += tw_hi * tw_hi;
            double zt_hi = r_z / (1.0 - 2.0 * hi);
            double f_hi = sum_hi - zt_hi * zt_hi;
            if (f_hi > 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_lam = warm_start[blk];
    if (warm_lam > lo && warm_lam < hi)
    {
        double sum_w = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double e_m = d_st / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double tt = rc_m * e_m / (e_m2 + 2.0 * warm_lam);
            sum_w += tt * tt;
        }
        double tw = r_w / (1.0 + 2.0 * warm_lam);
        sum_w += tw * tw;
        double zt = r_z / (1.0 - 2.0 * warm_lam);
        double f = sum_w - zt * zt;
        if (fabs(f) < 1e-12)
        {
            double p_w_w = r_w / (1.0 + 2.0 * warm_lam);
            double p_z_w = r_z / (1.0 - 2.0 * warm_lam);
            double p_s_w = (p_z_w + p_w_w) * INV_SQRT2;
            double p_t_w = (p_z_w - p_w_w) * INV_SQRT2;
            for (int m = 0; m < k; ++m)
            {
                double e_m = d_st / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double p_m = rc_m * e_m2 / (e_m2 + 2.0 * warm_lam);
                dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
            }
            dual_residual[start + k] = (r_s - p_s_w) * variable_rescaling[start + k];
            dual_residual[start + k + 1] = (r_t - p_t_w) * variable_rescaling[start + k + 1];
            return;
        }
        if (z_pos)
        {
            if (f > 0.0)
                lo = warm_lam;
            else
                hi = warm_lam;
        }
        else
        {
            if (f > 0.0)
                hi = warm_lam;
            else
                lo = warm_lam;
        }
    }

    for (int it = 0; it < 60; ++it)
    {
        double lam = 0.5 * (lo + hi);
        double sum = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double e_m = d_st / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double tt = rc_m * e_m / (e_m2 + 2.0 * lam);
            sum += tt * tt;
        }
        double tw = r_w / (1.0 + 2.0 * lam);
        sum += tw * tw;
        double zt = r_z / (1.0 - 2.0 * lam);
        double f = sum - zt * zt;
        if (z_pos)
        {
            if (f > 0.0)
                lo = lam;
            else
                hi = lam;
        }
        else
        {
            if (f > 0.0)
                hi = lam;
            else
                lo = lam;
        }
        if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
            break;
    }
    double lam = 0.5 * (lo + hi);
    warm_start[blk] = lam;

    double p_w = r_w / (1.0 + 2.0 * lam);
    double p_z = r_z / (1.0 - 2.0 * lam);
    double p_s = (p_z + p_w) * INV_SQRT2;
    double p_t = (p_z - p_w) * INV_SQRT2;

    for (int m = 0; m < k; ++m)
    {
        double e_m = d_st / variable_rescaling[start + m];
        double e_m2 = e_m * e_m;
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        double p_m = rc_m * e_m2 / (e_m2 + 2.0 * lam);
        dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
    }
    dual_residual[start + k] = (r_s - p_s) * variable_rescaling[start + k];
    dual_residual[start + k + 1] = (r_t - p_t) * variable_rescaling[start + k + 1];
}

__global__ void project_rotated_soc_grid_reduce_kernel(double *__restrict__ primal_solution,
                                                       double *__restrict__ workspace,
                                                       const int *__restrict__ start_idx,
                                                       const int *__restrict__ v_dim,
                                                       int num_cones,
                                                       int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    int part = blockIdx.x - cone * blocks_per_cone;
    if (cone >= num_cones)
        return;

    int start = start_idx[cone];
    int k = v_dim[cone];
    double sum = 0.0;
    for (int m = part * blockDim.x + threadIdx.x; m < k; m += blocks_per_cone * blockDim.x)
    {
        double value = primal_solution[start + m];
        sum += value * value;
    }
    sum = large_cone_block_sum(sum);
    if (threadIdx.x == 0)
        atomicAdd(workspace + cone, sum);
}

__global__ void project_rotated_soc_grid_finalize_kernel(double *__restrict__ primal_solution,
                                                         double *__restrict__ workspace,
                                                         const int *__restrict__ start_idx,
                                                         const int *__restrict__ v_dim,
                                                         int num_cones)
{
    int cone = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone >= num_cones)
        return;

    const double INV_SQRT2 = 0.7071067811865475;
    int start = start_idx[cone];
    int k = v_dim[cone];
    double s = primal_solution[start + k];
    double t = primal_solution[start + k + 1];
    double w = (s - t) * INV_SQRT2;
    double z = (s + t) * INV_SQRT2;
    double radius = sqrt(fmax(0.0, workspace[cone] + w * w));

    if (radius <= z)
    {
        workspace[cone] = 1.0;
        return;
    }
    if (radius <= -z)
    {
        workspace[cone] = 0.0;
        primal_solution[start + k] = 0.0;
        primal_solution[start + k + 1] = 0.0;
        return;
    }

    double scale = (z + radius) / (2.0 * radius);
    double w_new = scale * w;
    double z_new = scale * radius;
    workspace[cone] = scale;
    primal_solution[start + k] = (z_new + w_new) * INV_SQRT2;
    primal_solution[start + k + 1] = (z_new - w_new) * INV_SQRT2;
}

__global__ void project_rotated_soc_grid_apply_kernel(double *__restrict__ primal_solution,
                                                      const double *__restrict__ workspace,
                                                      const int *__restrict__ start_idx,
                                                      const int *__restrict__ v_dim,
                                                      int num_cones,
                                                      int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    int part = blockIdx.x - cone * blocks_per_cone;
    if (cone >= num_cones)
        return;

    double scale = workspace[cone];
    if (scale == 1.0)
        return;

    int start = start_idx[cone];
    int k = v_dim[cone];
    for (int m = part * blockDim.x + threadIdx.x; m < k; m += blocks_per_cone * blockDim.x)
    {
        primal_solution[start + m] *= scale;
    }
}

__global__ void compute_cone_dual_residual_grid_reduce_kernel(const double *__restrict__ objective_vector,
                                                              const double *__restrict__ dual_product,
                                                              double *__restrict__ workspace,
                                                              const int *__restrict__ start_idx,
                                                              const int *__restrict__ v_dim,
                                                              int num_cones,
                                                              int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    int part = blockIdx.x - cone * blocks_per_cone;
    if (cone >= num_cones)
        return;

    int start = start_idx[cone];
    int k = v_dim[cone];
    double sum = 0.0;
    for (int m = part * blockDim.x + threadIdx.x; m < k; m += blocks_per_cone * blockDim.x)
    {
        double residual = objective_vector[start + m] - dual_product[start + m];
        sum += residual * residual;
    }
    sum = large_cone_block_sum(sum);
    if (threadIdx.x == 0)
        atomicAdd(workspace + cone, sum);
}

__global__ void compute_cone_dual_residual_grid_finalize_kernel(double *__restrict__ dual_residual,
                                                                const double *__restrict__ objective_vector,
                                                                const double *__restrict__ dual_product,
                                                                const double *__restrict__ variable_rescaling,
                                                                double *__restrict__ workspace,
                                                                const int *__restrict__ start_idx,
                                                                const int *__restrict__ v_dim,
                                                                int num_cones)
{
    int cone = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone >= num_cones)
        return;

    const double INV_SQRT2 = 0.7071067811865475;
    int start = start_idx[cone];
    int k = v_dim[cone];
    double r_s = objective_vector[start + k] - dual_product[start + k];
    double r_t = objective_vector[start + k + 1] - dual_product[start + k + 1];
    double r_w = (r_s - r_t) * INV_SQRT2;
    double r_z = (r_s + r_t) * INV_SQRT2;
    double norm = sqrt(fmax(0.0, workspace[cone] + r_w * r_w));
    double factor;
    double p_s;
    double p_t;

    if (norm <= r_z)
    {
        factor = 0.0;
        p_s = r_s;
        p_t = r_t;
    }
    else if (norm <= -r_z)
    {
        factor = 1.0;
        p_s = 0.0;
        p_t = 0.0;
    }
    else
    {
        double scale = (r_z + norm) / (2.0 * norm);
        double w_new = scale * r_w;
        double z_new = scale * norm;
        factor = 1.0 - scale;
        p_s = (z_new + w_new) * INV_SQRT2;
        p_t = (z_new - w_new) * INV_SQRT2;
    }

    workspace[cone] = factor;
    dual_residual[start + k] = (r_s - p_s) * variable_rescaling[start + k];
    dual_residual[start + k + 1] = (r_t - p_t) * variable_rescaling[start + k + 1];
}

__global__ void compute_cone_dual_residual_grid_apply_kernel(double *__restrict__ dual_residual,
                                                             const double *__restrict__ objective_vector,
                                                             const double *__restrict__ dual_product,
                                                             const double *__restrict__ variable_rescaling,
                                                             const double *__restrict__ workspace,
                                                             const int *__restrict__ start_idx,
                                                             const int *__restrict__ v_dim,
                                                             int num_cones,
                                                             int blocks_per_cone)
{
    int cone = blockIdx.x / blocks_per_cone;
    int part = blockIdx.x - cone * blocks_per_cone;
    if (cone >= num_cones)
        return;

    double factor = workspace[cone];
    int start = start_idx[cone];
    int k = v_dim[cone];
    for (int m = part * blockDim.x + threadIdx.x; m < k; m += blocks_per_cone * blockDim.x)
    {
        int idx = start + m;
        double residual = objective_vector[idx] - dual_product[idx];
        dual_residual[idx] = residual * factor * variable_rescaling[idx];
    }
}

__global__ void project_rotated_soc_warp_kernel(double *__restrict__ primal_solution,
                                                const double *__restrict__ variable_rescaling,
                                                double *__restrict__ warm_start,
                                                const int *__restrict__ start_idx,
                                                const int *__restrict__ v_dim,
                                                const char *__restrict__ is_fixed,
                                                int num_blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk = tid >> 5;
    int lane = tid & 31;
    if (blk >= num_blocks)
        return;

    const double INV_SQRT2 = 0.7071067811865475;
    const unsigned MASK = 0xffffffffu;

    int start = start_idx[blk];
    int k = v_dim[blk];

    int has_fixed = lane == 0 ? cone_section_has_fixed(is_fixed, start, k + 2) : 0;
    has_fixed = __shfl_sync(MASK, has_fixed, 0);
    if (has_fixed)
    {
        if (lane == 0)
            project_rotated_soc_section_serial(
                primal_solution, variable_rescaling, NULL, 0.0, warm_start + blk, start, k, is_fixed);
        return;
    }

    double s_val = primal_solution[start + k];
    double t_val = primal_solution[start + k + 1];

    double w = (s_val - t_val) * INV_SQRT2;
    double z = (s_val + t_val) * INV_SQRT2;

    double d_s = variable_rescaling[start + k];
    double d_t = variable_rescaling[start + k + 1];
    double d_st = sqrt(d_s * d_t);

    int my_diff = 0;
    for (int m = lane; m < k; m += 32)
    {
        if (variable_rescaling[start + m] != d_st)
            my_diff = 1;
    }
    for (int o = 16; o > 0; o >>= 1)
        my_diff |= __shfl_xor_sync(MASK, my_diff, o);

    if (my_diff == 0)
    {
        double my_sumsq = (lane == 0) ? w * w : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double v_m = primal_solution[start + m];
            my_sumsq += v_m * v_m;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sumsq += __shfl_xor_sync(MASK, my_sumsq, o);
        double r = sqrt(my_sumsq);
        if (r <= z)
            return;
        if (r <= -z)
        {
            for (int m = lane; m < k; m += 32)
                primal_solution[start + m] = 0.0;
            if (lane == 0)
            {
                primal_solution[start + k] = 0.0;
                primal_solution[start + k + 1] = 0.0;
            }
            return;
        }
        double scale = (z + r) / (2.0 * r);
        for (int m = lane; m < k; m += 32)
            primal_solution[start + m] *= scale;
        double w_new = scale * w;
        double z_new = scale * r;
        if (lane == 0)
        {
            primal_solution[start + k] = (z_new + w_new) * INV_SQRT2;
            primal_solution[start + k + 1] = (z_new - w_new) * INV_SQRT2;
        }
        return;
    }

    double my_inv = (lane == 0) ? w * w : 0.0;
    double my_pos = (lane == 0) ? w * w : 0.0;
    for (int m = lane; m < k; m += 32)
    {
        double dh = variable_rescaling[start + m] / d_st;
        double v_m = primal_solution[start + m];
        my_inv += (v_m / dh) * (v_m / dh);
        my_pos += (v_m * dh) * (v_m * dh);
    }
    for (int o = 16; o > 0; o >>= 1)
    {
        my_inv += __shfl_xor_sync(MASK, my_inv, o);
        my_pos += __shfl_xor_sync(MASK, my_pos, o);
    }
    double r_inv = sqrt(my_inv);
    if (r_inv <= z)
        return;
    double r_pos = sqrt(my_pos);
    if (r_pos <= -z)
    {
        for (int m = lane; m < k; m += 32)
            primal_solution[start + m] = 0.0;
        if (lane == 0)
        {
            primal_solution[start + k] = 0.0;
            primal_solution[start + k + 1] = 0.0;
        }
        return;
    }

    double lo, hi;
    bool z_pos = (z > 0.0);
    if (z_pos)
    {
        lo = 0.0;
        hi = 0.5 - 1e-14;
    }
    else
    {
        lo = 0.5 + 1e-14;
        hi = 1.0;
        for (int doubling = 0; doubling < 60; ++doubling)
        {
            double my_sum = (lane == 0) ? (w / (1.0 + 2.0 * hi)) * (w / (1.0 + 2.0 * hi)) : 0.0;
            for (int m = lane; m < k; m += 32)
            {
                double dh = variable_rescaling[start + m] / d_st;
                double dh2 = dh * dh;
                double tt = primal_solution[start + m] * dh / (dh2 + 2.0 * hi);
                my_sum += tt * tt;
            }
            for (int o = 16; o > 0; o >>= 1)
                my_sum += __shfl_xor_sync(MASK, my_sum, o);
            double zt_hi = z / (1.0 - 2.0 * hi);
            double f_hi = my_sum - zt_hi * zt_hi;
            if (f_hi > 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_lam = warm_start[blk];
    if (warm_lam > lo && warm_lam < hi)
    {
        double my_sum = (lane == 0) ? (w / (1.0 + 2.0 * warm_lam)) * (w / (1.0 + 2.0 * warm_lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double dh = variable_rescaling[start + m] / d_st;
            double dh2 = dh * dh;
            double tt = primal_solution[start + m] * dh / (dh2 + 2.0 * warm_lam);
            my_sum += tt * tt;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(MASK, my_sum, o);
        double zt = z / (1.0 - 2.0 * warm_lam);
        double f = my_sum - zt * zt;
        if (fabs(f) < 1e-12)
        {
            double w_new = w / (1.0 + 2.0 * warm_lam);
            double z_new = z / (1.0 - 2.0 * warm_lam);
            for (int m = lane; m < k; m += 32)
            {
                double dh = variable_rescaling[start + m] / d_st;
                double dh2 = dh * dh;
                primal_solution[start + m] = primal_solution[start + m] * dh2 / (dh2 + 2.0 * warm_lam);
            }
            if (lane == 0)
            {
                primal_solution[start + k] = (z_new + w_new) * INV_SQRT2;
                primal_solution[start + k + 1] = (z_new - w_new) * INV_SQRT2;
            }
            return;
        }
        if (z_pos)
        {
            if (f > 0.0)
                lo = warm_lam;
            else
                hi = warm_lam;
        }
        else
        {
            if (f > 0.0)
                hi = warm_lam;
            else
                lo = warm_lam;
        }
    }

    for (int it = 0; it < 60; ++it)
    {
        double lam = 0.5 * (lo + hi);
        double my_sum = (lane == 0) ? (w / (1.0 + 2.0 * lam)) * (w / (1.0 + 2.0 * lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double dh = variable_rescaling[start + m] / d_st;
            double dh2 = dh * dh;
            double tt = primal_solution[start + m] * dh / (dh2 + 2.0 * lam);
            my_sum += tt * tt;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(MASK, my_sum, o);
        double zt = z / (1.0 - 2.0 * lam);
        double f = my_sum - zt * zt;
        if (z_pos)
        {
            if (f > 0.0)
                lo = lam;
            else
                hi = lam;
        }
        else
        {
            if (f > 0.0)
                hi = lam;
            else
                lo = lam;
        }
        if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
            break;
    }
    double lam = 0.5 * (lo + hi);
    if (lane == 0)
        warm_start[blk] = lam;

    double w_new = w / (1.0 + 2.0 * lam);
    double z_new = z / (1.0 - 2.0 * lam);
    for (int m = lane; m < k; m += 32)
    {
        double dh = variable_rescaling[start + m] / d_st;
        double dh2 = dh * dh;
        primal_solution[start + m] = primal_solution[start + m] * dh2 / (dh2 + 2.0 * lam);
    }
    if (lane == 0)
    {
        primal_solution[start + k] = (z_new + w_new) * INV_SQRT2;
        primal_solution[start + k + 1] = (z_new - w_new) * INV_SQRT2;
    }
}

__global__ void compute_cone_dual_residual_warp_kernel(double *__restrict__ dual_residual,
                                                       double *__restrict__ complementarity_residual,
                                                       const double *__restrict__ objective_vector,
                                                       const double *__restrict__ dual_product,
                                                       const double *__restrict__ variable_rescaling,
                                                       const double *__restrict__ primal_solution,
                                                       double *__restrict__ warm_start,
                                                       const int *__restrict__ start_idx,
                                                       const int *__restrict__ v_dim,
                                                       const char *__restrict__ is_fixed,
                                                       int num_blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk = tid >> 5;
    int lane = tid & 31;
    if (blk >= num_blocks)
        return;

    const double INV_SQRT2 = 0.7071067811865475;
    const unsigned MASK = 0xffffffffu;

    int start = start_idx[blk];
    int k = v_dim[blk];

    int has_fixed = lane == 0 ? cone_section_has_fixed(is_fixed, start, k + 2) : 0;
    has_fixed = __shfl_sync(MASK, has_fixed, 0);
    if (has_fixed)
    {
        if (lane == 0)
        {
            for (int slot = 0; slot < k + 2; ++slot)
            {
                int index = start + slot;
                double residual = objective_vector[index] - dual_product[index];
                dual_residual[index] = is_fixed[index] ? primal_solution[index] : primal_solution[index] - residual;
            }
            project_rotated_soc_section_serial(
                dual_residual, variable_rescaling, NULL, 0.0, warm_start + blk, start, k, is_fixed);
            for (int slot = 0; slot < k + 2; ++slot)
            {
                int index = start + slot;
                dual_residual[index] =
                    is_fixed[index] ? 0.0 : (primal_solution[index] - dual_residual[index]) * variable_rescaling[index];
            }
            complementarity_residual[blk] = 0.0;
        }
        return;
    }

    double r_s = objective_vector[start + k] - dual_product[start + k];
    double r_t = objective_vector[start + k + 1] - dual_product[start + k + 1];
    double r_w = (r_s - r_t) * INV_SQRT2;
    double r_z = (r_s + r_t) * INV_SQRT2;

    double d_s = variable_rescaling[start + k];
    double d_t = variable_rescaling[start + k + 1];
    double d_st = sqrt(d_s * d_t);

    int my_diff = 0;
    for (int m = lane; m < k; m += 32)
    {
        if (variable_rescaling[start + m] != d_st)
            my_diff = 1;
    }
    for (int o = 16; o > 0; o >>= 1)
        my_diff |= __shfl_xor_sync(MASK, my_diff, o);

    if (my_diff == 0)
    {
        double my_sumsq = (lane == 0) ? r_w * r_w : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            my_sumsq += rc_m * rc_m;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sumsq += __shfl_xor_sync(MASK, my_sumsq, o);
        double r_norm = sqrt(my_sumsq);

        double v_factor, p_s, p_t;
        if (r_norm <= r_z)
        {
            v_factor = 0.0;
            p_s = r_s;
            p_t = r_t;
        }
        else if (r_norm <= -r_z)
        {
            v_factor = 1.0;
            p_s = 0.0;
            p_t = 0.0;
        }
        else
        {
            double scale = (r_z + r_norm) / (2.0 * r_norm);
            v_factor = 1.0 - scale;
            double w_new = scale * r_w;
            double z_new = scale * r_norm;
            p_s = (z_new + w_new) * INV_SQRT2;
            p_t = (z_new - w_new) * INV_SQRT2;
        }

        for (int m = lane; m < k; m += 32)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = rc_m * v_factor * variable_rescaling[start + m];
        }
        if (lane == 0)
        {
            dual_residual[start + k] = (r_s - p_s) * variable_rescaling[start + k];
            dual_residual[start + k + 1] = (r_t - p_t) * variable_rescaling[start + k + 1];
        }
        return;
    }

    double my_inv = (lane == 0) ? r_w * r_w : 0.0;
    double my_pos = (lane == 0) ? r_w * r_w : 0.0;
    for (int m = lane; m < k; m += 32)
    {
        double e_m = d_st / variable_rescaling[start + m];
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        my_inv += (rc_m / e_m) * (rc_m / e_m);
        my_pos += (rc_m * e_m) * (rc_m * e_m);
    }
    for (int o = 16; o > 0; o >>= 1)
    {
        my_inv += __shfl_xor_sync(MASK, my_inv, o);
        my_pos += __shfl_xor_sync(MASK, my_pos, o);
    }
    double r_inv = sqrt(my_inv);
    double r_pos = sqrt(my_pos);

    if (r_inv <= r_z)
    {
        for (int m = lane; m < k; m += 32)
            dual_residual[start + m] = 0.0;
        if (lane == 0)
        {
            dual_residual[start + k] = 0.0;
            dual_residual[start + k + 1] = 0.0;
        }
        return;
    }
    if (r_pos <= -r_z)
    {
        for (int m = lane; m < k; m += 32)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = rc_m * variable_rescaling[start + m];
        }
        if (lane == 0)
        {
            dual_residual[start + k] = r_s * variable_rescaling[start + k];
            dual_residual[start + k + 1] = r_t * variable_rescaling[start + k + 1];
        }
        return;
    }

    double lo, hi;
    bool z_pos = (r_z > 0.0);
    if (z_pos)
    {
        lo = 0.0;
        hi = 0.5 - 1e-14;
    }
    else
    {
        lo = 0.5 + 1e-14;
        hi = 1.0;
        for (int doubling = 0; doubling < 60; ++doubling)
        {
            double my_sum = (lane == 0) ? (r_w / (1.0 + 2.0 * hi)) * (r_w / (1.0 + 2.0 * hi)) : 0.0;
            for (int m = lane; m < k; m += 32)
            {
                double e_m = d_st / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double tt = rc_m * e_m / (e_m2 + 2.0 * hi);
                my_sum += tt * tt;
            }
            for (int o = 16; o > 0; o >>= 1)
                my_sum += __shfl_xor_sync(MASK, my_sum, o);
            double zt_hi = r_z / (1.0 - 2.0 * hi);
            double f_hi = my_sum - zt_hi * zt_hi;
            if (f_hi > 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_lam = warm_start[blk];
    if (warm_lam > lo && warm_lam < hi)
    {
        double my_sum = (lane == 0) ? (r_w / (1.0 + 2.0 * warm_lam)) * (r_w / (1.0 + 2.0 * warm_lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double e_m = d_st / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double tt = rc_m * e_m / (e_m2 + 2.0 * warm_lam);
            my_sum += tt * tt;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(MASK, my_sum, o);
        double zt = r_z / (1.0 - 2.0 * warm_lam);
        double f = my_sum - zt * zt;
        if (fabs(f) < 1e-12)
        {
            double p_w_w = r_w / (1.0 + 2.0 * warm_lam);
            double p_z_w = r_z / (1.0 - 2.0 * warm_lam);
            double p_s_w = (p_z_w + p_w_w) * INV_SQRT2;
            double p_t_w = (p_z_w - p_w_w) * INV_SQRT2;
            for (int m = lane; m < k; m += 32)
            {
                double e_m = d_st / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double p_m = rc_m * e_m2 / (e_m2 + 2.0 * warm_lam);
                dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
            }
            if (lane == 0)
            {
                dual_residual[start + k] = (r_s - p_s_w) * variable_rescaling[start + k];
                dual_residual[start + k + 1] = (r_t - p_t_w) * variable_rescaling[start + k + 1];
            }
            return;
        }
        if (z_pos)
        {
            if (f > 0.0)
                lo = warm_lam;
            else
                hi = warm_lam;
        }
        else
        {
            if (f > 0.0)
                hi = warm_lam;
            else
                lo = warm_lam;
        }
    }

    for (int it = 0; it < 60; ++it)
    {
        double lam = 0.5 * (lo + hi);
        double my_sum = (lane == 0) ? (r_w / (1.0 + 2.0 * lam)) * (r_w / (1.0 + 2.0 * lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double e_m = d_st / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double tt = rc_m * e_m / (e_m2 + 2.0 * lam);
            my_sum += tt * tt;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(MASK, my_sum, o);
        double zt = r_z / (1.0 - 2.0 * lam);
        double f = my_sum - zt * zt;
        if (z_pos)
        {
            if (f > 0.0)
                lo = lam;
            else
                hi = lam;
        }
        else
        {
            if (f > 0.0)
                hi = lam;
            else
                lo = lam;
        }
        if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
            break;
    }
    double lam = 0.5 * (lo + hi);
    if (lane == 0)
        warm_start[blk] = lam;

    double p_w = r_w / (1.0 + 2.0 * lam);
    double p_z = r_z / (1.0 - 2.0 * lam);
    double p_s = (p_z + p_w) * INV_SQRT2;
    double p_t = (p_z - p_w) * INV_SQRT2;

    for (int m = lane; m < k; m += 32)
    {
        double e_m = d_st / variable_rescaling[start + m];
        double e_m2 = e_m * e_m;
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        double p_m = rc_m * e_m2 / (e_m2 + 2.0 * lam);
        dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
    }
    if (lane == 0)
    {
        dual_residual[start + k] = (r_s - p_s) * variable_rescaling[start + k];
        dual_residual[start + k + 1] = (r_t - p_t) * variable_rescaling[start + k + 1];
    }
}

__global__ void project_rotated_soc_diag_q_kernel(double *__restrict__ pdhg_primal,
                                                  double *__restrict__ reflected_primal,
                                                  const double *__restrict__ current_primal,
                                                  const double *__restrict__ variable_rescaling,
                                                  const double *__restrict__ Q_diag,
                                                  double tau,
                                                  double *__restrict__ warm_start,
                                                  const int *__restrict__ start_idx,
                                                  const int *__restrict__ v_dim,
                                                  const char *__restrict__ is_fixed,
                                                  int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    const double W_FLOOR = 1e-300;

    int start = start_idx[blk];
    int k = v_dim[blk];
    int len = k + 2;

    if (cone_section_has_fixed(is_fixed, start, len))
    {
        project_rotated_soc_section_serial(
            pdhg_primal, variable_rescaling, Q_diag, tau, warm_start + blk, start, k, is_fixed);
        for (int slot = 0; slot < len; ++slot)
        {
            int index = start + slot;
            reflected_primal[index] = 2.0 * pdhg_primal[index] - current_primal[index];
        }
        return;
    }

    double r_s = pdhg_primal[start + k];
    double r_t = pdhg_primal[start + k + 1];

    double q_s = Q_diag[start + k];
    double q_t = Q_diag[start + k + 1];
    double w_s = 1.0 + tau * q_s;
    double w_t = 1.0 + tau * q_t;
    if (!(w_s > W_FLOOR))
        w_s = W_FLOOR;
    if (!(w_t > W_FLOOR))
        w_t = W_FLOOR;
    double sigma = sqrt(w_s * w_t);
    double alpha = sqrt(w_t / w_s);
    double inv_alpha = 1.0 / alpha;

    double d_s = variable_rescaling[start + k];
    double d_t = variable_rescaling[start + k + 1];
    double d_st = sqrt(d_s * d_t);

    const double INV_SQRT2 = 0.7071067811865475;
    /* Fast path: no Q on cone slots (w_s = w_t = 1 and all w_v_i = 1) and uniform d_v = d_st.
       This is the COMMON case for QCQP transform aux vars. Reduces to LP-style RSOC closed form. */
    if (q_s == 0.0 && q_t == 0.0)
    {
        bool no_cone_Q = true;
        bool d_uniform = true;
        for (int m = 0; m < k; ++m)
        {
            if (Q_diag[start + m] != 0.0)
            {
                no_cone_Q = false;
                break;
            }
            if (variable_rescaling[start + m] != d_st)
            {
                d_uniform = false;
                break;
            }
        }
        if (no_cone_Q && d_uniform)
        {
            double w_val = (r_s - r_t) * INV_SQRT2;
            double z_val = (r_s + r_t) * INV_SQRT2;
            double sumsq = w_val * w_val;
            for (int m = 0; m < k; ++m)
            {
                double vm = pdhg_primal[start + m];
                sumsq += vm * vm;
            }
            double rnorm = sqrt(sumsq);
            if (rnorm <= z_val)
            {
                for (int m = 0; m < len; ++m)
                {
                    int idx = start + m;
                    reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
                }
                return;
            }
            if (rnorm <= -z_val)
            {
                for (int m = 0; m < k; ++m)
                {
                    pdhg_primal[start + m] = 0.0;
                    int idx = start + m;
                    reflected_primal[idx] = -current_primal[idx];
                }
                pdhg_primal[start + k] = 0.0;
                pdhg_primal[start + k + 1] = 0.0;
                reflected_primal[start + k] = -current_primal[start + k];
                reflected_primal[start + k + 1] = -current_primal[start + k + 1];
                return;
            }
            double scale = (z_val + rnorm) / (2.0 * rnorm);
            double w_new = scale * w_val;
            double z_new = scale * rnorm;
            for (int m = 0; m < k; ++m)
            {
                double v_new = scale * pdhg_primal[start + m];
                pdhg_primal[start + m] = v_new;
                int idx = start + m;
                reflected_primal[idx] = 2.0 * v_new - current_primal[idx];
            }
            double s_new = (z_new + w_new) * INV_SQRT2;
            double t_new = (z_new - w_new) * INV_SQRT2;
            pdhg_primal[start + k] = s_new;
            pdhg_primal[start + k + 1] = t_new;
            reflected_primal[start + k] = 2.0 * s_new - current_primal[start + k];
            reflected_primal[start + k + 1] = 2.0 * t_new - current_primal[start + k + 1];
            return;
        }
    }

    {
        double lhs = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double d_m = variable_rescaling[start + m];
            double Ds = d_st / d_m;
            double rv = pdhg_primal[start + m];
            double term = Ds * rv;
            lhs += term * term;
        }
        if (r_s >= 0.0 && r_t >= 0.0 && lhs <= 2.0 * r_s * r_t)
        {
            for (int m = 0; m < len; ++m)
            {
                int idx = start + m;
                double pv = pdhg_primal[idx];
                reflected_primal[idx] = 2.0 * pv - current_primal[idx];
            }
            return;
        }
    }

    if (r_s <= 0.0 && r_t <= 0.0)
    {
        double rhs = 2.0 * sigma * sigma * r_s * r_t;
        double lhs = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double d_m = variable_rescaling[start + m];
            double q_m = Q_diag[start + m];
            double w_m = 1.0 + tau * q_m;
            if (!(w_m > W_FLOOR))
                w_m = W_FLOOR;
            double rv = pdhg_primal[start + m];
            double term = d_m * w_m * rv / d_st;
            lhs += term * term;
        }
        if (lhs <= rhs)
        {
            for (int m = 0; m < k; ++m)
                pdhg_primal[start + m] = 0.0;
            pdhg_primal[start + k] = 0.0;
            pdhg_primal[start + k + 1] = 0.0;
            for (int m = 0; m < len; ++m)
            {
                int idx = start + m;
                reflected_primal[idx] = -current_primal[idx];
            }
            return;
        }
    }

    double lo, hi;
    int bracket_kind; /* 0: f increasing on bracket; 1: f decreasing. */
    bool need_doubling = false;
    double sum_alpha = r_s + alpha * r_t;

    if (r_s > 0.0 && r_t > 0.0)
    {
        lo = 0.0;
        hi = 1.0 - 1e-14;
        bracket_kind = 1;
    }
    else if (r_s < 0.0 && r_t < 0.0)
    {
        lo = 1.0 + 1e-14;
        hi = 2.0;
        bracket_kind = 0;
        need_doubling = true;
    }
    else if (r_s <= 0.0 && r_t >= 0.0)
    {
        if (sum_alpha <= 0.0)
        {
            lo = 1.0 + 1e-14;
            if (r_t == 0.0)
            {
                hi = 2.0;
                need_doubling = true;
            }
            else
            {
                hi = -r_s / (alpha * r_t);
                if (!(hi > lo))
                    hi = lo + 1.0;
            }
            bracket_kind = 0;
        }
        else
        {
            lo = (r_t > 0.0) ? (-r_s / (alpha * r_t)) : 0.0;
            if (!(lo >= 0.0))
                lo = 0.0;
            hi = 1.0 - 1e-14;
            if (!(lo < hi))
                lo = hi - 1e-7;
            bracket_kind = 1;
        }
    }
    else
    {
        if (sum_alpha <= 0.0)
        {
            lo = 1.0 + 1e-14;
            if (r_s == 0.0)
            {
                hi = 2.0;
                need_doubling = true;
            }
            else
            {
                hi = -alpha * r_t / r_s;
                if (!(hi > lo))
                    hi = lo + 1.0;
            }
            bracket_kind = 0;
        }
        else
        {
            lo = (r_s > 0.0) ? (-alpha * r_t / r_s) : 0.0;
            if (!(lo >= 0.0))
                lo = 0.0;
            hi = 1.0 - 1e-14;
            if (!(lo < hi))
                lo = hi - 1e-7;
            bracket_kind = 1;
        }
    }

#define ORACLE_EVAL(ZETA, F_OUT)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        double _zeta = (ZETA);                                                                                         \
        double _denom = 1.0 - _zeta * _zeta;                                                                           \
        double _s = (r_s + _zeta * alpha * r_t) / _denom;                                                              \
        double _t = (r_t + _zeta * inv_alpha * r_s) / _denom;                                                          \
        double _sv = 0.0;                                                                                              \
        for (int _m = 0; _m < k; ++_m)                                                                                 \
        {                                                                                                              \
            double _dm = variable_rescaling[start + _m];                                                               \
            double _Ds = d_st / _dm;                                                                                   \
            double _qm = Q_diag[start + _m];                                                                           \
            double _wm = 1.0 + tau * _qm;                                                                              \
            if (!(_wm > W_FLOOR))                                                                                      \
                _wm = W_FLOOR;                                                                                         \
            double _Dh2 = _Ds * _Ds * sigma / _wm;                                                                     \
            double _rv = pdhg_primal[start + _m];                                                                      \
            double _vz = _rv / (1.0 + _zeta * _Dh2);                                                                   \
            double _tm = _Ds * _vz;                                                                                    \
            _sv += _tm * _tm;                                                                                          \
        }                                                                                                              \
        (F_OUT) = _sv - 2.0 * _s * _t;                                                                                 \
    } while (0)

    if (need_doubling)
    {
        double f_hi;
        for (int dbl = 0; dbl < 60; ++dbl)
        {
            ORACLE_EVAL(hi, f_hi);
            if (f_hi >= 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_zeta = warm_start[blk];
    if (warm_zeta > lo && warm_zeta < hi)
    {
        double f_w;
        ORACLE_EVAL(warm_zeta, f_w);
        if (fabs(f_w) < 1e-12)
        {
            double zeta = warm_zeta;
            double denom = 1.0 - zeta * zeta;
            double s_new = (r_s + zeta * alpha * r_t) / denom;
            double t_new = (r_t + zeta * inv_alpha * r_s) / denom;
            for (int m = 0; m < k; ++m)
            {
                double d_m = variable_rescaling[start + m];
                double Ds = d_st / d_m;
                double q_m = Q_diag[start + m];
                double w_m = 1.0 + tau * q_m;
                if (!(w_m > W_FLOOR))
                    w_m = W_FLOOR;
                double Dh2 = Ds * Ds * sigma / w_m;
                double rv = pdhg_primal[start + m];
                pdhg_primal[start + m] = rv / (1.0 + zeta * Dh2);
            }
            pdhg_primal[start + k] = s_new;
            pdhg_primal[start + k + 1] = t_new;
            for (int m = 0; m < len; ++m)
            {
                int idx = start + m;
                double pv = pdhg_primal[idx];
                reflected_primal[idx] = 2.0 * pv - current_primal[idx];
            }
            return;
        }
        if (bracket_kind == 0)
        {
            if (f_w < 0.0)
                lo = warm_zeta;
            else
                hi = warm_zeta;
        }
        else
        {
            if (f_w > 0.0)
                lo = warm_zeta;
            else
                hi = warm_zeta;
        }
    }

    for (int it = 0; it < 80; ++it)
    {
        double mid = 0.5 * (lo + hi);
        double f_m;
        ORACLE_EVAL(mid, f_m);
        if (bracket_kind == 0)
        {
            if (f_m < 0.0)
                lo = mid;
            else
                hi = mid;
        }
        else
        {
            if (f_m > 0.0)
                lo = mid;
            else
                hi = mid;
        }
        if ((hi - lo) / (1.0 + fabs(hi) + fabs(lo)) < 1e-13)
            break;
    }
    double zeta = 0.5 * (lo + hi);
    warm_start[blk] = zeta;

    double denom = 1.0 - zeta * zeta;
    double s_new = (r_s + zeta * alpha * r_t) / denom;
    double t_new = (r_t + zeta * inv_alpha * r_s) / denom;
    for (int m = 0; m < k; ++m)
    {
        double d_m = variable_rescaling[start + m];
        double Ds = d_st / d_m;
        double q_m = Q_diag[start + m];
        double w_m = 1.0 + tau * q_m;
        if (!(w_m > W_FLOOR))
            w_m = W_FLOOR;
        double Dh2 = Ds * Ds * sigma / w_m;
        double rv = pdhg_primal[start + m];
        pdhg_primal[start + m] = rv / (1.0 + zeta * Dh2);
    }
    pdhg_primal[start + k] = s_new;
    pdhg_primal[start + k + 1] = t_new;

    for (int m = 0; m < len; ++m)
    {
        int idx = start + m;
        double pv = pdhg_primal[idx];
        reflected_primal[idx] = 2.0 * pv - current_primal[idx];
    }
#undef ORACLE_EVAL
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

static void launch_rotated_thread_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_rotated_soc_kernel<<<b, t>>>(p, vr, ws, si, vd, isf, n);
}
static void launch_rotated_warp_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n * 32 + t - 1) / t;
    project_rotated_soc_warp_kernel<<<b, t>>>(p, vr, ws, si, vd, isf, n);
}
static void launch_rotated_block_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    project_rotated_soc_block_kernel<<<n, THREADS_PER_BLOCK>>>(p, vr, NULL, 0.0, ws, si, vd, isf, n);
}
static void launch_rotated_grid_weighted_impl(double *p,
                                              const double *vr,
                                              const double *qd,
                                              double tau,
                                              double *ws,
                                              const int *si,
                                              const int *vd,
                                              const char *isf,
                                              int n)
{
    int threads = THREADS_PER_BLOCK;
    int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    int blocks = n * blocks_per_cone;
    CUDA_CHECK(cudaMemsetAsync(ws + n, 0, (size_t)5 * n * sizeof(double)));
    initialize_rotated_soc_grid_weighted_kernel<<<blocks, threads>>>(
        p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
    finalize_rotated_soc_grid_weighted_initialization_kernel<<<(n + threads - 1) / threads, threads>>>(
        p, vr, qd, tau, ws, si, vd, isf, n);
    for (int iteration = 0; iteration < PDHCG_CONE_GRID_ROOT_ITERATIONS; ++iteration)
    {
        CUDA_CHECK(cudaMemsetAsync(ws + n, 0, (size_t)2 * n * sizeof(double)));
        reduce_rotated_soc_grid_weighted_root_kernel<<<blocks, threads>>>(
            p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
        finalize_rotated_soc_grid_weighted_root_kernel<<<(n + threads - 1) / threads, threads>>>(
            p, vr, qd, tau, ws, si, vd, isf, n);
    }
    CUDA_CHECK(cudaMemsetAsync(ws + n, 0, (size_t)2 * n * sizeof(double)));
    reduce_rotated_soc_grid_axis_objective_kernel<<<blocks, threads>>>(
        p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
    finalize_rotated_soc_grid_axis_objective_kernel<<<(n + threads - 1) / threads, threads>>>(
        p, vr, qd, tau, ws, si, vd, n);
    apply_rotated_soc_grid_weighted_kernel<<<blocks, threads>>>(p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
}
static void launch_rotated_grid_weighted_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    launch_rotated_grid_weighted_impl(p, vr, NULL, 0.0, ws, si, vd, isf, n);
}
static void launch_rotated_grid_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)vr;
    (void)pa;
    (void)isf;
    int t = THREADS_PER_BLOCK;
    int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    int b = n * blocks_per_cone;
    CUDA_CHECK(cudaMemsetAsync(ws, 0, (size_t)n * sizeof(double)));
    project_rotated_soc_grid_reduce_kernel<<<b, t>>>(p, ws, si, vd, n, blocks_per_cone);
    project_rotated_soc_grid_finalize_kernel<<<(n + t - 1) / t, t>>>(p, ws, si, vd, n);
    project_rotated_soc_grid_apply_kernel<<<b, t>>>(p, ws, si, vd, n, blocks_per_cone);
}
static void launch_rotated_thread_dual(double *dr,
                                       double *cr,
                                       const double *obj,
                                       const double *dp,
                                       const double *vr,
                                       const double *ps,
                                       double *ws,
                                       const int *si,
                                       const int *vd,
                                       const double *pa,
                                       const char *isf,
                                       int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    compute_cone_dual_residual_kernel<<<b, t>>>(dr, cr, obj, dp, vr, ps, ws, si, vd, isf, n);
}
static void launch_rotated_warp_dual(double *dr,
                                     double *cr,
                                     const double *obj,
                                     const double *dp,
                                     const double *vr,
                                     const double *ps,
                                     double *ws,
                                     const int *si,
                                     const int *vd,
                                     const double *pa,
                                     const char *isf,
                                     int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n * 32 + t - 1) / t;
    compute_cone_dual_residual_warp_kernel<<<b, t>>>(dr, cr, obj, dp, vr, ps, ws, si, vd, isf, n);
}
static void launch_rotated_grid_dual(double *dr,
                                     double *cr,
                                     const double *obj,
                                     const double *dp,
                                     const double *vr,
                                     const double *ps,
                                     double *ws,
                                     const int *si,
                                     const int *vd,
                                     const double *pa,
                                     const char *isf,
                                     int n)
{
    (void)cr;
    (void)ps;
    (void)pa;
    (void)isf;
    int t = THREADS_PER_BLOCK;
    int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    int b = n * blocks_per_cone;
    CUDA_CHECK(cudaMemsetAsync(ws, 0, (size_t)n * sizeof(double)));
    compute_cone_dual_residual_grid_reduce_kernel<<<b, t>>>(obj, dp, ws, si, vd, n, blocks_per_cone);
    compute_cone_dual_residual_grid_finalize_kernel<<<(n + t - 1) / t, t>>>(dr, obj, dp, vr, ws, si, vd, n);
    compute_cone_dual_residual_grid_apply_kernel<<<b, t>>>(dr, obj, dp, vr, ws, si, vd, n, blocks_per_cone);
}
static void launch_rotated_thread_proj_diag_q(double *pp,
                                              double *rp,
                                              const double *cp,
                                              const double *vr,
                                              const double *qd,
                                              double tau,
                                              double *ws,
                                              const int *si,
                                              const int *vd,
                                              const double *pa,
                                              const char *isf,
                                              int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_rotated_soc_diag_q_kernel<<<b, t>>>(pp, rp, cp, vr, qd, tau, ws, si, vd, isf, n);
}
static void launch_rotated_block_proj_diag_q(double *pp,
                                             double *rp,
                                             const double *cp,
                                             const double *vr,
                                             const double *qd,
                                             double tau,
                                             double *ws,
                                             const int *si,
                                             const int *vd,
                                             const double *pa,
                                             const char *isf,
                                             int n)
{
    (void)pa;
    project_rotated_soc_block_kernel<<<n, THREADS_PER_BLOCK>>>(pp, vr, qd, tau, ws, si, vd, isf, n);
    launch_cone_reflection(PROJ_METHOD_BLOCK, rp, pp, cp, si, vd, n);
}
static void launch_rotated_grid_weighted_proj_diag_q(double *pp,
                                                     double *rp,
                                                     const double *cp,
                                                     const double *vr,
                                                     const double *qd,
                                                     double tau,
                                                     double *ws,
                                                     const int *si,
                                                     const int *vd,
                                                     const double *pa,
                                                     const char *isf,
                                                     int n)
{
    (void)pa;
    launch_rotated_grid_weighted_impl(pp, vr, qd, tau, ws, si, vd, isf, n);
    launch_cone_reflection(PROJ_METHOD_GRID, rp, pp, cp, si, vd, n);
}

extern const cone_kernel_ops_t pdhcg_rsoc_cone_kernel_ops = {
    {
        launch_rotated_thread_proj,
        launch_rotated_warp_proj,
        launch_rotated_block_proj,
        launch_rotated_grid_proj,
        launch_rotated_grid_weighted_proj,
    },
    {
        launch_rotated_thread_proj_diag_q,
        launch_rotated_block_proj_diag_q,
        launch_rotated_block_proj_diag_q,
        launch_rotated_grid_weighted_proj_diag_q,
        launch_rotated_grid_weighted_proj_diag_q,
    },
    {
        launch_rotated_thread_dual,
        launch_rotated_warp_dual,
        launch_block_projected_mapping_only_dual,
        launch_rotated_grid_dual,
        launch_grid_projected_mapping_only_dual,
    },
};
