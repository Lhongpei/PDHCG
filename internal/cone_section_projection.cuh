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

#include <float.h>
#include <math.h>

__device__ static inline double
cone_section_weight(const double *rescaling, const double *q_diag, double tau, int index)
{
    double metric = q_diag ? 1.0 + tau * q_diag[index] : 1.0;
    double d = rescaling[index];
    return fmax(metric * d * d, DBL_MIN);
}

__device__ static inline bool cone_section_has_fixed(const char *is_fixed, int start, int length)
{
    if (!is_fixed)
        return false;
    for (int slot = 0; slot < length; ++slot)
        if (is_fixed[start + slot])
            return true;
    return false;
}

__device__ static inline double cone_section_actual(const double *point, const double *rescaling, int index)
{
    return point[index] / rescaling[index];
}

/* For the negative scalar branch of a weighted SOC projection, return a
   multiplier at which the root residual is nonnegative. */
__device__ static inline double cone_section_negative_soc_upper(
    double singular_metric, double endpoint_polar, double fixed_norm2, double polar_norm2, double max_vector_metric)
{
    double upper;
    if (fixed_norm2 > 0.0)
    {
        upper = singular_metric + endpoint_polar / sqrt(fixed_norm2);
    }
    else
    {
        double polar_norm = sqrt(polar_norm2);
        double gap = polar_norm - endpoint_polar;
        if (!(gap > 0.0))
            return NAN;
        upper = (polar_norm / gap) * singular_metric + (endpoint_polar / gap) * max_vector_metric;
    }
    return upper * (1.0 + 64.0 * DBL_EPSILON);
}

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

/* Weighted projection onto an arbitrary nonempty fixed section of
   { (u,z) : ||u||_2 <= z }. The first k+1 coordinates form u. */
__device__ static inline void project_standard_soc_section_serial(double *point,
                                                                  const double *rescaling,
                                                                  const double *q_diag,
                                                                  double tau,
                                                                  double *warm_start,
                                                                  int start,
                                                                  int k,
                                                                  const char *is_fixed)
{
    int u_length = k + 1;
    int z_index = start + u_length;
    bool fixed_z = is_fixed[z_index] != 0;
    double fixed_norm2 = 0.0;
    double free_norm2 = 0.0;
    double polar_norm2 = 0.0;
    double max_omega = 0.0;
    int free_count = 0;
    for (int slot = 0; slot < u_length; ++slot)
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

    double z_input = cone_section_actual(point, rescaling, z_index);
    if (fixed_z)
    {
        double radius2 = fmax(0.0, z_input * z_input - fixed_norm2);
        if (free_count == 0 || free_norm2 <= radius2)
            return;
        if (!(radius2 > 0.0))
        {
            for (int slot = 0; slot < u_length; ++slot)
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
                for (int slot = 0; slot < u_length; ++slot)
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
            for (int slot = 0; slot < u_length; ++slot)
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
        for (int slot = 0; slot < u_length; ++slot)
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

    double total_norm2 = fixed_norm2 + free_norm2;
    if (z_input >= 0.0 && total_norm2 <= z_input * z_input)
        return;
    if (free_count == 0)
    {
        double projected_z = fmax(z_input, sqrt(fixed_norm2));
        point[z_index] = projected_z * rescaling[z_index];
        return;
    }

    double omega_z = cone_section_weight(rescaling, q_diag, tau, z_index);
    if (fixed_norm2 == 0.0)
    {
        if (-omega_z * z_input >= sqrt(polar_norm2))
        {
            for (int slot = 0; slot < u_length; ++slot)
                if (!is_fixed[start + slot])
                    point[start + slot] = 0.0;
            point[z_index] = 0.0;
            return;
        }
    }

    double lambda;
    if (z_input == 0.0)
    {
        lambda = omega_z;
        double norm2 = fixed_norm2;
        for (int slot = 0; slot < u_length; ++slot)
        {
            int index = start + slot;
            if (is_fixed[index])
                continue;
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            double value = cone_section_actual(point, rescaling, index) * omega / (omega + lambda);
            norm2 += value * value;
        }
        point[z_index] = sqrt(norm2) * rescaling[z_index];
    }
    else
    {
        bool lower_branch = z_input > 0.0;
        double lo;
        double hi;
        if (lower_branch)
        {
            lo = 0.0;
            hi = omega_z * (1.0 - 1e-14);
        }
        else
        {
            lo = omega_z * (1.0 + 1e-14);
            hi = cone_section_negative_soc_upper(omega_z, -omega_z * z_input, fixed_norm2, polar_norm2, max_omega);
            if (!(hi > lo) || !isfinite(hi))
            {
                hi = 2.0 * omega_z;
                for (int expansion = 0; expansion < 100; ++expansion)
                {
                    double norm2 = fixed_norm2;
                    for (int slot = 0; slot < u_length; ++slot)
                    {
                        int index = start + slot;
                        if (is_fixed[index])
                            continue;
                        double omega = cone_section_weight(rescaling, q_diag, tau, index);
                        double value = cone_section_actual(point, rescaling, index) * omega / (omega + hi);
                        norm2 += value * value;
                    }
                    double z = omega_z * z_input / (omega_z - hi);
                    if (norm2 >= z * z)
                        break;
                    hi *= 2.0;
                }
            }
        }

        if (warm_start && *warm_start > lo && *warm_start < hi && isfinite(*warm_start))
        {
            double norm2 = fixed_norm2;
            for (int slot = 0; slot < u_length; ++slot)
            {
                int index = start + slot;
                if (is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = cone_section_actual(point, rescaling, index) * omega / (omega + *warm_start);
                norm2 += value * value;
            }
            double z = omega_z * z_input / (omega_z - *warm_start);
            double f = norm2 - z * z;
            if ((lower_branch && f > 0.0) || (!lower_branch && f < 0.0))
                lo = *warm_start;
            else
                hi = *warm_start;
        }

        for (int iteration = 0; iteration < 80; ++iteration)
        {
            double trial = 0.5 * (lo + hi);
            double norm2 = fixed_norm2;
            for (int slot = 0; slot < u_length; ++slot)
            {
                int index = start + slot;
                if (is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = cone_section_actual(point, rescaling, index) * omega / (omega + trial);
                norm2 += value * value;
            }
            double z = omega_z * z_input / (omega_z - trial);
            double f = norm2 - z * z;
            if ((lower_branch && f > 0.0) || (!lower_branch && f < 0.0))
                lo = trial;
            else
                hi = trial;
            if ((hi - lo) <= 1e-13 * (1.0 + hi + lo))
                break;
        }
        lambda = 0.5 * (lo + hi);
        point[z_index] *= omega_z / (omega_z - lambda);
    }

    if (warm_start)
        *warm_start = lambda;
    for (int slot = 0; slot < u_length; ++slot)
    {
        int index = start + slot;
        if (!is_fixed[index])
        {
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            point[index] *= omega / (omega + lambda);
        }
    }
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
