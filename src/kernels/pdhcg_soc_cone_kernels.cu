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
#include "pdhcg_soc_cone_kernels.h"
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
#include "utils.h"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void project_standard_soc_kernel(double *__restrict__ primal_solution,
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

    int start = start_idx[blk];
    int k = v_dim[blk];
    if (cone_section_has_fixed(is_fixed, start, k + 2))
    {
        project_standard_soc_section_serial(
            primal_solution, variable_rescaling, NULL, 0.0, warm_start + blk, start, k, is_fixed);
        return;
    }
    double *v = primal_solution + start;
    double *wptr = primal_solution + start + k;
    double *zptr = primal_solution + start + k + 1;

    double w = *wptr;
    double z = *zptr;

    double d_z = variable_rescaling[start + k + 1];
    double dhat_w = variable_rescaling[start + k] / d_z;
    double dhat_w2 = dhat_w * dhat_w;

    bool diag_uniform = (dhat_w == 1.0);
    for (int m = 0; m < k && diag_uniform; ++m)
    {
        if (variable_rescaling[start + m] != d_z)
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
            *wptr = 0.0;
            *zptr = 0.0;
            return;
        }
        double scale = (z + r) / (2.0 * r);
        for (int m = 0; m < k; ++m)
            v[m] *= scale;
        *wptr = scale * w;
        *zptr = scale * r;
        return;
    }

    double r_inv_sq = (w / dhat_w) * (w / dhat_w);
    double r_pos_sq = (w * dhat_w) * (w * dhat_w);
    for (int m = 0; m < k; ++m)
    {
        double dh = variable_rescaling[start + m] / d_z;
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
        *wptr = 0.0;
        *zptr = 0.0;
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
                double dh = variable_rescaling[start + m] / d_z;
                double dh2 = dh * dh;
                double t = v[m] * dh / (dh2 + 2.0 * hi);
                sum_hi += t * t;
            }
            double tw_hi = w * dhat_w / (dhat_w2 + 2.0 * hi);
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
            double dh = variable_rescaling[start + m] / d_z;
            double dh2 = dh * dh;
            double t = v[m] * dh / (dh2 + 2.0 * warm_lam);
            sum_w += t * t;
        }
        double tw = w * dhat_w / (dhat_w2 + 2.0 * warm_lam);
        sum_w += tw * tw;
        double zt = z / (1.0 - 2.0 * warm_lam);
        double f = sum_w - zt * zt;
        if (fabs(f) < 1e-12)
        {
            *zptr = z / (1.0 - 2.0 * warm_lam);
            *wptr = w * dhat_w2 / (dhat_w2 + 2.0 * warm_lam);
            for (int m = 0; m < k; ++m)
            {
                double dh = variable_rescaling[start + m] / d_z;
                double dh2 = dh * dh;
                v[m] = v[m] * dh2 / (dh2 + 2.0 * warm_lam);
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
        double sum = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double dh = variable_rescaling[start + m] / d_z;
            double dh2 = dh * dh;
            double t = v[m] * dh / (dh2 + 2.0 * lam);
            sum += t * t;
        }
        double tw = w * dhat_w / (dhat_w2 + 2.0 * lam);
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

    *zptr = z / (1.0 - 2.0 * lam);
    *wptr = w * dhat_w2 / (dhat_w2 + 2.0 * lam);
    for (int m = 0; m < k; ++m)
    {
        double dh = variable_rescaling[start + m] / d_z;
        double dh2 = dh * dh;
        v[m] = v[m] * dh2 / (dh2 + 2.0 * lam);
    }
}

__global__ void compute_cone_dual_residual_standard_kernel(double *__restrict__ dual_residual,
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
        project_standard_soc_section_serial(
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

    double r_w = objective_vector[start + k] - dual_product[start + k];
    double r_z = objective_vector[start + k + 1] - dual_product[start + k + 1];

    double d_z = variable_rescaling[start + k + 1];
    double e_w = d_z / variable_rescaling[start + k];
    double e_w2 = e_w * e_w;

    bool diag_uniform = (e_w == 1.0);
    for (int m = 0; m < k && diag_uniform; ++m)
    {
        if (variable_rescaling[start + m] != d_z)
            diag_uniform = false;
    }

    if (diag_uniform)
    {
        double sumsq = r_w * r_w;
        for (int m = 0; m < k; ++m)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            sumsq += rc_m * rc_m;
        }
        double r = sqrt(sumsq);
        double v_factor, p_w, p_z;
        if (r <= r_z)
        {
            v_factor = 0.0;
            p_w = r_w;
            p_z = r_z;
        }
        else if (r <= -r_z)
        {
            v_factor = 1.0;
            p_w = 0.0;
            p_z = 0.0;
        }
        else
        {
            double scale = (r_z + r) / (2.0 * r);
            v_factor = 1.0 - scale;
            p_w = scale * r_w;
            p_z = scale * r;
        }
        for (int m = 0; m < k; ++m)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = rc_m * v_factor * variable_rescaling[start + m];
        }
        dual_residual[start + k] = (r_w - p_w) * variable_rescaling[start + k];
        dual_residual[start + k + 1] = (r_z - p_z) * variable_rescaling[start + k + 1];
        return;
    }

    double r_inv_sq = (r_w / e_w) * (r_w / e_w);
    double r_pos_sq = (r_w * e_w) * (r_w * e_w);
    for (int m = 0; m < k; ++m)
    {
        double e_m = d_z / variable_rescaling[start + m];
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
        dual_residual[start + k] = r_w * variable_rescaling[start + k];
        dual_residual[start + k + 1] = r_z * variable_rescaling[start + k + 1];
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
                double e_m = d_z / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double t = rc_m * e_m / (e_m2 + 2.0 * hi);
                sum_hi += t * t;
            }
            double tw_hi = r_w * e_w / (e_w2 + 2.0 * hi);
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
            double e_m = d_z / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double t = rc_m * e_m / (e_m2 + 2.0 * warm_lam);
            sum_w += t * t;
        }
        double tw = r_w * e_w / (e_w2 + 2.0 * warm_lam);
        sum_w += tw * tw;
        double zt = r_z / (1.0 - 2.0 * warm_lam);
        double f = sum_w - zt * zt;
        if (fabs(f) < 1e-12)
        {
            double p_z_w = r_z / (1.0 - 2.0 * warm_lam);
            double p_w_w = r_w * e_w2 / (e_w2 + 2.0 * warm_lam);
            for (int m = 0; m < k; ++m)
            {
                double e_m = d_z / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double p_m = rc_m * e_m2 / (e_m2 + 2.0 * warm_lam);
                dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
            }
            dual_residual[start + k] = (r_w - p_w_w) * variable_rescaling[start + k];
            dual_residual[start + k + 1] = (r_z - p_z_w) * variable_rescaling[start + k + 1];
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
            double e_m = d_z / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double t = rc_m * e_m / (e_m2 + 2.0 * lam);
            sum += t * t;
        }
        double tw = r_w * e_w / (e_w2 + 2.0 * lam);
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

    double p_z = r_z / (1.0 - 2.0 * lam);
    double p_w = r_w * e_w2 / (e_w2 + 2.0 * lam);

    for (int m = 0; m < k; ++m)
    {
        double e_m = d_z / variable_rescaling[start + m];
        double e_m2 = e_m * e_m;
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        double p_m = rc_m * e_m2 / (e_m2 + 2.0 * lam);
        dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
    }
    dual_residual[start + k] = (r_w - p_w) * variable_rescaling[start + k];
    dual_residual[start + k + 1] = (r_z - p_z) * variable_rescaling[start + k + 1];
}

__global__ void project_standard_soc_grid_reduce_kernel(double *__restrict__ primal_solution,
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

__global__ void project_standard_soc_grid_finalize_kernel(double *__restrict__ primal_solution,
                                                          double *__restrict__ workspace,
                                                          const int *__restrict__ start_idx,
                                                          const int *__restrict__ v_dim,
                                                          int num_cones)
{
    int cone = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone >= num_cones)
        return;

    int start = start_idx[cone];
    int k = v_dim[cone];
    double w = primal_solution[start + k];
    double z = primal_solution[start + k + 1];
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
    workspace[cone] = scale;
    primal_solution[start + k] = scale * w;
    primal_solution[start + k + 1] = scale * radius;
}

__global__ void project_standard_soc_grid_apply_kernel(double *__restrict__ primal_solution,
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

__global__ void compute_cone_dual_residual_standard_grid_reduce_kernel(const double *__restrict__ objective_vector,
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

__global__ void compute_cone_dual_residual_standard_grid_finalize_kernel(double *__restrict__ dual_residual,
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

    int start = start_idx[cone];
    int k = v_dim[cone];
    double r_w = objective_vector[start + k] - dual_product[start + k];
    double r_z = objective_vector[start + k + 1] - dual_product[start + k + 1];
    double radius = sqrt(fmax(0.0, workspace[cone] + r_w * r_w));
    double factor;
    double p_w;
    double p_z;

    if (radius <= r_z)
    {
        factor = 0.0;
        p_w = r_w;
        p_z = r_z;
    }
    else if (radius <= -r_z)
    {
        factor = 1.0;
        p_w = 0.0;
        p_z = 0.0;
    }
    else
    {
        double scale = (r_z + radius) / (2.0 * radius);
        factor = 1.0 - scale;
        p_w = scale * r_w;
        p_z = scale * radius;
    }

    workspace[cone] = factor;
    dual_residual[start + k] = (r_w - p_w) * variable_rescaling[start + k];
    dual_residual[start + k + 1] = (r_z - p_z) * variable_rescaling[start + k + 1];
}

__global__ void compute_cone_dual_residual_standard_grid_apply_kernel(double *__restrict__ dual_residual,
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

__global__ void project_standard_soc_warp_kernel(double *__restrict__ primal_solution,
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

    const unsigned MASK = 0xffffffffu;

    int start = start_idx[blk];
    int k = v_dim[blk];

    int has_fixed = lane == 0 ? cone_section_has_fixed(is_fixed, start, k + 2) : 0;
    has_fixed = __shfl_sync(MASK, has_fixed, 0);
    if (has_fixed)
    {
        if (lane == 0)
            project_standard_soc_section_serial(
                primal_solution, variable_rescaling, NULL, 0.0, warm_start + blk, start, k, is_fixed);
        return;
    }

    double w = primal_solution[start + k];
    double z = primal_solution[start + k + 1];

    double d_z = variable_rescaling[start + k + 1];
    double dhat_w = variable_rescaling[start + k] / d_z;
    double dhat_w2 = dhat_w * dhat_w;

    int my_diff = (lane == 0 && dhat_w != 1.0) ? 1 : 0;
    for (int m = lane; m < k; m += 32)
    {
        if (variable_rescaling[start + m] != d_z)
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
        if (lane == 0)
        {
            primal_solution[start + k] = scale * w;
            primal_solution[start + k + 1] = scale * r;
        }
        return;
    }

    double my_inv = (lane == 0) ? (w / dhat_w) * (w / dhat_w) : 0.0;
    double my_pos = (lane == 0) ? (w * dhat_w) * (w * dhat_w) : 0.0;
    for (int m = lane; m < k; m += 32)
    {
        double dh = variable_rescaling[start + m] / d_z;
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
            double my_sum =
                (lane == 0) ? (w * dhat_w / (dhat_w2 + 2.0 * hi)) * (w * dhat_w / (dhat_w2 + 2.0 * hi)) : 0.0;
            for (int m = lane; m < k; m += 32)
            {
                double dh = variable_rescaling[start + m] / d_z;
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
        double my_sum =
            (lane == 0) ? (w * dhat_w / (dhat_w2 + 2.0 * warm_lam)) * (w * dhat_w / (dhat_w2 + 2.0 * warm_lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double dh = variable_rescaling[start + m] / d_z;
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
            for (int m = lane; m < k; m += 32)
            {
                double dh = variable_rescaling[start + m] / d_z;
                double dh2 = dh * dh;
                primal_solution[start + m] = primal_solution[start + m] * dh2 / (dh2 + 2.0 * warm_lam);
            }
            if (lane == 0)
            {
                primal_solution[start + k + 1] = z / (1.0 - 2.0 * warm_lam);
                primal_solution[start + k] = w * dhat_w2 / (dhat_w2 + 2.0 * warm_lam);
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
        double my_sum = (lane == 0) ? (w * dhat_w / (dhat_w2 + 2.0 * lam)) * (w * dhat_w / (dhat_w2 + 2.0 * lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double dh = variable_rescaling[start + m] / d_z;
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

    for (int m = lane; m < k; m += 32)
    {
        double dh = variable_rescaling[start + m] / d_z;
        double dh2 = dh * dh;
        primal_solution[start + m] = primal_solution[start + m] * dh2 / (dh2 + 2.0 * lam);
    }
    if (lane == 0)
    {
        primal_solution[start + k + 1] = z / (1.0 - 2.0 * lam);
        primal_solution[start + k] = w * dhat_w2 / (dhat_w2 + 2.0 * lam);
    }
}

__global__ void compute_cone_dual_residual_standard_warp_kernel(double *__restrict__ dual_residual,
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
            project_standard_soc_section_serial(
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

    double r_w = objective_vector[start + k] - dual_product[start + k];
    double r_z = objective_vector[start + k + 1] - dual_product[start + k + 1];

    double d_z = variable_rescaling[start + k + 1];
    double e_w = d_z / variable_rescaling[start + k];
    double e_w2 = e_w * e_w;

    int my_diff = (lane == 0 && e_w != 1.0) ? 1 : 0;
    for (int m = lane; m < k; m += 32)
    {
        if (variable_rescaling[start + m] != d_z)
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
        double r = sqrt(my_sumsq);
        double v_factor, p_w, p_z;
        if (r <= r_z)
        {
            v_factor = 0.0;
            p_w = r_w;
            p_z = r_z;
        }
        else if (r <= -r_z)
        {
            v_factor = 1.0;
            p_w = 0.0;
            p_z = 0.0;
        }
        else
        {
            double scale = (r_z + r) / (2.0 * r);
            v_factor = 1.0 - scale;
            p_w = scale * r_w;
            p_z = scale * r;
        }
        for (int m = lane; m < k; m += 32)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = rc_m * v_factor * variable_rescaling[start + m];
        }
        if (lane == 0)
        {
            dual_residual[start + k] = (r_w - p_w) * variable_rescaling[start + k];
            dual_residual[start + k + 1] = (r_z - p_z) * variable_rescaling[start + k + 1];
        }
        return;
    }

    double my_inv = (lane == 0) ? (r_w / e_w) * (r_w / e_w) : 0.0;
    double my_pos = (lane == 0) ? (r_w * e_w) * (r_w * e_w) : 0.0;
    for (int m = lane; m < k; m += 32)
    {
        double e_m = d_z / variable_rescaling[start + m];
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
            dual_residual[start + k] = r_w * variable_rescaling[start + k];
            dual_residual[start + k + 1] = r_z * variable_rescaling[start + k + 1];
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
            double my_sum = (lane == 0) ? (r_w * e_w / (e_w2 + 2.0 * hi)) * (r_w * e_w / (e_w2 + 2.0 * hi)) : 0.0;
            for (int m = lane; m < k; m += 32)
            {
                double e_m = d_z / variable_rescaling[start + m];
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
        double my_sum =
            (lane == 0) ? (r_w * e_w / (e_w2 + 2.0 * warm_lam)) * (r_w * e_w / (e_w2 + 2.0 * warm_lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double e_m = d_z / variable_rescaling[start + m];
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
            double p_z_w = r_z / (1.0 - 2.0 * warm_lam);
            double p_w_w = r_w * e_w2 / (e_w2 + 2.0 * warm_lam);
            for (int m = lane; m < k; m += 32)
            {
                double e_m = d_z / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double p_m = rc_m * e_m2 / (e_m2 + 2.0 * warm_lam);
                dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
            }
            if (lane == 0)
            {
                dual_residual[start + k] = (r_w - p_w_w) * variable_rescaling[start + k];
                dual_residual[start + k + 1] = (r_z - p_z_w) * variable_rescaling[start + k + 1];
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
        double my_sum = (lane == 0) ? (r_w * e_w / (e_w2 + 2.0 * lam)) * (r_w * e_w / (e_w2 + 2.0 * lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double e_m = d_z / variable_rescaling[start + m];
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

    double p_z = r_z / (1.0 - 2.0 * lam);
    double p_w = r_w * e_w2 / (e_w2 + 2.0 * lam);

    for (int m = lane; m < k; m += 32)
    {
        double e_m = d_z / variable_rescaling[start + m];
        double e_m2 = e_m * e_m;
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        double p_m = rc_m * e_m2 / (e_m2 + 2.0 * lam);
        dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
    }
    if (lane == 0)
    {
        dual_residual[start + k] = (r_w - p_w) * variable_rescaling[start + k];
        dual_residual[start + k + 1] = (r_z - p_z) * variable_rescaling[start + k + 1];
    }
}

/* Project onto D K_exp via Parikh-Boyd Newton on rho = u_1/u_2 (u = D^{-1} x). */

__global__ void project_standard_soc_diag_q_kernel(double *__restrict__ pdhg_primal,
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

    int start = start_idx[blk];
    int k = v_dim[blk];
    int w_off = start + k;
    int z_off = start + k + 1;

    if (cone_section_has_fixed(is_fixed, start, k + 2))
    {
        project_standard_soc_section_serial(
            pdhg_primal, variable_rescaling, Q_diag, tau, warm_start + blk, start, k, is_fixed);
        for (int slot = 0; slot < k + 2; ++slot)
        {
            int index = start + slot;
            reflected_primal[index] = 2.0 * pdhg_primal[index] - current_primal[index];
        }
        return;
    }

    double r_w = pdhg_primal[w_off];
    double r_z = pdhg_primal[z_off];

    double d_z = variable_rescaling[z_off];
    double w_w = 1.0 + tau * Q_diag[w_off];
    double w_z = 1.0 + tau * Q_diag[z_off];
    double sqrt_w_w = sqrt(w_w);
    double sqrt_w_z = sqrt(w_z);
    double e_z = sqrt_w_z * d_z;
    double e_w = sqrt_w_w * variable_rescaling[w_off];
    double eh_w = e_w / e_z;
    double eh_w2 = eh_w * eh_w;

    double r_inv_sq = w_w * (r_w / eh_w) * (r_w / eh_w);
    double r_pos_sq = w_w * (r_w * eh_w) * (r_w * eh_w);
    for (int m = 0; m < k; ++m)
    {
        double w_m = 1.0 + tau * Q_diag[start + m];
        double e_m = sqrt(w_m) * variable_rescaling[start + m];
        double eh_m = e_m / e_z;
        double r_m = pdhg_primal[start + m];
        r_inv_sq += w_m * (r_m / eh_m) * (r_m / eh_m);
        r_pos_sq += w_m * (r_m * eh_m) * (r_m * eh_m);
    }
    double w_z_r_z_sq = w_z * r_z * r_z;

    if (r_inv_sq <= w_z_r_z_sq && r_z >= 0.0)
    {
        for (int m = 0; m < k; ++m)
        {
            int idx = start + m;
            reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
        }
        reflected_primal[w_off] = 2.0 * r_w - current_primal[w_off];
        reflected_primal[z_off] = 2.0 * r_z - current_primal[z_off];
        return;
    }

    if (r_pos_sq <= w_z_r_z_sq && r_z <= 0.0)
    {
        for (int m = 0; m < k; ++m)
        {
            int idx = start + m;
            pdhg_primal[idx] = 0.0;
            reflected_primal[idx] = -current_primal[idx];
        }
        pdhg_primal[w_off] = 0.0;
        pdhg_primal[z_off] = 0.0;
        reflected_primal[w_off] = -current_primal[w_off];
        reflected_primal[z_off] = -current_primal[z_off];
        return;
    }

    /* Fast path: no Q on cone slots and uniform d_v = d_z (LP-style symmetric case). */
    if (Q_diag[w_off] == 0.0 && Q_diag[z_off] == 0.0)
    {
        bool no_cone_Q = true;
        bool d_uniform = (variable_rescaling[w_off] == d_z);
        for (int m = 0; m < k; ++m)
        {
            if (Q_diag[start + m] != 0.0)
            {
                no_cone_Q = false;
                break;
            }
            if (variable_rescaling[start + m] != d_z)
            {
                d_uniform = false;
                break;
            }
        }
        if (no_cone_Q && d_uniform)
        {
            double sumsq = r_w * r_w;
            for (int m = 0; m < k; ++m)
            {
                double vm = pdhg_primal[start + m];
                sumsq += vm * vm;
            }
            double rnorm = sqrt(sumsq);
            /* in-cone (rnorm <= r_z, r_z >= 0) and at-origin (rnorm <= -r_z, r_z <= 0) handled above */
            double scale = (r_z + rnorm) / (2.0 * rnorm);
            for (int m = 0; m < k; ++m)
            {
                double v_new = scale * pdhg_primal[start + m];
                pdhg_primal[start + m] = v_new;
                int idx = start + m;
                reflected_primal[idx] = 2.0 * v_new - current_primal[idx];
            }
            double w_new = scale * r_w;
            double z_new = scale * rnorm;
            pdhg_primal[w_off] = w_new;
            pdhg_primal[z_off] = z_new;
            reflected_primal[w_off] = 2.0 * w_new - current_primal[w_off];
            reflected_primal[z_off] = 2.0 * z_new - current_primal[z_off];
            return;
        }
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
                double w_m = 1.0 + tau * Q_diag[start + m];
                double e_m = sqrt(w_m) * variable_rescaling[start + m];
                double eh_m = e_m / e_z;
                double eh_m2 = eh_m * eh_m;
                double r_m = pdhg_primal[start + m];
                double t = sqrt(w_m) * r_m * eh_m / (eh_m2 + 2.0 * hi);
                sum_hi += t * t;
            }
            double tw_hi = sqrt_w_w * r_w * eh_w / (eh_w2 + 2.0 * hi);
            sum_hi += tw_hi * tw_hi;
            double tz_hi = sqrt_w_z * r_z / (1.0 - 2.0 * hi);
            double f_hi = sum_hi - tz_hi * tz_hi;
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
            double w_m = 1.0 + tau * Q_diag[start + m];
            double e_m = sqrt(w_m) * variable_rescaling[start + m];
            double eh_m = e_m / e_z;
            double eh_m2 = eh_m * eh_m;
            double r_m = pdhg_primal[start + m];
            double t = sqrt(w_m) * r_m * eh_m / (eh_m2 + 2.0 * warm_lam);
            sum_w += t * t;
        }
        double tw = sqrt_w_w * r_w * eh_w / (eh_w2 + 2.0 * warm_lam);
        sum_w += tw * tw;
        double tz = sqrt_w_z * r_z / (1.0 - 2.0 * warm_lam);
        double f = sum_w - tz * tz;
        if (fabs(f) < 1e-12)
        {
            double new_z = r_z / (1.0 - 2.0 * warm_lam);
            double new_w = r_w * eh_w2 / (eh_w2 + 2.0 * warm_lam);
            pdhg_primal[z_off] = new_z;
            pdhg_primal[w_off] = new_w;
            reflected_primal[z_off] = 2.0 * new_z - current_primal[z_off];
            reflected_primal[w_off] = 2.0 * new_w - current_primal[w_off];
            for (int m = 0; m < k; ++m)
            {
                int idx = start + m;
                double w_m = 1.0 + tau * Q_diag[idx];
                double e_m = sqrt(w_m) * variable_rescaling[idx];
                double eh_m = e_m / e_z;
                double eh_m2 = eh_m * eh_m;
                double r_m = pdhg_primal[idx];
                double new_m = r_m * eh_m2 / (eh_m2 + 2.0 * warm_lam);
                pdhg_primal[idx] = new_m;
                reflected_primal[idx] = 2.0 * new_m - current_primal[idx];
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
        double sum = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double w_m = 1.0 + tau * Q_diag[start + m];
            double e_m = sqrt(w_m) * variable_rescaling[start + m];
            double eh_m = e_m / e_z;
            double eh_m2 = eh_m * eh_m;
            double r_m = pdhg_primal[start + m];
            double t = sqrt(w_m) * r_m * eh_m / (eh_m2 + 2.0 * lam);
            sum += t * t;
        }
        double tw = sqrt_w_w * r_w * eh_w / (eh_w2 + 2.0 * lam);
        sum += tw * tw;
        double tz = sqrt_w_z * r_z / (1.0 - 2.0 * lam);
        double f = sum - tz * tz;
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

    double new_z = r_z / (1.0 - 2.0 * lam);
    double new_w = r_w * eh_w2 / (eh_w2 + 2.0 * lam);
    pdhg_primal[z_off] = new_z;
    pdhg_primal[w_off] = new_w;
    reflected_primal[z_off] = 2.0 * new_z - current_primal[z_off];
    reflected_primal[w_off] = 2.0 * new_w - current_primal[w_off];
    for (int m = 0; m < k; ++m)
    {
        int idx = start + m;
        double w_m = 1.0 + tau * Q_diag[idx];
        double e_m = sqrt(w_m) * variable_rescaling[idx];
        double eh_m = e_m / e_z;
        double eh_m2 = eh_m * eh_m;
        double r_m = pdhg_primal[idx];
        double new_m = r_m * eh_m2 / (eh_m2 + 2.0 * lam);
        pdhg_primal[idx] = new_m;
        reflected_primal[idx] = 2.0 * new_m - current_primal[idx];
    }
}

/* Weighted prox onto D K_exp; coordinate change y_i = sqrt(w_i) x_i gives e_i = sqrt(w_i) d_i. */

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

static void launch_standard_thread_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_standard_soc_kernel<<<b, t>>>(p, vr, ws, si, vd, isf, n);
}
static void launch_standard_warp_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n * 32 + t - 1) / t;
    project_standard_soc_warp_kernel<<<b, t>>>(p, vr, ws, si, vd, isf, n);
}
static void launch_standard_block_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    project_standard_soc_block_kernel<<<n, THREADS_PER_BLOCK>>>(p, vr, NULL, 0.0, ws, si, vd, isf, n);
}
static void launch_standard_grid_weighted_impl(double *p,
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
    initialize_standard_soc_grid_weighted_kernel<<<blocks, threads>>>(
        p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
    finalize_standard_soc_grid_weighted_initialization_kernel<<<(n + threads - 1) / threads, threads>>>(
        p, vr, qd, tau, ws, si, vd, isf, n);
    for (int iteration = 0; iteration < PDHCG_CONE_GRID_ROOT_ITERATIONS; ++iteration)
    {
        CUDA_CHECK(cudaMemsetAsync(ws + n, 0, (size_t)2 * n * sizeof(double)));
        reduce_standard_soc_grid_weighted_root_kernel<<<blocks, threads>>>(
            p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
        finalize_standard_soc_grid_weighted_root_kernel<<<(n + threads - 1) / threads, threads>>>(
            p, vr, qd, tau, ws, si, vd, n);
    }
    apply_standard_soc_grid_weighted_kernel<<<blocks, threads>>>(p, vr, qd, tau, ws, si, vd, isf, n, blocks_per_cone);
}
static void launch_standard_grid_weighted_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    launch_standard_grid_weighted_impl(p, vr, NULL, 0.0, ws, si, vd, isf, n);
}
static void launch_standard_grid_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)vr;
    (void)pa;
    (void)isf;
    int t = THREADS_PER_BLOCK;
    int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
    int b = n * blocks_per_cone;
    CUDA_CHECK(cudaMemsetAsync(ws, 0, (size_t)n * sizeof(double)));
    project_standard_soc_grid_reduce_kernel<<<b, t>>>(p, ws, si, vd, n, blocks_per_cone);
    project_standard_soc_grid_finalize_kernel<<<(n + t - 1) / t, t>>>(p, ws, si, vd, n);
    project_standard_soc_grid_apply_kernel<<<b, t>>>(p, ws, si, vd, n, blocks_per_cone);
}
static void launch_standard_thread_dual(double *dr,
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
    compute_cone_dual_residual_standard_kernel<<<b, t>>>(dr, cr, obj, dp, vr, ps, ws, si, vd, isf, n);
}
static void launch_standard_warp_dual(double *dr,
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
    compute_cone_dual_residual_standard_warp_kernel<<<b, t>>>(dr, cr, obj, dp, vr, ps, ws, si, vd, isf, n);
}
static void launch_standard_grid_dual(double *dr,
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
    compute_cone_dual_residual_standard_grid_reduce_kernel<<<b, t>>>(obj, dp, ws, si, vd, n, blocks_per_cone);
    compute_cone_dual_residual_standard_grid_finalize_kernel<<<(n + t - 1) / t, t>>>(dr, obj, dp, vr, ws, si, vd, n);
    compute_cone_dual_residual_standard_grid_apply_kernel<<<b, t>>>(dr, obj, dp, vr, ws, si, vd, n, blocks_per_cone);
}
static void launch_standard_thread_proj_diag_q(double *pp,
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
    project_standard_soc_diag_q_kernel<<<b, t>>>(pp, rp, cp, vr, qd, tau, ws, si, vd, isf, n);
}
static void launch_standard_block_proj_diag_q(double *pp,
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
    project_standard_soc_block_kernel<<<n, THREADS_PER_BLOCK>>>(pp, vr, qd, tau, ws, si, vd, isf, n);
    launch_cone_reflection(PROJ_METHOD_BLOCK, rp, pp, cp, si, vd, n);
}
static void launch_standard_grid_weighted_proj_diag_q(double *pp,
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
    launch_standard_grid_weighted_impl(pp, vr, qd, tau, ws, si, vd, isf, n);
    launch_cone_reflection(PROJ_METHOD_GRID, rp, pp, cp, si, vd, n);
}

extern const cone_kernel_ops_t pdhcg_soc_cone_kernel_ops = {
    {
        launch_standard_thread_proj,
        launch_standard_warp_proj,
        launch_standard_block_proj,
        launch_standard_grid_proj,
        launch_standard_grid_weighted_proj,
    },
    {
        launch_standard_thread_proj_diag_q,
        launch_standard_block_proj_diag_q,
        launch_standard_block_proj_diag_q,
        launch_standard_grid_weighted_proj_diag_q,
        launch_standard_grid_weighted_proj_diag_q,
    },
    {
        launch_standard_thread_dual,
        launch_standard_warp_dual,
        launch_block_projected_mapping_only_dual,
        launch_standard_grid_dual,
        launch_grid_projected_mapping_only_dual,
    },
};
