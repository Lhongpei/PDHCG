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
__device__ static inline void project_exp_cone_point(
    double r1, double r2, double r3, double d1, double d2, double d3, double *xo, double *yo, double *zo)
{
    const double E_CONST = 2.718281828459045;
    double rr1 = r1 / d1, rr2 = r2 / d2, rr3 = r3 / d3;

    if (rr2 > 0.0)
    {
        double ratio = rr1 / rr2;
        if (ratio < 700.0 && rr2 * exp(ratio) <= rr3)
        {
            *xo = r1;
            *yo = r2;
            *zo = r3;
            return;
        }
    }
    else if (rr2 == 0.0 && rr1 <= 0.0 && rr3 >= 0.0)
    {
        *xo = r1;
        *yo = r2;
        *zo = r3;
        return;
    }

    if (r1 > 0.0)
    {
        double ratio = (d2 * r2) / (d1 * r1);
        if (ratio < 700.0 && d1 * r1 * exp(ratio) + E_CONST * d3 * r3 <= 0.0)
        {
            *xo = 0.0;
            *yo = 0.0;
            *zo = 0.0;
            return;
        }
    }
    else if (r1 == 0.0 && r2 <= 0.0 && r3 <= 0.0)
    {
        *xo = 0.0;
        *yo = 0.0;
        *zo = 0.0;
        return;
    }

    if (rr1 <= 0.0 && rr2 <= 0.0)
    {
        *xo = r1;
        *yo = 0.0;
        *zo = (rr3 < 0.0) ? 0.0 : r3;
        return;
    }

    double alpha = (d3 / d1) * (d3 / d1);
    double beta = (d3 / d2) * (d3 / d2);
    double rho = 0.0;
    bool diverged = false;
    for (int it = 0; it < 100; ++it)
    {
        if (rho >= 299.0 || rho <= -299.0)
        {
            diverged = true;
            break;
        }
        double e_rho = exp(rho);
        double e_2rho = e_rho * e_rho;
        double one_m_rho = 1.0 - rho;

        double a_term = alpha - beta * rho * one_m_rho;
        double b_term = beta * rr1 * one_m_rho - alpha * rr2;
        double f = rr1 - rr2 * rho + rr3 * e_rho * a_term + e_2rho * b_term;

        double da_drho = -beta * (1.0 - 2.0 * rho);
        double db_drho = -beta * rr1;
        double df = -rr2 + rr3 * e_rho * (a_term + da_drho) + e_2rho * (2.0 * b_term + db_drho);

        if (fabs(df) < 1e-300)
            break;
        double step = f / df;
        if (step > 10.0)
            step = 10.0;
        if (step < -10.0)
            step = -10.0;
        rho -= step;
        if (fabs(step) < 1e-13 * (1.0 + fabs(rho)))
            break;
    }

    double e_rho = exp(rho);
    double denom = rho + alpha * e_rho * e_rho;
    double u2 = (rr1 + alpha * rr3 * e_rho) / denom;

    if (diverged || !isfinite(u2) || u2 <= 0.0)
    {
        *xo = (r1 < 0.0) ? r1 : 0.0;
        *yo = 0.0;
        *zo = (r3 > 0.0) ? r3 : 0.0;
        return;
    }

    double u1 = rho * u2;
    double u3 = u2 * e_rho;

    *xo = d1 * u1;
    *yo = d2 * u2;
    *zo = d3 * u3;
}

/* y-fixed cross-section of D K_exp: weighted 1D Newton-bisection on u = exp((rz/d_r)/y_eff). */
__device__ static inline void project_2d_exp_persp(
    double rz0, double ry, double rt0, double d_r, double d_y, double d_t, double *warm_start, double *rzo, double *rto)
{
    if (d_r <= 0.0 || d_y <= 0.0 || d_t <= 0.0)
    {
        *rzo = rz0;
        *rto = rt0;
        return;
    }
    double y_eff = ry / d_y;
    if (y_eff <= 0.0)
    {
        *rzo = (rz0 < 0.0) ? rz0 : 0.0;
        *rto = (rt0 > 0.0) ? rt0 : 0.0;
        return;
    }

    double arg = (rz0 / d_r) / y_eff;
    if (arg < 700.0)
    {
        double rhs = y_eff * d_t * exp(arg);
        if (rhs <= rt0)
        {
            *rzo = rz0;
            *rto = rt0;
            return;
        }
    }

    double a = d_t * d_t * y_eff;
    double b = d_t * rt0;
    double c = d_r * d_r * y_eff;
    double e = d_r * rz0;

    double u = *warm_start;
    double u_lo = 1e-30;
    double u_hi = 1.0;
    for (int g = 0; g < 200; ++g)
    {
        double lu = log(u_hi);
        double f_hi = a * u_hi * u_hi - b * u_hi + c * lu - e;
        if (isfinite(f_hi) && f_hi > 0.0)
            break;
        u_hi *= 4.0;
        if (u_hi > 1e150)
            break;
    }
    for (int g = 0; g < 200; ++g)
    {
        double lu = log(u_lo);
        double f_lo = a * u_lo * u_lo - b * u_lo + c * lu - e;
        if (isfinite(f_lo) && f_lo < 0.0)
            break;
        u_lo *= 0.25;
        if (u_lo < 1e-300)
            break;
    }
    if (u_lo >= u_hi)
    {
        *rzo = rz0;
        *rto = rt0;
        return;
    }

    if (!(u > u_lo && u < u_hi) || !isfinite(u))
        u = exp(0.5 * (log(u_lo) + log(u_hi)));

    for (int it = 0; it < 80; ++it)
    {
        double lu = log(u);
        double f = a * u * u - b * u + c * lu - e;
        double df = 2.0 * a * u - b + c / u;
        if (f > 0.0)
            u_hi = u;
        else
            u_lo = u;
        double u_new;
        if (df > 1e-300 && isfinite(df) && isfinite(f))
        {
            u_new = u - f / df;
            if (!isfinite(u_new) || u_new <= u_lo || u_new >= u_hi)
                u_new = exp(0.5 * (log(u_lo) + log(u_hi)));
        }
        else
        {
            u_new = exp(0.5 * (log(u_lo) + log(u_hi)));
        }
        if (fabs(u_new - u) < 1e-14 * (1.0 + fabs(u_new)))
        {
            u = u_new;
            break;
        }
        u = u_new;
    }
    *warm_start = u;
    *rzo = d_r * y_eff * log(u);
    *rto = d_t * y_eff * u;
}

__device__ static inline double exp_cone_boundary(double x, double y)
{
    if (!(y > 0.0))
        return x <= 0.0 ? 0.0 : INFINITY;
    double exponent = x / y;
    double log_value = log(y) + exponent;
    if (log_value >= log(DBL_MAX))
        return INFINITY;
    if (log_value <= log(DBL_MIN))
        return 0.0;
    return exp(log_value);
}

__device__ static inline bool exp_cone_contains_point(double x, double y, double z)
{
    if (y > 0.0 && z > 0.0)
    {
        double lhs = log(y) + x / y;
        double rhs = log(z);
        double tolerance = 64.0 * DBL_EPSILON * (1.0 + fabs(lhs) + fabs(rhs));
        return lhs <= rhs + tolerance;
    }
    return y == 0.0 && x <= 0.0 && z >= 0.0;
}

__device__ static inline double
exp_fixed_x_objective(double y, double x, double input_y, double input_z, double weight_y, double weight_z)
{
    double z = exp_cone_boundary(x, y);
    if (!isfinite(z))
        return INFINITY;
    double dy = y - input_y;
    double dz = z - input_z;
    return weight_y * dy * dy + weight_z * dz * dz;
}

__device__ static inline double
exp_fixed_z_objective(double y, double z, double input_x, double input_y, double weight_x, double weight_y)
{
    double x = y > 0.0 ? y * (log(z) - log(y)) : 0.0;
    double dx = x - input_x;
    double dy = y - input_y;
    return weight_x * dx * dx + weight_y * dy * dy;
}

__device__ static inline double exp_xz_log_violation(double y, double x, double z)
{
    if (!(y > 0.0) || !(z > 0.0))
        return x <= 0.0 ? -INFINITY : INFINITY;
    return log(y) + x / y - log(z);
}

__device__ static inline void project_exp_cone_section(double *point,
                                                       const double *rescaling,
                                                       const double *q_diag,
                                                       double tau,
                                                       double *warm_start,
                                                       int start,
                                                       const char *is_fixed)
{
    bool fixed_x = is_fixed[start + 0] != 0;
    bool fixed_y = is_fixed[start + 1] != 0;
    bool fixed_z = is_fixed[start + 2] != 0;
    double input_x = point[start + 0] / rescaling[start + 0];
    double input_y = point[start + 1] / rescaling[start + 1];
    double input_z = point[start + 2] / rescaling[start + 2];

    if (exp_cone_contains_point(input_x, input_y, input_z) || (fixed_x && fixed_y && fixed_z))
        return;

    double weight_x = cone_section_weight(rescaling, q_diag, tau, start + 0);
    double weight_y = cone_section_weight(rescaling, q_diag, tau, start + 1);
    double weight_z = cone_section_weight(rescaling, q_diag, tau, start + 2);
    double output_x = input_x;
    double output_y = input_y;
    double output_z = input_z;

    if (fixed_x && fixed_y)
    {
        output_z = fmax(input_z, exp_cone_boundary(input_x, input_y));
    }
    else if (fixed_y && fixed_z)
    {
        if (input_y == 0.0)
            output_x = fmin(input_x, 0.0);
        else
            output_x = fmin(input_x, input_y * (log(input_z) - log(input_y)));
    }
    else if (fixed_x && fixed_z)
    {
        if (input_z == 0.0)
        {
            output_y = 0.0;
        }
        else if (input_x > 0.0)
        {
            double center = input_x;
            double left = fmax(DBL_MIN, input_x / 1024.0);
            while (exp_xz_log_violation(left, input_x, input_z) <= 0.0 && left > DBL_MIN)
                left *= 0.5;
            double lo = left;
            double hi = center;
            for (int iteration = 0; iteration < 100; ++iteration)
            {
                double mid = 0.5 * (lo + hi);
                if (exp_xz_log_violation(mid, input_x, input_z) > 0.0)
                    lo = mid;
                else
                    hi = mid;
            }
            double lower = 0.5 * (lo + hi);

            lo = center;
            hi = fmax(2.0 * center, input_z);
            while (exp_xz_log_violation(hi, input_x, input_z) < 0.0 && hi < DBL_MAX / 4.0)
                hi *= 2.0;
            for (int iteration = 0; iteration < 100; ++iteration)
            {
                double mid = 0.5 * (lo + hi);
                if (exp_xz_log_violation(mid, input_x, input_z) <= 0.0)
                    lo = mid;
                else
                    hi = mid;
            }
            double upper = 0.5 * (lo + hi);
            output_y = fmin(fmax(input_y, lower), upper);
        }
        else
        {
            double lo = 0.0;
            double hi = fmax(1.0, fmax(input_z, fabs(input_x)));
            while (exp_xz_log_violation(hi, input_x, input_z) < 0.0 && hi < DBL_MAX / 4.0)
                hi *= 2.0;
            for (int iteration = 0; iteration < 100; ++iteration)
            {
                double mid = 0.5 * (lo + hi);
                if (exp_xz_log_violation(mid, input_x, input_z) <= 0.0)
                    lo = mid;
                else
                    hi = mid;
            }
            output_y = fmin(fmax(input_y, 0.0), 0.5 * (lo + hi));
        }
    }
    else if (fixed_y)
    {
        if (!(input_y > 0.0))
        {
            output_x = fmin(input_x, 0.0);
            output_z = fmax(input_z, 0.0);
        }
        else
        {
            double effective_x = sqrt(weight_x);
            double effective_y = sqrt(weight_y);
            double effective_z = sqrt(weight_z);
            double scaled_x;
            double scaled_z;
            project_2d_exp_persp(effective_x * input_x,
                                 effective_y * input_y,
                                 effective_z * input_z,
                                 effective_x,
                                 effective_y,
                                 effective_z,
                                 warm_start,
                                 &scaled_x,
                                 &scaled_z);
            output_x = scaled_x / effective_x;
            output_z = scaled_z / effective_z;
        }
    }
    else if (fixed_x)
    {
        double scale = 1.0 + fabs(input_x) + fabs(input_y) + fabs(input_z);
        double lo = input_x > 0.0 ? fmax(DBL_MIN, input_x / 700.0) : 0.0;
        double hi = scale;
        double previous = exp_fixed_x_objective(0.5 * hi, input_x, input_y, input_z, weight_y, weight_z);
        double current = exp_fixed_x_objective(hi, input_x, input_y, input_z, weight_y, weight_z);
        for (int expansion = 0; expansion < 80 && current < previous && hi < DBL_MAX / 4.0; ++expansion)
        {
            previous = current;
            hi *= 2.0;
            current = exp_fixed_x_objective(hi, input_x, input_y, input_z, weight_y, weight_z);
        }
        const double ratio = 0.6180339887498948482;
        double a = lo;
        double b = hi;
        double c = b - ratio * (b - a);
        double d = a + ratio * (b - a);
        double fc = exp_fixed_x_objective(c, input_x, input_y, input_z, weight_y, weight_z);
        double fd = exp_fixed_x_objective(d, input_x, input_y, input_z, weight_y, weight_z);
        for (int iteration = 0; iteration < 100; ++iteration)
        {
            if (fc <= fd)
            {
                b = d;
                d = c;
                fd = fc;
                c = b - ratio * (b - a);
                fc = exp_fixed_x_objective(c, input_x, input_y, input_z, weight_y, weight_z);
            }
            else
            {
                a = c;
                c = d;
                fc = fd;
                d = a + ratio * (b - a);
                fd = exp_fixed_x_objective(d, input_x, input_y, input_z, weight_y, weight_z);
            }
        }
        output_y = 0.5 * (a + b);
        output_z = exp_cone_boundary(input_x, output_y);
        if (input_x <= 0.0)
        {
            double closure_z = fmax(input_z, 0.0);
            double closure_objective =
                weight_y * input_y * input_y + weight_z * (closure_z - input_z) * (closure_z - input_z);
            double smooth_objective = exp_fixed_x_objective(output_y, input_x, input_y, input_z, weight_y, weight_z);
            if (closure_objective <= smooth_objective)
            {
                output_y = 0.0;
                output_z = closure_z;
            }
        }
    }
    else if (fixed_z)
    {
        if (input_z == 0.0)
        {
            output_x = fmin(input_x, 0.0);
            output_y = 0.0;
        }
        else
        {
            double scale = 1.0 + fabs(input_x) + fabs(input_y) + input_z;
            double lo = 0.0;
            double hi = scale;
            double previous = exp_fixed_z_objective(0.5 * hi, input_z, input_x, input_y, weight_x, weight_y);
            double current = exp_fixed_z_objective(hi, input_z, input_x, input_y, weight_x, weight_y);
            for (int expansion = 0; expansion < 80 && current < previous && hi < DBL_MAX / 4.0; ++expansion)
            {
                previous = current;
                hi *= 2.0;
                current = exp_fixed_z_objective(hi, input_z, input_x, input_y, weight_x, weight_y);
            }
            const double ratio = 0.6180339887498948482;
            double a = lo;
            double b = hi;
            double c = b - ratio * (b - a);
            double d = a + ratio * (b - a);
            double fc = exp_fixed_z_objective(c, input_z, input_x, input_y, weight_x, weight_y);
            double fd = exp_fixed_z_objective(d, input_z, input_x, input_y, weight_x, weight_y);
            for (int iteration = 0; iteration < 100; ++iteration)
            {
                if (fc <= fd)
                {
                    b = d;
                    d = c;
                    fd = fc;
                    c = b - ratio * (b - a);
                    fc = exp_fixed_z_objective(c, input_z, input_x, input_y, weight_x, weight_y);
                }
                else
                {
                    a = c;
                    c = d;
                    fc = fd;
                    d = a + ratio * (b - a);
                    fd = exp_fixed_z_objective(d, input_z, input_x, input_y, weight_x, weight_y);
                }
            }
            output_y = 0.5 * (a + b);
            output_x = output_y > 0.0 ? output_y * (log(input_z) - log(output_y)) : 0.0;
            double closure_x = fmin(input_x, 0.0);
            double closure_objective =
                weight_x * (closure_x - input_x) * (closure_x - input_x) + weight_y * input_y * input_y;
            double smooth_objective = exp_fixed_z_objective(output_y, input_z, input_x, input_y, weight_x, weight_y);
            if (closure_objective <= smooth_objective)
            {
                output_x = closure_x;
                output_y = 0.0;
            }
        }
    }

    if (!fixed_x)
        point[start + 0] = output_x * rescaling[start + 0];
    if (!fixed_y)
        point[start + 1] = output_y * rescaling[start + 1];
    if (!fixed_z)
        point[start + 2] = output_z * rescaling[start + 2];
}

__global__ void project_exp_cone_kernel(double *__restrict__ primal_solution,
                                        const double *__restrict__ variable_rescaling,
                                        double *__restrict__ warm_start,
                                        const int *__restrict__ start_idx,
                                        const int *__restrict__ v_dim,
                                        const char *__restrict__ is_fixed,
                                        int num_blocks)
{
    (void)v_dim;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int s_idx = start_idx[blk];
    double r1 = primal_solution[s_idx + 0];
    double r2 = primal_solution[s_idx + 1];
    double r3 = primal_solution[s_idx + 2];

    double d1 = variable_rescaling[s_idx + 0];
    double d2 = variable_rescaling[s_idx + 1];
    double d3 = variable_rescaling[s_idx + 2];

    if (cone_section_has_fixed(is_fixed, s_idx, 3))
    {
        project_exp_cone_section(primal_solution, variable_rescaling, NULL, 0.0, warm_start + blk, s_idx, is_fixed);
        return;
    }

    double xo, yo, zo;
    project_exp_cone_point(r1, r2, r3, d1, d2, d3, &xo, &yo, &zo);

    primal_solution[s_idx + 0] = xo;
    primal_solution[s_idx + 1] = yo;
    primal_solution[s_idx + 2] = zo;
}

__global__ void compute_cone_dual_residual_exp_kernel(double *__restrict__ dual_residual,
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
    (void)v_dim;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int s_idx = start_idx[blk];
    double r1 = objective_vector[s_idx + 0] - dual_product[s_idx + 0];
    double r2 = objective_vector[s_idx + 1] - dual_product[s_idx + 1];
    double r3 = objective_vector[s_idx + 2] - dual_product[s_idx + 2];

    if (cone_section_has_fixed(is_fixed, s_idx, 3))
    {
        const double residual[3] = {r1, r2, r3};
        for (int slot = 0; slot < 3; ++slot)
        {
            int index = s_idx + slot;
            dual_residual[index] = is_fixed[index] ? primal_solution[index] : primal_solution[index] - residual[slot];
        }
        project_exp_cone_section(dual_residual, variable_rescaling, NULL, 0.0, warm_start + blk, s_idx, is_fixed);
        for (int slot = 0; slot < 3; ++slot)
        {
            int index = s_idx + slot;
            dual_residual[index] =
                is_fixed[index] ? 0.0 : (primal_solution[index] - dual_residual[index]) * variable_rescaling[index];
        }
        complementarity_residual[blk] = 0.0;
        return;
    }

    double d1 = 1.0 / variable_rescaling[s_idx + 0];
    double d2 = 1.0 / variable_rescaling[s_idx + 1];
    double d3 = 1.0 / variable_rescaling[s_idx + 2];

    /* Moreau: dist(r, K_exp^*) = ||-proj_{K_exp}(-r)|| with inverse-scaled d. */
    double xo, yo, zo;
    project_exp_cone_point(-r1, -r2, -r3, d1, d2, d3, &xo, &yo, &zo);

    dual_residual[s_idx + 0] = -xo * variable_rescaling[s_idx + 0];
    dual_residual[s_idx + 1] = -yo * variable_rescaling[s_idx + 1];
    dual_residual[s_idx + 2] = -zo * variable_rescaling[s_idx + 2];
}

/* 3-dim alpha-power cone K_a = {(x,y,z) : x >= 0, y >= 0, x^a * y^(1-a) >= |z|}.
   Weighted projection: solves
     min_{(x,y,z) in K_a}  0.5 * ( wx*(x-rx)^2 + wy*(y-ry)^2 + wz*(z-rz)^2 )
   with wi > 0. In-cone test is metric-independent; opposite-cone test is not.
   Bisection on rho = |z_proj| in [0, |r_z|] using KKT-derived formulas
     x(rho) = 0.5 (rx + sqrt(rx^2 + 4 a (wz/wx) rho (|rz|-rho)))
     y(rho) = 0.5 (ry + sqrt(ry^2 + 4 (1-a) (wz/wy) rho (|rz|-rho)))
     G(rho) = x^a y^(1-a) - rho. */
__device__ static inline double positive_quadratic_root(double r, double q)
{
    if (!(q > 0.0))
        return fmax(r, 0.0);
    double disc = hypot(r, 2.0 * sqrt(q));
    if (r >= 0.0)
        return 0.5 * (r + disc);
    return (2.0 * q) / (disc - r);
}

__device__ static inline void project_power_cone_point_normalized(
    double rx, double ry, double rz, double wx, double wy, double wz, double alpha, double *xo, double *yo, double *zo)
{
    double abs_rz = fabs(rz);
    double sgn_rz = (rz >= 0.0) ? 1.0 : -1.0;
    double om = 1.0 - alpha;

    if (abs_rz == 0.0)
    {
        *xo = fmax(rx, 0.0);
        *yo = fmax(ry, 0.0);
        *zo = 0.0;
        return;
    }

    if (rx > 0.0 && ry > 0.0)
    {
        if (alpha * log(rx) + om * log(ry) >= log(abs_rz))
        {
            *xo = rx;
            *yo = ry;
            *zo = rz;
            return;
        }
    }

    /* Opposite cone under weighted inner product:
       proj^w(r) = 0 iff (wx*rx, wy*ry, wz*rz) in -K_a^*, i.e.,
         (-wx*rx)/a)^a * ((-wy*ry)/(1-a))^(1-a) >= wz*|rz|,  rx <= 0, ry <= 0. */
    if (rx <= 0.0 && ry <= 0.0)
    {
        double u = (rx < 0.0) ? (-wx * rx) / alpha : 0.0;
        double v = (ry < 0.0) ? (-wy * ry) / om : 0.0;
        if (u > 0.0 && v > 0.0 && alpha * log(u) + om * log(v) >= log(wz) + log(abs_rz))
        {
            *xo = 0.0;
            *yo = 0.0;
            *zo = 0.0;
            return;
        }
    }

    double c_x = 4.0 * alpha * (wz / wx);
    double c_y = 4.0 * om * (wz / wy);

    /*
     * Bisect in log(rho).  When one input axis is negative and alpha is close
     * to an endpoint, the positive root can be many orders of magnitude below
     * |r_z|.  A linear relative floor would then converge to an infeasible
     * point instead of the nonzero root.
     */
    double lo = log(DBL_MIN);
    double hi = log(abs_rz);
    if (!(hi > lo))
    {
        *xo = fmax(rx, 0.0);
        *yo = fmax(ry, 0.0);
        *zo = 0.0;
        return;
    }

    for (int it = 0; it < 80; ++it)
    {
        double log_rho = lo + 0.5 * (hi - lo);
        double rho = exp(log_rho);
        double x = positive_quadratic_root(rx, 0.25 * c_x * rho * (abs_rz - rho));
        double y = positive_quadratic_root(ry, 0.25 * c_y * rho * (abs_rz - rho));
        bool above_boundary = x > 0.0 && y > 0.0 && alpha * log(x) + om * log(y) > log_rho;
        if (above_boundary)
            lo = log_rho;
        else
            hi = log_rho;
    }
    double rho = exp(lo + 0.5 * (hi - lo));
    *xo = positive_quadratic_root(rx, 0.25 * c_x * rho * (abs_rz - rho));
    *yo = positive_quadratic_root(ry, 0.25 * c_y * rho * (abs_rz - rho));
    double log_bound = alpha * log(*xo) + om * log(*yo);
    *zo = sgn_rz * fmin(rho, exp(log_bound));
}

__device__ static inline void project_power_cone_point(
    double rx, double ry, double rz, double wx, double wy, double wz, double alpha, double *xo, double *yo, double *zo)
{
    /* The cone and weighted projection are positively homogeneous.  Normalize
       the point so products such as rho * (|r_z| - rho) cannot overflow. */
    double scale = fmax(fabs(rx), fmax(fabs(ry), fabs(rz)));
    if (!(scale > 0.0) || !isfinite(scale))
    {
        project_power_cone_point_normalized(rx, ry, rz, wx, wy, wz, alpha, xo, yo, zo);
        return;
    }

    double xn, yn, zn;
    project_power_cone_point_normalized(rx / scale, ry / scale, rz / scale, wx, wy, wz, alpha, &xn, &yn, &zn);
    *xo = xn * scale;
    *yo = yn * scale;
    *zo = zn * scale;
}

/* Project x,y while z is fixed. The active boundary is x^a y^(1-a) = |z|. */
__device__ static inline double
power_xy_log_boundary(double lambda, double rx, double ry, double wx, double wy, double alpha)
{
    double om = 1.0 - alpha;
    double x = positive_quadratic_root(rx, (lambda / wx) * alpha);
    double y = positive_quadratic_root(ry, (lambda / wy) * om);
    if (!(x > 0.0) || !(y > 0.0))
        return -INFINITY;
    return alpha * log(x) + om * log(y);
}

__device__ static inline void project_power_xy_fixed_z_normalized(
    double rx, double ry, double fixed_z, double wx, double wy, double alpha, double *xo, double *yo)
{
    double c = fabs(fixed_z);
    double om = 1.0 - alpha;
    if (c == 0.0)
    {
        *xo = fmax(rx, 0.0);
        *yo = fmax(ry, 0.0);
        return;
    }

    if (rx > 0.0 && ry > 0.0 && alpha * log(rx) + om * log(ry) >= log(c))
    {
        *xo = rx;
        *yo = ry;
        return;
    }

    double target = log(c);
    double hi = fmin(wx, wy);
    if (!(hi > 0.0) || !isfinite(hi))
        hi = 1.0;
    for (int it = 0; it < 2048; ++it)
    {
        double log_boundary = power_xy_log_boundary(hi, rx, ry, wx, wy, alpha);
        if (log_boundary >= target || isnan(log_boundary))
            break;
        if (hi >= 0.5 * DBL_MAX)
        {
            hi = DBL_MAX;
            break;
        }
        hi *= 2.0;
    }

    double lambda;
    double floor_log_boundary = power_xy_log_boundary(DBL_MIN, rx, ry, wx, wy, alpha);
    if (floor_log_boundary >= target)
    {
        double lo = 0.0;
        double floor_hi = DBL_MIN;
        for (int it = 0; it < 80; ++it)
        {
            double candidate = lo + 0.5 * (floor_hi - lo);
            if (power_xy_log_boundary(candidate, rx, ry, wx, wy, alpha) < target)
                lo = candidate;
            else
                floor_hi = candidate;
        }
        lambda = lo + 0.5 * (floor_hi - lo);
    }
    else
    {
        double log_lo = log(DBL_MIN);
        double log_hi = log(hi);
        for (int it = 0; it < 96; ++it)
        {
            double log_lambda = log_lo + 0.5 * (log_hi - log_lo);
            double candidate = exp(log_lambda);
            if (power_xy_log_boundary(candidate, rx, ry, wx, wy, alpha) < target)
                log_lo = log_lambda;
            else
                log_hi = log_lambda;
        }
        lambda = exp(log_lo + 0.5 * (log_hi - log_lo));
    }
    *xo = positive_quadratic_root(rx, (lambda / wx) * alpha);
    *yo = positive_quadratic_root(ry, (lambda / wy) * om);
}

__device__ static inline void project_power_xy_fixed_z(
    double rx, double ry, double fixed_z, double wx, double wy, double alpha, double *xo, double *yo)
{
    double scale = fmax(fabs(rx), fmax(fabs(ry), fabs(fixed_z)));
    if (!(scale > 0.0) || !isfinite(scale))
    {
        project_power_xy_fixed_z_normalized(rx, ry, fixed_z, wx, wy, alpha, xo, yo);
        return;
    }

    double xn, yn;
    project_power_xy_fixed_z_normalized(rx / scale, ry / scale, fixed_z / scale, wx, wy, alpha, &xn, &yn);
    *xo = xn * scale;
    *yo = yn * scale;
}

/* With one nonnegative axis fixed, project the other axis and z onto
   |z| <= fixed_axis^fixed_exp * other^other_exp. On the active boundary,
   direct bisection in other is stable even when the KKT multiplier is tiny. */
__device__ static inline double power_exp_from_log(double log_value)
{
    if (log_value >= log(DBL_MAX))
        return INFINITY;
    if (log_value <= log(DBL_MIN))
        return 0.0;
    return exp(log_value);
}

__device__ static inline double power_section_derivative(
    double other, double r_other, double abs_rz, double w_other, double wz, double log_coefficient, double other_exp)
{
    if (!(other > 0.0))
        return -INFINITY;

    double log_other = log(other);
    double bound = power_exp_from_log(log_coefficient + other_exp * log_other);
    double slope = power_exp_from_log(log_coefficient + log(other_exp) + (other_exp - 1.0) * log_other);
    double linear_term = w_other * (other - r_other);
    double gap = bound - abs_rz;
    if (gap == 0.0 || slope == 0.0)
        return linear_term;
    if (!isfinite(slope))
        return copysign(INFINITY, gap);
    return linear_term + wz * gap * slope;
}

__device__ static inline void project_power_section_fixed_axis_normalized(double fixed_axis,
                                                                          double r_other,
                                                                          double rz,
                                                                          double w_other,
                                                                          double wz,
                                                                          double fixed_exp,
                                                                          double other_exp,
                                                                          double *other_out,
                                                                          double *z_out)
{
    double abs_rz = fabs(rz);
    if (!(fixed_axis > 0.0) || abs_rz == 0.0)
    {
        *other_out = fmax(r_other, 0.0);
        *z_out = 0.0;
        return;
    }
    double log_coefficient = fixed_exp * log(fixed_axis);

    if (r_other > 0.0 && log_coefficient + other_exp * log(r_other) >= log(abs_rz))
    {
        *other_out = r_other;
        *z_out = rz;
        return;
    }

    double log_feasible_other = (log(abs_rz) - log_coefficient) / other_exp;
    double feasible_other = power_exp_from_log(log_feasible_other);
    if (feasible_other == 0.0)
    {
        *other_out = 0.0;
        *z_out = 0.0;
        return;
    }

    double lo = 0.0;
    double hi = fmax(1.0, fmax(r_other, 0.0));
    if (isfinite(feasible_other))
        hi = fmin(hi, feasible_other);
    for (int it = 0; it < 1024; ++it)
    {
        double derivative = power_section_derivative(hi, r_other, abs_rz, w_other, wz, log_coefficient, other_exp);
        if (!(derivative < 0.0))
            break;
        if (isfinite(feasible_other) && hi >= feasible_other)
            break;
        double next_hi = hi * 2.0;
        if (!isfinite(next_hi))
        {
            hi = isfinite(feasible_other) ? feasible_other : DBL_MAX;
            break;
        }
        hi = isfinite(feasible_other) ? fmin(next_hi, feasible_other) : next_hi;
    }

    for (int it = 0; it < 80; ++it)
    {
        double other = lo + 0.5 * (hi - lo);
        if (other == 0.0)
            break;
        double derivative = power_section_derivative(other, r_other, abs_rz, w_other, wz, log_coefficient, other_exp);
        if (derivative < 0.0)
            lo = other;
        else
            hi = other;
    }
    double other = lo + 0.5 * (hi - lo);
    double projected_abs_z = other > 0.0 ? power_exp_from_log(log_coefficient + other_exp * log(other)) : 0.0;
    *other_out = other;
    *z_out = copysign(fmin(projected_abs_z, abs_rz), rz);
}

__device__ static inline void project_power_section_fixed_axis(double fixed_axis,
                                                               double r_other,
                                                               double rz,
                                                               double w_other,
                                                               double wz,
                                                               double fixed_exp,
                                                               double other_exp,
                                                               double *other_out,
                                                               double *z_out)
{
    double scale = fmax(fixed_axis, fmax(fabs(r_other), fabs(rz)));
    if (!(scale > 0.0) || !isfinite(scale))
    {
        project_power_section_fixed_axis_normalized(
            fixed_axis, r_other, rz, w_other, wz, fixed_exp, other_exp, other_out, z_out);
        return;
    }

    double normalized_other, normalized_z;
    project_power_section_fixed_axis_normalized(fixed_axis / scale,
                                                r_other / scale,
                                                rz / scale,
                                                w_other,
                                                wz,
                                                fixed_exp,
                                                other_exp,
                                                &normalized_other,
                                                &normalized_z);
    *other_out = normalized_other * scale;
    *z_out = normalized_z * scale;
}

__device__ static inline void project_power_cone_point_with_fixed(double rx,
                                                                  double ry,
                                                                  double rz,
                                                                  double wx,
                                                                  double wy,
                                                                  double wz,
                                                                  double alpha,
                                                                  bool fixed_x,
                                                                  bool fixed_y,
                                                                  bool fixed_z,
                                                                  double *xo,
                                                                  double *yo,
                                                                  double *zo)
{
    double om = 1.0 - alpha;
    *xo = rx;
    *yo = ry;
    *zo = rz;

    if (!fixed_x && !fixed_y && !fixed_z)
    {
        project_power_cone_point(rx, ry, rz, wx, wy, wz, alpha, xo, yo, zo);
        return;
    }

    if (fixed_z)
    {
        if (fixed_x && fixed_y)
            return;
        if (fixed_x)
        {
            double lower = fabs(rz) == 0.0 ? 0.0 : exp((log(fabs(rz)) - alpha * log(rx)) / om);
            *yo = fmax(ry, lower);
            return;
        }
        if (fixed_y)
        {
            double lower = fabs(rz) == 0.0 ? 0.0 : exp((log(fabs(rz)) - om * log(ry)) / alpha);
            *xo = fmax(rx, lower);
            return;
        }
        project_power_xy_fixed_z(rx, ry, rz, wx, wy, alpha, xo, yo);
        return;
    }

    if (fixed_x && fixed_y)
    {
        double bound = 0.0;
        if (rx > 0.0 && ry > 0.0)
        {
            double log_bound = alpha * log(rx) + om * log(ry);
            bound = log_bound < log(DBL_MAX) ? exp(log_bound) : INFINITY;
        }
        *zo = fmax(-bound, fmin(rz, bound));
        return;
    }
    if (fixed_x)
    {
        project_power_section_fixed_axis(rx, ry, rz, wy, wz, alpha, om, yo, zo);
        return;
    }
    if (fixed_y)
    {
        project_power_section_fixed_axis(ry, rx, rz, wx, wz, om, alpha, xo, zo);
        return;
    }
}

__global__ void project_power_cone_kernel(double *__restrict__ primal_solution,
                                          const double *__restrict__ variable_rescaling,
                                          double *__restrict__ warm_start,
                                          const int *__restrict__ start_idx,
                                          const int *__restrict__ v_dim,
                                          const double *__restrict__ power_alpha,
                                          const char *__restrict__ is_fixed,
                                          int num_blocks)
{
    (void)v_dim;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int s_idx = start_idx[blk];
    double r1 = primal_solution[s_idx + 0];
    double r2 = primal_solution[s_idx + 1];
    double r3 = primal_solution[s_idx + 2];

    double d1 = variable_rescaling[s_idx + 0];
    double d2 = variable_rescaling[s_idx + 1];
    double d3 = variable_rescaling[s_idx + 2];
    double alpha = power_alpha[blk];

    /* Prox in scaled space with metric I equals prox in actual space with metric diag(d^2). */
    double rx = r1 / d1;
    double ry = r2 / d2;
    double rz = r3 / d3;
    double wx = d1 * d1;
    double wy = d2 * d2;
    double wz = d3 * d3;
    double xo, yo, zo;
    project_power_cone_point_with_fixed(rx,
                                        ry,
                                        rz,
                                        wx,
                                        wy,
                                        wz,
                                        alpha,
                                        is_fixed && is_fixed[s_idx + 0],
                                        is_fixed && is_fixed[s_idx + 1],
                                        is_fixed && is_fixed[s_idx + 2],
                                        &xo,
                                        &yo,
                                        &zo);
    primal_solution[s_idx + 0] = xo * d1;
    primal_solution[s_idx + 1] = yo * d2;
    primal_solution[s_idx + 2] = zo * d3;
}

__global__ void compute_cone_dual_residual_power_kernel(double *__restrict__ dual_residual,
                                                        double *__restrict__ complementarity_residual,
                                                        const double *__restrict__ objective_vector,
                                                        const double *__restrict__ dual_product,
                                                        const double *__restrict__ variable_rescaling,
                                                        const double *__restrict__ primal_solution,
                                                        double *__restrict__ warm_start,
                                                        const int *__restrict__ start_idx,
                                                        const int *__restrict__ v_dim,
                                                        const double *__restrict__ power_alpha,
                                                        const char *__restrict__ is_fixed,
                                                        int num_blocks)
{
    (void)warm_start;
    (void)v_dim;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int s_idx = start_idx[blk];
    double r1 = objective_vector[s_idx + 0] - dual_product[s_idx + 0];
    double r2 = objective_vector[s_idx + 1] - dual_product[s_idx + 1];
    double r3 = objective_vector[s_idx + 2] - dual_product[s_idx + 2];
    double alpha = power_alpha[blk];

    bool fixed_x = is_fixed && is_fixed[s_idx + 0];
    bool fixed_y = is_fixed && is_fixed[s_idx + 1];
    bool fixed_z = is_fixed && is_fixed[s_idx + 2];
    if (fixed_x || fixed_y || fixed_z)
    {
        double d1 = variable_rescaling[s_idx + 0];
        double d2 = variable_rescaling[s_idx + 1];
        double d3 = variable_rescaling[s_idx + 2];
        double x = primal_solution[s_idx + 0] / d1;
        double y = primal_solution[s_idx + 1] / d2;
        double z = primal_solution[s_idx + 2] / d3;
        double q1 = r1 * d1;
        double q2 = r2 * d2;
        double q3 = r3 * d3;

        dual_residual[s_idx + 0] = fixed_x ? 0.0 : q1;
        dual_residual[s_idx + 1] = fixed_y ? 0.0 : q2;
        dual_residual[s_idx + 2] = fixed_z ? 0.0 : q3;
        if (fixed_x && fixed_y && fixed_z)
            return;

        double abs_z = fabs(z);
        double bound = 0.0;
        bool regular = x > 0.0 && y > 0.0 && isfinite(x) && isfinite(y) && isfinite(z);
        if (regular)
        {
            double log_bound = alpha * log(x) + (1.0 - alpha) * log(y);
            bound = exp(log_bound);
            regular = isfinite(bound) && bound > 0.0;
        }

        if (regular && abs_z > 0.0)
        {
            double normal[3] = {
                -alpha * bound / x,
                -(1.0 - alpha) * bound / y,
                copysign(1.0, z),
            };
            double q[3] = {q1, q2, q3};
            bool fixed[3] = {fixed_x, fixed_y, fixed_z};
            double normal_scale = 0.0;
            for (int i = 0; i < 3; ++i)
            {
                if (!fixed[i])
                    normal_scale = fmax(normal_scale, fabs(normal[i]));
            }
            if (!(normal_scale > 0.0) || !isfinite(normal_scale))
            {
                regular = false;
            }

            double dot = 0.0;
            double normal2 = 0.0;
            for (int i = 0; i < 3 && regular; ++i)
            {
                if (!fixed[i])
                {
                    double scaled_normal = normal[i] / normal_scale;
                    dot += q[i] * scaled_normal;
                    normal2 += scaled_normal * scaled_normal;
                }
            }
            if (regular)
            {
                double scaled_lambda = (dot < 0.0 && normal2 > 0.0) ? -dot / normal2 : 0.0;
                double lambda = scaled_lambda / normal_scale;
                for (int i = 0; i < 3; ++i)
                {
                    if (!fixed[i])
                        dual_residual[s_idx + i] = q[i] + scaled_lambda * (normal[i] / normal_scale);
                }
                double slack_scale = fmax(1.0, fmax(bound, abs_z));
                double complementarity = lambda * (fmax(bound - abs_z, 0.0) / slack_scale);
                complementarity_residual[blk] = complementarity;
                return;
            }
        }

        if (regular)
            return;

        /* Degenerate axes are nonsmooth.  A unit metric projection supplies a
           scale-independent KKT guard without changing the adaptive mapping. */
        double rx = x - (fixed_x ? 0.0 : r1 / d1);
        double ry = y - (fixed_y ? 0.0 : r2 / d2);
        double rz = z - (fixed_z ? 0.0 : r3 / d3);
        double xo, yo, zo;
        project_power_cone_point_with_fixed(
            rx, ry, rz, d1 * d1, d2 * d2, d3 * d3, alpha, fixed_x, fixed_y, fixed_z, &xo, &yo, &zo);
        if (!fixed_x)
            dual_residual[s_idx + 0] = (x - xo) * d1 * d1;
        if (!fixed_y)
            dual_residual[s_idx + 1] = (y - yo) * d2 * d2;
        if (!fixed_z)
            dual_residual[s_idx + 2] = (z - zo) * d3 * d3;
        return;
    }

    double vr1 = variable_rescaling[s_idx + 0];
    double vr2 = variable_rescaling[s_idx + 1];
    double vr3 = variable_rescaling[s_idx + 2];

    /* Moreau via primal projection: dual_res = -Proj_K(-r * vr). */
    double xo, yo, zo;
    project_power_cone_point(-r1 * vr1, -r2 * vr2, -r3 * vr3, 1.0, 1.0, 1.0, alpha, &xo, &yo, &zo);

    dual_residual[s_idx + 0] = -xo;
    dual_residual[s_idx + 1] = -yo;
    dual_residual[s_idx + 2] = -zo;
}

__global__ void compute_power_cone_primal_violation_kernel(double *__restrict__ absolute_violation,
                                                           double *__restrict__ relative_violation,
                                                           const double *__restrict__ primal_solution,
                                                           const double *__restrict__ variable_rescaling,
                                                           const int *__restrict__ start_idx,
                                                           const double *__restrict__ power_alpha,
                                                           double homogeneous_scale,
                                                           int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int start = start_idx[blk];
    double x = primal_solution[start + 0] / variable_rescaling[start + 0];
    double y = primal_solution[start + 1] / variable_rescaling[start + 1];
    double z = primal_solution[start + 2] / variable_rescaling[start + 2];
    if (!isfinite(x) || !isfinite(y) || !isfinite(z))
    {
        absolute_violation[blk] = INFINITY;
        relative_violation[blk] = INFINITY;
        return;
    }
    double violation = fmax(-x, -y);
    double abs_z = fabs(z);
    if (abs_z > 0.0)
    {
        double bound = 0.0;
        if (x > 0.0 && y > 0.0)
        {
            double alpha = power_alpha[blk];
            double log_bound = alpha * log(x) + (1.0 - alpha) * log(y);
            double log_abs_z = log(abs_z);
            double roundoff_tolerance = 64.0 * DBL_EPSILON * (1.0 + fabs(log_bound) + fabs(log_abs_z));
            if (log_bound + roundoff_tolerance >= log_abs_z)
            {
                violation = fmax(violation, 0.0);
                absolute_violation[blk] = violation;
                relative_violation[blk] = violation / (homogeneous_scale + fmax(fabs(x), fmax(fabs(y), abs_z)));
                return;
            }
            bound = exp(log_bound);
        }
        violation = fmax(violation, abs_z - bound);
    }
    violation = fmax(violation, 0.0);
    absolute_violation[blk] = violation;
    relative_violation[blk] = violation / (homogeneous_scale + fmax(fabs(x), fmax(fabs(y), abs_z)));
}

__global__ void project_power_cone_diag_q_kernel(double *__restrict__ pdhg_primal,
                                                 double *__restrict__ reflected_primal,
                                                 const double *__restrict__ current_primal,
                                                 const double *__restrict__ variable_rescaling,
                                                 const double *__restrict__ Q_diag,
                                                 double tau,
                                                 double *__restrict__ warm_start,
                                                 const int *__restrict__ start_idx,
                                                 const int *__restrict__ v_dim,
                                                 const double *__restrict__ power_alpha,
                                                 const char *__restrict__ is_fixed,
                                                 int num_blocks)
{
    (void)warm_start;
    (void)v_dim;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int s_idx = start_idx[blk];
    double r1 = pdhg_primal[s_idx + 0];
    double r2 = pdhg_primal[s_idx + 1];
    double r3 = pdhg_primal[s_idx + 2];

    double d1 = variable_rescaling[s_idx + 0];
    double d2 = variable_rescaling[s_idx + 1];
    double d3 = variable_rescaling[s_idx + 2];
    double alpha = power_alpha[blk];

    /* Effective weight in actual space: omega_i = (1 + tau*Q_ii) * d_i^2. */
    double w1 = 1.0 + tau * Q_diag[s_idx + 0];
    double w2 = 1.0 + tau * Q_diag[s_idx + 1];
    double w3 = 1.0 + tau * Q_diag[s_idx + 2];
    double om_x = w1 * d1 * d1;
    double om_y = w2 * d2 * d2;
    double om_z = w3 * d3 * d3;
    double rx = r1 / d1;
    double ry = r2 / d2;
    double rz = r3 / d3;
    double xo, yo, zo;
    project_power_cone_point_with_fixed(rx,
                                        ry,
                                        rz,
                                        om_x,
                                        om_y,
                                        om_z,
                                        alpha,
                                        is_fixed && is_fixed[s_idx + 0],
                                        is_fixed && is_fixed[s_idx + 1],
                                        is_fixed && is_fixed[s_idx + 2],
                                        &xo,
                                        &yo,
                                        &zo);
    pdhg_primal[s_idx + 0] = xo * d1;
    pdhg_primal[s_idx + 1] = yo * d2;
    pdhg_primal[s_idx + 2] = zo * d3;
    for (int m = 0; m < 3; ++m)
    {
        int idx = s_idx + m;
        reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
    }
}

__global__ void set_cone_dual_slack_kernel(double *__restrict__ dual_slack,
                                           const double *__restrict__ objective_vector,
                                           const double *__restrict__ dual_product,
                                           const int *__restrict__ start_idx,
                                           const int *__restrict__ v_dim,
                                           int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;
    int start = start_idx[blk];
    int k = v_dim[blk];
    for (int m = 0; m < k + 2; ++m)
    {
        int idx = start + m;
        dual_slack[idx] = objective_vector[idx] - dual_product[idx];
    }
}

__global__ void set_cone_dual_slack_grid_kernel(double *__restrict__ dual_slack,
                                                const double *__restrict__ objective_vector,
                                                const double *__restrict__ dual_product,
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
    int n = v_dim[cone] + 2;
    for (int m = part * blockDim.x + threadIdx.x; m < n; m += blocks_per_cone * blockDim.x)
    {
        int idx = start + m;
        dual_slack[idx] = objective_vector[idx] - dual_product[idx];
    }
}

__global__ void set_cone_dual_slack_warp_kernel(double *__restrict__ dual_slack,
                                                const double *__restrict__ objective_vector,
                                                const double *__restrict__ dual_product,
                                                const int *__restrict__ start_idx,
                                                const int *__restrict__ v_dim,
                                                int num_cones)
{
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int cone = global_thread >> 5;
    if (cone >= num_cones)
        return;

    int lane = global_thread & 31;
    int start = start_idx[cone];
    int n = v_dim[cone] + 2;
    for (int m = lane; m < n; m += 32)
    {
        int idx = start + m;
        dual_slack[idx] = objective_vector[idx] - dual_product[idx];
    }
}

__global__ void recompute_reflected_at_cone_kernel(double *__restrict__ reflected_primal,
                                                   const double *__restrict__ pdhg_primal,
                                                   const double *__restrict__ current_primal,
                                                   const int *__restrict__ start_idx,
                                                   const int *__restrict__ v_dim,
                                                   int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;
    int start = start_idx[blk];
    int k = v_dim[blk];
    for (int m = 0; m < k + 2; ++m)
    {
        int idx = start + m;
        reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
    }
}

__global__ void recompute_reflected_at_cone_warp_kernel(double *__restrict__ reflected_primal,
                                                        const double *__restrict__ pdhg_primal,
                                                        const double *__restrict__ current_primal,
                                                        const int *__restrict__ start_idx,
                                                        const int *__restrict__ v_dim,
                                                        int num_cones)
{
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int cone = global_thread >> 5;
    if (cone >= num_cones)
        return;

    int lane = global_thread & 31;
    int start = start_idx[cone];
    int n = v_dim[cone] + 2;
    for (int m = lane; m < n; m += 32)
    {
        int idx = start + m;
        reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
    }
}

__global__ void recompute_reflected_at_cone_grid_kernel(double *__restrict__ reflected_primal,
                                                        const double *__restrict__ pdhg_primal,
                                                        const double *__restrict__ current_primal,
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
    int n = v_dim[cone] + 2;
    for (int m = part * blockDim.x + threadIdx.x; m < n; m += blocks_per_cone * blockDim.x)
    {
        int idx = start + m;
        reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
    }
}

/* Weighted prox onto D K_soc; effective rescaling e_i = sqrt(w_i) d_i, w_i = 1 + tau Q_i. */
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
__global__ void project_exp_cone_diag_q_kernel(double *__restrict__ pdhg_primal,
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
    (void)v_dim;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int s_idx = start_idx[blk];

    if (cone_section_has_fixed(is_fixed, s_idx, 3))
    {
        project_exp_cone_section(pdhg_primal, variable_rescaling, Q_diag, tau, warm_start + blk, s_idx, is_fixed);
        for (int slot = 0; slot < 3; ++slot)
        {
            int index = s_idx + slot;
            reflected_primal[index] = 2.0 * pdhg_primal[index] - current_primal[index];
        }
        return;
    }

    double r1 = pdhg_primal[s_idx + 0];
    double r2 = pdhg_primal[s_idx + 1];
    double r3 = pdhg_primal[s_idx + 2];

    double d1 = variable_rescaling[s_idx + 0];
    double d2 = variable_rescaling[s_idx + 1];
    double d3 = variable_rescaling[s_idx + 2];

    double w1 = 1.0 + tau * Q_diag[s_idx + 0];
    double w2 = 1.0 + tau * Q_diag[s_idx + 1];
    double w3 = 1.0 + tau * Q_diag[s_idx + 2];

    /* Clamp guards against negative drift in Q_diag invalidating sqrt(w_i). */
    if (!(w1 > 0.0))
        w1 = 1.0;
    if (!(w2 > 0.0))
        w2 = 1.0;
    if (!(w3 > 0.0))
        w3 = 1.0;

    double sw1 = sqrt(w1);
    double sw2 = sqrt(w2);
    double sw3 = sqrt(w3);

    double e1 = sw1 * d1;
    double e2 = sw2 * d2;
    double e3 = sw3 * d3;

    double u1 = sw1 * r1;
    double u2 = sw2 * r2;
    double u3 = sw3 * r3;
    double y1_out, y2_out, y3_out;
    project_exp_cone_point(u1, u2, u3, e1, e2, e3, &y1_out, &y2_out, &y3_out);
    double x1 = y1_out / sw1;
    double x2 = y2_out / sw2;
    double x3 = y3_out / sw3;

    pdhg_primal[s_idx + 0] = x1;
    pdhg_primal[s_idx + 1] = x2;
    pdhg_primal[s_idx + 2] = x3;

    reflected_primal[s_idx + 0] = 2.0 * x1 - current_primal[s_idx + 0];
    reflected_primal[s_idx + 1] = 2.0 * x2 - current_primal[s_idx + 1];
    reflected_primal[s_idx + 2] = 2.0 * x3 - current_primal[s_idx + 2];
}

/* Direct (s,t) bisection in zeta = xi/sqrt(w_s w_t); alpha = sqrt(w_t/w_s) carries asymmetry. */
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
