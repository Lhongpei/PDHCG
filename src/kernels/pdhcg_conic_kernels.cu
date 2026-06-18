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

#include "pdhcg_kernels.cuh"
#include <cuda_runtime.h>
#include <math.h>

/* Conic projection and dual-residual kernels (split from pdhcg_kernels.cu). */

__global__ void project_rotated_soc_kernel(double *__restrict__ primal_solution,
                                           const double *__restrict__ variable_rescaling,
                                           double *__restrict__ warm_start,
                                           const int *__restrict__ start_idx,
                                           const int *__restrict__ v_dim,
                                           int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    const double INV_SQRT2 = 0.7071067811865475;

    int start = start_idx[blk];
    int k = v_dim[blk];
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
                                                  const double *__restrict__ objective_vector,
                                                  const double *__restrict__ dual_product,
                                                  const double *__restrict__ variable_rescaling,
                                                  double *__restrict__ warm_start,
                                                  const int *__restrict__ start_idx,
                                                  const int *__restrict__ v_dim,
                                                  int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    const double INV_SQRT2 = 0.7071067811865475;
    int start = start_idx[blk];
    int k = v_dim[blk];

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
                                            int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int start = start_idx[blk];
    int k = v_dim[blk];
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
                                                           const double *__restrict__ objective_vector,
                                                           const double *__restrict__ dual_product,
                                                           const double *__restrict__ variable_rescaling,
                                                           double *__restrict__ warm_start,
                                                           const int *__restrict__ start_idx,
                                                           const int *__restrict__ v_dim,
                                                           int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int start = start_idx[blk];
    int k = v_dim[blk];

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

__global__ void project_rotated_soc_warp_kernel(double *__restrict__ primal_solution,
                                                const double *__restrict__ variable_rescaling,
                                                double *__restrict__ warm_start,
                                                const int *__restrict__ start_idx,
                                                const int *__restrict__ v_dim,
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
                                                       const double *__restrict__ objective_vector,
                                                       const double *__restrict__ dual_product,
                                                       const double *__restrict__ variable_rescaling,
                                                       double *__restrict__ warm_start,
                                                       const int *__restrict__ start_idx,
                                                       const int *__restrict__ v_dim,
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
                                                                const double *__restrict__ objective_vector,
                                                                const double *__restrict__ dual_product,
                                                                const double *__restrict__ variable_rescaling,
                                                                double *__restrict__ warm_start,
                                                                const int *__restrict__ start_idx,
                                                                const int *__restrict__ v_dim,
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

/* Project (r1, r2, r3) onto D * K_exp with D = diag(d1, d2, d3).
   Reduces to standard projection onto K_exp when d1=d2=d3=1.
   Algorithm: change variables u = D^{-1} x to convert to weighted projection onto K_exp.
   With weights d_i^2, KKT gives a 1D Newton equation in rho = u_1/u_2 with parameters
   alpha = (d3/d1)^2, beta = (d3/d2)^2. */
__device__ static inline void project_exp_cone_point(
    double r1, double r2, double r3, double d1, double d2, double d3, double *xo, double *yo, double *zo)
{
    const double E_CONST = 2.718281828459045;
    double rr1 = r1 / d1, rr2 = r2 / d2, rr3 = r3 / d3;

    /* Case 1: r in D K_exp  <=>  D^{-1} r in K_exp. */
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

    /* Case 2: r in polar = -D^{-1} K_exp^*.
       -D r in K_exp^*: -d1 r1 < 0 (so r1>0), and  -(-d1 r1) exp(-d2 r2/(-d1 r1)) <= e * (-d3 r3)
                       i.e.  d1 r1 * exp(d2 r2/(d1 r1)) <= -e d3 r3. */
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

    /* Case 3: recession edge.  D K_exp contains {(c1, 0, c3) : c1 <= 0, c3 >= 0}
       (no scaling needed since y=0). Project x and z components, zero out y. */
    if (rr1 <= 0.0 && rr2 <= 0.0)
    {
        *xo = r1;
        *yo = 0.0;
        *zo = (rr3 < 0.0) ? 0.0 : r3;
        return;
    }

    /* Case 4: Newton iteration on f(rho)=0 with rho = u_1/u_2 (u = D^{-1} x). */
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
        /* Damp aggressive steps to prevent runaway divergence. */
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

    /* Robustness: if Newton failed to find a valid cone point, fall back to
       projection onto the recession edge {(x, 0, z) : x <= 0, z >= 0}. */
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

/* Project (rz0, rt0) onto the y-fixed cross-section of D K_exp with D = diag(d_r, d_y, d_t),
   y slot held at scaled value ry. In scaled coordinates the constraint is
       d_t * y_eff * exp((rz/d_r) / y_eff) <= rt,    y_eff = ry / d_y > 0.
   Parameterize the boundary by u = exp((rz/d_r) / y_eff) > 0:
       rz_s = d_r * y_eff * log(u),   rt_s = d_t * y_eff * u.
   The KKT stationarity for distance^2 = (rz - rz0)^2 + (rt - rt0)^2 gives
       f(u) = d_t^2 * y_eff * u^2 - d_t * rt0 * u + d_r^2 * y_eff * log(u) - d_r * rz0 = 0.
   We bracket [u_lo, u_hi] with f(u_lo) < 0, f(u_hi) > 0 and run Newton with bisection
   fallback. */
__device__ static inline void
project_2d_exp_persp(double rz0, double ry, double rt0, double d_r, double d_y, double d_t, double *rzo, double *rto)
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
        /* Degenerate: K_exp interior needs y > 0. Project to recession edge. */
        *rzo = (rz0 < 0.0) ? rz0 : 0.0;
        *rto = (rt0 > 0.0) ? rt0 : 0.0;
        return;
    }

    /* In-cone test. */
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

    double a = d_t * d_t * y_eff; /* > 0 */
    double b = d_t * rt0;         /* any sign */
    double c = d_r * d_r * y_eff; /* > 0 */
    double e = d_r * rz0;         /* any sign */

    /* Bracket. f(u_lo) < 0, f(u_hi) > 0. */
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

    /* Geometric-mean init keeps log(u) well-conditioned. */
    double u = exp(0.5 * (log(u_lo) + log(u_hi)));

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
    *rzo = d_r * y_eff * log(u);
    *rto = d_t * y_eff * u;
}

__global__ void project_exp_cone_kernel(double *__restrict__ primal_solution,
                                        const double *__restrict__ variable_rescaling,
                                        double *__restrict__ warm_start,
                                        const int *__restrict__ start_idx,
                                        const int *__restrict__ v_dim,
                                        const char *__restrict__ is_fixed,
                                        int num_blocks)
{
    (void)warm_start;
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

    /* If the y slot is fixed, do a 2D cross-section projection at y = r2 (constant). */
    if (is_fixed && is_fixed[s_idx + 1])
    {
        double zo, to;
        project_2d_exp_persp(r1, r2, r3, d1, d2, d3, &zo, &to);
        primal_solution[s_idx + 0] = zo;
        /* primal_solution[s_idx + 1] left untouched (= r2). */
        primal_solution[s_idx + 2] = to;
        return;
    }

    double xo, yo, zo;
    project_exp_cone_point(r1, r2, r3, d1, d2, d3, &xo, &yo, &zo);

    primal_solution[s_idx + 0] = xo;
    primal_solution[s_idx + 1] = yo;
    primal_solution[s_idx + 2] = zo;
}

__global__ void compute_cone_dual_residual_exp_kernel(double *__restrict__ dual_residual,
                                                      const double *__restrict__ objective_vector,
                                                      const double *__restrict__ dual_product,
                                                      const double *__restrict__ variable_rescaling,
                                                      double *__restrict__ warm_start,
                                                      const int *__restrict__ start_idx,
                                                      const int *__restrict__ v_dim,
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

    /* y fixed: dual cross-section of K_exp^* projected onto (z, t) plane is {u <= 0, w >= 0}
       (since y free in K_exp^* allows any feasible (u, w) with u <= 0, w >= 0). Residual
       outside this set = (max(r_z, 0), 0, min(r_t, 0)). */
    if (is_fixed && is_fixed[s_idx + 1])
    {
        dual_residual[s_idx + 0] = ((r1 > 0.0) ? r1 : 0.0) * variable_rescaling[s_idx + 0];
        dual_residual[s_idx + 1] = 0.0;
        dual_residual[s_idx + 2] = ((r3 < 0.0) ? r3 : 0.0) * variable_rescaling[s_idx + 2];
        return;
    }

    /* Inverse scaling for dual cone projection (K_exp^* lives in dual space). */
    double d1 = 1.0 / variable_rescaling[s_idx + 0];
    double d2 = 1.0 / variable_rescaling[s_idx + 1];
    double d3 = 1.0 / variable_rescaling[s_idx + 2];

    /* dist(r, K_exp^*) = || -proj_{K_exp}(-r) || via Moreau identity. */
    double xo, yo, zo;
    project_exp_cone_point(-r1, -r2, -r3, d1, d2, d3, &xo, &yo, &zo);

    dual_residual[s_idx + 0] = -xo * variable_rescaling[s_idx + 0];
    dual_residual[s_idx + 1] = -yo * variable_rescaling[s_idx + 1];
    dual_residual[s_idx + 2] = -zo * variable_rescaling[s_idx + 2];
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

// Diagonal-Q variant of project_standard_soc_kernel.
__global__ void project_standard_soc_diag_q_kernel(double *__restrict__ pdhg_primal,
                                                   double *__restrict__ reflected_primal,
                                                   const double *__restrict__ current_primal,
                                                   const double *__restrict__ variable_rescaling,
                                                   const double *__restrict__ Q_diag,
                                                   double tau,
                                                   double *__restrict__ warm_start,
                                                   const int *__restrict__ start_idx,
                                                   const int *__restrict__ v_dim,
                                                   int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int start = start_idx[blk];
    int k = v_dim[blk];
    int w_off = start + k;
    int z_off = start + k + 1;

    // Capture r (the temp values) into locals before any writes to pdhg_primal.
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

    // Fast-path tests are exactly f(lam=0) <= 0 (in-cone) and the recession KKT.
    //   f(lam) = sum_m w_m r_m^2 eh_m^2 / (eh_m^2 + 2 lam)^2
    //          + w_w r_w^2 eh_w^2 / (eh_w^2 + 2 lam)^2
    //          - w_z r_z^2 / (1 - 2 lam)^2
    // f(0) = sum_m w_m r_m^2 / eh_m^2 + w_w r_w^2 / eh_w^2 - w_z r_z^2.
    // Recession (x* = 0): -DWr in K_soc <=> sum_m w_m r_m^2 eh_m^2 + w_w r_w^2 eh_w^2
    //                                       <= w_z r_z^2 with r_z <= 0.

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
        // Already in DK: projection is the identity. Still need reflected writes.
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
        // Recession: projection is zero.
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

/* Weighted-prox projection onto D K_exp for the diagonal-Q PDHG primal update.

   At call time pdhg_primal already holds the per-coordinate temp value
       temp_i = (current_i - tau (c_i - dual_product_i)) / (1 + tau Q_i),
   produced by diag_q_primal_update. The remaining step is

       min   sum_i w_i (x_i - temp_i)^2,    w_i = 1 + tau Q_i,
       s.t.  x in D K_exp.

   Coordinate transform y_i = sqrt(w_i) x_i, u_i = sqrt(w_i) temp_i turns the
   objective into the unweighted sum_i (y_i - u_i)^2, while the constraint
   x in D K_exp <=> v in K_exp with x_i = d_i v_i becomes y_i = e_i v_i with
       e_i = sqrt(w_i) * d_i,
   i.e. y in E K_exp. project_exp_cone_point already solves the unweighted
   projection onto E K_exp by taking (e_1, e_2, e_3) as its scaling vector,
   so we just feed (u_1, u_2, u_3, e_1, e_2, e_3) in and recover x_i = y_i / sqrt(w_i).

   y-fixed sub-case (is_fixed[y]): the y slot is held at temp_2 = r_2 (no prox
   weight applies because the slot cannot move). The 2D weighted prox in the
   (x_1, x_3) plane reduces analogously by transforming only the variable
   coordinates. The y_eff parameter inside project_2d_exp_persp depends on
   ry / d_y, which uses the unscaled d_2, so we pass (e_1, d_2, e_3) as the
   scaling triple along with (u_1, r_2, u_3) for the inputs.

   Fast paths and edge cases (in-cone, polar, recession edge, Newton diverged)
   are handled inside project_exp_cone_point / project_2d_exp_persp; we just
   transform on the way in and out. Reflected primal write is fused:
       reflected_primal[i] = 2 * proj_i - current_primal[i]
   on every cone slot, including the in-cone fast path. */
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

    double w1 = 1.0 + tau * Q_diag[s_idx + 0];
    double w2 = 1.0 + tau * Q_diag[s_idx + 1];
    double w3 = 1.0 + tau * Q_diag[s_idx + 2];

    /* Diagonal Q on a convex problem should give w_i >= 1, but clamp to guard
       against tiny negative drift in Q_diag (which would invalidate the sqrt). */
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

    double x1, x2, x3;

    if (is_fixed && is_fixed[s_idx + 1])
    {
        /* y slot is fixed at r2 (untransformed). Only the (x_1, x_3) coordinates
           are subject to the prox; transform them and project onto the y=r2
           cross-section of E' K_exp with E' = diag(e_1, d_2, e_3). */
        double u1 = sw1 * r1;
        double u3 = sw3 * r3;
        double y1_out, y3_out;
        project_2d_exp_persp(u1, r2, u3, e1, d2, e3, &y1_out, &y3_out);
        x1 = y1_out / sw1;
        x2 = r2;
        x3 = y3_out / sw3;
    }
    else
    {
        /* Full 3D weighted prox: transform input, project onto E K_exp, untransform. */
        double u1 = sw1 * r1;
        double u2 = sw2 * r2;
        double u3 = sw3 * r3;
        double y1_out, y2_out, y3_out;
        project_exp_cone_point(u1, u2, u3, e1, e2, e3, &y1_out, &y2_out, &y3_out);
        x1 = y1_out / sw1;
        x2 = y2_out / sw2;
        x3 = y3_out / sw3;
    }

    pdhg_primal[s_idx + 0] = x1;
    pdhg_primal[s_idx + 1] = x2;
    pdhg_primal[s_idx + 2] = x3;

    reflected_primal[s_idx + 0] = 2.0 * x1 - current_primal[s_idx + 0];
    reflected_primal[s_idx + 1] = 2.0 * x2 - current_primal[s_idx + 1];
    reflected_primal[s_idx + 2] = 2.0 * x3 - current_primal[s_idx + 2];
}

/* Rotated SOC projection with diagonal-Q prox weights w_i = 1 + tau * Q_diag[i].
 * Minimizes sum w_i (x_i - temp_i)^2 over x in (D K_rsoc) where temp_i is
 * already in pdhg_primal[start+i]. Requires Q_diag[start+k] == Q_diag[start+k+1]
 * (so the rotation (s,t) -> (w,z) preserves the prox metric).
 *
 * After projecting, fuses the reflected-primal write:
 *   reflected_primal[i] = 2 * pdhg_primal[i] - current_primal[i] for every cone slot.
 */
__global__ void project_rotated_soc_diag_q_kernel(double *__restrict__ pdhg_primal,
                                                  double *__restrict__ reflected_primal,
                                                  const double *__restrict__ current_primal,
                                                  const double *__restrict__ variable_rescaling,
                                                  const double *__restrict__ Q_diag,
                                                  double tau,
                                                  double *__restrict__ warm_start,
                                                  const int *__restrict__ start_idx,
                                                  const int *__restrict__ v_dim,
                                                  int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    const double INV_SQRT2 = 0.7071067811865475;

    int start = start_idx[blk];
    int k = v_dim[blk];
    int len = k + 2;
    double *v = pdhg_primal + start;
    double *sptr = pdhg_primal + start + k;
    double *tptr = pdhg_primal + start + k + 1;

    double q_s = Q_diag[start + k];
    double q_t = Q_diag[start + k + 1];

    /* Limitation: rotation only decouples prox metric when Q_s == Q_t.
     * If violated, leave temp values in place and still write reflected. */
    if (q_s != q_t)
    {
        for (int m = 0; m < len; ++m)
        {
            int idx = start + m;
            double pv = pdhg_primal[idx];
            reflected_primal[idx] = 2.0 * pv - current_primal[idx];
        }
        return;
    }

    double w_st = 1.0 + tau * q_s;
    /* Guard against tiny negative drift in Q_diag breaking sqrt(w_st). */
    if (!(w_st > 0.0))
        w_st = 1.0;

    double r_s = *sptr;
    double r_t = *tptr;
    double r_w = (r_s - r_t) * INV_SQRT2;
    double r_z = (r_s + r_t) * INV_SQRT2;

    double d_s = variable_rescaling[start + k];
    double d_t = variable_rescaling[start + k + 1];
    double d_st = sqrt(d_s * d_t);

    double sqrt_w_st = sqrt(w_st);

    /* Uniform fast path: when eh_m == 1 for all v slots, the SOC bisection
     * collapses to the closed-form scalar projection. */
    bool diag_uniform = true;
    for (int m = 0; m < k && diag_uniform; ++m)
    {
        double w_m = 1.0 + tau * Q_diag[start + m];
        if (!(w_m > 0.0))
            w_m = 1.0;
        double d_m = variable_rescaling[start + m];
        double eh_m = sqrt(w_m / w_st) * (d_m / d_st);
        if (eh_m != 1.0)
            diag_uniform = false;
    }

    if (diag_uniform)
    {
        double sumsq = r_w * r_w;
        for (int m = 0; m < k; ++m)
            sumsq += v[m] * v[m];
        double r = sqrt(sumsq);
        if (r <= r_z)
        {
            /* Already in scaled cone; temp values are the projection. */
            for (int m = 0; m < len; ++m)
            {
                int idx = start + m;
                double pv = pdhg_primal[idx];
                reflected_primal[idx] = 2.0 * pv - current_primal[idx];
            }
            return;
        }
        if (r <= -r_z)
        {
            for (int m = 0; m < k; ++m)
                v[m] = 0.0;
            *sptr = 0.0;
            *tptr = 0.0;
            for (int m = 0; m < len; ++m)
            {
                int idx = start + m;
                reflected_primal[idx] = -current_primal[idx];
            }
            return;
        }
        double scale = (r_z + r) / (2.0 * r);
        for (int m = 0; m < k; ++m)
            v[m] *= scale;
        double w_new = scale * r_w;
        double z_new = scale * r;
        *sptr = (z_new + w_new) * INV_SQRT2;
        *tptr = (z_new - w_new) * INV_SQRT2;
        for (int m = 0; m < len; ++m)
        {
            int idx = start + m;
            double pv = pdhg_primal[idx];
            reflected_primal[idx] = 2.0 * pv - current_primal[idx];
        }
        return;
    }

    /* General path: KKT bisection on
     *   f(lam) = sum_m t_m^2 + t_w^2 - t_z^2,
     * with
     *   t_m = sqrt(w_m) r_m eh_m / (eh_m^2 + 2 lam),
     *   t_w = sqrt(w_st) r_w / (1 + 2 lam),
     *   t_z = sqrt(w_st) r_z / (1 - 2 lam).
     *
     * Cone fast-check uses r_inv (eh_m -> 0 limit yields large weight) and
     * r_pos (eh_m -> infty) as worst-case bounds on the radius. */
    double rw_sq = w_st * r_w * r_w;
    double r_inv_sq = rw_sq;
    double r_pos_sq = rw_sq;
    for (int m = 0; m < k; ++m)
    {
        double w_m = 1.0 + tau * Q_diag[start + m];
        if (!(w_m > 0.0))
            w_m = 1.0;
        double d_m = variable_rescaling[start + m];
        double eh_m = sqrt(w_m / w_st) * (d_m / d_st);
        double sqrt_w_m = sqrt(w_m);
        double v_m = v[m];
        double a_lo = sqrt_w_m * v_m / eh_m; /* eh_m^2 + 2lam -> 2lam as eh_m -> 0 */
        double a_hi = sqrt_w_m * v_m * eh_m; /* eh_m^2 + 2lam -> eh_m^2 as eh_m -> infty */
        r_inv_sq += a_lo * a_lo;
        r_pos_sq += a_hi * a_hi;
    }
    double r_inv = sqrt(r_inv_sq);
    double sw_rz = sqrt_w_st * r_z;
    if (r_inv <= sw_rz)
    {
        for (int m = 0; m < len; ++m)
        {
            int idx = start + m;
            double pv = pdhg_primal[idx];
            reflected_primal[idx] = 2.0 * pv - current_primal[idx];
        }
        return;
    }
    double r_pos = sqrt(r_pos_sq);
    if (r_pos <= -sw_rz)
    {
        for (int m = 0; m < k; ++m)
            v[m] = 0.0;
        *sptr = 0.0;
        *tptr = 0.0;
        for (int m = 0; m < len; ++m)
        {
            int idx = start + m;
            reflected_primal[idx] = -current_primal[idx];
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
            double sum_hi = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double w_m = 1.0 + tau * Q_diag[start + m];
                if (!(w_m > 0.0))
                    w_m = 1.0;
                double d_m = variable_rescaling[start + m];
                double eh_m = sqrt(w_m / w_st) * (d_m / d_st);
                double eh_m2 = eh_m * eh_m;
                double tt = sqrt(w_m) * v[m] * eh_m / (eh_m2 + 2.0 * hi);
                sum_hi += tt * tt;
            }
            double tw_hi = sqrt_w_st * r_w / (1.0 + 2.0 * hi);
            sum_hi += tw_hi * tw_hi;
            double zt_hi = sqrt_w_st * r_z / (1.0 - 2.0 * hi);
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
            double w_m = 1.0 + tau * Q_diag[start + m];
            if (!(w_m > 0.0))
                w_m = 1.0;
            double d_m = variable_rescaling[start + m];
            double eh_m = sqrt(w_m / w_st) * (d_m / d_st);
            double eh_m2 = eh_m * eh_m;
            double tt = sqrt(w_m) * v[m] * eh_m / (eh_m2 + 2.0 * warm_lam);
            sum_w += tt * tt;
        }
        double tw = sqrt_w_st * r_w / (1.0 + 2.0 * warm_lam);
        sum_w += tw * tw;
        double zt = sqrt_w_st * r_z / (1.0 - 2.0 * warm_lam);
        double f = sum_w - zt * zt;
        if (fabs(f) < 1e-12)
        {
            double x_w_new = r_w / (1.0 + 2.0 * warm_lam);
            double x_z_new = r_z / (1.0 - 2.0 * warm_lam);
            for (int m = 0; m < k; ++m)
            {
                double w_m = 1.0 + tau * Q_diag[start + m];
                if (!(w_m > 0.0))
                    w_m = 1.0;
                double d_m = variable_rescaling[start + m];
                double eh_m = sqrt(w_m / w_st) * (d_m / d_st);
                double eh_m2 = eh_m * eh_m;
                v[m] = v[m] * eh_m2 / (eh_m2 + 2.0 * warm_lam);
            }
            *sptr = (x_z_new + x_w_new) * INV_SQRT2;
            *tptr = (x_z_new - x_w_new) * INV_SQRT2;
            for (int m = 0; m < len; ++m)
            {
                int idx = start + m;
                double pv = pdhg_primal[idx];
                reflected_primal[idx] = 2.0 * pv - current_primal[idx];
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
            if (!(w_m > 0.0))
                w_m = 1.0;
            double d_m = variable_rescaling[start + m];
            double eh_m = sqrt(w_m / w_st) * (d_m / d_st);
            double eh_m2 = eh_m * eh_m;
            double tt = sqrt(w_m) * v[m] * eh_m / (eh_m2 + 2.0 * lam);
            sum += tt * tt;
        }
        double tw = sqrt_w_st * r_w / (1.0 + 2.0 * lam);
        sum += tw * tw;
        double zt = sqrt_w_st * r_z / (1.0 - 2.0 * lam);
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

    double x_w_new = r_w / (1.0 + 2.0 * lam);
    double x_z_new = r_z / (1.0 - 2.0 * lam);
    for (int m = 0; m < k; ++m)
    {
        double w_m = 1.0 + tau * Q_diag[start + m];
        if (!(w_m > 0.0))
            w_m = 1.0;
        double d_m = variable_rescaling[start + m];
        double eh_m = sqrt(w_m / w_st) * (d_m / d_st);
        double eh_m2 = eh_m * eh_m;
        v[m] = v[m] * eh_m2 / (eh_m2 + 2.0 * lam);
    }
    *sptr = (x_z_new + x_w_new) * INV_SQRT2;
    *tptr = (x_z_new - x_w_new) * INV_SQRT2;

    for (int m = 0; m < len; ++m)
    {
        int idx = start + m;
        double pv = pdhg_primal[idx];
        reflected_primal[idx] = 2.0 * pv - current_primal[idx];
    }
}
