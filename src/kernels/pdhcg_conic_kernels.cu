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
    double *v = primal_solution + start;
    double *sptr = primal_solution + start + k;
    double *tptr = primal_solution + start + k + 1;

    double s = *sptr;
    double t = *tptr;

    /* Fast path: s, t pinned via is_fixed. Cone reduces to ball on v with R^2 = 2 s t / (d_s d_t).
       s, t kept at their input (pinned) values; only v is projected. */
    if (is_fixed && is_fixed[start + k] && is_fixed[start + k + 1])
    {
        double d_s_p = variable_rescaling[start + k];
        double d_t_p = variable_rescaling[start + k + 1];
        double R2 = 2.0 * (s / d_s_p) * (t / d_t_p);
        if (!(R2 > 0.0))
        {
            for (int m = 0; m < k; ++m)
                v[m] = 0.0;
            return;
        }
        /* In-cone: sum (v_i/d_v_i)^2 <= R^2 -> identity */
        double lhs = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double d_m = variable_rescaling[start + m];
            double y = v[m] / d_m;
            lhs += y * y;
        }
        if (lhs <= R2)
            return;
        double scale = sqrt(R2 / lhs);
        for (int m = 0; m < k; ++m)
            v[m] *= scale;
        return;
    }

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

    /* Fast path: s, t pinned via is_fixed. Cone reduces to ball ||v/d||^2 <= R^2 in v slots.
       Normal cone at v*: {0} if interior, R_+ . (v/d^2) if boundary. Dual residual is
       r - proj_{N_K(v*)}(r); report it scaled by variable_rescaling like the LP-style kernel. */
    if (is_fixed && is_fixed[start + k] && is_fixed[start + k + 1])
    {
        /* k==1 closed form: lambda absorbs r iff r and v have opposite signs (inward gradient). */
        if (k == 1)
        {
            int idx = start;
            double r = objective_vector[idx] - dual_product[idx];
            double v = primal_solution[idx];
            double d_0 = variable_rescaling[idx];
            dual_residual[idx] = (r * v >= 0.0) ? r * d_0 : 0.0;
            dual_residual[start + 1] = 0.0;
            dual_residual[start + 2] = 0.0;
            return;
        }
        /* KKT on v: r_v + lambda * n = 0 with lambda >= 0, n = outward normal (v / d^2).
           Choose lambda = max(0, -r . n / n . n); residual = r + lambda * n; scale by d.
           Fused single pass over k slots (was two; redundant reads removed). */
        double dot = 0.0, n2 = 0.0;
        for (int m = 0; m < k; ++m)
        {
            int idx = start + m;
            double d_m = variable_rescaling[idx];
            double n_m = primal_solution[idx] / (d_m * d_m);
            double r = objective_vector[idx] - dual_product[idx];
            dot += r * n_m;
            n2 += n_m * n_m;
        }
        double lambda = (dot < 0.0 && n2 > 0.0) ? -dot / n2 : 0.0;
        for (int m = 0; m < k; ++m)
        {
            int idx = start + m;
            double d_m = variable_rescaling[idx];
            double n_m = primal_solution[idx] / (d_m * d_m);
            double r = objective_vector[idx] - dual_product[idx];
            dual_residual[idx] = (r + lambda * n_m) * d_m;
        }
        dual_residual[start + k] = 0.0;
        dual_residual[start + k + 1] = 0.0;
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
    double *v = primal_solution + start;
    double *wptr = primal_solution + start + k;
    double *zptr = primal_solution + start + k + 1;

    double w = *wptr;
    double z = *zptr;

    /* Fast path: w, z pinned via is_fixed. SOC ||v,w|| <= z reduces to ball
       sum(v_actual)^2 <= z_actual^2 - w_actual^2 (assumes uniform d on cone slots). */
    if (is_fixed && is_fixed[start + k] && is_fixed[start + k + 1])
    {
        double d_z_p = variable_rescaling[start + k + 1];
        double d_w_p = variable_rescaling[start + k];
        double z_a = z / d_z_p;
        double w_a = w / d_w_p;
        double R2 = z_a * z_a - w_a * w_a;
        if (!(R2 > 0.0))
        {
            for (int m = 0; m < k; ++m)
                v[m] = 0.0;
            return;
        }
        double lhs = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double d_m = variable_rescaling[start + m];
            double y = v[m] / d_m;
            lhs += y * y;
        }
        if (lhs <= R2)
            return;
        double scale = sqrt(R2 / lhs);
        for (int m = 0; m < k; ++m)
            v[m] *= scale;
        return;
    }

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
    int w_off = start + k;
    int z_off = start + k + 1;

    /* Fast path: w, z pinned via is_fixed. SOC reduces to ball on v; dual residual for
       v projects onto ball's normal cone at v*; w, z slots have no dual condition. */
    if (is_fixed && is_fixed[w_off] && is_fixed[z_off])
    {
        if (k == 1)
        {
            int idx = start;
            double r = objective_vector[idx] - dual_product[idx];
            double v = primal_solution[idx];
            double d_0 = variable_rescaling[idx];
            dual_residual[idx] = (r * v >= 0.0) ? r * d_0 : 0.0;
            dual_residual[w_off] = 0.0;
            dual_residual[z_off] = 0.0;
            return;
        }
        double dot = 0.0, n2 = 0.0;
        for (int m = 0; m < k; ++m)
        {
            int idx = start + m;
            double d_m = variable_rescaling[idx];
            double n_m = primal_solution[idx] / (d_m * d_m);
            double r = objective_vector[idx] - dual_product[idx];
            dot += r * n_m;
            n2 += n_m * n_m;
        }
        double lambda = (dot < 0.0 && n2 > 0.0) ? -dot / n2 : 0.0;
        for (int m = 0; m < k; ++m)
        {
            int idx = start + m;
            double d_m = variable_rescaling[idx];
            double n_m = primal_solution[idx] / (d_m * d_m);
            double r = objective_vector[idx] - dual_product[idx];
            dual_residual[idx] = (r + lambda * n_m) * d_m;
        }
        dual_residual[w_off] = 0.0;
        dual_residual[z_off] = 0.0;
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

    double s_val = primal_solution[start + k];
    double t_val = primal_solution[start + k + 1];

    if (is_fixed && is_fixed[start + k] && is_fixed[start + k + 1])
    {
        double d_s_p = variable_rescaling[start + k];
        double d_t_p = variable_rescaling[start + k + 1];
        double R2 = 2.0 * (s_val / d_s_p) * (t_val / d_t_p);
        if (!(R2 > 0.0))
        {
            for (int m = lane; m < k; m += 32)
                primal_solution[start + m] = 0.0;
            return;
        }
        double lhs_p = 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double d_m = variable_rescaling[start + m];
            double y = primal_solution[start + m] / d_m;
            lhs_p += y * y;
        }
        for (int off = 16; off > 0; off >>= 1)
            lhs_p += __shfl_xor_sync(MASK, lhs_p, off);
        if (lhs_p <= R2)
            return;
        double scale = sqrt(R2 / lhs_p);
        for (int m = lane; m < k; m += 32)
            primal_solution[start + m] *= scale;
        return;
    }

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

    if (is_fixed && is_fixed[start + k] && is_fixed[start + k + 1])
    {
        if (k == 1)
        {
            if (lane == 0)
            {
                int idx = start;
                double r = objective_vector[idx] - dual_product[idx];
                double v = primal_solution[idx];
                double d_0 = variable_rescaling[idx];
                dual_residual[idx] = (r * v >= 0.0) ? r * d_0 : 0.0;
                dual_residual[start + 1] = 0.0;
                dual_residual[start + 2] = 0.0;
            }
            return;
        }

        double dot_p = 0.0, n2_p = 0.0;
        for (int m = lane; m < k; m += 32)
        {
            int idx = start + m;
            double d_m = variable_rescaling[idx];
            double n_m = primal_solution[idx] / (d_m * d_m);
            double r = objective_vector[idx] - dual_product[idx];
            dot_p += r * n_m;
            n2_p += n_m * n_m;
        }
        for (int off = 16; off > 0; off >>= 1)
        {
            dot_p += __shfl_xor_sync(MASK, dot_p, off);
            n2_p += __shfl_xor_sync(MASK, n2_p, off);
        }
        double lambda = (dot_p < 0.0 && n2_p > 0.0) ? -dot_p / n2_p : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            int idx = start + m;
            double d_m = variable_rescaling[idx];
            double n_m = primal_solution[idx] / (d_m * d_m);
            double r = objective_vector[idx] - dual_product[idx];
            dual_residual[idx] = (r + lambda * n_m) * d_m;
        }
        if (lane == 0)
        {
            dual_residual[start + k] = 0.0;
            dual_residual[start + k + 1] = 0.0;
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

    double w = primal_solution[start + k];
    double z = primal_solution[start + k + 1];

    if (is_fixed && is_fixed[start + k] && is_fixed[start + k + 1])
    {
        double d_z_p = variable_rescaling[start + k + 1];
        double d_w_p = variable_rescaling[start + k];
        double z_a = z / d_z_p;
        double w_a = w / d_w_p;
        double R2 = z_a * z_a - w_a * w_a;
        if (!(R2 > 0.0))
        {
            for (int m = lane; m < k; m += 32)
                primal_solution[start + m] = 0.0;
            return;
        }
        double lhs_p = 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double d_m = variable_rescaling[start + m];
            double y = primal_solution[start + m] / d_m;
            lhs_p += y * y;
        }
        for (int off = 16; off > 0; off >>= 1)
            lhs_p += __shfl_xor_sync(MASK, lhs_p, off);
        if (lhs_p <= R2)
            return;
        double scale = sqrt(R2 / lhs_p);
        for (int m = lane; m < k; m += 32)
            primal_solution[start + m] *= scale;
        return;
    }

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
    int w_off = start + k;
    int z_off = start + k + 1;

    if (is_fixed && is_fixed[w_off] && is_fixed[z_off])
    {
        double dot_p = 0.0, n2_p = 0.0;
        for (int m = lane; m < k; m += 32)
        {
            int idx = start + m;
            double d_m = variable_rescaling[idx];
            double n_m = primal_solution[idx] / (d_m * d_m);
            double r = objective_vector[idx] - dual_product[idx];
            dot_p += r * n_m;
            n2_p += n_m * n_m;
        }
        for (int off = 16; off > 0; off >>= 1)
        {
            dot_p += __shfl_xor_sync(MASK, dot_p, off);
            n2_p += __shfl_xor_sync(MASK, n2_p, off);
        }
        double lambda = (dot_p < 0.0 && n2_p > 0.0) ? -dot_p / n2_p : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            int idx = start + m;
            double d_m = variable_rescaling[idx];
            double n_m = primal_solution[idx] / (d_m * d_m);
            double r = objective_vector[idx] - dual_product[idx];
            dual_residual[idx] = (r + lambda * n_m) * d_m;
        }
        if (lane == 0)
        {
            dual_residual[w_off] = 0.0;
            dual_residual[z_off] = 0.0;
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

    if (is_fixed && is_fixed[s_idx + 1])
    {
        double zo, to;
        project_2d_exp_persp(r1, r2, r3, d1, d2, d3, &zo, &to);
        primal_solution[s_idx + 0] = zo;
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

    if (is_fixed && is_fixed[s_idx + 1])
    {
        dual_residual[s_idx + 0] = ((r1 > 0.0) ? r1 : 0.0) * variable_rescaling[s_idx + 0];
        dual_residual[s_idx + 1] = 0.0;
        dual_residual[s_idx + 2] = ((r3 < 0.0) ? r3 : 0.0) * variable_rescaling[s_idx + 2];
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

    double r_w = pdhg_primal[w_off];
    double r_z = pdhg_primal[z_off];

    /* Fast path: w, z pinned via is_fixed. Cone reduces to ball on v with R^2 = z_actual^2 - w_actual^2. */
    if (is_fixed && is_fixed[w_off] && is_fixed[z_off])
    {
        double d_z_p = variable_rescaling[z_off];
        double d_w_p = variable_rescaling[w_off];
        double z_a = r_z / d_z_p;
        double w_a = r_w / d_w_p;
        double R2 = z_a * z_a - w_a * w_a;
        if (!(R2 > 0.0))
        {
            for (int m = 0; m < k; ++m)
            {
                pdhg_primal[start + m] = 0.0;
                int idx = start + m;
                reflected_primal[idx] = -current_primal[idx];
            }
            reflected_primal[w_off] = 2.0 * r_w - current_primal[w_off];
            reflected_primal[z_off] = 2.0 * r_z - current_primal[z_off];
            return;
        }
        double lhs = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double d_m = variable_rescaling[start + m];
            double y = pdhg_primal[start + m] / d_m;
            lhs += y * y;
        }
        if (lhs <= R2)
        {
            for (int m = 0; m < k + 2; ++m)
            {
                int idx = start + m;
                reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
            }
            return;
        }
        double scale = sqrt(R2 / lhs);
        for (int m = 0; m < k; ++m)
        {
            double v_new = pdhg_primal[start + m] * scale;
            pdhg_primal[start + m] = v_new;
            int idx = start + m;
            reflected_primal[idx] = 2.0 * v_new - current_primal[idx];
        }
        reflected_primal[w_off] = 2.0 * r_w - current_primal[w_off];
        reflected_primal[z_off] = 2.0 * r_z - current_primal[z_off];
        return;
    }

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

    double x1, x2, x3;

    if (is_fixed && is_fixed[s_idx + 1])
    {
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

    double r_s = pdhg_primal[start + k];
    double r_t = pdhg_primal[start + k + 1];

    /* Fast path: both s and t slots pinned via is_fixed. Cone reduces to weighted ball
       projection on v slots with radius R^2 = 2 (S/d_s)(T/d_t); s, t stay at pinned r_s, r_t. */
    if (is_fixed && is_fixed[start + k] && is_fixed[start + k + 1])
    {
        double d_s = variable_rescaling[start + k];
        double d_t = variable_rescaling[start + k + 1];
        double R2 = 2.0 * (r_s / d_s) * (r_t / d_t);
        if (!(R2 > 0.0))
        {
            for (int m = 0; m < k; ++m)
            {
                pdhg_primal[start + m] = 0.0;
                int idx = start + m;
                reflected_primal[idx] = -current_primal[idx];
            }
            reflected_primal[start + k] = 2.0 * r_s - current_primal[start + k];
            reflected_primal[start + k + 1] = 2.0 * r_t - current_primal[start + k + 1];
            return;
        }
        /* k==1 closed form. KKT collapses: w cancels, v_new = sign(r) * min(|r|, d*R). */
        if (k == 1)
        {
            double d_0 = variable_rescaling[start];
            double r_0 = pdhg_primal[start];
            double bound = d_0 * sqrt(R2);
            double v_new = (fabs(r_0) <= bound) ? r_0 : ((r_0 >= 0.0) ? bound : -bound);
            pdhg_primal[start] = v_new;
            reflected_primal[start] = 2.0 * v_new - current_primal[start];
            reflected_primal[start + 1] = 2.0 * r_s - current_primal[start + 1];
            reflected_primal[start + 2] = 2.0 * r_t - current_primal[start + 2];
            return;
        }
        /* In-cone test: sum (r_v_i / d_v_i)^2 <= R^2 */
        double lhs0 = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double d_m = variable_rescaling[start + m];
            double y = pdhg_primal[start + m] / d_m;
            lhs0 += y * y;
        }
        if (lhs0 <= R2)
        {
            for (int m = 0; m < len; ++m)
            {
                int idx = start + m;
                reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
            }
            return;
        }
        /* Weighted ball: bisect lambda; v_i = r_i e_i^2 / (e_i^2 + lambda),  e_i^2 = w_i d_i^2.
           Constraint sum (v_i/d_v_i)^2 = R^2  =>  sum r_i^2 w_i^2 d_i^2 / (e_i^2 + lambda)^2 = R^2. */
        double lo_l = 0.0, hi_l = 1.0;
        for (int it = 0; it < 80; ++it)
        {
            double s = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double w_m = 1.0 + tau * Q_diag[start + m];
                if (!(w_m > W_FLOOR))
                    w_m = W_FLOOR;
                double d_m = variable_rescaling[start + m];
                double e2 = w_m * d_m * d_m;
                double tt = pdhg_primal[start + m] * w_m * d_m / (e2 + hi_l);
                s += tt * tt;
            }
            if (s < R2)
                break;
            lo_l = hi_l;
            hi_l *= 2.0;
        }
        for (int it = 0; it < 80; ++it)
        {
            double lam = 0.5 * (lo_l + hi_l);
            double s = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double w_m = 1.0 + tau * Q_diag[start + m];
                if (!(w_m > W_FLOOR))
                    w_m = W_FLOOR;
                double d_m = variable_rescaling[start + m];
                double e2 = w_m * d_m * d_m;
                double tt = pdhg_primal[start + m] * w_m * d_m / (e2 + lam);
                s += tt * tt;
            }
            if (s > R2)
                lo_l = lam;
            else
                hi_l = lam;
            if ((hi_l - lo_l) / (1.0 + hi_l + lo_l) < 1e-13)
                break;
        }
        double lam = 0.5 * (lo_l + hi_l);
        for (int m = 0; m < k; ++m)
        {
            double w_m = 1.0 + tau * Q_diag[start + m];
            if (!(w_m > W_FLOOR))
                w_m = W_FLOOR;
            double d_m = variable_rescaling[start + m];
            double e2 = w_m * d_m * d_m;
            double v_new = pdhg_primal[start + m] * e2 / (e2 + lam);
            pdhg_primal[start + m] = v_new;
            int idx = start + m;
            reflected_primal[idx] = 2.0 * v_new - current_primal[idx];
        }
        reflected_primal[start + k] = 2.0 * r_s - current_primal[start + k];
        reflected_primal[start + k + 1] = 2.0 * r_t - current_primal[start + k + 1];
        return;
    }

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
