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
#include "cone_projection_utils.h"
#include "pdhcg_exp_cone_kernels.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

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

static void launch_exp_thread_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    (void)pa;
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_exp_cone_kernel<<<b, t>>>(p, vr, ws, si, vd, isf, n);
}
static void launch_exp_thread_dual(double *dr,
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
    compute_cone_dual_residual_exp_kernel<<<b, t>>>(dr, cr, obj, dp, vr, ps, ws, si, vd, isf, n);
}
static void launch_exp_thread_proj_diag_q(double *pp,
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
    project_exp_cone_diag_q_kernel<<<b, t>>>(pp, rp, cp, vr, qd, tau, ws, si, vd, isf, n);
}

extern const cone_kernel_ops_t pdhcg_exp_cone_kernel_ops = {
    {
        launch_exp_thread_proj,
        NULL,
        NULL,
        NULL,
        NULL,
    },
    {
        launch_exp_thread_proj_diag_q,
        NULL,
        NULL,
        NULL,
        NULL,
    },
    {
        launch_exp_thread_dual,
        NULL,
        NULL,
        NULL,
        NULL,
    },
};
