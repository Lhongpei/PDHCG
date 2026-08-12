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
#include "pdhcg_power_cone_kernels.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

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

static void launch_power_thread_proj(
    double *p, const double *vr, double *ws, const int *si, const int *vd, const double *pa, const char *isf, int n)
{
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_power_cone_kernel<<<b, t>>>(p, vr, ws, si, vd, pa, isf, n);
}

static void launch_power_thread_dual(double *dr,
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
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    compute_cone_dual_residual_power_kernel<<<b, t>>>(dr, cr, obj, dp, vr, ps, ws, si, vd, pa, isf, n);
}

static void launch_power_thread_proj_diag_q(double *pp,
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
    int t = THREADS_PER_BLOCK;
    int b = (n + t - 1) / t;
    project_power_cone_diag_q_kernel<<<b, t>>>(pp, rp, cp, vr, qd, tau, ws, si, vd, pa, isf, n);
}

void launch_power_cone_primal_violation(double *absolute_violation,
                                        double *relative_violation,
                                        const double *primal_solution,
                                        const double *variable_rescaling,
                                        const int *start_idx,
                                        const double *power_alpha,
                                        double homogeneous_scale,
                                        int count)
{
    int threads = THREADS_PER_BLOCK;
    int blocks = (count + threads - 1) / threads;
    compute_power_cone_primal_violation_kernel<<<blocks, threads>>>(absolute_violation,
                                                                    relative_violation,
                                                                    primal_solution,
                                                                    variable_rescaling,
                                                                    start_idx,
                                                                    power_alpha,
                                                                    homogeneous_scale,
                                                                    count);
}

extern const cone_kernel_ops_t pdhcg_power_cone_kernel_ops = {
    {
        launch_power_thread_proj,
        NULL,
        NULL,
        NULL,
        NULL,
    },
    {
        launch_power_thread_proj_diag_q,
        NULL,
        NULL,
        NULL,
        NULL,
    },
    {
        launch_power_thread_dual,
        NULL,
        NULL,
        NULL,
        NULL,
    },
};
