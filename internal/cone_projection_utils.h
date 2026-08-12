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
