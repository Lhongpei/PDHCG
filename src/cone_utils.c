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

#include "cone_utils.h"
#include "utils.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cone_length(cone_type_t type, int v_dim)
{
    if (type == CONE_EXPONENTIAL || type == CONE_POWER)
        return 3;
    if (type == CONE_STANDARD_SOC || type == CONE_ROTATED_SOC)
        return v_dim >= 0 && v_dim <= INT_MAX - 2 ? v_dim + 2 : -1;
    if (type == CONE_PSD && v_dim > 0)
    {
        long long length = (long long)v_dim * (v_dim + 1LL) / 2LL;
        return length <= INT_MAX ? (int)length : -1;
    }
    return -1;
}

int cone_block_length(const cone_blocks_t *blocks, int block)
{
    if (!blocks || block < 0 || block >= blocks->num_cones || !blocks->type || !blocks->v_dim)
        return -1;
    return cone_length(blocks->type[block], blocks->v_dim[block]);
}

int cone_blocks_init_from_specs(cone_blocks_t *blocks,
                                int num_cones,
                                const cone_spec_t *specs,
                                int ambient_dimension,
                                bool allow_fixed,
                                const char *label)
{
    if (!blocks)
        return -1;
    if (num_cones < 0 || ambient_dimension < 0 || (num_cones > 0 && !specs))
    {
        fprintf(stderr,
                "[create_qp_problem] Invalid %s cone metadata "
                "(num_cones=%d, ambient_dimension=%d, specs=%p).\n",
                label ? label : "",
                num_cones,
                ambient_dimension,
                (const void *)specs);
        return -1;
    }

    cone_blocks_free(blocks);
    if (num_cones == 0)
        return 0;

    const char *kind = label ? label : "";
    char *owner = ambient_dimension > 0 ? (char *)safe_calloc((size_t)ambient_dimension, sizeof(char)) : NULL;
    int any_fixed = 0;
    int any_power = 0;

    for (int cone = 0; cone < num_cones; ++cone)
    {
        int length = cone_length(specs[cone].type, specs[cone].v_dim);
        int start = specs[cone].start_idx;
        if (length <= 0 || start < 0 || (long long)start + length > ambient_dimension)
        {
            fprintf(stderr,
                    "[create_qp_problem] %s cone %d has invalid type, size, or range "
                    "(start=%d, length=%d, ambient_dimension=%d).\n",
                    kind,
                    cone,
                    start,
                    length,
                    ambient_dimension);
            free(owner);
            return -1;
        }
        if (!allow_fixed && specs[cone].is_fixed)
        {
            fprintf(stderr, "[create_qp_problem] %s cone %d does not support fixed slots.\n", kind, cone);
            free(owner);
            return -1;
        }
        if (specs[cone].type == CONE_PSD && specs[cone].is_fixed)
        {
            int has_fixed = 0;
            for (int slot = 0; slot < length; ++slot)
                has_fixed |= specs[cone].is_fixed[slot] != 0;
            if (has_fixed)
            {
                fprintf(stderr, "[create_qp_problem] %s PSD cone %d does not support fixed slots.\n", kind, cone);
                free(owner);
                return -1;
            }
        }
        if (specs[cone].type == CONE_POWER &&
            !(isfinite(specs[cone].power_alpha) && specs[cone].power_alpha > 0.0 && specs[cone].power_alpha < 1.0))
        {
            fprintf(stderr,
                    "[create_qp_problem] %s power cone %d requires alpha in (0,1); got %.6g.\n",
                    kind,
                    cone,
                    specs[cone].power_alpha);
            free(owner);
            return -1;
        }
        for (int index = start; index < start + length; ++index)
        {
            if (owner[index])
            {
                fprintf(stderr, "[create_qp_problem] %s cone %d overlaps at index %d.\n", kind, cone, index);
                free(owner);
                return -1;
            }
            owner[index] = 1;
        }
        any_fixed |= specs[cone].is_fixed != NULL;
        any_power |= specs[cone].type == CONE_POWER;
    }
    free(owner);

    size_t count = (size_t)num_cones;
    blocks->num_cones = num_cones;
    blocks->start_idx = (int *)safe_malloc(count * sizeof(int));
    blocks->v_dim = (int *)safe_malloc(count * sizeof(int));
    blocks->type = (cone_type_t *)safe_malloc(count * sizeof(cone_type_t));
    blocks->power_alpha = any_power ? (double *)safe_calloc(count, sizeof(double)) : NULL;
    if (any_fixed)
    {
        blocks->fixed_mask_size = ambient_dimension;
        blocks->is_fixed = (char *)safe_calloc((size_t)ambient_dimension, sizeof(char));
    }

    for (int cone = 0; cone < num_cones; ++cone)
    {
        int length = cone_length(specs[cone].type, specs[cone].v_dim);
        blocks->start_idx[cone] = specs[cone].start_idx;
        blocks->v_dim[cone] =
            (specs[cone].type == CONE_EXPONENTIAL || specs[cone].type == CONE_POWER) ? 1 : specs[cone].v_dim;
        blocks->type[cone] = specs[cone].type;
        if (blocks->power_alpha && specs[cone].type == CONE_POWER)
            blocks->power_alpha[cone] = specs[cone].power_alpha;
        if (blocks->is_fixed && specs[cone].is_fixed)
        {
            int start = specs[cone].start_idx;
            for (int slot = 0; slot < length; ++slot)
                blocks->is_fixed[start + slot] = specs[cone].is_fixed[slot] ? 1 : 0;
        }
    }
    return 0;
}

void cone_blocks_clone(cone_blocks_t *dst, const cone_blocks_t *src)
{
    if (!dst || !src)
        return;
    if (dst == src)
        return;
    cone_blocks_free(dst);
    dst->num_cones = src->num_cones;
    if (src->num_cones <= 0)
        return;

    size_t count = (size_t)src->num_cones;
    dst->start_idx = (int *)safe_malloc(count * sizeof(int));
    dst->v_dim = (int *)safe_malloc(count * sizeof(int));
    dst->type = (cone_type_t *)safe_malloc(count * sizeof(cone_type_t));
    memcpy(dst->start_idx, src->start_idx, count * sizeof(int));
    memcpy(dst->v_dim, src->v_dim, count * sizeof(int));
    memcpy(dst->type, src->type, count * sizeof(cone_type_t));

    if (src->power_alpha)
    {
        dst->power_alpha = (double *)safe_malloc(count * sizeof(double));
        memcpy(dst->power_alpha, src->power_alpha, count * sizeof(double));
    }
    if (src->is_fixed && src->fixed_mask_size > 0)
    {
        dst->fixed_mask_size = src->fixed_mask_size;
        dst->is_fixed = (char *)safe_malloc((size_t)src->fixed_mask_size * sizeof(char));
        memcpy(dst->is_fixed, src->is_fixed, (size_t)src->fixed_mask_size * sizeof(char));
    }
}

void cone_blocks_free(cone_blocks_t *blocks)
{
    if (!blocks)
        return;
    free(blocks->start_idx);
    free(blocks->v_dim);
    free(blocks->type);
    free(blocks->power_alpha);
    free(blocks->is_fixed);
    memset(blocks, 0, sizeof(*blocks));
}
