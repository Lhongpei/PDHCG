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

#include "distributed_interface.h"
#include <stdbool.h>
#include <stdio.h>

void pdhcg_all_reduce_array(
    grid_context_t *ctx, double *buf, int count, pdhcg_reduce_op_t op, pdhcg_comm_scope_t scope, void *stream)
{
    (void)ctx;
    (void)buf;
    (void)count;
    (void)op;
    (void)scope;
    (void)stream;
}

void pdhcg_all_reduce_scalar(
    grid_context_t *ctx, double *value, pdhcg_reduce_op_t op, pdhcg_comm_scope_t scope, bool on_device)
{
    (void)ctx;
    (void)value;
    (void)op;
    (void)scope;
    (void)on_device;
}

int pdhcg_get_grid_p_col(struct grid_context_s *ctx)
{
    (void)ctx;
    return 1;
}

int pdhcg_get_grid_row_coord(struct grid_context_s *ctx)
{
    (void)ctx;
    return 0;
}

int pdhcg_get_global_num_variables(grid_context_t *ctx)
{
    (void)ctx;
    return 0;
}

int pdhcg_get_variable_start(grid_context_t *ctx)
{
    (void)ctx;
    return 0;
}

int pdhcg_get_global_num_cones(grid_context_t *ctx)
{
    (void)ctx;
    return 0;
}

int pdhcg_get_global_num_affine_cones(grid_context_t *ctx)
{
    (void)ctx;
    return 0;
}

pdhcg_result_t *pdhcg_distributed_optimize(const pdhg_parameters_t *params, const qp_problem_t *original_problem)
{
    (void)params;
    (void)original_problem;
    fprintf(stderr, "[interface] distributed support is not enabled in this build.\n");
    return NULL;
}
