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
#include "distributed_solver.h"
#include "distributed_types.h"
#include "distributed_utils.h"
#include <mpi.h>
#include <nccl.h>

void pdhcg_all_reduce_array(
    grid_context_t *ctx, double *buf, int count, pdhcg_reduce_op_t op, pdhcg_comm_scope_t scope, void *stream)
{
    if (!ctx || count <= 0)
        return;

    ncclRedOp_t nccl_op = (op == PDHCG_OP_MAX) ? ncclMax : ncclSum;
    ncclComm_t comm;

    if (scope == PDHCG_SCOPE_GLOBAL && (ctx->dims[0] * ctx->dims[1] > 1))
        comm = ctx->nccl_global;
    else if (scope == PDHCG_SCOPE_COL && ctx->dims[0] > 1)
        comm = ctx->nccl_col;
    else if (scope == PDHCG_SCOPE_ROW && ctx->dims[1] > 1)
        comm = ctx->nccl_row;
    else
        return;

    NCCL_CHECK(ncclAllReduce(buf, buf, count, ncclDouble, nccl_op, comm, (cudaStream_t)stream));
}

void pdhcg_all_reduce_scalar(
    grid_context_t *ctx, double *value, pdhcg_reduce_op_t op, pdhcg_comm_scope_t scope, bool on_device)
{
    if (!ctx)
        return;

    if (on_device)
    {
        ncclRedOp_t nccl_op = (op == PDHCG_OP_MAX) ? ncclMax : ncclSum;
        ncclComm_t comm;

        if (scope == PDHCG_SCOPE_GLOBAL && (ctx->dims[0] * ctx->dims[1] > 1))
            comm = ctx->nccl_global;
        else if (scope == PDHCG_SCOPE_COL && ctx->dims[0] > 1)
            comm = ctx->nccl_col;
        else if (scope == PDHCG_SCOPE_ROW && ctx->dims[1] > 1)
            comm = ctx->nccl_row;
        else
            return;

        NCCL_CHECK(ncclAllReduce(value, value, 1, ncclDouble, nccl_op, comm, 0));
    }
    else
    {
        MPI_Op mpi_op = (op == PDHCG_OP_MAX) ? MPI_MAX : MPI_SUM;
        MPI_Comm comm;

        if (scope == PDHCG_SCOPE_GLOBAL && (ctx->dims[0] * ctx->dims[1] > 1))
            comm = ctx->comm_global;
        else if (scope == PDHCG_SCOPE_COL && ctx->dims[0] > 1)
            comm = ctx->comm_col;
        else if (scope == PDHCG_SCOPE_ROW && ctx->dims[1] > 1)
            comm = ctx->comm_row;
        else
            return;

        double local_val = *value;
        MPI_Allreduce(&local_val, value, 1, MPI_DOUBLE, mpi_op, comm);
    }
}

int pdhcg_get_grid_p_col(struct grid_context_s *ctx)
{
    return ctx ? ctx->dims[1] : 1;
}

int pdhcg_get_grid_row_coord(struct grid_context_s *ctx)
{
    return ctx ? ctx->coords[0] : 0;
}

int pdhcg_get_global_num_variables(grid_context_t *ctx)
{
    return ctx ? ctx->global_num_variables : 0;
}

int pdhcg_get_variable_start(grid_context_t *ctx)
{
    return ctx ? ctx->n_start : 0;
}

int pdhcg_get_global_num_cones(grid_context_t *ctx)
{
    return ctx ? ctx->global_num_cones : 0;
}

int pdhcg_get_global_num_affine_cones(grid_context_t *ctx)
{
    return ctx ? ctx->global_num_affine_cones : 0;
}

pdhcg_result_t *pdhcg_distributed_optimize(const pdhg_parameters_t *params, const qp_problem_t *original_problem)
{
    return distributed_optimize(params, original_problem);
}
