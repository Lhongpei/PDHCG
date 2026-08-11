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

#include "pdhcg_types.h"
#include <mpi.h>
#include <nccl.h>

typedef struct distributed_cone_partition_s
{
    int num_cones;
    int *v_dim;
    cone_type_t *type;
    unsigned char *fixed_mask;
    int *local_start;
    int *local_first;
    int *local_count;
} distributed_cone_partition_t;

enum
{
    PDHCG_DIST_CONE_FIXED_AUX0 = 1,
    PDHCG_DIST_CONE_FIXED_AUX1 = 2,
    PDHCG_DIST_CONE_FIXED_VECTOR = 4
};

struct grid_context_s
{
    MPI_Comm comm_global;
    MPI_Comm comm_row;
    MPI_Comm comm_col;
    ncclComm_t nccl_row;
    ncclComm_t nccl_col;
    ncclComm_t nccl_global;
    int rank_global;
    int coords[2];
    int dims[2];
    int global_num_variables;
    int global_num_cones;
    int global_num_affine_cones;
    int n_start;
    int n_end;
    int *variable_cuts;
    int *constraint_cuts;
    distributed_cone_partition_t split_cones;
    distributed_cone_partition_t split_affine_cones;
};
