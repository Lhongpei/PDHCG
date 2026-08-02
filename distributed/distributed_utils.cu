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

#include "distributed_types.h"
#include "distributed_utils.h"
#include "internal_types.h"
#include "solver_state.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C"
{
    ncclComm_t init_nccl(MPI_Comm mpi_comm)
    {
        ncclUniqueId id;
        ncclComm_t nccl_comm;
        int rank, nranks;

        MPI_Comm_rank(mpi_comm, &rank);
        MPI_Comm_size(mpi_comm, &nranks);

        if (rank == 0)
        {
            NCCL_CHECK(ncclGetUniqueId(&id));
        }

        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);
        NCCL_CHECK(ncclCommInitRank(&nccl_comm, nranks, id, rank));

        return nccl_comm;
    }

    grid_context_t initialize_parallel_context(int P_row, int P_col)
    {
        grid_context_t grid;
        memset(&grid, 0, sizeof(grid));
        int initialized;
        int world_size;

        MPI_Initialized(&initialized);
        if (!initialized)
        {
            MPI_Init(NULL, NULL);
        }

        grid.comm_global = MPI_COMM_WORLD;
        MPI_Comm_rank(grid.comm_global, &grid.rank_global);
        MPI_Comm_size(grid.comm_global, &world_size);

        grid.dims[0] = P_row;
        grid.dims[1] = P_col;

        int num_devices;
        CUDA_CHECK(cudaGetDeviceCount(&num_devices));
        int local_device_id = grid.rank_global % num_devices;
        CUDA_CHECK(cudaSetDevice(local_device_id));

        char hostname[MPI_MAX_PROCESSOR_NAME];
        int hostname_len;
        MPI_Get_processor_name(hostname, &hostname_len);

        int *all_device_ids = (int *)malloc(world_size * sizeof(int));
        char *all_hostnames = (char *)malloc(world_size * MPI_MAX_PROCESSOR_NAME);

        MPI_Allgather(&local_device_id, 1, MPI_INT, all_device_ids, 1, MPI_INT, grid.comm_global);
        MPI_Allgather(hostname,
                      MPI_MAX_PROCESSOR_NAME,
                      MPI_CHAR,
                      all_hostnames,
                      MPI_MAX_PROCESSOR_NAME,
                      MPI_CHAR,
                      grid.comm_global);

        if (grid.rank_global == 0)
        {
            int conflict_found = 0;
            for (int i = 0; i < world_size; i++)
            {
                for (int j = i + 1; j < world_size; j++)
                {
                    char *host_i = &all_hostnames[i * MPI_MAX_PROCESSOR_NAME];
                    char *host_j = &all_hostnames[j * MPI_MAX_PROCESSOR_NAME];

                    if (strcmp(host_i, host_j) == 0 && all_device_ids[i] == all_device_ids[j])
                    {
                        fprintf(stderr,
                                "\n[WARNING] GPU CONFLICT: Rank %d and Rank %d both bound to GPU %d on %s\n",
                                i,
                                j,
                                all_device_ids[i],
                                host_i);
                        conflict_found = 1;
                    }
                }
            }
            if (conflict_found)
            {
                fprintf(stderr,
                        "[WARNING] Multiple ranks sharing GPU will cause severe performance degradation \n"
                        "          and potential OOM (like your current situation). \n"
                        "[HINT]    Ensure: mpirun -n <num_procs_per_node> ≤ %d (GPUs per node)\n"
                        "          Or use CUDA MPS: export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps\n",
                        num_devices);
            }
        }

        free(all_device_ids);
        free(all_hostnames);
        MPI_Barrier(grid.comm_global);

        int my_row = grid.rank_global / P_col;
        int my_col = grid.rank_global % P_col;

        grid.coords[0] = my_row;
        grid.coords[1] = my_col;

        MPI_Comm_split(grid.comm_global, my_row, grid.rank_global, &grid.comm_row);
        MPI_Comm_split(grid.comm_global, my_col, grid.rank_global, &grid.comm_col);
        grid.nccl_row = init_nccl(grid.comm_row);
        grid.nccl_col = init_nccl(grid.comm_col);
        grid.nccl_global = init_nccl(grid.comm_global);
        return grid;
    }
}

void destroy_parallel_context(grid_context_t *grid)
{
    if (!grid)
        return;

    free(grid->variable_cuts);
    free(grid->constraint_cuts);
    free(grid->split_cones.global_start);
    free(grid->split_cones.v_dim);
    free(grid->split_cones.type);
    free(grid->split_cones.fixed_mask);
    free(grid->split_cones.local_start);
    free(grid->split_cones.local_first);
    free(grid->split_cones.local_count);

    NCCL_CHECK(ncclCommDestroy(grid->nccl_row));
    NCCL_CHECK(ncclCommDestroy(grid->nccl_col));
    NCCL_CHECK(ncclCommDestroy(grid->nccl_global));
    if (grid->comm_row != MPI_COMM_NULL)
        MPI_Comm_free(&grid->comm_row);
    if (grid->comm_col != MPI_COMM_NULL)
        MPI_Comm_free(&grid->comm_col);
    memset(grid, 0, sizeof(*grid));
}

int *get_balanced_cuts(const int *weights, int total_dim, int num_partitions)
{
    int *cuts = (int *)malloc((num_partitions + 1) * sizeof(int));
    cuts[0] = 0;
    cuts[num_partitions] = total_dim;

    if (num_partitions == 1)
        return cuts;

    long long total_weight = 0;
    for (int i = 0; i < total_dim; i++)
        total_weight += weights[i];

    double target_per_part = (double)total_weight / num_partitions;
    long long current_cumulative = 0;
    int partition_idx = 1;

    for (int i = 0; i < total_dim; i++)
    {
        current_cumulative += weights[i];

        if (current_cumulative >= partition_idx * target_per_part)
        {
            cuts[partition_idx] = i + 1;
            partition_idx++;
            if (partition_idx >= num_partitions)
                break;
        }
    }

    while (partition_idx < num_partitions)
    {
        cuts[partition_idx] = total_dim;
        partition_idx++;
    }

    return cuts;
}

static int distributed_cone_length(const cone_blocks_t *cones, int cone)
{
    return (cones->type[cone] == CONE_EXPONENTIAL || cones->type[cone] == CONE_POWER) ? 3 : cones->v_dim[cone] + 2;
}

static unsigned char distributed_cone_fixed_mask(const qp_problem_t *problem, int cone)
{
    if (!problem->cones.is_fixed)
        return 0;

    int start = problem->cones.start_idx[cone];
    int k = problem->cones.v_dim[cone];
    int length = distributed_cone_length(&problem->cones, cone);
    unsigned char mask = 0;
    for (int slot = 0; slot < length - 2; ++slot)
        if (problem->cones.is_fixed[start + slot])
            mask |= PDHCG_DIST_CONE_FIXED_VECTOR;
    if (problem->cones.is_fixed[start + k])
        mask |= PDHCG_DIST_CONE_FIXED_AUX0;
    if (problem->cones.is_fixed[start + k + 1])
        mask |= PDHCG_DIST_CONE_FIXED_AUX1;
    return mask;
}

static bool objective_is_pure_diagonal(const qp_problem_t *problem)
{
    if (problem->num_rank_lowrank_obj > 0 || !problem->objective_sparse_matrix)
        return problem->num_rank_lowrank_obj == 0;
    for (int row = 0; row < problem->num_variables; ++row)
    {
        for (int nz = problem->objective_sparse_matrix->row_ptr[row];
             nz < problem->objective_sparse_matrix->row_ptr[row + 1];
             ++nz)
        {
            if (problem->objective_sparse_matrix->col_ind[nz] != row &&
                problem->objective_sparse_matrix->val[nz] != 0.0)
                return false;
        }
    }
    return true;
}

static bool cone_has_diagonal_objective(const qp_problem_t *problem, int cone)
{
    if (!problem->objective_sparse_matrix || !objective_is_pure_diagonal(problem))
        return false;
    int start = problem->cones.start_idx[cone];
    int end = start + distributed_cone_length(&problem->cones, cone);
    for (int row = start; row < end; ++row)
    {
        for (int nz = problem->objective_sparse_matrix->row_ptr[row];
             nz < problem->objective_sparse_matrix->row_ptr[row + 1];
             ++nz)
        {
            if (problem->objective_sparse_matrix->col_ind[nz] == row &&
                problem->objective_sparse_matrix->val[nz] != 0.0)
                return true;
        }
    }
    return false;
}

static bool cone_can_span_gpus(const qp_problem_t *problem, int cone, const pdhg_parameters_t *params)
{
    cone_type_t type = problem->cones.type[cone];
    if (type != CONE_STANDARD_SOC && type != CONE_ROTATED_SOC)
        return false;
    if (params->heterogeneous_cone_scaling || cone_has_diagonal_objective(problem, cone))
        return false;

    unsigned char fixed = distributed_cone_fixed_mask(problem, cone);
    if (fixed & PDHCG_DIST_CONE_FIXED_VECTOR)
        return false;
    if (type == CONE_STANDARD_SOC)
        return fixed == 0 || fixed == PDHCG_DIST_CONE_FIXED_AUX1 ||
            fixed == (PDHCG_DIST_CONE_FIXED_AUX0 | PDHCG_DIST_CONE_FIXED_AUX1);
    return fixed == 0 || fixed == (PDHCG_DIST_CONE_FIXED_AUX0 | PDHCG_DIST_CONE_FIXED_AUX1);
}

static int *get_uniform_cuts(int total_dim, int num_partitions)
{
    int *cuts = (int *)malloc((size_t)(num_partitions + 1) * sizeof(int));
    cuts[0] = 0;
    cuts[num_partitions] = total_dim;
    int chunk = total_dim / num_partitions;
    for (int part = 1; part < num_partitions; ++part)
        cuts[part] = part * chunk;
    return cuts;
}

static int find_partition(const int *cuts, int num_partitions, int index)
{
    for (int part = 0; part < num_partitions; ++part)
        if (index >= cuts[part] && index < cuts[part + 1])
            return part;
    return num_partitions - 1;
}

void configure_partition_metadata(const qp_problem_t *problem, grid_context_t *grid, const pdhg_parameters_t *params)
{
    int n = problem->num_variables;
    int m = problem->num_constraints;
    int P_cols = grid->dims[1];
    int P_rows = grid->dims[0];
    grid->global_num_cones = problem->cones.num_cones;

    free(grid->variable_cuts);
    free(grid->constraint_cuts);
    grid->variable_cuts = NULL;
    grid->constraint_cuts = NULL;

    if (params->partition_method == NNZ_BALANCE_PARTITION)
    {
        int *col_weights = (int *)calloc((size_t)n, sizeof(int));
        int *row_weights = (int *)calloc((size_t)m, sizeof(int));
        if (problem->constraint_matrix)
        {
            for (int row = 0; row < m; ++row)
            {
                row_weights[row] =
                    problem->constraint_matrix->row_ptr[row + 1] - problem->constraint_matrix->row_ptr[row];
                for (int nz = problem->constraint_matrix->row_ptr[row];
                     nz < problem->constraint_matrix->row_ptr[row + 1];
                     ++nz)
                    ++col_weights[problem->constraint_matrix->col_ind[nz]];
            }
        }
        grid->variable_cuts = get_balanced_cuts(col_weights, n, P_cols);
        grid->constraint_cuts = get_balanced_cuts(row_weights, m, P_rows);
        free(col_weights);
        free(row_weights);
    }
    else
    {
        grid->variable_cuts = get_uniform_cuts(n, P_cols);
        grid->constraint_cuts = get_uniform_cuts(m, P_rows);
    }

    int target_variables = (n + P_cols - 1) / P_cols;
    for (int cone = 0; cone < problem->cones.num_cones; ++cone)
    {
        int start = problem->cones.start_idx[cone];
        int end = start + distributed_cone_length(&problem->cones, cone);
        bool may_split = cone_can_span_gpus(problem, cone, params) && (end - start > target_variables);
        if (may_split)
            continue;

        int midpoint = start + (end - start) / 2;
        for (int cut = 1; cut < P_cols; ++cut)
        {
            if (grid->variable_cuts[cut] > start && grid->variable_cuts[cut] < end)
                grid->variable_cuts[cut] = (grid->variable_cuts[cut] <= midpoint) ? start : end;
        }
    }
    for (int cut = 1; cut < P_cols; ++cut)
        if (grid->variable_cuts[cut] < grid->variable_cuts[cut - 1])
            grid->variable_cuts[cut] = grid->variable_cuts[cut - 1];

    int empty_variable_partition = 0;
    int empty_constraint_partition = 0;
    for (int part = 0; part < P_cols; ++part)
        empty_variable_partition |= grid->variable_cuts[part] == grid->variable_cuts[part + 1];
    for (int part = 0; part < P_rows; ++part)
        empty_constraint_partition |= grid->constraint_cuts[part] == grid->constraint_cuts[part + 1];
    if (empty_variable_partition || empty_constraint_partition)
    {
        if (grid->rank_global == 0)
        {
            fprintf(stderr,
                    "Error: the requested %d x %d process grid creates an empty %s partition "
                    "(problem dimensions %d x %d). Use fewer row/column tiles; zero-width local "
                    "partitions are not supported.\n",
                    P_rows,
                    P_cols,
                    empty_variable_partition ? "variable" : "constraint",
                    m,
                    n);
        }
        MPI_Abort(grid->comm_global, EXIT_FAILURE);
    }

    int my_col = grid->coords[1];
    grid->n_start = grid->variable_cuts[my_col];
    grid->n_end = grid->variable_cuts[my_col + 1];

    distributed_cone_partition_t *split = &grid->split_cones;
    free(split->global_start);
    free(split->v_dim);
    free(split->type);
    free(split->fixed_mask);
    free(split->local_start);
    free(split->local_first);
    free(split->local_count);
    memset(split, 0, sizeof(*split));

    for (int cone = 0; cone < problem->cones.num_cones; ++cone)
    {
        int start = problem->cones.start_idx[cone];
        int end = start + distributed_cone_length(&problem->cones, cone);
        if (find_partition(grid->variable_cuts, P_cols, start) != find_partition(grid->variable_cuts, P_cols, end - 1))
            ++split->num_cones;
    }

    int K = split->num_cones;
    if (K == 0)
        return;
    split->global_start = (int *)malloc((size_t)K * sizeof(int));
    split->v_dim = (int *)malloc((size_t)K * sizeof(int));
    split->type = (cone_type_t *)malloc((size_t)K * sizeof(cone_type_t));
    split->fixed_mask = (unsigned char *)malloc((size_t)K * sizeof(unsigned char));
    split->local_start = (int *)malloc((size_t)K * sizeof(int));
    split->local_first = (int *)malloc((size_t)K * sizeof(int));
    split->local_count = (int *)malloc((size_t)K * sizeof(int));

    int out = 0;
    for (int cone = 0; cone < problem->cones.num_cones; ++cone)
    {
        int start = problem->cones.start_idx[cone];
        int length = distributed_cone_length(&problem->cones, cone);
        int end = start + length;
        if (find_partition(grid->variable_cuts, P_cols, start) == find_partition(grid->variable_cuts, P_cols, end - 1))
            continue;
        if (!cone_can_span_gpus(problem, cone, params))
        {
            fprintf(stderr, "Error: unsupported cone %d crossed a GPU partition boundary.\n", cone);
            MPI_Abort(grid->comm_global, EXIT_FAILURE);
        }

        int intersection_start = (start > grid->n_start) ? start : grid->n_start;
        int intersection_end = (end < grid->n_end) ? end : grid->n_end;
        int local_count = (intersection_end > intersection_start) ? intersection_end - intersection_start : 0;
        split->global_start[out] = start;
        split->v_dim[out] = problem->cones.v_dim[cone];
        split->type[out] = problem->cones.type[cone];
        split->fixed_mask[out] = distributed_cone_fixed_mask(problem, cone);
        split->local_start[out] = local_count > 0 ? intersection_start - grid->n_start : 0;
        split->local_first[out] = local_count > 0 ? intersection_start - start : 0;
        split->local_count[out] = local_count;
        ++out;
    }
}

CsrComponent *
extract_csr_component(int row_start, int row_end, int col_start, int col_end, const CsrComponent *src, int *out_nnz)
{
    if (!src || !src->row_ptr)
    {
        *out_nnz = 0;
        return NULL;
    }

    int m_sub = row_end - row_start;
    int nnz_count = 0;

    for (int i = row_start; i < row_end; i++)
    {
        for (int jj = src->row_ptr[i]; jj < src->row_ptr[i + 1]; jj++)
        {
            int col = src->col_ind[jj];
            if (col >= col_start && col < col_end)
            {
                nnz_count++;
            }
        }
    }
    *out_nnz = nnz_count;

    CsrComponent *sub = (CsrComponent *)malloc(sizeof(CsrComponent));
    sub->row_ptr = (int *)malloc((m_sub + 1) * sizeof(int));

    int alloc_nnz = (nnz_count > 0) ? nnz_count : 1;
    sub->col_ind = (int *)malloc(alloc_nnz * sizeof(int));
    sub->val = (double *)malloc(alloc_nnz * sizeof(double));
    if (nnz_count == 0)
    {
        sub->col_ind[0] = 0;
        sub->val[0] = 0.0;
    }

    sub->row_ptr[0] = 0;
    int current_nnz = 0;

    for (int i = row_start; i < row_end; i++)
    {
        for (int jj = src->row_ptr[i]; jj < src->row_ptr[i + 1]; jj++)
        {
            int col = src->col_ind[jj];
            if (col >= col_start && col < col_end)
            {
                sub->col_ind[current_nnz] = col - col_start;
                sub->val[current_nnz] = src->val[jj];
                current_nnz++;
            }
        }
        sub->row_ptr[i - row_start + 1] = current_nnz;
    }

    return sub;
}

double *copy_slice(const double *src, int start, int count)
{
    if (count <= 0)
        return NULL;
    double *dst = (double *)malloc(count * sizeof(double));
    memcpy(dst, src + start, count * sizeof(double));
    return dst;
}

qp_problem_t *partition_qp_problem(const qp_problem_t *global_qp,
                                   const grid_context_t *grid,
                                   partition_method_t method,
                                   int *out_n_start,
                                   int *out_m_start)
{
    qp_problem_t *loc = (qp_problem_t *)calloc(1, sizeof(qp_problem_t));

    int my_row_idx = grid->coords[0];
    int my_col_idx = grid->coords[1];
    int P_rows = grid->dims[0];
    int P_cols = grid->dims[1];

    int n_total = global_qp->num_variables;
    int m_total = global_qp->num_constraints;
    int n_start, n_end, m_start, m_end;

    int *owned_col_cuts = NULL;
    int *owned_row_cuts = NULL;
    const int *col_cuts = grid->variable_cuts;
    const int *row_cuts = grid->constraint_cuts;

    if (!col_cuts || !row_cuts)
    {
        if (method == NNZ_BALANCE_PARTITION)
        {
            int *col_weights = (int *)calloc((size_t)n_total, sizeof(int));
            int *row_weights = (int *)calloc((size_t)m_total, sizeof(int));
            if (global_qp->constraint_matrix)
            {
                for (int i = 0; i < m_total; ++i)
                {
                    row_weights[i] =
                        global_qp->constraint_matrix->row_ptr[i + 1] - global_qp->constraint_matrix->row_ptr[i];
                    for (int nz = global_qp->constraint_matrix->row_ptr[i];
                         nz < global_qp->constraint_matrix->row_ptr[i + 1];
                         ++nz)
                        ++col_weights[global_qp->constraint_matrix->col_ind[nz]];
                }
            }
            owned_col_cuts = get_balanced_cuts(col_weights, n_total, P_cols);
            owned_row_cuts = get_balanced_cuts(row_weights, m_total, P_rows);
            free(col_weights);
            free(row_weights);
        }
        else
        {
            owned_col_cuts = get_uniform_cuts(n_total, P_cols);
            owned_row_cuts = get_uniform_cuts(m_total, P_rows);
        }
        col_cuts = owned_col_cuts;
        row_cuts = owned_row_cuts;
    }

    n_start = col_cuts[my_col_idx];
    n_end = col_cuts[my_col_idx + 1];
    m_start = row_cuts[my_row_idx];
    m_end = row_cuts[my_row_idx + 1];

    if (out_n_start)
        *out_n_start = n_start;
    if (out_m_start)
        *out_m_start = m_start;

    loc->num_variables = n_end - n_start;
    loc->num_constraints = m_end - m_start;
    loc->objective_constant = global_qp->objective_constant;

    loc->constraint_matrix = extract_csr_component(
        m_start, m_end, n_start, n_end, global_qp->constraint_matrix, &loc->constraint_matrix_num_nonzeros);

    loc->objective_sparse_matrix = extract_csr_component(
        0, n_total, n_start, n_end, global_qp->objective_sparse_matrix, &loc->objective_sparse_matrix_num_nonzeros);

    loc->num_rank_lowrank_obj = global_qp->num_rank_lowrank_obj;
    loc->objective_lowrank_matrix = extract_csr_component(0,
                                                          loc->num_rank_lowrank_obj,
                                                          n_start,
                                                          n_end,
                                                          global_qp->objective_lowrank_matrix,
                                                          &loc->objective_lowrank_matrix_num_nonzeros);

    loc->objective_lowrank_middle_matrix = NULL;
    loc->objective_lowrank_middle_matrix_num_nonzeros = 0;
    if (global_qp->objective_lowrank_middle_matrix && loc->num_rank_lowrank_obj > 0 &&
        global_qp->objective_lowrank_middle_matrix_num_nonzeros > 0)
    {
        int k = loc->num_rank_lowrank_obj;
        int nnz = global_qp->objective_lowrank_middle_matrix_num_nonzeros;
        loc->objective_lowrank_middle_matrix_num_nonzeros = nnz;
        loc->objective_lowrank_middle_matrix = (CsrComponent *)malloc(sizeof(CsrComponent));
        loc->objective_lowrank_middle_matrix->row_ptr = (int *)malloc((size_t)(k + 1) * sizeof(int));
        loc->objective_lowrank_middle_matrix->col_ind = (int *)malloc((size_t)nnz * sizeof(int));
        loc->objective_lowrank_middle_matrix->val = (double *)malloc((size_t)nnz * sizeof(double));
        memcpy(loc->objective_lowrank_middle_matrix->row_ptr,
               global_qp->objective_lowrank_middle_matrix->row_ptr,
               (size_t)(k + 1) * sizeof(int));
        memcpy(loc->objective_lowrank_middle_matrix->col_ind,
               global_qp->objective_lowrank_middle_matrix->col_ind,
               (size_t)nnz * sizeof(int));
        memcpy(loc->objective_lowrank_middle_matrix->val,
               global_qp->objective_lowrank_middle_matrix->val,
               (size_t)nnz * sizeof(double));
    }

    loc->objective_vector = copy_slice(global_qp->objective_vector, n_start, loc->num_variables);
    loc->variable_lower_bound = copy_slice(global_qp->variable_lower_bound, n_start, loc->num_variables);
    loc->variable_upper_bound = copy_slice(global_qp->variable_upper_bound, n_start, loc->num_variables);
    loc->constraint_lower_bound = copy_slice(global_qp->constraint_lower_bound, m_start, loc->num_constraints);
    loc->constraint_upper_bound = copy_slice(global_qp->constraint_upper_bound, m_start, loc->num_constraints);

    if (global_qp->primal_start)
        loc->primal_start = copy_slice(global_qp->primal_start, n_start, loc->num_variables);
    if (global_qp->dual_start)
        loc->dual_start = copy_slice(global_qp->dual_start, m_start, loc->num_constraints);

    loc->num_original_variables = 0;
    if (global_qp->num_original_variables > n_start)
    {
        int original_end = global_qp->num_original_variables < n_end ? global_qp->num_original_variables : n_end;
        loc->num_original_variables = original_end - n_start;
    }

    int local_cones = 0;
    for (int cone = 0; cone < global_qp->cones.num_cones; ++cone)
    {
        int start = global_qp->cones.start_idx[cone];
        int end = start + distributed_cone_length(&global_qp->cones, cone);
        if (start >= n_start && end <= n_end)
            ++local_cones;
    }
    loc->cones.num_cones = local_cones;
    if (local_cones > 0)
    {
        loc->cones.start_idx = (int *)malloc((size_t)local_cones * sizeof(int));
        loc->cones.v_dim = (int *)malloc((size_t)local_cones * sizeof(int));
        loc->cones.type = (cone_type_t *)malloc((size_t)local_cones * sizeof(cone_type_t));
        loc->cones.power_alpha = (double *)calloc((size_t)local_cones, sizeof(double));
        int out = 0;
        for (int cone = 0; cone < global_qp->cones.num_cones; ++cone)
        {
            int start = global_qp->cones.start_idx[cone];
            int end = start + distributed_cone_length(&global_qp->cones, cone);
            if (start < n_start || end > n_end)
                continue;
            loc->cones.start_idx[out] = start - n_start;
            loc->cones.v_dim[out] = global_qp->cones.v_dim[cone];
            loc->cones.type[out] = global_qp->cones.type[cone];
            if (global_qp->cones.power_alpha)
                loc->cones.power_alpha[out] = global_qp->cones.power_alpha[cone];
            ++out;
        }
    }
    if (global_qp->cones.is_fixed)
    {
        loc->cones.is_fixed = (char *)malloc((size_t)loc->num_variables * sizeof(char));
        memcpy(loc->cones.is_fixed, global_qp->cones.is_fixed + n_start, (size_t)loc->num_variables * sizeof(char));
    }

    free(owned_col_cuts);
    free(owned_row_cuts);

    return loc;
}

rescale_info_t *partition_rescale_info(rescale_info_t *global_info,
                                       const grid_context_t *grid,
                                       partition_method_t method,
                                       int *out_n_start,
                                       int *out_m_start)
{
    rescale_info_t *loc_info = (rescale_info_t *)calloc(1, sizeof(rescale_info_t));

    int n_start, m_start;

    loc_info->scaled_problem = partition_qp_problem(global_info->scaled_problem, grid, method, &n_start, &m_start);
    qp_problem_t *loc_lp = loc_info->scaled_problem;

    loc_info->var_rescale = copy_slice(global_info->var_rescale, n_start, loc_lp->num_variables);
    loc_info->con_rescale = copy_slice(global_info->con_rescale, m_start, loc_lp->num_constraints);

    if (out_n_start)
        *out_n_start = n_start;
    if (out_m_start)
        *out_m_start = m_start;

    loc_info->con_bound_rescale = global_info->con_bound_rescale;
    loc_info->obj_vec_rescale = global_info->obj_vec_rescale;
    loc_info->rescaling_time_sec = global_info->rescaling_time_sec;

    processed_qp_problem_t *global_processed = global_info->processed_problem;
    processed_qp_problem_t *loc_processed = (processed_qp_problem_t *)safe_calloc(1, sizeof(processed_qp_problem_t));

    loc_processed->num_variables = loc_lp->num_variables;
    loc_processed->num_constraints = loc_lp->num_constraints;
    loc_processed->num_rank_lowrank_obj = loc_lp->num_rank_lowrank_obj;
    loc_processed->objective_constant = loc_lp->objective_constant;

    loc_processed->constraint_matrix_num_nonzeros = loc_lp->constraint_matrix_num_nonzeros;
    loc_processed->objective_sparse_matrix_num_nonzeros = loc_lp->objective_sparse_matrix_num_nonzeros;
    loc_processed->objective_lowrank_matrix_num_nonzeros = loc_lp->objective_lowrank_matrix_num_nonzeros;

    loc_processed->variable_lower_bound = loc_lp->variable_lower_bound;
    loc_processed->variable_upper_bound = loc_lp->variable_upper_bound;
    loc_processed->objective_vector = loc_lp->objective_vector;
    loc_processed->constraint_lower_bound = loc_lp->constraint_lower_bound;
    loc_processed->constraint_upper_bound = loc_lp->constraint_upper_bound;
    loc_processed->primal_start = loc_lp->primal_start;
    loc_processed->dual_start = loc_lp->dual_start;
    loc_processed->constraint_matrix = loc_lp->constraint_matrix;
    loc_processed->objective_sparse_matrix = loc_lp->objective_sparse_matrix;
    loc_processed->objective_lowrank_matrix = loc_lp->objective_lowrank_matrix;
    loc_processed->quad_type = global_processed->quad_type;

    loc_processed->objective_lowrank_middle_kind = global_processed->objective_lowrank_middle_kind;
    loc_processed->objective_lowrank_middle_diag = NULL;
    loc_processed->objective_lowrank_middle_dense = NULL;
    if (global_processed->objective_lowrank_middle_kind == PDHCG_D_DIAG &&
        global_processed->objective_lowrank_middle_diag && loc_lp->num_rank_lowrank_obj > 0)
    {
        size_t bytes = (size_t)loc_lp->num_rank_lowrank_obj * sizeof(double);
        loc_processed->objective_lowrank_middle_diag = (double *)malloc(bytes);
        memcpy(loc_processed->objective_lowrank_middle_diag, global_processed->objective_lowrank_middle_diag, bytes);
    }
    else if (global_processed->objective_lowrank_middle_kind == PDHCG_D_DENSE &&
             global_processed->objective_lowrank_middle_dense && loc_lp->num_rank_lowrank_obj > 0)
    {
        size_t bytes = (size_t)loc_lp->num_rank_lowrank_obj * (size_t)loc_lp->num_rank_lowrank_obj * sizeof(double);
        loc_processed->objective_lowrank_middle_dense = (double *)malloc(bytes);
        memcpy(loc_processed->objective_lowrank_middle_dense, global_processed->objective_lowrank_middle_dense, bytes);
    }

    if (global_processed->quad_type == PDHCG_DIAG_Q && global_processed->diagonal_quad_objective != NULL)
    {
        loc_processed->diagonal_quad_objective =
            copy_slice(global_processed->diagonal_quad_objective, n_start, loc_lp->num_variables);
    }
    else
    {
        loc_processed->diagonal_quad_objective = NULL;
    }

    loc_info->processed_problem = loc_processed;

    return loc_info;
}

size_t get_qp_problem_size(const qp_problem_t *qp)
{
    if (!qp)
        return 0;
    size_t size = 0;

    size += sizeof(int) * 6 + sizeof(double);
    size += sizeof(int) * 2;

    int num_cones = qp->cones.num_cones;
    size += (size_t)num_cones * (sizeof(int) * 2 + sizeof(cone_type_t));
    size += sizeof(int);
    if (qp->cones.power_alpha)
        size += (size_t)num_cones * sizeof(double);
    size += sizeof(int);
    if (qp->cones.is_fixed)
        size += (size_t)qp->num_variables * sizeof(char);

    size += sizeof(double) * qp->num_variables * 3;
    size += sizeof(double) * qp->num_constraints * 2;

#define ADD_CSR_SIZE(csr, num_rows, nnz)                                                                               \
    {                                                                                                                  \
        size += sizeof(int);                                                                                           \
        if ((csr) && (csr)->row_ptr)                                                                                   \
        {                                                                                                              \
            int safe_nnz = (nnz) > 0 ? (nnz) : 1;                                                                      \
            size += sizeof(int) * ((num_rows) + 1);                                                                    \
            size += sizeof(int) * safe_nnz;                                                                            \
            size += sizeof(double) * safe_nnz;                                                                         \
        }                                                                                                              \
    }

    ADD_CSR_SIZE(qp->constraint_matrix, qp->num_constraints, qp->constraint_matrix_num_nonzeros);
    ADD_CSR_SIZE(qp->objective_sparse_matrix, qp->num_variables, qp->objective_sparse_matrix_num_nonzeros);
    ADD_CSR_SIZE(qp->objective_lowrank_matrix, qp->num_rank_lowrank_obj, qp->objective_lowrank_matrix_num_nonzeros);

    size += sizeof(int);
    ADD_CSR_SIZE(qp->objective_lowrank_middle_matrix,
                 qp->num_rank_lowrank_obj,
                 qp->objective_lowrank_middle_matrix_num_nonzeros);

    size += sizeof(int) * 2;
    if (qp->primal_start)
        size += sizeof(double) * qp->num_variables;
    if (qp->dual_start)
        size += sizeof(double) * qp->num_constraints;

    return size;
}

#define S_CSR(csr, num_rows, nnz)                                                                                      \
    {                                                                                                                  \
        int has_csr = ((csr) != NULL && (csr)->row_ptr != NULL);                                                       \
        S_COPY(has_csr, int);                                                                                          \
        if (has_csr)                                                                                                   \
        {                                                                                                              \
            S_ARR(csr->row_ptr, num_rows + 1, int);                                                                    \
            if ((nnz) > 0)                                                                                             \
            {                                                                                                          \
                S_ARR(csr->col_ind, nnz, int);                                                                         \
                S_ARR(csr->val, nnz, double);                                                                          \
            }                                                                                                          \
            else                                                                                                       \
            {                                                                                                          \
                S_COPY(0, int);                                                                                        \
                S_COPY(0.0, double);                                                                                   \
            }                                                                                                          \
        }                                                                                                              \
    }

void serialize_qp_problem_to_ptr(const qp_problem_t *qp, char **ptr_ref)
{
    char *ptr = *ptr_ref;

#define S_COPY(val, type)                                                                                              \
    {                                                                                                                  \
        *((type *)ptr) = val;                                                                                          \
        ptr += sizeof(type);                                                                                           \
    }
#define S_ARR(arr, count, type)                                                                                        \
    {                                                                                                                  \
        memcpy(ptr, arr, sizeof(type) * (count));                                                                      \
        ptr += sizeof(type) * (count);                                                                                 \
    }

    S_COPY(qp->num_variables, int);
    S_COPY(qp->num_constraints, int);
    S_COPY(qp->num_rank_lowrank_obj, int);
    S_COPY(qp->constraint_matrix_num_nonzeros, int);
    S_COPY(qp->objective_sparse_matrix_num_nonzeros, int);
    S_COPY(qp->objective_lowrank_matrix_num_nonzeros, int);
    S_COPY(qp->objective_constant, double);
    S_COPY(qp->num_original_variables, int);
    S_COPY(qp->cones.num_cones, int);
    if (qp->cones.num_cones > 0)
    {
        S_ARR(qp->cones.start_idx, qp->cones.num_cones, int);
        S_ARR(qp->cones.v_dim, qp->cones.num_cones, int);
        S_ARR(qp->cones.type, qp->cones.num_cones, cone_type_t);
    }
    int has_power_alpha = qp->cones.power_alpha != NULL;
    S_COPY(has_power_alpha, int);
    if (has_power_alpha)
        S_ARR(qp->cones.power_alpha, qp->cones.num_cones, double);
    int has_cone_fixed = qp->cones.is_fixed != NULL;
    S_COPY(has_cone_fixed, int);
    if (has_cone_fixed)
        S_ARR(qp->cones.is_fixed, qp->num_variables, char);

    S_ARR(qp->objective_vector, qp->num_variables, double);
    S_ARR(qp->variable_lower_bound, qp->num_variables, double);
    S_ARR(qp->variable_upper_bound, qp->num_variables, double);
    S_ARR(qp->constraint_lower_bound, qp->num_constraints, double);
    S_ARR(qp->constraint_upper_bound, qp->num_constraints, double);

    S_CSR(qp->constraint_matrix, qp->num_constraints, qp->constraint_matrix_num_nonzeros);
    S_CSR(qp->objective_sparse_matrix, qp->num_variables, qp->objective_sparse_matrix_num_nonzeros);
    S_CSR(qp->objective_lowrank_matrix, qp->num_rank_lowrank_obj, qp->objective_lowrank_matrix_num_nonzeros);

    S_COPY(qp->objective_lowrank_middle_matrix_num_nonzeros, int);
    S_CSR(qp->objective_lowrank_middle_matrix,
          qp->num_rank_lowrank_obj,
          qp->objective_lowrank_middle_matrix_num_nonzeros);

    int has_primal = (qp->primal_start != NULL);
    int has_dual = (qp->dual_start != NULL);
    S_COPY(has_primal, int);
    S_COPY(has_dual, int);
    if (has_primal)
        S_ARR(qp->primal_start, qp->num_variables, double);
    if (has_dual)
        S_ARR(qp->dual_start, qp->num_constraints, double);

    *ptr_ref = ptr;
}
#define D_CSR(target_ptr, num_rows, nnz)                                                                               \
    {                                                                                                                  \
        int has_csr;                                                                                                   \
        D_VAL(has_csr, int);                                                                                           \
        if (has_csr)                                                                                                   \
        {                                                                                                              \
            target_ptr = (CsrComponent *)malloc(sizeof(CsrComponent));                                                 \
            int safe_nnz = nnz > 0 ? nnz : 1;                                                                          \
            D_ARR(target_ptr->row_ptr, num_rows + 1, int);                                                             \
            D_ARR(target_ptr->col_ind, safe_nnz, int);                                                                 \
            D_ARR(target_ptr->val, safe_nnz, double);                                                                  \
        }                                                                                                              \
    }

qp_problem_t *deserialize_qp_problem_from_ptr(const char **ptr_ref)
{
    const char *ptr = *ptr_ref;
    qp_problem_t *qp = (qp_problem_t *)calloc(1, sizeof(qp_problem_t));

#define D_VAL(var, type)                                                                                               \
    {                                                                                                                  \
        var = *((type *)ptr);                                                                                          \
        ptr += sizeof(type);                                                                                           \
    }
#define D_ARR(dest, count, type)                                                                                       \
    {                                                                                                                  \
        dest = (type *)malloc(sizeof(type) * (count));                                                                 \
        memcpy(dest, ptr, sizeof(type) * (count));                                                                     \
        ptr += sizeof(type) * (count);                                                                                 \
    }

    D_VAL(qp->num_variables, int);
    D_VAL(qp->num_constraints, int);
    D_VAL(qp->num_rank_lowrank_obj, int);
    D_VAL(qp->constraint_matrix_num_nonzeros, int);
    D_VAL(qp->objective_sparse_matrix_num_nonzeros, int);
    D_VAL(qp->objective_lowrank_matrix_num_nonzeros, int);
    D_VAL(qp->objective_constant, double);
    D_VAL(qp->num_original_variables, int);
    D_VAL(qp->cones.num_cones, int);
    if (qp->cones.num_cones > 0)
    {
        D_ARR(qp->cones.start_idx, qp->cones.num_cones, int);
        D_ARR(qp->cones.v_dim, qp->cones.num_cones, int);
        D_ARR(qp->cones.type, qp->cones.num_cones, cone_type_t);
    }
    int has_power_alpha;
    D_VAL(has_power_alpha, int);
    if (has_power_alpha)
        D_ARR(qp->cones.power_alpha, qp->cones.num_cones, double);
    int has_cone_fixed;
    D_VAL(has_cone_fixed, int);
    if (has_cone_fixed)
        D_ARR(qp->cones.is_fixed, qp->num_variables, char);

    D_ARR(qp->objective_vector, qp->num_variables, double);
    D_ARR(qp->variable_lower_bound, qp->num_variables, double);
    D_ARR(qp->variable_upper_bound, qp->num_variables, double);
    D_ARR(qp->constraint_lower_bound, qp->num_constraints, double);
    D_ARR(qp->constraint_upper_bound, qp->num_constraints, double);

    D_CSR(qp->constraint_matrix, qp->num_constraints, qp->constraint_matrix_num_nonzeros);
    D_CSR(qp->objective_sparse_matrix, qp->num_variables, qp->objective_sparse_matrix_num_nonzeros);
    D_CSR(qp->objective_lowrank_matrix, qp->num_rank_lowrank_obj, qp->objective_lowrank_matrix_num_nonzeros);

    D_VAL(qp->objective_lowrank_middle_matrix_num_nonzeros, int);
    qp->objective_lowrank_middle_matrix = NULL;
    D_CSR(qp->objective_lowrank_middle_matrix,
          qp->num_rank_lowrank_obj,
          qp->objective_lowrank_middle_matrix_num_nonzeros);

    int has_primal, has_dual;
    D_VAL(has_primal, int);
    D_VAL(has_dual, int);
    if (has_primal)
        D_ARR(qp->primal_start, qp->num_variables, double);
    if (has_dual)
        D_ARR(qp->dual_start, qp->num_constraints, double);

    *ptr_ref = ptr;
    return qp;
}

size_t get_rescale_info_size(const rescale_info_t *info)
{
    if (!info)
        return 0;
    size_t size = 0;
    size += sizeof(double) * 3;
    int n = info->scaled_problem->num_variables;
    int m = info->scaled_problem->num_constraints;
    size += sizeof(double) * (n + m);

    size += get_qp_problem_size(info->scaled_problem);

    return size;
}

void serialize_rescale_info(const rescale_info_t *info, char *buffer)
{
    char *ptr = buffer;

    S_COPY(info->con_bound_rescale, double);
    S_COPY(info->obj_vec_rescale, double);
    S_COPY(info->rescaling_time_sec, double);

    serialize_qp_problem_to_ptr(info->scaled_problem, &ptr);

    int n = info->scaled_problem->num_variables;
    int m = info->scaled_problem->num_constraints;
    S_ARR(info->var_rescale, n, double);
    S_ARR(info->con_rescale, m, double);
}

rescale_info_t *deserialize_rescale_info(const char *buffer)
{
    const char *ptr = buffer;
    rescale_info_t *info = (rescale_info_t *)calloc(1, sizeof(rescale_info_t));

    D_VAL(info->con_bound_rescale, double);
    D_VAL(info->obj_vec_rescale, double);
    D_VAL(info->rescaling_time_sec, double);

    info->scaled_problem = deserialize_qp_problem_from_ptr(&ptr);

    int n = info->scaled_problem->num_variables;
    int m = info->scaled_problem->num_constraints;
    D_ARR(info->var_rescale, n, double);
    D_ARR(info->con_rescale, m, double);

    return info;
}

#define CHUNK_SIZE (1024 * 1024 * 1024)

void big_bcast_bytes(void **buffer_ptr, size_t *size_ptr, int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    int is_root = (rank == root);

    unsigned long long total_len = is_root ? *size_ptr : 0;
    MPI_Bcast(&total_len, 1, MPI_UNSIGNED_LONG_LONG, root, comm);

    if (!is_root)
    {
        *size_ptr = (size_t)total_len;
        *buffer_ptr = malloc(total_len);
    }

    char *buf = (char *)(*buffer_ptr);
    size_t offset = 0;

    while (offset < total_len)
    {
        size_t remaining = total_len - offset;
        int current_chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : (int)remaining;

        MPI_Bcast(buf + offset, current_chunk, MPI_BYTE, root, comm);

        offset += current_chunk;
    }
}

void big_send_bytes(const void *buffer, size_t size, int dest, MPI_Comm comm)
{
    unsigned long long total_len = size;
    MPI_Send(&total_len, 1, MPI_UNSIGNED_LONG_LONG, dest, 0, comm);

    const char *buf = (const char *)buffer;
    size_t offset = 0;
    while (offset < total_len)
    {
        size_t remaining = total_len - offset;
        int current_chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : (int)remaining;
        MPI_Send(buf + offset, current_chunk, MPI_BYTE, dest, 1, comm);
        offset += current_chunk;
    }
}

void big_recv_bytes(void **buffer_ptr, size_t *size_ptr, int source, MPI_Comm comm)
{
    unsigned long long total_len = 0;
    MPI_Recv(&total_len, 1, MPI_UNSIGNED_LONG_LONG, source, 0, comm, MPI_STATUS_IGNORE);

    *size_ptr = (size_t)total_len;
    *buffer_ptr = malloc((size_t)total_len);

    char *buf = (char *)(*buffer_ptr);
    size_t offset = 0;
    while (offset < total_len)
    {
        size_t remaining = total_len - offset;
        int current_chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : (int)remaining;
        MPI_Recv(buf + offset, current_chunk, MPI_BYTE, source, 1, comm, MPI_STATUS_IGNORE);
        offset += current_chunk;
    }
}

big_request_t big_isend_bytes(const void *buffer, size_t size, int dest, MPI_Comm comm)
{
    big_request_t breq;
    int num_chunks = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    breq.num_reqs = 1 + num_chunks;
    breq.reqs = (MPI_Request *)malloc(breq.num_reqs * sizeof(MPI_Request));

    unsigned long long *p_len = (unsigned long long *)malloc(sizeof(unsigned long long));
    *p_len = size;

    MPI_Isend(p_len, 1, MPI_UNSIGNED_LONG_LONG, dest, 0, comm, &breq.reqs[0]);

    const char *buf = (const char *)buffer;
    size_t offset = 0;
    int req_idx = 1;
    while (offset < size)
    {
        size_t remaining = size - offset;
        int current_chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : (int)remaining;
        MPI_Isend(buf + offset, current_chunk, MPI_BYTE, dest, 1, comm, &breq.reqs[req_idx++]);
        offset += current_chunk;
    }
    return breq;
}

void big_wait_bytes(big_request_t *breq, unsigned long long *p_len)
{
    if (breq->num_reqs > 0)
    {
        MPI_Waitall(breq->num_reqs, breq->reqs, MPI_STATUSES_IGNORE);
        free(breq->reqs);
        breq->num_reqs = 0;
    }
    if (p_len)
        free(p_len);
}

void distribute_data_bcast_then_partition(const qp_problem_t *working_problem,
                                          rescale_info_t *rescale_info,
                                          grid_context_t *grid_context,
                                          const pdhg_parameters_t *params,
                                          qp_problem_t **out_local_qp,
                                          rescale_info_t **out_local_resc)
{
    double t_start = MPI_Wtime();
    const qp_problem_t *current_working_problem = working_problem;
    rescale_info_t *current_rescale_info = rescale_info;
    int real_n_start = 0;

    {
        char *buf = NULL;
        size_t sz = 0;
        if (grid_context->rank_global == 0)
        {
            sz = get_qp_problem_size(current_working_problem);
            buf = (char *)malloc(sz);
            char *ptr_tmp = buf;
            serialize_qp_problem_to_ptr(current_working_problem, &ptr_tmp);
        }
        big_bcast_bytes((void **)&buf, &sz, 0, grid_context->comm_global);

        if (grid_context->rank_global != 0)
        {
            const char *ptr_tmp = buf;
            current_working_problem = deserialize_qp_problem_from_ptr(&ptr_tmp);
        }
        if (buf)
            free(buf);
    }

    grid_context->global_num_variables = current_working_problem->num_variables;
    configure_partition_metadata(current_working_problem, grid_context, params);

    {
        char *buf = NULL;
        size_t sz = 0;
        if (grid_context->rank_global == 0)
        {
            sz = get_rescale_info_size(current_rescale_info);
            buf = (char *)malloc(sz);
            serialize_rescale_info(current_rescale_info, buf);
        }
        big_bcast_bytes((void **)&buf, &sz, 0, grid_context->comm_global);

        if (grid_context->rank_global != 0)
        {
            current_rescale_info = deserialize_rescale_info(buf);
            current_rescale_info->processed_problem = preprocess_qp_problem(current_rescale_info->scaled_problem);
        }
        if (buf)
            free(buf);
    }

    *out_local_resc =
        partition_rescale_info(current_rescale_info, grid_context, params->partition_method, &real_n_start, NULL);
    *out_local_qp = partition_qp_problem(current_working_problem, grid_context, params->partition_method, NULL, NULL);
    grid_context->n_start = real_n_start;
    grid_context->n_end = real_n_start + (*out_local_qp)->num_variables;

    if (grid_context->rank_global != 0)
    {
        rescale_info_free(current_rescale_info);
        qp_problem_free((qp_problem_t *)current_working_problem);
    }

    double t_end = MPI_Wtime();
    if (params->verbose && grid_context->rank_global == 0)
    {
        printf("[Timer] Data Distribution (Bcast -> Partition) took %.3f seconds.\n", t_end - t_start);
    }
}

void distribute_data_partition_then_send(const qp_problem_t *working_problem,
                                         rescale_info_t *rescale_info,
                                         grid_context_t *grid_context,
                                         const pdhg_parameters_t *params,
                                         qp_problem_t **out_local_qp,
                                         rescale_info_t **out_local_resc)
{
    double t_start = MPI_Wtime();
    int world_size;
    MPI_Comm_size(grid_context->comm_global, &world_size);

    if (grid_context->rank_global == 0)
    {
        char *prev_buf_lp = NULL;
        char *prev_buf_resc = NULL;
        big_request_t req_lp = {NULL, 0};
        big_request_t req_resc = {NULL, 0};
        unsigned long long *prev_len_lp = NULL;
        unsigned long long *prev_len_resc = NULL;

        for (int r = 0; r < world_size; ++r)
        {
            grid_context_t target_grid = *grid_context;
            target_grid.rank_global = r;
            target_grid.coords[0] = r / grid_context->dims[1];
            target_grid.coords[1] = r % grid_context->dims[1];

            qp_problem_t *sub_qp =
                partition_qp_problem(working_problem, &target_grid, params->partition_method, NULL, NULL);
            rescale_info_t *sub_rescale =
                partition_rescale_info(rescale_info, &target_grid, params->partition_method, NULL, NULL);

            if (r == 0)
            {
                *out_local_qp = sub_qp;
                *out_local_resc = sub_rescale;
                continue;
            }

            size_t sz_lp = get_qp_problem_size(sub_qp);
            char *buf_lp = (char *)malloc(sz_lp);
            char *ptr_lp = buf_lp;
            serialize_qp_problem_to_ptr(sub_qp, &ptr_lp);

            size_t sz_resc = get_rescale_info_size(sub_rescale);
            char *buf_resc = (char *)malloc(sz_resc);
            serialize_rescale_info(sub_rescale, buf_resc);

            qp_problem_free(sub_qp);
            rescale_info_free(sub_rescale);

            if (req_lp.num_reqs > 0 || req_resc.num_reqs > 0)
            {
                big_wait_bytes(&req_lp, prev_len_lp);
                big_wait_bytes(&req_resc, prev_len_resc);
                free(prev_buf_lp);
                free(prev_buf_resc);
            }

            prev_len_lp = (unsigned long long *)malloc(sizeof(unsigned long long));
            *prev_len_lp = sz_lp;
            req_lp = big_isend_bytes(buf_lp, sz_lp, r, grid_context->comm_global);

            prev_len_resc = (unsigned long long *)malloc(sizeof(unsigned long long));
            *prev_len_resc = sz_resc;
            req_resc = big_isend_bytes(buf_resc, sz_resc, r, grid_context->comm_global);

            prev_buf_lp = buf_lp;
            prev_buf_resc = buf_resc;
        }

        if (req_lp.num_reqs > 0 || req_resc.num_reqs > 0)
        {
            big_wait_bytes(&req_lp, prev_len_lp);
            big_wait_bytes(&req_resc, prev_len_resc);
            free(prev_buf_lp);
            free(prev_buf_resc);
        }

        rescale_info_free(rescale_info);
    }
    else
    {
        char *buf_lp = NULL;
        size_t sz_lp = 0;
        big_recv_bytes((void **)&buf_lp, &sz_lp, 0, grid_context->comm_global);
        const char *ptr_lp = buf_lp;
        *out_local_qp = deserialize_qp_problem_from_ptr(&ptr_lp);
        free(buf_lp);

        char *buf_resc = NULL;
        size_t sz_resc = 0;
        big_recv_bytes((void **)&buf_resc, &sz_resc, 0, grid_context->comm_global);
        *out_local_resc = deserialize_rescale_info(buf_resc);
        free(buf_resc);
    }

    double t_end = MPI_Wtime();
    if (params->verbose && grid_context->rank_global == 0)
    {
        printf("[Timer] Data Distribution (Partition -> P2P Send) took %.3f "
               "seconds.\n",
               t_end - t_start);
    }
}

double compute_global_norm(cublasHandle_t blas_handle, int m_local, double *d_vec, MPI_Comm comm)
{
    double local_norm_sq = 0.0;
    double global_norm_sq = 0.0;

    CUBLAS_CHECK(cublasDdot(blas_handle, m_local, d_vec, 1, d_vec, 1, &local_norm_sq));

    MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, comm);

    return sqrt(global_norm_sq);
}

double compute_global_dot(cublasHandle_t blas_handle, int m_local, double *d_vec1, double *d_vec2, MPI_Comm comm)
{
    double local_dot = 0.0;
    double global_dot = 0.0;

    CUBLAS_CHECK(cublasDdot(blas_handle, m_local, d_vec1, 1, d_vec2, 1, &local_dot));
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);

    return global_dot;
}

void gather_distributed_vector(
    double *d_local_vec, int local_len, MPI_Comm comm_check, MPI_Comm comm_gather, double **result_ptr)
{
    int rank_check;
    MPI_Comm_rank(comm_check, &rank_check);

    if (rank_check == 0)
    {
        double *h_local = (double *)malloc(local_len * sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_local, d_local_vec, local_len * sizeof(double), cudaMemcpyDeviceToHost));

        int size_gather, rank_gather;
        MPI_Comm_size(comm_gather, &size_gather);
        MPI_Comm_rank(comm_gather, &rank_gather);

        int *counts = NULL;
        int *displs = NULL;
        double *h_global = NULL;

        if (rank_gather == 0)
        {
            counts = (int *)malloc(size_gather * sizeof(int));
            displs = (int *)malloc(size_gather * sizeof(int));
        }

        MPI_Gather(&local_len, 1, MPI_INT, counts, 1, MPI_INT, 0, comm_gather);

        if (rank_gather == 0)
        {
            int total_len = 0;
            for (int i = 0; i < size_gather; ++i)
            {
                displs[i] = total_len;
                total_len += counts[i];
            }
            h_global = (double *)malloc(total_len * sizeof(double));
        }
        MPI_Gatherv(h_local, local_len, MPI_DOUBLE, h_global, counts, displs, MPI_DOUBLE, 0, comm_gather);

        free(h_local);
        if (counts)
            free(counts);
        if (displs)
            free(displs);

        if (rank_gather == 0 && result_ptr != NULL)
        {
            *result_ptr = h_global;
        }
        else if (h_global)
        {
            free(h_global);
        }
    }
}

void print_distributed_params(const pdhg_parameters_t *params)
{
    if (params->verbose < 2)
        return;
    printf("------------------------------------ Distributed Configuration "
           "------------------------------------\n");

    if (params->grid_size.decided)
    {
        printf(
            "  Grid Size          : %d x %d (Rows x Cols)\n", params->grid_size.row_dims, params->grid_size.col_dims);
    }
    else
    {
        printf("  Grid Size          : Auto-detect (implementation dependent)\n");
    }

    printf("  Partition Method   : ");
    switch (params->partition_method)
    {
        case UNIFORM_PARTITION:
            printf("Uniform\n");
            break;
        case NNZ_BALANCE_PARTITION:
            printf("NNZ Balance\n");
            break;
        default:
            printf("Unknown (%d)\n", params->partition_method);
            break;
    }

    printf("  Permute Method     : ");
    switch (params->permute_method)
    {
        case NO_PERMUTATION:
            printf("None (Original ordering)\n");
            break;
        case FULL_RANDOM_PERMUTATION:
            printf("Full Random (Full Random shuffle)\n");
            break;
        case BLOCK_RANDOM_PERMUTATION:
            printf("Block Random (Block-wise Random shuffle)\n");
            break;
        default:
            printf("Unknown (%d)\n", params->permute_method);
            break;
    }

    printf("---------------------------------------------------------------------------------------------------\n");
}
