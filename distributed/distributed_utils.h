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
#include "distributed_types.h"
#include "internal_types.h"
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C"
{
#endif
#define NCCL_CHECK(cmd)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess)                                                                                          \
        {                                                                                                              \
            printf("NCCL failure %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                            \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
    grid_context_t initialize_parallel_context(int P_row, int P_col);
    void destroy_parallel_context(grid_context_t *grid);
    void
    configure_partition_metadata(const qp_problem_t *problem, grid_context_t *grid, const pdhg_parameters_t *params);
    rescale_info_t *partition_rescale_info(rescale_info_t *global_info,
                                           const grid_context_t *grid,
                                           partition_method_t method,
                                           int *out_n_start,
                                           int *out_m_start);
    qp_problem_t *partition_qp_problem(const qp_problem_t *global_qp,
                                       const grid_context_t *grid,
                                       partition_method_t method,
                                       int *out_n_start,
                                       int *out_m_start);

    rescale_info_t *deserialize_rescale_info(const char *buffer);
    void serialize_rescale_info(const rescale_info_t *info, char *buffer);
    size_t get_rescale_info_size(const rescale_info_t *info);
    qp_problem_t *deserialize_qp_problem_from_ptr(const char **ptr_ref);
    void serialize_qp_problem_to_ptr(const qp_problem_t *qp, char **ptr_ref);
    size_t get_qp_problem_size(const qp_problem_t *qp);
    void big_bcast_bytes(void **buffer_ptr, size_t *size_ptr, int root, MPI_Comm comm);
    void big_send_bytes(const void *buffer, size_t size, int dest, MPI_Comm comm);
    void big_recv_bytes(void **buffer_ptr, size_t *size_ptr, int source, MPI_Comm comm);
    big_request_t big_isend_bytes(const void *buffer, size_t size, int dest, MPI_Comm comm);
    void big_wait_bytes(big_request_t *breq, unsigned long long *p_len);
    void distribute_data_bcast_then_partition(const qp_problem_t *working_problem,
                                              rescale_info_t *rescale_info,
                                              grid_context_t *grid_context,
                                              const pdhg_parameters_t *params,
                                              qp_problem_t **out_local_qp,
                                              rescale_info_t **out_local_resc);
    void distribute_data_partition_then_send(const qp_problem_t *working_problem,
                                             rescale_info_t *rescale_info,
                                             grid_context_t *grid_context,
                                             const pdhg_parameters_t *params,
                                             qp_problem_t **out_local_qp,
                                             rescale_info_t **out_local_resc);
    void gather_distributed_vector(
        double *d_local_vec, int local_len, MPI_Comm comm_check, MPI_Comm comm_gather, double **result_ptr);
    void print_distributed_params(const pdhg_parameters_t *params);
#ifdef __cplusplus
}
#endif
