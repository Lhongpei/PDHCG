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

#include "internal_types.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C"
{
#endif

    bool permute_problem(qp_problem_t *qp, int *row_perm, int *col_perm);

    qp_problem_t *permute_problem_return_new(const qp_problem_t *qp, int *row_perm, int *col_perm);

    void generate_cone_aware_permutation(const qp_problem_t *qp, permute_method_t method, int block_size, int *perm);
    void generate_affine_cone_aware_row_permutation(const qp_problem_t *qp,
                                                    permute_method_t method,
                                                    int block_size,
                                                    int *perm);
    bool validate_cone_permutation(const qp_problem_t *qp, const int *col_perm);
    bool validate_affine_cone_row_permutation(const qp_problem_t *qp, const int *row_perm);
    void repermute_solution(pdhcg_result_t *result, int *row_perm, int *col_perm);
#ifdef __cplusplus
}

#endif
