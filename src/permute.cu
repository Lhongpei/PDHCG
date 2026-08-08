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
#include "pdhcg.h"
#include "permute.h"
#include "utils.h"
#include <math.h>
#include <random>
#include <vector>

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

typedef struct
{
    int new_col;
    double val;
} permute_tuple_t;

static void generate_random_permutation(int n, int *perm);
static void generate_block_permutation(int n, int block_size, int *perm);

static int cmp_tuples(const void *a, const void *b)
{
    return ((permute_tuple_t *)a)->new_col - ((permute_tuple_t *)b)->new_col;
}

static void col_permute_in_place(int m, int *Ap, int *Aj, double *Ax, const int *old_col_to_new)
{
    int max_row_nnz = 0;
    for (int i = 0; i < m; i++)
    {
        int len = Ap[i + 1] - Ap[i];
        if (len > max_row_nnz)
            max_row_nnz = len;
    }

    permute_tuple_t *buffer = (permute_tuple_t *)malloc(max_row_nnz * sizeof(permute_tuple_t));

    for (int r = 0; r < m; r++)
    {
        int start = Ap[r];
        int end = Ap[r + 1];
        int len = end - start;

        if (len == 0)
            continue;

        for (int k = 0; k < len; k++)
        {
            int current_idx = start + k;
            int old_col = Aj[current_idx];

            buffer[k].new_col = old_col_to_new[old_col];
            buffer[k].val = Ax[current_idx];
        }

        if (len > 1)
        {
            qsort(buffer, len, sizeof(permute_tuple_t), cmp_tuples);
        }

        for (int k = 0; k < len; k++)
        {
            int current_idx = start + k;
            Aj[current_idx] = buffer[k].new_col;
            Ax[current_idx] = buffer[k].val;
        }
    }

    free(buffer);
}

static void permute_csr_rows_structural(CsrComponent *csr, int num_rows, int nnz, const int *row_perm)
{
    if (!csr || nnz == 0)
        return;

    int *new_Ap = (int *)malloc((num_rows + 1) * sizeof(int));
    int *new_Aj = (int *)malloc(nnz * sizeof(int));
    double *new_Ax = (double *)malloc(nnz * sizeof(double));

    new_Ap[0] = 0;
    int current_nz = 0;

    for (int i = 0; i < num_rows; i++)
    {
        int old_row_idx = row_perm[i];

        int start = csr->row_ptr[old_row_idx];
        int len = csr->row_ptr[old_row_idx + 1] - start;

        if (len > 0)
        {
            memcpy(&new_Aj[current_nz], &csr->col_ind[start], len * sizeof(int));
            memcpy(&new_Ax[current_nz], &csr->val[start], len * sizeof(double));
            current_nz += len;
        }

        new_Ap[i + 1] = current_nz;
    }

    free(csr->row_ptr);
    free(csr->col_ind);
    free(csr->val);

    csr->row_ptr = new_Ap;
    csr->col_ind = new_Aj;
    csr->val = new_Ax;
}

static void permute_double_array(double *arr, int n, const int *perm)
{
    if (!arr)
        return;
    double *tmp = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        tmp[i] = arr[perm[i]];
    memcpy(arr, tmp, n * sizeof(double));
    free(tmp);
}

static void compute_inv_perm(int n, const int *perm, int *inv_perm)
{
    for (int i = 0; i < n; i++)
        inv_perm[perm[i]] = i;
}

bool permute_problem(qp_problem_t *qp, int *row_perm, int *col_perm)
{
    if (!qp || (qp->num_constraints > 0 && !row_perm) || (qp->num_variables > 0 && !col_perm))
        return false;
    int m = qp->num_constraints;
    int n = qp->num_variables;

    if (!validate_cone_permutation(qp, col_perm))
    {
        fprintf(stderr, "Error: column permutation splits or reorders a cone block.\n");
        return false;
    }
    if (!validate_affine_cone_row_permutation(qp, row_perm))
    {
        fprintf(stderr, "Error: row permutation splits or reorders an affine cone block.\n");
        return false;
    }

    permute_double_array(qp->objective_vector, n, col_perm);
    permute_double_array(qp->variable_lower_bound, n, col_perm);
    permute_double_array(qp->variable_upper_bound, n, col_perm);
    if (qp->primal_start)
        permute_double_array(qp->primal_start, n, col_perm);

    permute_double_array(qp->constraint_lower_bound, m, row_perm);
    permute_double_array(qp->constraint_upper_bound, m, row_perm);
    permute_double_array(qp->affine_cone_offset, m, row_perm);
    if (qp->dual_start)
        permute_double_array(qp->dual_start, m, row_perm);

    int *inv_row_perm = (int *)malloc((size_t)m * sizeof(int));
    compute_inv_perm(m, row_perm, inv_row_perm);
    for (int cone = 0; cone < qp->affine_cones.num_cones; ++cone)
    {
        int old_start = qp->affine_cones.start_idx[cone];
        qp->affine_cones.start_idx[cone] = inv_row_perm[old_start];
    }

    int *inv_col_perm = (int *)malloc(n * sizeof(int));
    compute_inv_perm(n, col_perm, inv_col_perm);

    if (qp->constraint_matrix && qp->constraint_matrix_num_nonzeros > 0)
    {
        permute_csr_rows_structural(qp->constraint_matrix, m, qp->constraint_matrix_num_nonzeros, row_perm);
        col_permute_in_place(m,
                             qp->constraint_matrix->row_ptr,
                             qp->constraint_matrix->col_ind,
                             qp->constraint_matrix->val,
                             inv_col_perm);
    }

    if (qp->objective_sparse_matrix && qp->objective_sparse_matrix_num_nonzeros > 0)
    {
        permute_csr_rows_structural(qp->objective_sparse_matrix, n, qp->objective_sparse_matrix_num_nonzeros, col_perm);
        col_permute_in_place(n,
                             qp->objective_sparse_matrix->row_ptr,
                             qp->objective_sparse_matrix->col_ind,
                             qp->objective_sparse_matrix->val,
                             inv_col_perm);
    }

    if (qp->objective_lowrank_matrix && qp->objective_lowrank_matrix_num_nonzeros > 0)
    {
        col_permute_in_place(qp->num_rank_lowrank_obj,
                             qp->objective_lowrank_matrix->row_ptr,
                             qp->objective_lowrank_matrix->col_ind,
                             qp->objective_lowrank_matrix->val,
                             inv_col_perm);
    }

    if (qp->cones.num_cones > 0)
    {
        for (int cone = 0; cone < qp->cones.num_cones; ++cone)
            qp->cones.start_idx[cone] = inv_col_perm[qp->cones.start_idx[cone]];

        if (qp->cones.is_fixed)
        {
            char *tmp = (char *)malloc((size_t)n * sizeof(char));
            for (int i = 0; i < n; ++i)
                tmp[i] = qp->cones.is_fixed[col_perm[i]];
            memcpy(qp->cones.is_fixed, tmp, (size_t)n * sizeof(char));
            free(tmp);
        }
    }

    free(inv_row_perm);
    free(inv_col_perm);
    return true;
}

typedef struct
{
    int start;
    int length;
} permutation_unit_t;

static int compare_units_by_start(const void *a, const void *b)
{
    const permutation_unit_t *ua = (const permutation_unit_t *)a;
    const permutation_unit_t *ub = (const permutation_unit_t *)b;
    return (ua->start > ub->start) - (ua->start < ub->start);
}

static bool build_checked_inverse_permutation(int size, const int *permutation, int **inverse_out)
{
    *inverse_out = NULL;
    if (size <= 0)
        return true;
    if (!permutation)
        return false;

    int *inverse = (int *)malloc((size_t)size * sizeof(int));
    if (!inverse)
        return false;
    for (int index = 0; index < size; ++index)
        inverse[index] = -1;
    for (int index = 0; index < size; ++index)
    {
        int value = permutation[index];
        if (value < 0 || value >= size || inverse[value] >= 0)
        {
            free(inverse);
            return false;
        }
        inverse[value] = index;
    }
    *inverse_out = inverse;
    return true;
}

bool validate_cone_permutation(const qp_problem_t *qp, const int *col_perm)
{
    if (!qp)
        return false;
    int *inverse = NULL;
    if (!build_checked_inverse_permutation(qp->num_variables, col_perm, &inverse))
        return false;
    if (qp->cones.num_cones <= 0)
    {
        free(inverse);
        return true;
    }

    bool valid = true;
    for (int cone = 0; cone < qp->cones.num_cones && valid; ++cone)
    {
        int old_start = qp->cones.start_idx[cone];
        int length = cone_block_length(&qp->cones, cone);
        int new_start = inverse[old_start];
        for (int slot = 1; slot < length; ++slot)
        {
            if (inverse[old_start + slot] != new_start + slot)
            {
                valid = false;
                break;
            }
        }
    }
    free(inverse);
    return valid;
}

bool validate_affine_cone_row_permutation(const qp_problem_t *qp, const int *row_perm)
{
    if (!qp)
        return false;
    int *inverse = NULL;
    if (!build_checked_inverse_permutation(qp->num_constraints, row_perm, &inverse))
        return false;
    if (qp->affine_cones.num_cones <= 0)
    {
        free(inverse);
        return true;
    }
    bool valid = true;
    for (int cone = 0; cone < qp->affine_cones.num_cones && valid; ++cone)
    {
        int old_start = qp->affine_cones.start_idx[cone];
        int length = cone_block_length(&qp->affine_cones, cone);
        int new_start = inverse[old_start];
        for (int slot = 1; slot < length; ++slot)
        {
            if (inverse[old_start + slot] != new_start + slot)
            {
                valid = false;
                break;
            }
        }
    }
    free(inverse);
    return valid;
}

static void generate_cone_aware_vector_permutation(
    int n, const cone_blocks_t *cones, permute_method_t method, int block_size, int *perm)
{
    if (method == NO_PERMUTATION || n <= 1)
    {
        for (int i = 0; i < n; ++i)
            perm[i] = i;
        return;
    }

    if (cones->num_cones <= 0)
    {
        if (method == FULL_RANDOM_PERMUTATION)
            generate_random_permutation(n, perm);
        else
            generate_block_permutation(n, block_size, perm);
        return;
    }

    int K = cones->num_cones;
    permutation_unit_t *cone_units = (permutation_unit_t *)malloc((size_t)K * sizeof(permutation_unit_t));
    for (int cone = 0; cone < K; ++cone)
    {
        cone_units[cone].start = cones->start_idx[cone];
        cone_units[cone].length = cone_block_length(cones, cone);
    }
    qsort(cone_units, (size_t)K, sizeof(permutation_unit_t), compare_units_by_start);

    std::vector<permutation_unit_t> units;
    int cursor = 0;
    int free_block = (method == FULL_RANDOM_PERMUTATION) ? 1 : ((block_size > 0) ? block_size : 1);
    for (int cone = 0; cone < K; ++cone)
    {
        int cone_start = cone_units[cone].start;
        while (cursor < cone_start)
        {
            int length = MIN(free_block, cone_start - cursor);
            units.push_back({cursor, length});
            cursor += length;
        }
        units.push_back(cone_units[cone]);
        cursor = cone_start + cone_units[cone].length;
    }
    while (cursor < n)
    {
        int length = MIN(free_block, n - cursor);
        units.push_back({cursor, length});
        cursor += length;
    }
    free(cone_units);

    for (int i = (int)units.size() - 1; i > 0; --i)
    {
        int j = rand() % (i + 1);
        permutation_unit_t tmp = units[i];
        units[i] = units[j];
        units[j] = tmp;
    }

    int out = 0;
    for (const permutation_unit_t &unit : units)
        for (int slot = 0; slot < unit.length; ++slot)
            perm[out++] = unit.start + slot;
}

void generate_cone_aware_permutation(const qp_problem_t *qp, permute_method_t method, int block_size, int *perm)
{
    generate_cone_aware_vector_permutation(qp->num_variables, &qp->cones, method, block_size, perm);
}

void generate_affine_cone_aware_row_permutation(const qp_problem_t *qp,
                                                permute_method_t method,
                                                int block_size,
                                                int *perm)
{
    generate_cone_aware_vector_permutation(qp->num_constraints, &qp->affine_cones, method, block_size, perm);
}

qp_problem_t *permute_problem_return_new(const qp_problem_t *qp, int *row_perm, int *col_perm)
{
    if (!qp)
        return NULL;

    qp_problem_t *new_qp = deepcopy_problem(qp);

    if (!permute_problem(new_qp, row_perm, col_perm))
    {
        qp_problem_free(new_qp);
        return NULL;
    }

    return new_qp;
}

static void generate_random_permutation(int n, int *perm)
{
    for (int i = 0; i < n; i++)
        perm[i] = i;
    for (int i = n - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
}

static void generate_block_permutation(int n, int block_size, int *perm)
{
    if (block_size <= 0)
        block_size = 1;
    int num_blocks = (n + block_size - 1) / block_size;

    int *block_indices = (int *)malloc(num_blocks * sizeof(int));
    if (!block_indices)
        return;

    for (int i = 0; i < num_blocks; i++)
    {
        block_indices[i] = i;
    }

    for (int i = num_blocks - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int temp = block_indices[i];
        block_indices[i] = block_indices[j];
        block_indices[j] = temp;
    }

    int current_pos = 0;

    for (int i = 0; i < num_blocks; i++)
    {
        int b_idx = block_indices[i];

        int start_val = b_idx * block_size;
        int end_val = MIN((b_idx + 1) * block_size, n);

        for (int val = start_val; val < end_val; val++)
        {
            perm[current_pos++] = val;
        }
    }
    free(block_indices);
}

void repermute_solution(pdhcg_result_t *result, int *row_perm, int *col_perm)
{
    int *inv_col_perm = (int *)malloc(result->num_variables * sizeof(int));
    int *inv_row_perm = (int *)malloc(result->num_constraints * sizeof(int));
    compute_inv_perm(result->num_variables, col_perm, inv_col_perm);
    compute_inv_perm(result->num_constraints, row_perm, inv_row_perm);
    permute_double_array(result->primal_solution, result->num_variables, inv_col_perm);
    permute_double_array(result->dual_solution, result->num_constraints, inv_row_perm);
    permute_double_array(result->reduced_cost, result->num_variables, inv_col_perm);
    free(inv_col_perm);
    free(inv_row_perm);
}
