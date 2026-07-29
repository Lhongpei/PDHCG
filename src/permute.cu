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
#include "pdhcg_types.h"
#include "permute.h"
#include "utils.h"
#include <math.h>
#include <random>
#include <vector>

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

int cmp_tuples(const void *a, const void *b)
{
    return ((permute_tuple_t *)a)->new_col - ((permute_tuple_t *)b)->new_col;
}

void col_permute_in_place(int m, int *Ap, int *Aj, double *Ax, const int *old_col_to_new)
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

void permute_csr_rows_structural(CsrComponent *csr, int num_rows, int nnz, const int *row_perm)
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

void permute_double_array(double *arr, int n, const int *perm)
{
    if (!arr)
        return;
    double *tmp = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        tmp[i] = arr[perm[i]];
    memcpy(arr, tmp, n * sizeof(double));
    free(tmp);
}

void compute_inv_perm(int n, const int *perm, int *inv_perm)
{
    for (int i = 0; i < n; i++)
        inv_perm[perm[i]] = i;
}

void permute_problem(qp_problem_t *qp, int *row_perm, int *col_perm)
{
    int m = qp->num_constraints;
    int n = qp->num_variables;

    if (!validate_cone_permutation(qp, col_perm))
    {
        fprintf(stderr, "Error: column permutation splits or reorders a cone block.\n");
        return;
    }

    permute_double_array(qp->objective_vector, n, col_perm);
    permute_double_array(qp->variable_lower_bound, n, col_perm);
    permute_double_array(qp->variable_upper_bound, n, col_perm);
    if (qp->primal_start)
        permute_double_array(qp->primal_start, n, col_perm);

    permute_double_array(qp->constraint_lower_bound, m, row_perm);
    permute_double_array(qp->constraint_upper_bound, m, row_perm);
    if (qp->dual_start)
        permute_double_array(qp->dual_start, m, row_perm);

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

    free(inv_col_perm);
}

typedef struct
{
    int start;
    int length;
} permutation_unit_t;

static int cone_length(const cone_blocks_t *cones, int cone)
{
    return (cones->type[cone] == CONE_EXPONENTIAL || cones->type[cone] == CONE_POWER) ? 3 : cones->v_dim[cone] + 2;
}

static int compare_units_by_start(const void *a, const void *b)
{
    const permutation_unit_t *ua = (const permutation_unit_t *)a;
    const permutation_unit_t *ub = (const permutation_unit_t *)b;
    return (ua->start > ub->start) - (ua->start < ub->start);
}

bool validate_cone_permutation(const qp_problem_t *qp, const int *col_perm)
{
    if (!qp || qp->cones.num_cones <= 0)
        return true;

    int n = qp->num_variables;
    int *inverse = (int *)malloc((size_t)n * sizeof(int));
    if (!inverse)
        return false;
    compute_inv_perm(n, col_perm, inverse);

    bool valid = true;
    for (int cone = 0; cone < qp->cones.num_cones && valid; ++cone)
    {
        int old_start = qp->cones.start_idx[cone];
        int length = cone_length(&qp->cones, cone);
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

void generate_cone_aware_permutation(const qp_problem_t *qp, permute_method_t method, int block_size, int *perm)
{
    int n = qp->num_variables;
    if (method == NO_PERMUTATION || n <= 1)
    {
        for (int i = 0; i < n; ++i)
            perm[i] = i;
        return;
    }

    if (qp->cones.num_cones <= 0)
    {
        if (method == FULL_RANDOM_PERMUTATION)
            generate_random_permutation(n, perm);
        else
            generate_block_permutation(n, block_size, perm);
        return;
    }

    int K = qp->cones.num_cones;
    permutation_unit_t *cone_units = (permutation_unit_t *)malloc((size_t)K * sizeof(permutation_unit_t));
    for (int cone = 0; cone < K; ++cone)
    {
        cone_units[cone].start = qp->cones.start_idx[cone];
        cone_units[cone].length = cone_length(&qp->cones, cone);
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

qp_problem_t *permute_problem_return_new(const qp_problem_t *qp, int *row_perm, int *col_perm)
{
    if (!qp)
        return NULL;

    qp_problem_t *new_qp = deepcopy_problem(qp);

    permute_problem(new_qp, row_perm, col_perm);

    return new_qp;
}

void generate_random_permutation(int n, int *perm)
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

void randomly_permute_problem(qp_problem_t *qp, int **out_row_perm, int **out_col_perm)
{
    int m = qp->num_constraints;
    int n = qp->num_variables;

    int *row_perm = (int *)malloc(m * sizeof(int));
    int *col_perm = (int *)malloc(n * sizeof(int));

    generate_random_permutation(m, row_perm);
    generate_cone_aware_permutation(qp, FULL_RANDOM_PERMUTATION, 1, col_perm);

    permute_problem(qp, row_perm, col_perm);

    *out_row_perm = row_perm;
    *out_col_perm = col_perm;
}

void generate_block_permutation(int n, int block_size, int *perm)
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

void randomly_block_permute_problem(
    qp_problem_t *qp, int row_block_size, int col_block_size, int **out_row_perm, int **out_col_perm)
{
    int m = qp->num_constraints;
    int n = qp->num_variables;

    int *row_perm = (int *)malloc(m * sizeof(int));
    int *col_perm = (int *)malloc(n * sizeof(int));

    generate_block_permutation(m, row_block_size, row_perm);
    generate_cone_aware_permutation(qp, BLOCK_RANDOM_PERMUTATION, col_block_size, col_perm);

    permute_problem(qp, row_perm, col_perm);

    *out_row_perm = row_perm;
    *out_col_perm = col_perm;
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
