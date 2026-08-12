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

#include "presolve_wrapper.h"
#include "cone_utils.h"

#ifdef PREFOS_AVAILABLE

#include <PreFOS/PreFOS.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct
{
    PreFOSProblemData problem;
    int *owned_A_row_pointers;
    int *owned_A_column_indices;
    double *owned_A_values;
    double *owned_constraint_lower;
    double *owned_constraint_upper;
    int *owned_Q_row_pointers;
    int *owned_Q_column_indices;
    double *owned_Q_values;
    double *owned_D;
} PreFOSInputAdapter;

#define PDHCG_INFINITY_SENTINEL 1e20

static double normalize_lower_bound(double value)
{
    return value <= -PDHCG_INFINITY_SENTINEL ? -INFINITY : value;
}

static double normalize_upper_bound(double value)
{
    return value >= PDHCG_INFINITY_SENTINEL ? INFINITY : value;
}

static double shifted_constraint_bound(double bound, double constant, int is_lower)
{
    if (isfinite(bound))
        bound -= constant;
    return is_lower ? normalize_lower_bound(bound) : normalize_upper_bound(bound);
}

static void *allocate_array(size_t count, size_t element_size)
{
    if (count == 0)
        return NULL;
    if (element_size != 0 && count > SIZE_MAX / element_size)
        return NULL;
    return malloc(count * element_size);
}

static PreFOSCsrMatrix csr_view(const CsrComponent *matrix, size_t rows, size_t cols, size_t nnz)
{
    PreFOSCsrMatrix view;
    view.rows = rows;
    view.cols = cols;
    view.nnz = nnz;
    view.values = matrix ? matrix->val : NULL;
    view.column_indices = matrix ? matrix->col_ind : NULL;
    view.row_pointers = matrix ? matrix->row_ptr : NULL;
    return view;
}

static void free_input_adapter(PreFOSInputAdapter *adapter)
{
    size_t cone;
    if (!adapter)
        return;
    for (cone = 0; cone < adapter->problem.n_cones; ++cone)
        free(adapter->problem.cones[cone].indices);
    free(adapter->problem.cones);
    free(adapter->problem.box_indices);
    free(adapter->problem.box_lower);
    free(adapter->problem.box_upper);
    free(adapter->owned_A_row_pointers);
    free(adapter->owned_A_column_indices);
    free(adapter->owned_A_values);
    free(adapter->owned_constraint_lower);
    free(adapter->owned_constraint_upper);
    free(adapter->owned_Q_row_pointers);
    free(adapter->owned_Q_column_indices);
    free(adapter->owned_Q_values);
    free(adapter->owned_D);
    memset(adapter, 0, sizeof(*adapter));
}

static int initialize_full_q(const qp_problem_t *source, PreFOSInputAdapter *adapter)
{
    const CsrComponent *Q = source->objective_sparse_matrix;
    size_t n = (size_t)source->num_variables;
    size_t nnz =
        source->objective_sparse_matrix_num_nonzeros > 0 ? (size_t)source->objective_sparse_matrix_num_nonzeros : 0;
    int all_upper = 1;
    int all_lower = 1;
    int has_off_diagonal = 0;
    size_t expanded_nnz = nnz;
    size_t row;

    adapter->problem.Q = csr_view(Q, n, n, nnz);
    adapter->problem.q_storage = PREFOS_Q_FULL;
    if (nnz == 0)
        return 1;
    if (!Q || !Q->row_ptr || !Q->col_ind || !Q->val)
        return 0;

    for (row = 0; row < n; ++row)
    {
        int p;
        for (p = Q->row_ptr[row]; p < Q->row_ptr[row + 1]; ++p)
        {
            int column = Q->col_ind[p];
            if (column < 0 || (size_t)column >= n)
                return 0;
            if (column < (int)row)
                all_upper = 0;
            if (column > (int)row)
                all_lower = 0;
            if (column != (int)row)
            {
                has_off_diagonal = 1;
                ++expanded_nnz;
            }
        }
    }

    if (!has_off_diagonal || (!all_upper && !all_lower))
        return 1;
    if (expanded_nnz > (size_t)INT_MAX)
        return 0;

    adapter->owned_Q_row_pointers = (int *)calloc(n + 1, sizeof(int));
    adapter->owned_Q_column_indices = (int *)allocate_array(expanded_nnz, sizeof(int));
    adapter->owned_Q_values = (double *)allocate_array(expanded_nnz, sizeof(double));
    if (!adapter->owned_Q_row_pointers ||
        (expanded_nnz > 0 && (!adapter->owned_Q_column_indices || !adapter->owned_Q_values)))
        return 0;

    for (row = 0; row < n; ++row)
    {
        int p;
        for (p = Q->row_ptr[row]; p < Q->row_ptr[row + 1]; ++p)
        {
            int column = Q->col_ind[p];
            ++adapter->owned_Q_row_pointers[row + 1];
            if (column != (int)row)
                ++adapter->owned_Q_row_pointers[(size_t)column + 1];
        }
    }
    for (row = 0; row < n; ++row)
        adapter->owned_Q_row_pointers[row + 1] += adapter->owned_Q_row_pointers[row];

    {
        int *next = (int *)allocate_array(n, sizeof(int));
        if (n > 0 && !next)
            return 0;
        if (n > 0)
            memcpy(next, adapter->owned_Q_row_pointers, n * sizeof(int));
        for (row = 0; row < n; ++row)
        {
            int p;
            for (p = Q->row_ptr[row]; p < Q->row_ptr[row + 1]; ++p)
            {
                int column = Q->col_ind[p];
                int position = next[row]++;
                adapter->owned_Q_column_indices[position] = column;
                adapter->owned_Q_values[position] = Q->val[p];
                if (column != (int)row)
                {
                    position = next[column]++;
                    adapter->owned_Q_column_indices[position] = (int)row;
                    adapter->owned_Q_values[position] = Q->val[p];
                }
            }
        }
        free(next);
    }

    adapter->problem.Q.rows = n;
    adapter->problem.Q.cols = n;
    adapter->problem.Q.nnz = expanded_nnz;
    adapter->problem.Q.row_pointers = adapter->owned_Q_row_pointers;
    adapter->problem.Q.column_indices = adapter->owned_Q_column_indices;
    adapter->problem.Q.values = adapter->owned_Q_values;
    return 1;
}

static int initialize_diagonal_d(const qp_problem_t *source, PreFOSInputAdapter *adapter)
{
    const CsrComponent *middle = source->objective_lowrank_middle_matrix;
    size_t rank = source->num_rank_lowrank_obj > 0 ? (size_t)source->num_rank_lowrank_obj : 0;
    size_t row;

    if (rank == 0)
        return 1;
    adapter->owned_D = (double *)allocate_array(rank, sizeof(double));
    if (!adapter->owned_D)
        return 0;

    if (!middle || !middle->row_ptr || source->objective_lowrank_middle_matrix_num_nonzeros == 0)
    {
        for (row = 0; row < rank; ++row)
            adapter->owned_D[row] = 1.0;
    }
    else
    {
        memset(adapter->owned_D, 0, rank * sizeof(double));
        for (row = 0; row < rank; ++row)
        {
            int p;
            for (p = middle->row_ptr[row]; p < middle->row_ptr[row + 1]; ++p)
            {
                if (middle->col_ind[p] != (int)row)
                    return -1;
                adapter->owned_D[row] = middle->val[p];
            }
        }
    }
    adapter->problem.D = adapter->owned_D;
    return 1;
}

static int append_fixed_cone_rows(const qp_problem_t *source, PreFOSInputAdapter *adapter, int *prefos_rows)
{
    const CsrComponent *A = source->constraint_matrix;
    size_t original_rows = (size_t)source->num_constraints;
    size_t original_nnz =
        source->constraint_matrix_num_nonzeros > 0 ? (size_t)source->constraint_matrix_num_nonzeros : 0;
    size_t fixed_count = 0;
    int normalize_original_bounds = 0;
    size_t variable;
    size_t row;

    if (source->cones.is_fixed)
        for (variable = 0; variable < (size_t)source->num_variables; ++variable)
            if (source->cones.is_fixed[variable])
                ++fixed_count;

    for (row = 0; row < original_rows; ++row)
        if (source->affine_cone_offset[row] != 0.0 ||
            normalize_lower_bound(source->constraint_lower_bound[row]) != source->constraint_lower_bound[row] ||
            normalize_upper_bound(source->constraint_upper_bound[row]) != source->constraint_upper_bound[row])
            normalize_original_bounds = 1;

    if (fixed_count == 0 && !normalize_original_bounds)
    {
        adapter->problem.A = csr_view(A, original_rows, (size_t)source->num_variables, original_nnz);
        adapter->problem.constraint_lower = source->constraint_lower_bound;
        adapter->problem.constraint_upper = source->constraint_upper_bound;
        *prefos_rows = source->num_constraints;
        return 1;
    }
    if (original_rows + fixed_count > (size_t)INT_MAX || original_nnz + fixed_count > (size_t)INT_MAX)
        return 0;
    if (original_rows > 0 && (!A || !A->row_ptr))
        return 0;

    adapter->owned_A_row_pointers = (int *)calloc(original_rows + fixed_count + 1, sizeof(int));
    adapter->owned_A_column_indices = (int *)allocate_array(original_nnz + fixed_count, sizeof(int));
    adapter->owned_A_values = (double *)allocate_array(original_nnz + fixed_count, sizeof(double));
    adapter->owned_constraint_lower = (double *)allocate_array(original_rows + fixed_count, sizeof(double));
    adapter->owned_constraint_upper = (double *)allocate_array(original_rows + fixed_count, sizeof(double));
    if (!adapter->owned_A_row_pointers ||
        (original_nnz + fixed_count > 0 && (!adapter->owned_A_column_indices || !adapter->owned_A_values)) ||
        !adapter->owned_constraint_lower || !adapter->owned_constraint_upper)
        return 0;

    if (original_rows > 0)
    {
        memcpy(adapter->owned_A_row_pointers, A->row_ptr, (original_rows + 1) * sizeof(int));
        for (row = 0; row < original_rows; ++row)
        {
            adapter->owned_constraint_lower[row] =
                shifted_constraint_bound(source->constraint_lower_bound[row], source->affine_cone_offset[row], 1);
            adapter->owned_constraint_upper[row] =
                shifted_constraint_bound(source->constraint_upper_bound[row], source->affine_cone_offset[row], 0);
        }
    }
    if (original_nnz > 0)
    {
        memcpy(adapter->owned_A_column_indices, A->col_ind, original_nnz * sizeof(int));
        memcpy(adapter->owned_A_values, A->val, original_nnz * sizeof(double));
    }

    row = original_rows;
    for (variable = 0; variable < (size_t)source->num_variables; ++variable)
    {
        size_t position;
        double value;
        if (!source->cones.is_fixed[variable])
            continue;
        position = original_nnz + row - original_rows;
        value = source->primal_start ? source->primal_start[variable] : 0.0;
        adapter->owned_A_column_indices[position] = (int)variable;
        adapter->owned_A_values[position] = 1.0;
        adapter->owned_A_row_pointers[row + 1] = (int)(position + 1);
        adapter->owned_constraint_lower[row] = value;
        adapter->owned_constraint_upper[row] = value;
        ++row;
    }

    adapter->problem.A.rows = original_rows + fixed_count;
    adapter->problem.A.cols = (size_t)source->num_variables;
    adapter->problem.A.nnz = original_nnz + fixed_count;
    adapter->problem.A.row_pointers = adapter->owned_A_row_pointers;
    adapter->problem.A.column_indices = adapter->owned_A_column_indices;
    adapter->problem.A.values = adapter->owned_A_values;
    adapter->problem.constraint_lower = adapter->owned_constraint_lower;
    adapter->problem.constraint_upper = adapter->owned_constraint_upper;
    *prefos_rows = (int)(original_rows + fixed_count);
    return 1;
}

static size_t psd_column_major_slot(size_t order, size_t row, size_t column)
{
    return column * (2 * order - column + 1) / 2 + row - column;
}

static int initialize_domains(const qp_problem_t *source, PreFOSInputAdapter *adapter)
{
    size_t n = (size_t)source->num_variables;
    size_t cone_count = source->cones.num_cones > 0 ? (size_t)source->cones.num_cones : 0;
    unsigned char *owner = (unsigned char *)calloc(n, sizeof(unsigned char));
    size_t cone;
    size_t cone_variables = 0;
    size_t box_write = 0;
    size_t variable;

    if (n > 0 && !owner)
        return 0;
    if (cone_count > 0 && (!source->cones.start_idx || !source->cones.v_dim || !source->cones.type))
    {
        free(owner);
        return 0;
    }
    adapter->problem.n_cones = cone_count;
    adapter->problem.cones = (PreFOSConeBlock *)calloc(cone_count, sizeof(PreFOSConeBlock));
    if (cone_count > 0 && !adapter->problem.cones)
    {
        free(owner);
        return 0;
    }

    for (cone = 0; cone < cone_count; ++cone)
    {
        PreFOSConeBlock *target = &adapter->problem.cones[cone];
        int start = source->cones.start_idx[cone];
        int vector_dimension = source->cones.v_dim[cone];
        int block_length = cone_block_length(&source->cones, (int)cone);
        size_t dimension;
        size_t index;

        if (block_length <= 0)
        {
            free(owner);
            return 0;
        }
        dimension = (size_t)block_length;
        if (start < 0 || (size_t)start > n || dimension > n - (size_t)start)
        {
            free(owner);
            return 0;
        }

        target->dimension = dimension;
        target->matrix_order = 0;
        target->indices = (int *)allocate_array(dimension, sizeof(int));
        if (!target->indices)
        {
            free(owner);
            return 0;
        }
        switch (source->cones.type[cone])
        {
            case CONE_STANDARD_SOC:
                target->type = PREFOS_CONE_SECOND_ORDER;
                target->indices[0] = start + vector_dimension + 1;
                for (index = 1; index < dimension; ++index)
                    target->indices[index] = start + (int)index - 1;
                break;
            case CONE_ROTATED_SOC:
                target->type = PREFOS_CONE_ROTATED_SECOND_ORDER;
                target->indices[0] = start + vector_dimension;
                target->indices[1] = start + vector_dimension + 1;
                for (index = 2; index < dimension; ++index)
                    target->indices[index] = start + (int)index - 2;
                break;
            case CONE_EXPONENTIAL:
                target->type = PREFOS_CONE_EXPONENTIAL;
                target->indices[0] = start;
                target->indices[1] = start + 1;
                target->indices[2] = start + 2;
                break;
            case CONE_POWER:
                target->type = PREFOS_CONE_POWER;
                target->power_alpha = source->cones.power_alpha ? source->cones.power_alpha[cone] : 0.0;
                target->indices[0] = start;
                target->indices[1] = start + 1;
                target->indices[2] = start + 2;
                break;
            case CONE_PSD:
            {
                size_t row;
                size_t column;
                size_t position = 0;
                size_t order = (size_t)vector_dimension;
                target->type = PREFOS_CONE_POSITIVE_SEMIDEFINITE;
                target->matrix_order = order;
                for (row = 0; row < order; ++row)
                    for (column = 0; column <= row; ++column)
                        target->indices[position++] = start + (int)psd_column_major_slot(order, row, column);
                break;
            }
            default:
                free(owner);
                return 0;
        }
        for (index = 0; index < dimension; ++index)
        {
            int column = target->indices[index];
            double lower = normalize_lower_bound(source->variable_lower_bound[column]);
            double upper = normalize_upper_bound(source->variable_upper_bound[column]);
            if (owner[column] || isfinite(lower) || isfinite(upper))
            {
                free(owner);
                return 0;
            }
            owner[column] = 1;
            ++cone_variables;
        }
    }

    adapter->problem.n_box = n - cone_variables;
    adapter->problem.box_indices = (int *)allocate_array(adapter->problem.n_box, sizeof(int));
    adapter->problem.box_lower = (double *)allocate_array(adapter->problem.n_box, sizeof(double));
    adapter->problem.box_upper = (double *)allocate_array(adapter->problem.n_box, sizeof(double));
    if (adapter->problem.n_box > 0 &&
        (!adapter->problem.box_indices || !adapter->problem.box_lower || !adapter->problem.box_upper))
    {
        free(owner);
        return 0;
    }
    for (variable = 0; variable < n; ++variable)
    {
        if (owner[variable])
            continue;
        adapter->problem.box_indices[box_write] = (int)variable;
        adapter->problem.box_lower[box_write] = normalize_lower_bound(source->variable_lower_bound[variable]);
        adapter->problem.box_upper[box_write] = normalize_upper_bound(source->variable_upper_bound[variable]);
        ++box_write;
    }
    free(owner);
    return box_write == adapter->problem.n_box;
}

static int initialize_prefos_input(const qp_problem_t *source, PreFOSInputAdapter *adapter, int *prefos_rows)
{
    int d_status;
    memset(adapter, 0, sizeof(*adapter));
    if (!source || source->num_variables < 0 || source->num_constraints < 0 || source->num_rank_lowrank_obj < 0 ||
        source->cones.num_cones < 0 || source->constraint_matrix_num_nonzeros < 0 ||
        source->objective_sparse_matrix_num_nonzeros < 0 || source->objective_lowrank_matrix_num_nonzeros < 0 ||
        source->objective_lowrank_middle_matrix_num_nonzeros < 0 ||
        (source->num_variables > 0 &&
         (!source->objective_vector || !source->variable_lower_bound || !source->variable_upper_bound)) ||
        (source->num_constraints > 0 &&
         (!source->constraint_matrix || !source->constraint_lower_bound || !source->constraint_upper_bound ||
          !source->affine_cone_offset)))
        return 0;

    adapter->problem.n = (size_t)source->num_variables;
    adapter->problem.c = source->objective_vector;
    adapter->problem.objective_offset = source->objective_constant;
    adapter->problem.R = csr_view(
        source->objective_lowrank_matrix,
        source->num_rank_lowrank_obj > 0 ? (size_t)source->num_rank_lowrank_obj : 0,
        (size_t)source->num_variables,
        source->objective_lowrank_matrix_num_nonzeros > 0 ? (size_t)source->objective_lowrank_matrix_num_nonzeros : 0);

    if (!initialize_full_q(source, adapter) || !initialize_domains(source, adapter) ||
        !append_fixed_cone_rows(source, adapter, prefos_rows))
        return 0;
    d_status = initialize_diagonal_d(source, adapter);
    if (d_status <= 0)
        return d_status;
    return 1;
}

static CsrComponent *wrap_csr(const PreFOSCsrMatrix *matrix)
{
    CsrComponent *wrapper = (CsrComponent *)calloc(1, sizeof(CsrComponent));
    if (!wrapper)
        return NULL;
    wrapper->row_ptr = matrix->row_pointers;
    wrapper->col_ind = matrix->column_indices;
    wrapper->val = matrix->values;
    return wrapper;
}

static int convert_cone_to_pdhcg(const PreFOSConeBlock *source, cone_blocks_t *target, size_t cone)
{
    size_t dimension = source->dimension;
    size_t i;
    int start;
    if (!source->indices || dimension == 0)
        return 0;

    switch (source->type)
    {
        case PREFOS_CONE_SECOND_ORDER:
            if (dimension < 2)
                return 0;
            start = source->indices[1];
            for (i = 1; i < dimension; ++i)
                if (source->indices[i] != start + (int)i - 1)
                    return 0;
            if (source->indices[0] != start + (int)dimension - 1)
                return 0;
            target->type[cone] = CONE_STANDARD_SOC;
            target->v_dim[cone] = (int)dimension - 2;
            break;
        case PREFOS_CONE_ROTATED_SECOND_ORDER:
            if (dimension < 3)
                return 0;
            start = source->indices[2];
            for (i = 2; i < dimension; ++i)
                if (source->indices[i] != start + (int)i - 2)
                    return 0;
            if (source->indices[0] != start + (int)dimension - 2 || source->indices[1] != start + (int)dimension - 1)
                return 0;
            target->type[cone] = CONE_ROTATED_SOC;
            target->v_dim[cone] = (int)dimension - 2;
            break;
        case PREFOS_CONE_EXPONENTIAL:
        case PREFOS_CONE_POWER:
            if (dimension != 3 || source->indices[1] != source->indices[0] + 1 ||
                source->indices[2] != source->indices[0] + 2)
                return 0;
            start = source->indices[0];
            target->type[cone] = source->type == PREFOS_CONE_EXPONENTIAL ? CONE_EXPONENTIAL : CONE_POWER;
            target->v_dim[cone] = 1;
            if (source->type == PREFOS_CONE_POWER)
                target->power_alpha[cone] = source->power_alpha;
            break;
        case PREFOS_CONE_POSITIVE_SEMIDEFINITE:
        {
            size_t row;
            size_t column;
            size_t position = 0;
            size_t order = source->matrix_order;
            if (order == 0 || order > (size_t)INT_MAX || order * (order + 1) / 2 != dimension)
                return 0;
            start = source->indices[0];
            if (start < 0)
                return 0;
            for (row = 0; row < order; ++row)
            {
                for (column = 0; column <= row; ++column)
                {
                    size_t slot = psd_column_major_slot(order, row, column);
                    if ((long long)source->indices[position++] != (long long)start + (long long)slot)
                        return 0;
                }
            }
            target->type[cone] = CONE_PSD;
            target->v_dim[cone] = (int)order;
            break;
        }
        default:
            return 0;
    }
    target->start_idx[cone] = start;
    return 1;
}

static qp_problem_t *convert_prefos_to_pdhcg(const PreFOSPresolvedProblem *source)
{
    qp_problem_t *target;
    size_t variable;
    size_t box;
    size_t cone;
    size_t rank;

    if (!source || source->n > (size_t)INT_MAX || source->A.rows > (size_t)INT_MAX || source->A.nnz > (size_t)INT_MAX ||
        source->Q.nnz > (size_t)INT_MAX || source->R.rows > (size_t)INT_MAX || source->R.nnz > (size_t)INT_MAX ||
        source->n_cones > (size_t)INT_MAX || source->n_affine_cones > 0 || source->affine_cone_matrix.rows > 0 ||
        source->q_storage != PREFOS_Q_FULL)
        return NULL;

    target = (qp_problem_t *)calloc(1, sizeof(qp_problem_t));
    if (!target)
        return NULL;
    target->num_variables = (int)source->n;
    target->num_constraints = (int)source->A.rows;
    target->affine_cone_offset =
        target->num_constraints > 0 ? (double *)calloc((size_t)target->num_constraints, sizeof(double)) : NULL;
    if (target->num_constraints > 0 && !target->affine_cone_offset)
    {
        free(target);
        return NULL;
    }
    target->constraint_matrix_num_nonzeros = (int)source->A.nnz;
    target->objective_sparse_matrix_num_nonzeros = (int)source->Q.nnz;
    target->objective_lowrank_matrix_num_nonzeros = (int)source->R.nnz;
    target->num_rank_lowrank_obj = (int)source->R.rows;
    target->objective_constant = source->objective_offset;
    target->objective_vector = source->c;
    target->constraint_lower_bound = source->constraint_lower;
    target->constraint_upper_bound = source->constraint_upper;
    target->constraint_matrix = wrap_csr(&source->A);
    target->objective_sparse_matrix = wrap_csr(&source->Q);
    target->objective_lowrank_matrix = wrap_csr(&source->R);
    if (!target->constraint_matrix || !target->objective_sparse_matrix || !target->objective_lowrank_matrix)
        goto failure;

    target->variable_lower_bound = (double *)allocate_array(source->n, sizeof(double));
    target->variable_upper_bound = (double *)allocate_array(source->n, sizeof(double));
    if (source->n > 0 && (!target->variable_lower_bound || !target->variable_upper_bound))
        goto failure;
    for (variable = 0; variable < source->n; ++variable)
    {
        target->variable_lower_bound[variable] = -INFINITY;
        target->variable_upper_bound[variable] = INFINITY;
    }
    for (box = 0; box < source->n_box; ++box)
    {
        int index = source->box_indices[box];
        if (index < 0 || (size_t)index >= source->n)
            goto failure;
        target->variable_lower_bound[index] = source->box_lower[box];
        target->variable_upper_bound[index] = source->box_upper[box];
    }

    target->cones.num_cones = (int)source->n_cones;
    target->cones.start_idx = (int *)allocate_array(source->n_cones, sizeof(int));
    target->cones.v_dim = (int *)allocate_array(source->n_cones, sizeof(int));
    target->cones.type = (cone_type_t *)allocate_array(source->n_cones, sizeof(cone_type_t));
    target->cones.power_alpha = (double *)calloc(source->n_cones, sizeof(double));
    if (source->n_cones > 0 &&
        (!target->cones.start_idx || !target->cones.v_dim || !target->cones.type || !target->cones.power_alpha))
        goto failure;
    for (cone = 0; cone < source->n_cones; ++cone)
        if (!convert_cone_to_pdhcg(&source->cones[cone], &target->cones, cone))
            goto failure;

    rank = source->R.rows;
    if (rank > 0)
    {
        CsrComponent *middle = (CsrComponent *)calloc(1, sizeof(CsrComponent));
        if (!middle)
            goto failure;
        target->objective_lowrank_middle_matrix = middle;
        target->objective_lowrank_middle_matrix_num_nonzeros = (int)rank;
        middle->row_ptr = (int *)allocate_array(rank + 1, sizeof(int));
        middle->col_ind = (int *)allocate_array(rank, sizeof(int));
        middle->val = (double *)allocate_array(rank, sizeof(double));
        if (!middle->row_ptr || !middle->col_ind || !middle->val)
            goto failure;
        for (variable = 0; variable < rank; ++variable)
        {
            middle->row_ptr[variable] = (int)variable;
            middle->col_ind[variable] = (int)variable;
            middle->val[variable] = source->D[variable];
        }
        middle->row_ptr[rank] = (int)rank;
    }

    target->num_original_variables = target->num_variables;
    return target;

failure:
    if (target)
    {
        free(target->constraint_matrix);
        free(target->objective_sparse_matrix);
        free(target->objective_lowrank_matrix);
        if (target->objective_lowrank_middle_matrix)
        {
            free(target->objective_lowrank_middle_matrix->row_ptr);
            free(target->objective_lowrank_middle_matrix->col_ind);
            free(target->objective_lowrank_middle_matrix->val);
            free(target->objective_lowrank_middle_matrix);
        }
        free(target->variable_lower_bound);
        free(target->variable_upper_bound);
        free(target->affine_cone_offset);
        cone_blocks_free(&target->cones);
        free(target);
    }
    return NULL;
}

static void free_converted_problem(qp_problem_t *problem)
{
    if (!problem)
        return;
    free(problem->constraint_matrix);
    free(problem->objective_sparse_matrix);
    free(problem->objective_lowrank_matrix);
    if (problem->objective_lowrank_middle_matrix)
    {
        free(problem->objective_lowrank_middle_matrix->row_ptr);
        free(problem->objective_lowrank_middle_matrix->col_ind);
        free(problem->objective_lowrank_middle_matrix->val);
        free(problem->objective_lowrank_middle_matrix);
    }
    free(problem->variable_lower_bound);
    free(problem->variable_upper_bound);
    free(problem->affine_cone_offset);
    cone_blocks_free(&problem->cones);
    free(problem);
}

static pdhcg_presolve_status_t map_prefos_status(PreFOSStatus status)
{
    switch (status)
    {
        case PREFOS_STATUS_OK:
            return PDHCG_PRESOLVE_STATUS_UNCHANGED;
        case PREFOS_STATUS_REDUCED:
            return PDHCG_PRESOLVE_STATUS_REDUCED;
        case PREFOS_STATUS_PRIMAL_INFEASIBLE:
            return PDHCG_PRESOLVE_STATUS_PRIMAL_INFEASIBLE;
        default:
            return PDHCG_PRESOLVE_STATUS_ERROR;
    }
}

const char *pdhcg_get_presolve_status_str(int status)
{
    switch ((pdhcg_presolve_status_t)status)
    {
        case PDHCG_PRESOLVE_STATUS_UNCHANGED:
            return "UNCHANGED";
        case PDHCG_PRESOLVE_STATUS_REDUCED:
            return "REDUCED";
        case PDHCG_PRESOLVE_STATUS_PRIMAL_INFEASIBLE:
            return "PRIMAL_INFEASIBLE";
        case PDHCG_PRESOLVE_STATUS_ERROR:
            return "ERROR";
        case PDHCG_PRESOLVE_STATUS_NOT_AVAILABLE:
            return "NOT_AVAILABLE";
        default:
            return "UNKNOWN_STATUS";
    }
}

pdhcg_presolve_info_t *pdhcg_presolve(const qp_problem_t *original_problem, const pdhg_parameters_t *parameters)
{
    PreFOSInputAdapter adapter;
    PreFOSSettings settings = prefos_default_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    pdhcg_presolve_info_t *info;
    PreFOSStatus status;
    clock_t start;
    int adapter_status;
    int prefos_rows = 0;

    if (!original_problem)
        return NULL;
    start = clock();
    adapter_status = initialize_prefos_input(original_problem, &adapter, &prefos_rows);
    if (adapter_status <= 0)
    {
        if (!parameters || parameters->verbose > 0)
        {
            if (adapter_status < 0)
                fprintf(stderr, "PreFOS presolve skipped: R^T D R currently requires diagonal D.\n");
            else
                fprintf(stderr, "PreFOS presolve skipped: the PDHCG model could not be adapted safely.\n");
        }
        free_input_adapter(&adapter);
        return NULL;
    }

    /* PDHCG reports standard row/domain multipliers. Keep transformations whose
       postsolve maps back to standard original-cone normals. */
    settings.rsoc_face_reduction = 0;
    settings.psd_face_reduction = 0;
    settings.exponential_face_reduction = 0;
    settings.power_face_reduction = 0;
    settings.affine_cone_coordinate_aggregation = 0;
    settings.propagated_bound_policy = PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER;
#ifdef PDHCG_PREFOS_CUDA_ENABLED
    settings.linear_propagation_gpu = 1;
#endif

    info = (pdhcg_presolve_info_t *)calloc(1, sizeof(pdhcg_presolve_info_t));
    if (!info)
    {
        free_input_adapter(&adapter);
        return NULL;
    }
    info->prefos_original_rows = prefos_rows;
    info->postsolve_tolerance = 1e-8;
    if (parameters && parameters->termination_criteria.eps_feasible_relative > 0.0)
        info->postsolve_tolerance = fmax(1e-10, parameters->termination_criteria.eps_feasible_relative);

    status = prefos_create_presolver(&adapter.problem, &settings, &presolver);
    free_input_adapter(&adapter);
    if (status == PREFOS_STATUS_OK && presolver)
        status = prefos_run_presolve(presolver);
    else if (status == PREFOS_STATUS_OK)
        status = PREFOS_STATUS_OUT_OF_MEMORY;
    info->presolve_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    info->presolve_status = map_prefos_status(status);
    info->presolver = presolver;

    if (status == PREFOS_STATUS_PRIMAL_INFEASIBLE)
    {
        info->problem_solved_during_presolve = true;
        return info;
    }
    if (status != PREFOS_STATUS_OK && status != PREFOS_STATUS_REDUCED)
    {
        if (!parameters || parameters->verbose > 0)
            fprintf(stderr, "PreFOS presolve failed: %s. Continuing without presolve.\n", prefos_status_string(status));
        pdhcg_presolve_info_free(info);
        return NULL;
    }

    reduced = prefos_get_reduced_problem(presolver);
    if (!reduced)
    {
        pdhcg_presolve_info_free(info);
        return NULL;
    }
    if (status == PREFOS_STATUS_REDUCED && reduced->n == 0)
    {
        info->problem_solved_during_presolve = true;
    }
    else if (status == PREFOS_STATUS_REDUCED)
    {
        info->reduced_problem = convert_prefos_to_pdhcg(reduced);
        if (!info->reduced_problem)
        {
            if (!parameters || parameters->verbose > 0)
                fprintf(stderr,
                        "PreFOS reduced model is not representable by the PDHCG direct-cone interface; "
                        "continuing without presolve.\n");
            pdhcg_presolve_info_free(info);
            return NULL;
        }
    }

    if (parameters && parameters->verbose > 1)
    {
        const PreFOSStats *stats = prefos_get_stats(presolver);
        printf("\nRunning presolver (PreFOS %s)...\n", PREFOS_VERSION);
        printf("  %-15s : %s\n", "status", pdhcg_get_presolve_status_str(info->presolve_status));
        printf("  %-15s : %.3g sec\n", "presolve time", info->presolve_time);
        if (stats && settings.linear_propagation_gpu)
        {
            printf("  %-15s : %zu rounds, %zu fallbacks\n",
                   "GPU propagation",
                   stats->linear_gpu_rounds,
                   stats->linear_gpu_fallbacks);
            printf("  %-15s : %.3g setup, %.3g transfer, %.3g kernel sec\n",
                   "GPU time",
                   stats->linear_gpu_setup_milliseconds * 1e-3,
                   stats->linear_gpu_transfer_milliseconds * 1e-3,
                   stats->linear_gpu_kernel_milliseconds * 1e-3);
        }
        printf("  %-15s : %zu rows, %zu columns, %zu nonzeros\n",
               "reduced problem",
               reduced->A.rows,
               reduced->n,
               reduced->A.nnz);
    }
    return info;
}

static void initialize_result_dimensions(pdhcg_result_t *result,
                                         const pdhcg_presolve_info_t *info,
                                         const qp_problem_t *original_problem)
{
    const PreFOSPresolvedProblem *reduced =
        info->presolver ? prefos_get_reduced_problem((const PreFOSPresolver *)info->presolver) : NULL;
    result->num_variables = original_problem->num_variables;
    result->num_constraints = original_problem->num_constraints;
    result->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;
    if (reduced)
    {
        result->num_reduced_variables = (int)reduced->n;
        result->num_reduced_constraints = (int)reduced->A.rows;
        result->num_reduced_nonzeros = (int)reduced->A.nnz;
    }
    result->presolve_status = (int)info->presolve_status;
    result->presolve_time = info->presolve_time;
}

pdhcg_result_t *pdhcg_create_result_from_presolve(const pdhcg_presolve_info_t *info,
                                                  const qp_problem_t *original_problem)
{
    pdhcg_result_t *result;
    if (!info || !original_problem)
        return NULL;
    result = (pdhcg_result_t *)calloc(1, sizeof(pdhcg_result_t));
    if (!result)
        return NULL;
    initialize_result_dimensions(result, info, original_problem);

    if (info->presolve_status == PDHCG_PRESOLVE_STATUS_PRIMAL_INFEASIBLE)
    {
        result->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
        result->absolute_primal_residual = INFINITY;
        result->relative_primal_residual = INFINITY;
        result->absolute_dual_residual = INFINITY;
        result->relative_dual_residual = INFINITY;
        result->primal_objective_value = INFINITY;
        result->dual_objective_value = -INFINITY;
        result->objective_gap = INFINITY;
        result->relative_objective_gap = INFINITY;
        return result;
    }

    result->termination_reason = TERMINATION_REASON_OPTIMAL;
    if (!pdhcg_postsolve(info, result, original_problem))
        result->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    return result;
}

int pdhcg_postsolve(const pdhcg_presolve_info_t *info, pdhcg_result_t *result, const qp_problem_t *original_problem)
{
    const PreFOSPresolver *presolver;
    const PreFOSPresolvedProblem *reduced;
    double *reduced_y = NULL;
    double *reduced_z = NULL;
    double *original_x = NULL;
    double *prefos_y = NULL;
    double *prefos_z = NULL;
    double *original_y = NULL;
    double *original_z = NULL;
    PreFOSStatus status;
    size_t i;
    int dual_recovered = 1;

    if (!info || !info->presolver || !result || !original_problem)
        return 0;
    presolver = (const PreFOSPresolver *)info->presolver;
    reduced = prefos_get_reduced_problem(presolver);
    if (!reduced)
        return 0;
    if ((reduced->n > 0 && (!result->primal_solution || !result->reduced_cost)) ||
        (reduced->A.rows > 0 && !result->dual_solution))
        return 0;

    reduced_y = (double *)allocate_array(reduced->A.rows, sizeof(double));
    reduced_z = (double *)allocate_array(reduced->n, sizeof(double));
    original_x = (double *)calloc((size_t)original_problem->num_variables, sizeof(double));
    prefos_y = (double *)calloc((size_t)info->prefos_original_rows, sizeof(double));
    prefos_z = (double *)calloc((size_t)original_problem->num_variables, sizeof(double));
    if ((reduced->A.rows > 0 && !reduced_y) || (reduced->n > 0 && !reduced_z) ||
        (original_problem->num_variables > 0 && (!original_x || !prefos_z)) ||
        (info->prefos_original_rows > 0 && !prefos_y))
        goto failure;

    for (i = 0; i < reduced->A.rows; ++i)
        reduced_y[i] = -result->dual_solution[i];
    for (i = 0; i < reduced->n; ++i)
        reduced_z[i] = -result->reduced_cost[i];

    status = prefos_postsolve_primal_dual(presolver,
                                          result->primal_solution,
                                          reduced_y,
                                          reduced_z,
                                          info->postsolve_tolerance,
                                          original_x,
                                          prefos_y,
                                          prefos_z);
    if (status == PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE)
        status = prefos_postsolve_extended_dual(presolver,
                                                result->primal_solution,
                                                reduced_y,
                                                reduced_z,
                                                info->postsolve_tolerance,
                                                original_x,
                                                prefos_y,
                                                prefos_z);
    if (status != PREFOS_STATUS_OK)
    {
        dual_recovered = 0;
        if (original_problem->num_variables > 0)
            memset(original_x, 0, (size_t)original_problem->num_variables * sizeof(double));
        status = prefos_postsolve_primal(presolver, result->primal_solution, original_x);
        if (status != PREFOS_STATUS_OK)
            goto failure;
        if (info->prefos_original_rows > 0)
            memset(prefos_y, 0, (size_t)info->prefos_original_rows * sizeof(double));
        if (original_problem->num_variables > 0)
            memset(prefos_z, 0, (size_t)original_problem->num_variables * sizeof(double));
    }

    original_y = (double *)allocate_array((size_t)original_problem->num_constraints, sizeof(double));
    original_z = (double *)allocate_array((size_t)original_problem->num_variables, sizeof(double));
    if ((original_problem->num_constraints > 0 && !original_y) || (original_problem->num_variables > 0 && !original_z))
        goto failure;
    for (i = 0; i < (size_t)original_problem->num_constraints; ++i)
        original_y[i] = -prefos_y[i];
    for (i = 0; i < (size_t)original_problem->num_variables; ++i)
        original_z[i] = -prefos_z[i];

    free(result->primal_solution);
    free(result->dual_solution);
    free(result->reduced_cost);
    result->primal_solution = original_x;
    result->dual_solution = original_y;
    result->reduced_cost = original_z;
    original_x = NULL;
    original_y = NULL;
    original_z = NULL;

    initialize_result_dimensions(result, info, original_problem);
    if (reduced->n == 0)
    {
        result->primal_objective_value = reduced->objective_offset;
        result->dual_objective_value = reduced->objective_offset;
    }
    free(reduced_y);
    free(reduced_z);
    free(prefos_y);
    free(prefos_z);
    if (!dual_recovered)
        fprintf(stderr, "Warning: PreFOS recovered the primal solution but not a valid original dual solution.\n");
    return dual_recovered;

failure:
    free(reduced_y);
    free(reduced_z);
    free(original_x);
    free(prefos_y);
    free(prefos_z);
    free(original_y);
    free(original_z);
    return 0;
}

void pdhcg_presolve_info_free(pdhcg_presolve_info_t *info)
{
    if (!info)
        return;
    free_converted_problem(info->reduced_problem);
    if (info->presolver)
        prefos_free_presolver((PreFOSPresolver *)info->presolver);
    free(info);
}

const char *pdhcg_presolve_version(void)
{
    return "PreFOS " PREFOS_VERSION;
}

int pdhcg_presolve_available(void)
{
    return 1;
}

#else

#include <stdio.h>

const char *pdhcg_get_presolve_status_str(int status)
{
    (void)status;
    return "NOT_AVAILABLE";
}

pdhcg_presolve_info_t *pdhcg_presolve(const qp_problem_t *original_problem, const pdhg_parameters_t *parameters)
{
    (void)original_problem;
    (void)parameters;
    fprintf(stderr, "Warning: PreFOS not available; presolving disabled.\n");
    return NULL;
}

pdhcg_result_t *pdhcg_create_result_from_presolve(const pdhcg_presolve_info_t *info,
                                                  const qp_problem_t *original_problem)
{
    (void)info;
    (void)original_problem;
    return NULL;
}

int pdhcg_postsolve(const pdhcg_presolve_info_t *info, pdhcg_result_t *result, const qp_problem_t *original_problem)
{
    (void)info;
    (void)result;
    (void)original_problem;
    return 0;
}

void pdhcg_presolve_info_free(pdhcg_presolve_info_t *info)
{
    (void)info;
}

const char *pdhcg_presolve_version(void)
{
    return "PreFOS not available";
}

int pdhcg_presolve_available(void)
{
    return 0;
}

#endif
