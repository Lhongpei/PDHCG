/*
Copyright 2025 Haihao Lu
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

#include "pdhcg.h"
#include "cone_utils.h"
#include "distributed_interface.h"
#include "solver.h"
#include "utils.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

volatile sig_atomic_t g_pdhcg_cancel_request = 0;

static void csr_component_free(CsrComponent *csr);

static int validate_matrix_descriptor(const matrix_desc_t *desc, const char *name)
{
    if (!desc)
        return 0;
    if (desc->m < 0 || desc->n < 0)
    {
        fprintf(stderr, "[create_qp_problem] %s matrix has negative shape (%d, %d).\n", name, desc->m, desc->n);
        return -1;
    }

    switch (desc->fmt)
    {
        case matrix_dense:
            if (desc->m > 0 && desc->n > INT_MAX / desc->m)
            {
                fprintf(stderr, "[create_qp_problem] %s dense matrix is too large to index with int.\n", name);
                return -1;
            }
            if (desc->m > 0 && desc->n > 0 && !desc->data.dense.A)
            {
                fprintf(stderr, "[create_qp_problem] %s dense matrix data is NULL.\n", name);
                return -1;
            }
            return 0;

        case matrix_csr:
        {
            int nnz = desc->data.csr.nnz;
            const int *row_ptr = desc->data.csr.row_ptr;
            if (nnz < 0 || !row_ptr || (nnz > 0 && (!desc->data.csr.col_ind || !desc->data.csr.vals)))
            {
                fprintf(stderr, "[create_qp_problem] %s CSR storage is incomplete.\n", name);
                return -1;
            }
            if (row_ptr[0] != 0 || row_ptr[desc->m] != nnz)
            {
                fprintf(stderr, "[create_qp_problem] %s CSR row pointers do not span [0, nnz].\n", name);
                return -1;
            }
            for (int row = 0; row < desc->m; ++row)
            {
                if (row_ptr[row] > row_ptr[row + 1] || row_ptr[row] < 0 || row_ptr[row + 1] > nnz)
                {
                    fprintf(stderr, "[create_qp_problem] %s CSR row pointers are invalid at row %d.\n", name, row);
                    return -1;
                }
            }
            for (int entry = 0; entry < nnz; ++entry)
            {
                int column = desc->data.csr.col_ind[entry];
                if (column < 0 || column >= desc->n)
                {
                    fprintf(stderr,
                            "[create_qp_problem] %s CSR column index %d is out of range [0, %d).\n",
                            name,
                            column,
                            desc->n);
                    return -1;
                }
            }
            return 0;
        }

        case matrix_csc:
        {
            int nnz = desc->data.csc.nnz;
            const int *col_ptr = desc->data.csc.col_ptr;
            if (nnz < 0 || !col_ptr || (nnz > 0 && (!desc->data.csc.row_ind || !desc->data.csc.vals)))
            {
                fprintf(stderr, "[create_qp_problem] %s CSC storage is incomplete.\n", name);
                return -1;
            }
            if (col_ptr[0] != 0 || col_ptr[desc->n] != nnz)
            {
                fprintf(stderr, "[create_qp_problem] %s CSC column pointers do not span [0, nnz].\n", name);
                return -1;
            }
            for (int column = 0; column < desc->n; ++column)
            {
                if (col_ptr[column] > col_ptr[column + 1] || col_ptr[column] < 0 || col_ptr[column + 1] > nnz)
                {
                    fprintf(
                        stderr, "[create_qp_problem] %s CSC column pointers are invalid at column %d.\n", name, column);
                    return -1;
                }
            }
            for (int entry = 0; entry < nnz; ++entry)
            {
                int row = desc->data.csc.row_ind[entry];
                if (row < 0 || row >= desc->m)
                {
                    fprintf(stderr,
                            "[create_qp_problem] %s CSC row index %d is out of range [0, %d).\n",
                            name,
                            row,
                            desc->m);
                    return -1;
                }
            }
            return 0;
        }

        case matrix_coo:
        {
            int nnz = desc->data.coo.nnz;
            if (nnz < 0 || (nnz > 0 && (!desc->data.coo.row_ind || !desc->data.coo.col_ind || !desc->data.coo.vals)))
            {
                fprintf(stderr, "[create_qp_problem] %s COO storage is incomplete.\n", name);
                return -1;
            }
            for (int entry = 0; entry < nnz; ++entry)
            {
                int row = desc->data.coo.row_ind[entry];
                int column = desc->data.coo.col_ind[entry];
                if (row < 0 || row >= desc->m || column < 0 || column >= desc->n)
                {
                    fprintf(stderr,
                            "[create_qp_problem] %s COO index (%d, %d) is out of range for shape (%d, %d).\n",
                            name,
                            row,
                            column,
                            desc->m,
                            desc->n);
                    return -1;
                }
            }
            return 0;
        }

        default:
            fprintf(stderr, "[create_qp_problem] %s matrix has unsupported format %d.\n", name, (int)desc->fmt);
            return -1;
    }
}

static int validate_problem_matrix_shapes(const matrix_desc_t *A_desc,
                                          const matrix_desc_t *F_desc,
                                          const matrix_desc_t *Q_desc,
                                          const matrix_desc_t *R_desc,
                                          const matrix_desc_t *D_desc,
                                          int *num_variables,
                                          int *num_scalar_constraints,
                                          int *num_affine_constraints)
{
    if (!A_desc && !F_desc && !Q_desc && !R_desc)
    {
        fprintf(stderr, "[create_qp_problem] at least one of A, F, Q, or R must be provided.\n");
        return -1;
    }
    if (validate_matrix_descriptor(A_desc, "A") != 0 || validate_matrix_descriptor(F_desc, "F") != 0 ||
        validate_matrix_descriptor(Q_desc, "Q") != 0 || validate_matrix_descriptor(R_desc, "R") != 0)
        return -1;

    int n = A_desc ? A_desc->n : (F_desc ? F_desc->n : (Q_desc ? Q_desc->n : R_desc->n));
    int m = A_desc ? A_desc->m : 0;
    int p = F_desc ? F_desc->m : 0;
    if (F_desc && F_desc->n != n)
    {
        fprintf(stderr, "[create_qp_problem] F matrix shape (%d, %d) must have %d columns.\n", F_desc->m, F_desc->n, n);
        return -1;
    }
    if (Q_desc && (Q_desc->m != n || Q_desc->n != n))
    {
        fprintf(stderr, "[create_qp_problem] Q matrix shape (%d, %d) must be (%d, %d).\n", Q_desc->m, Q_desc->n, n, n);
        return -1;
    }
    if (R_desc && R_desc->n != n)
    {
        fprintf(stderr, "[create_qp_problem] R matrix shape (%d, %d) must have %d columns.\n", R_desc->m, R_desc->n, n);
        return -1;
    }
    if (D_desc && R_desc && R_desc->m > 0)
    {
        int rank = R_desc->m;
        if (validate_matrix_descriptor(D_desc, "D") != 0 || D_desc->m != rank || D_desc->n != rank)
        {
            fprintf(stderr,
                    "[create_qp_problem] D matrix shape (%d, %d) must be (%d, %d).\n",
                    D_desc->m,
                    D_desc->n,
                    rank,
                    rank);
            return -1;
        }
    }
    if (p > INT_MAX - m)
    {
        fprintf(stderr, "[create_qp_problem] combined A and F matrices have too many rows.\n");
        return -1;
    }

    *num_variables = n;
    *num_scalar_constraints = m;
    *num_affine_constraints = p;
    return 0;
}

static int
copy_matrix_desc_to_csr(const matrix_desc_t *desc, const char *name, CsrComponent *destination, int *num_nonzeros)
{
    int rc = 0;
    switch (desc->fmt)
    {
        case matrix_dense:
            rc = dense_to_csr(desc, &destination->row_ptr, &destination->col_ind, &destination->val, num_nonzeros);
            break;
        case matrix_csc:
            rc = csc_to_csr(desc, &destination->row_ptr, &destination->col_ind, &destination->val, num_nonzeros);
            break;
        case matrix_coo:
            rc = coo_to_csr(desc, &destination->row_ptr, &destination->col_ind, &destination->val, num_nonzeros);
            break;
        case matrix_csr:
        {
            int nnz = desc->data.csr.nnz;
            destination->row_ptr = (int *)safe_malloc((size_t)(desc->m + 1) * sizeof(int));
            memcpy(destination->row_ptr, desc->data.csr.row_ptr, (size_t)(desc->m + 1) * sizeof(int));
            if (nnz > 0)
            {
                destination->col_ind = (int *)safe_malloc((size_t)nnz * sizeof(int));
                destination->val = (double *)safe_malloc((size_t)nnz * sizeof(double));
                memcpy(destination->col_ind, desc->data.csr.col_ind, (size_t)nnz * sizeof(int));
                memcpy(destination->val, desc->data.csr.vals, (size_t)nnz * sizeof(double));
            }
            *num_nonzeros = nnz;
            break;
        }
        default:
            rc = -1;
            break;
    }
    if (rc != 0)
    {
        fprintf(stderr, "[create_qp_problem] failed to convert %s matrix to CSR.\n", name);
        csr_component_free(destination);
    }
    return rc;
}

static void initialize_empty_csr(CsrComponent *component, int num_rows, int *num_nonzeros)
{
    component->row_ptr = (int *)safe_calloc((size_t)num_rows + 1, sizeof(int));
    *num_nonzeros = 0;
}

static int append_matrix_desc_to_csr(
    const matrix_desc_t *desc, const char *name, int current_rows, CsrComponent *destination, int *num_nonzeros)
{
    if (!desc)
        return 0;

    CsrComponent converted = {0};
    const int *suffix_row_ptr = NULL;
    const int *suffix_col_ind = NULL;
    const double *suffix_values = NULL;
    int suffix_nonzeros = 0;
    if (desc->fmt == matrix_csr)
    {
        suffix_row_ptr = desc->data.csr.row_ptr;
        suffix_col_ind = desc->data.csr.col_ind;
        suffix_values = desc->data.csr.vals;
        suffix_nonzeros = desc->data.csr.nnz;
    }
    else
    {
        if (copy_matrix_desc_to_csr(desc, name, &converted, &suffix_nonzeros) != 0)
            return -1;
        suffix_row_ptr = converted.row_ptr;
        suffix_col_ind = converted.col_ind;
        suffix_values = converted.val;
    }
    if (suffix_nonzeros > INT_MAX - *num_nonzeros)
    {
        fprintf(stderr, "[create_qp_problem] combined A and F matrices have too many nonzeros.\n");
        csr_component_free(&converted);
        return -1;
    }

    int initial_nonzeros = *num_nonzeros;
    int total_nonzeros = initial_nonzeros + suffix_nonzeros;
    size_t total_rows = (size_t)current_rows + (size_t)desc->m;
    destination->row_ptr = (int *)safe_realloc(destination->row_ptr, (total_rows + 1) * sizeof(int));
    for (int row = 1; row <= desc->m; ++row)
        destination->row_ptr[current_rows + row] = initial_nonzeros + suffix_row_ptr[row];

    if (suffix_nonzeros > 0)
    {
        destination->col_ind = (int *)safe_realloc(destination->col_ind, (size_t)total_nonzeros * sizeof(int));
        destination->val = (double *)safe_realloc(destination->val, (size_t)total_nonzeros * sizeof(double));
        memcpy(destination->col_ind + initial_nonzeros, suffix_col_ind, (size_t)suffix_nonzeros * sizeof(int));
        memcpy(destination->val + initial_nonzeros, suffix_values, (size_t)suffix_nonzeros * sizeof(double));
    }
    *num_nonzeros = total_nonzeros;
    csr_component_free(&converted);
    return 0;
}

static int initialize_affine_cones(qp_problem_t *prob,
                                   int num_scalar_constraints,
                                   int num_affine_constraints,
                                   int num_affine_cones,
                                   const cone_spec_t *affine_cones,
                                   const double *affine_cone_offset)
{
    if (cone_blocks_init_from_specs(
            &prob->affine_cones, num_affine_cones, affine_cones, num_affine_constraints, false, "affine") != 0)
        return -1;

    int covered_rows = 0;
    for (int cone = 0; cone < num_affine_cones; ++cone)
    {
        int length = cone_block_length(&prob->affine_cones, cone);
        covered_rows += length;
        prob->affine_cones.start_idx[cone] += num_scalar_constraints;
    }
    if (covered_rows != num_affine_constraints)
    {
        fprintf(stderr,
                "[create_qp_problem] affine cone blocks cover %d of %d rows of F.\n",
                covered_rows,
                num_affine_constraints);
        return -1;
    }
    if (affine_cone_offset && num_affine_constraints > 0)
        memcpy(prob->affine_cone_offset + num_scalar_constraints,
               affine_cone_offset,
               (size_t)num_affine_constraints * sizeof(double));
    return 0;
}

qp_problem_t *create_qp_problem(const double *objective_c,
                                const matrix_desc_t *Q_desc,
                                const matrix_desc_t *R_desc,
                                const matrix_desc_t *D_desc,
                                const matrix_desc_t *A_desc,
                                const double *con_lb,
                                const double *con_ub,
                                const double *var_lb,
                                const double *var_ub,
                                const double *objective_constant,
                                int num_var_cones,
                                const cone_spec_t *var_cones,
                                const matrix_desc_t *affine_cone_matrix_desc,
                                const double *affine_cone_offset,
                                int num_affine_cones,
                                const cone_spec_t *affine_cones)
{
    qp_problem_t *prob = (qp_problem_t *)safe_calloc(1, sizeof(qp_problem_t));
    int n = 0;
    int m = 0;
    int p = 0;
    if (!affine_cone_matrix_desc && (affine_cone_offset || num_affine_cones != 0 || affine_cones))
    {
        fprintf(stderr, "[create_qp_problem] affine cone data requires affine_cone_matrix_desc.\n");
        goto failure;
    }
    if (validate_problem_matrix_shapes(A_desc, affine_cone_matrix_desc, Q_desc, R_desc, D_desc, &n, &m, &p) != 0)
        goto failure;

    prob->num_variables = n;
    prob->num_constraints = m + p;
    prob->affine_cone_offset =
        prob->num_constraints > 0 ? (double *)safe_calloc((size_t)prob->num_constraints, sizeof(double)) : NULL;

    prob->constraint_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
    if (A_desc)
    {
        if (copy_matrix_desc_to_csr(A_desc, "A", prob->constraint_matrix, &prob->constraint_matrix_num_nonzeros) != 0)
            goto failure;
    }
    else
        initialize_empty_csr(prob->constraint_matrix, m, &prob->constraint_matrix_num_nonzeros);
    if (append_matrix_desc_to_csr(
            affine_cone_matrix_desc, "F", m, prob->constraint_matrix, &prob->constraint_matrix_num_nonzeros) != 0)
        goto failure;

    prob->objective_sparse_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
    if (Q_desc)
    {
        if (copy_matrix_desc_to_csr(
                Q_desc, "Q", prob->objective_sparse_matrix, &prob->objective_sparse_matrix_num_nonzeros) != 0)
            goto failure;
    }
    else
        initialize_empty_csr(prob->objective_sparse_matrix, n, &prob->objective_sparse_matrix_num_nonzeros);

    prob->objective_lowrank_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
    prob->num_rank_lowrank_obj = 0;

    if (R_desc)
    {
        prob->num_rank_lowrank_obj = R_desc->m;
        if (copy_matrix_desc_to_csr(
                R_desc, "R", prob->objective_lowrank_matrix, &prob->objective_lowrank_matrix_num_nonzeros) != 0)
            goto failure;
    }
    else
        initialize_empty_csr(prob->objective_lowrank_matrix, 0, &prob->objective_lowrank_matrix_num_nonzeros);

    prob->objective_lowrank_middle_matrix = NULL;
    prob->objective_lowrank_middle_matrix_num_nonzeros = 0;
    if (D_desc)
    {
        int k = prob->num_rank_lowrank_obj;
        if (k <= 0)
            fprintf(stderr, "[interface] D matrix ignored: problem has no low-rank component.\n");
        else
        {
            prob->objective_lowrank_middle_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
            if (copy_matrix_desc_to_csr(D_desc,
                                        "D",
                                        prob->objective_lowrank_middle_matrix,
                                        &prob->objective_lowrank_middle_matrix_num_nonzeros) != 0)
                goto failure;
        }
    }

    prob->objective_constant = objective_constant ? *objective_constant : 0.0;
    fill_or_copy(&prob->objective_vector, prob->num_variables, objective_c, 0.0);
    fill_or_copy(&prob->variable_lower_bound, prob->num_variables, var_lb, -INFINITY);
    fill_or_copy(&prob->variable_upper_bound, prob->num_variables, var_ub, INFINITY);
    fill_or_copy(&prob->constraint_lower_bound, prob->num_constraints, NULL, -INFINITY);
    fill_or_copy(&prob->constraint_upper_bound, prob->num_constraints, NULL, INFINITY);
    if (m > 0 && con_lb)
        memcpy(prob->constraint_lower_bound, con_lb, (size_t)m * sizeof(double));
    if (m > 0 && con_ub)
        memcpy(prob->constraint_upper_bound, con_ub, (size_t)m * sizeof(double));

    if (initialize_affine_cones(prob, m, p, num_affine_cones, affine_cones, affine_cone_offset) != 0)
        goto failure;
    if (cone_blocks_init_from_specs(&prob->cones, num_var_cones, var_cones, n, true, "variable") != 0)
        goto failure;

    /* A finite var bound on a cone slot makes proj_K ∘ proj_Box != proj_{K ∩ Box}; the
       caller must lift such variables with an auxiliary (x_cone = x_box) so cone slots
       stay free. Treat |bound| >= 1e30 as "free" (matches the +/- INFINITY sentinel and
       the 1e30 used in tests). */
    int n_orig = n;
    for (int cone = 0; cone < prob->cones.num_cones; ++cone)
    {
        int start = prob->cones.start_idx[cone];
        int length = cone_block_length(&prob->cones, cone);
        for (int variable = start; variable < start + length; ++variable)
        {
            double lo = prob->variable_lower_bound[variable];
            double hi = prob->variable_upper_bound[variable];
            int lo_finite = isfinite(lo) && lo > -1e30;
            int hi_finite = isfinite(hi) && hi < 1e30;
            if (lo_finite || hi_finite)
            {
                fprintf(stderr,
                        "[create_qp_problem] cone %d slot %d has a finite box bound "
                        "(lb=%.6g, ub=%.6g); cone variables must be free. Introduce an "
                        "auxiliary x_cone with x_cone = x_box and put the box on the "
                        "non-cone copy.\n",
                        cone,
                        variable,
                        lo,
                        hi);
                goto failure;
            }
        }
        if (start < n_orig)
            n_orig = start;
    }
    if (prob->cones.num_cones > 0)
        prob->num_original_variables = n_orig;

    return prob;

failure:
    qp_problem_free(prob);
    return NULL;
}

void pdhcg_result_free(pdhcg_result_t *results)
{
    if (results == NULL)
    {
        return;
    }

    free(results->primal_solution);
    free(results->dual_solution);
    free(results->reduced_cost);
    free(results);
}

static void csr_component_free(CsrComponent *csr)
{
    if (!csr)
        return;
    free(csr->row_ptr);
    free(csr->col_ind);
    free(csr->val);
    memset(csr, 0, sizeof(*csr));
}
void qp_problem_free(qp_problem_t *prob)
{
    if (!prob)
        return;
    csr_component_free(prob->objective_sparse_matrix);
    free(prob->objective_sparse_matrix);
    csr_component_free(prob->objective_lowrank_matrix);
    free(prob->objective_lowrank_matrix);
    csr_component_free(prob->constraint_matrix);
    free(prob->constraint_matrix);
    free(prob->variable_lower_bound);
    free(prob->variable_upper_bound);
    free(prob->objective_vector);
    free(prob->constraint_lower_bound);
    free(prob->constraint_upper_bound);
    free(prob->affine_cone_offset);
    free(prob->primal_start);
    free(prob->dual_start);
    csr_component_free(prob->objective_lowrank_middle_matrix);
    free(prob->objective_lowrank_middle_matrix);
    if (prob->quadratic_constraint_matrices)
    {
        for (int i = 0; i < prob->num_quadratic_constraints; ++i)
        {
            csr_component_free(prob->quadratic_constraint_matrices[i]);
            free(prob->quadratic_constraint_matrices[i]);
        }
        free(prob->quadratic_constraint_matrices);
    }
    free(prob->quadratic_constraint_row_indices);
    free(prob->quadratic_constraint_matrix_num_nonzeros);
    cone_blocks_free(&prob->cones);
    cone_blocks_free(&prob->affine_cones);
    memset(prob, 0, sizeof(*prob));
    free(prob);
}

void set_start_values(qp_problem_t *prob, const double *primal, const double *dual)
{
    if (!prob)
        return;

    int n = prob->num_variables;
    int m = prob->num_constraints;

    if (primal && prob->cones.is_fixed && prob->primal_start)
    {
        for (int i = 0; i < n; ++i)
        {
            if (prob->cones.is_fixed[i] && primal[i] != prob->primal_start[i])
            {
                fprintf(stderr,
                        "[set_start_values] slot %d is fixed at %.17g but caller provided %.17g; "
                        "rejecting (use set_cone_fixed to change fixed value).\n",
                        i,
                        prob->primal_start[i],
                        primal[i]);
                return;
            }
        }
    }

    double *new_primal_start = NULL;
    double *new_dual_start = NULL;
    if (primal)
    {
        new_primal_start = (double *)safe_malloc((size_t)n * sizeof(double));
        memcpy(new_primal_start, primal, (size_t)n * sizeof(double));
    }
    else if (prob->cones.is_fixed && prob->primal_start)
    {
        new_primal_start = (double *)safe_calloc((size_t)n, sizeof(double));
        for (int i = 0; i < n; ++i)
            if (prob->cones.is_fixed[i])
                new_primal_start[i] = prob->primal_start[i];
    }
    if (dual)
    {
        new_dual_start = (double *)safe_malloc((size_t)m * sizeof(double));
        memcpy(new_dual_start, dual, (size_t)m * sizeof(double));
    }

    free(prob->primal_start);
    free(prob->dual_start);
    prob->primal_start = new_primal_start;
    prob->dual_start = new_dual_start;
}

int set_cone_fixed(qp_problem_t *prob, int cone_idx, int slot, double value)
{
    if (!prob)
    {
        fprintf(stderr, "[set_cone_fixed] prob is NULL\n");
        return -1;
    }
    if (cone_idx < 0 || cone_idx >= prob->cones.num_cones)
    {
        fprintf(stderr, "[set_cone_fixed] cone_idx %d out of range [0, %d)\n", cone_idx, prob->cones.num_cones);
        return -1;
    }
    int len = cone_block_length(&prob->cones, cone_idx);
    if (slot < 0 || slot >= len)
    {
        fprintf(stderr, "[set_cone_fixed] slot %d out of range [0, %d) for cone %d\n", slot, len, cone_idx);
        return -1;
    }
    int idx = prob->cones.start_idx[cone_idx] + slot;
    if (idx < 0 || idx >= prob->num_variables)
    {
        fprintf(stderr, "[set_cone_fixed] computed index %d out of range [0, %d)\n", idx, prob->num_variables);
        return -1;
    }
    if (!isfinite(value))
    {
        fprintf(stderr, "[set_cone_fixed] fixed value must be finite; got %.17g\n", value);
        return -1;
    }

    if (!prob->cones.is_fixed)
    {
        prob->cones.is_fixed = (char *)safe_calloc(prob->num_variables, sizeof(char));
        prob->cones.fixed_mask_size = prob->num_variables;
    }
    prob->cones.is_fixed[idx] = 1;

    if (!prob->primal_start)
        prob->primal_start = (double *)safe_calloc(prob->num_variables, sizeof(double));
    prob->primal_start[idx] = value;
    return 0;
}

static double fixed_vector_norm(const qp_problem_t *problem, int start, int length)
{
    double norm = 0.0;
    for (int slot = 0; slot < length; ++slot)
    {
        int index = start + slot;
        if (problem->cones.is_fixed[index])
        {
            double value = problem->primal_start ? problem->primal_start[index] : 0.0;
            norm = hypot(norm, value);
        }
    }
    return norm;
}

static int fixed_exp_section_is_nonempty(const qp_problem_t *problem, int start)
{
    int fixed_x = problem->cones.is_fixed[start + 0] != 0;
    int fixed_y = problem->cones.is_fixed[start + 1] != 0;
    int fixed_z = problem->cones.is_fixed[start + 2] != 0;
    double x = problem->primal_start ? problem->primal_start[start + 0] : 0.0;
    double y = problem->primal_start ? problem->primal_start[start + 1] : 0.0;
    double z = problem->primal_start ? problem->primal_start[start + 2] : 0.0;

    if ((fixed_x && !isfinite(x)) || (fixed_y && !isfinite(y)) || (fixed_z && !isfinite(z)))
        return 0;

    if (fixed_y)
    {
        if (y < 0.0)
            return 0;
        if (y == 0.0)
            return (!fixed_x || x <= 0.0) && (!fixed_z || z >= 0.0);
        if (fixed_z && !(z > 0.0))
            return 0;
        if (fixed_x && fixed_z)
        {
            double log_bound = log(y) + x / y;
            double log_z = log(z);
            double tolerance = 64.0 * DBL_EPSILON * (1.0 + fabs(log_bound) + fabs(log_z));
            return log_bound <= log_z + tolerance;
        }
        return 1;
    }

    if (!fixed_z)
        return 1;
    if (z < 0.0)
        return 0;
    if (z == 0.0)
        return !fixed_x || x <= 0.0;
    if (!fixed_x || x <= 0.0)
        return 1;

    /* min_{y > 0} y exp(x / y) = e x for x > 0. */
    double log_minimum = 1.0 + log(x);
    double log_z = log(z);
    double tolerance = 64.0 * DBL_EPSILON * (1.0 + fabs(log_minimum) + fabs(log_z));
    return log_minimum <= log_z + tolerance;
}

int pdhcg_validate_fixed_cone_sections(const qp_problem_t *problem)
{
    if (!problem || !problem->cones.is_fixed)
        return 0;
    if (problem->cones.fixed_mask_size != problem->num_variables)
    {
        fprintf(stderr,
                "[solve_qp_problem] variable cone fixed mask has size %d; expected %d.\n",
                problem->cones.fixed_mask_size,
                problem->num_variables);
        return -1;
    }

    for (int cone = 0; cone < problem->cones.num_cones; ++cone)
    {
        int start = problem->cones.start_idx[cone];
        int vector_dimension = problem->cones.v_dim[cone];
        int length = cone_block_length(&problem->cones, cone);
        int any_fixed = 0;
        for (int slot = 0; slot < length; ++slot)
            any_fixed |= problem->cones.is_fixed[start + slot] != 0;
        if (!any_fixed)
            continue;

        if (problem->cones.type[cone] == CONE_EXPONENTIAL)
        {
            if (!fixed_exp_section_is_nonempty(problem, start))
            {
                fprintf(
                    stderr, "[solve_qp_problem] exponential cone %d has an empty or non-finite fixed section.\n", cone);
                return -1;
            }
            continue;
        }

        if (problem->cones.type[cone] == CONE_STANDARD_SOC)
        {
            int w_index = start + vector_dimension;
            int z_index = w_index + 1;
            int fixed_z = problem->cones.is_fixed[z_index] != 0;
            double z = problem->primal_start ? problem->primal_start[z_index] : 0.0;
            for (int slot = 0; slot < vector_dimension + 2; ++slot)
            {
                int index = start + slot;
                double value = problem->primal_start ? problem->primal_start[index] : 0.0;
                if (problem->cones.is_fixed[index] && !isfinite(value))
                {
                    fprintf(stderr, "[solve_qp_problem] standard SOC %d has a non-finite fixed value.\n", cone);
                    return -1;
                }
            }
            double fixed_norm = fixed_vector_norm(problem, start, vector_dimension + 1);
            if (fixed_z && (!(z >= 0.0) || fixed_norm > z))
            {
                fprintf(stderr,
                        "[solve_qp_problem] standard SOC %d has an empty fixed section "
                        "(fixed vector norm=%.17g, fixed z=%.17g).\n",
                        cone,
                        fixed_norm,
                        z);
                return -1;
            }
            continue;
        }

        if (problem->cones.type[cone] == CONE_ROTATED_SOC)
        {
            int s_index = start + vector_dimension;
            int t_index = s_index + 1;
            int fixed_s = problem->cones.is_fixed[s_index] != 0;
            int fixed_t = problem->cones.is_fixed[t_index] != 0;
            double s = problem->primal_start ? problem->primal_start[s_index] : 0.0;
            double t = problem->primal_start ? problem->primal_start[t_index] : 0.0;
            for (int slot = 0; slot < vector_dimension + 2; ++slot)
            {
                int index = start + slot;
                double value = problem->primal_start ? problem->primal_start[index] : 0.0;
                if (problem->cones.is_fixed[index] && !isfinite(value))
                {
                    fprintf(stderr, "[solve_qp_problem] rotated SOC %d has a non-finite fixed value.\n", cone);
                    return -1;
                }
            }
            double fixed_norm = fixed_vector_norm(problem, start, vector_dimension);
            int empty = (fixed_s && s < 0.0) || (fixed_t && t < 0.0);
            if (!empty && fixed_s && fixed_t)
                empty = fixed_norm > 1.41421356237309504880 * sqrt(s) * sqrt(t);
            else if (fixed_s && s == 0.0)
                empty |= fixed_norm > 0.0;
            else if (fixed_t && t == 0.0)
                empty |= fixed_norm > 0.0;
            if (empty)
            {
                fprintf(stderr,
                        "[solve_qp_problem] rotated SOC %d has an empty fixed section "
                        "(fixed vector norm=%.17g, s=%.17g%s, t=%.17g%s).\n",
                        cone,
                        fixed_norm,
                        s,
                        fixed_s ? " fixed" : "",
                        t,
                        fixed_t ? " fixed" : "");
                return -1;
            }
            continue;
        }

        if (problem->cones.type[cone] != CONE_POWER)
            continue;

        int fixed_x = problem->cones.is_fixed[start + 0] != 0;
        int fixed_y = problem->cones.is_fixed[start + 1] != 0;
        int fixed_z = problem->cones.is_fixed[start + 2] != 0;
        double x = problem->primal_start ? problem->primal_start[start + 0] : 0.0;
        double y = problem->primal_start ? problem->primal_start[start + 1] : 0.0;
        double z = problem->primal_start ? problem->primal_start[start + 2] : 0.0;

        if ((fixed_x && (!isfinite(x) || x < 0.0)) || (fixed_y && (!isfinite(y) || y < 0.0)) ||
            (fixed_z && !isfinite(z)))
        {
            fprintf(stderr,
                    "[solve_qp_problem] power cone %d has an invalid fixed value "
                    "(x=%.17g%s, y=%.17g%s, z=%.17g%s).\n",
                    cone,
                    x,
                    fixed_x ? " fixed" : "",
                    y,
                    fixed_y ? " fixed" : "",
                    z,
                    fixed_z ? " fixed" : "");
            return -1;
        }

        if (fixed_z && z != 0.0 && ((fixed_x && x == 0.0) || (fixed_y && y == 0.0)))
        {
            fprintf(stderr,
                    "[solve_qp_problem] power cone %d has an empty fixed section: "
                    "|z| is positive while a fixed nonnegative axis is zero.\n",
                    cone);
            return -1;
        }

        if (fixed_x && fixed_y && fixed_z && z != 0.0)
        {
            double alpha = problem->cones.power_alpha[cone];
            double log_bound = (x > 0.0 && y > 0.0) ? alpha * log(x) + (1.0 - alpha) * log(y) : -INFINITY;
            double log_abs_z = log(fabs(z));
            double roundoff_tolerance = 64.0 * DBL_EPSILON * (1.0 + fabs(log_bound) + fabs(log_abs_z));
            if (log_bound + roundoff_tolerance < log_abs_z)
            {
                fprintf(stderr,
                        "[solve_qp_problem] power cone %d has an infeasible fully fixed point "
                        "(x=%.17g, y=%.17g, z=%.17g, alpha=%.17g).\n",
                        cone,
                        x,
                        y,
                        z,
                        alpha);
                return -1;
            }
        }
    }
    return 0;
}

pdhcg_result_t *solve_qp_problem(const qp_problem_t *prob, const pdhg_parameters_t *params)
{
    if (!prob)
    {
        fprintf(stderr, "[interface] solve_qp_problem: invalid arguments.\n");
        return NULL;
    }
    if (pdhcg_validate_fixed_cone_sections(prob) != 0)
        return NULL;

    pdhg_parameters_t local_params;
    if (params)
    {
        local_params = *params;
    }
    else
    {
        set_default_parameters(&local_params);
    }

    pdhcg_result_t *res = optimize(&local_params, prob);
    if (!res)
    {
        fprintf(stderr, "[interface] optimize returned NULL.\n");
        return NULL;
    }

    return res;
}

pdhcg_result_t *solve_qp_problem_distributed(const pdhg_parameters_t *params, const qp_problem_t *original_problem)
{
    return pdhcg_distributed_optimize(params, original_problem);
}
