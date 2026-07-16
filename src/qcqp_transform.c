/*
Copyright 2026 Hongpei Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0
*/

#include "qcqp_transform.h"
#include "pdhcg.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void restore_qcqp_result_dimensions(pdhcg_result_t *result, const qp_problem_t *original)
{
    if (!result || !original)
        return;

    int n_orig = original->num_variables;
    int m_orig = original->num_constraints;

    if (n_orig >= 0 && n_orig < result->num_variables)
    {
        if (result->primal_solution)
        {
            double *new_primal = n_orig > 0 ? (double *)safe_malloc((size_t)n_orig * sizeof(double)) : NULL;
            if (n_orig > 0)
                memcpy(new_primal, result->primal_solution, (size_t)n_orig * sizeof(double));
            free(result->primal_solution);
            result->primal_solution = new_primal;
        }

        if (result->reduced_cost)
        {
            double *new_rc = n_orig > 0 ? (double *)safe_malloc((size_t)n_orig * sizeof(double)) : NULL;
            if (n_orig > 0)
                memcpy(new_rc, result->reduced_cost, (size_t)n_orig * sizeof(double));
            free(result->reduced_cost);
            result->reduced_cost = new_rc;
        }
        result->num_variables = n_orig;
    }
    if (m_orig >= 0 && m_orig < result->num_constraints)
    {
        if (result->dual_solution)
        {
            double *new_dual = m_orig > 0 ? (double *)safe_malloc((size_t)m_orig * sizeof(double)) : NULL;
            if (m_orig > 0)
                memcpy(new_dual, result->dual_solution, (size_t)m_orig * sizeof(double));
            free(result->dual_solution);
            result->dual_solution = new_dual;
        }
        result->num_constraints = m_orig;
    }
    result->num_nonzeros = original->constraint_matrix_num_nonzeros;
}

static int
extract_diag_signed(const CsrComponent *Q, int n, int nnz_max, int *out_cols, double *out_vals, int *sign_out)
{
    int count = 0;
    int sign = 0;
    for (int row = 0; row < n; ++row)
    {
        int start = Q->row_ptr[row];
        int end = Q->row_ptr[row + 1];
        for (int k = start; k < end; ++k)
        {
            int col = Q->col_ind[k];
            double val = Q->val[k];
            if (col != row)
                return -1;
            if (val == 0.0)
                continue;
            int s = (val > 0.0) ? +1 : -1;
            if (sign == 0)
                sign = s;
            else if (sign != s)
                return -1;
            if (count >= nnz_max)
                return -1;
            out_cols[count] = col;
            out_vals[count] = (val > 0.0) ? val : -val;
            count++;
        }
    }
    *sign_out = sign;
    return count;
}

qp_problem_t *qcqp_to_socp_qp(const qp_problem_t *orig, cone_type_t default_type)
{
    if (!orig)
        return NULL;
    if (orig->num_quadratic_constraints == 0)
    {
        fprintf(stderr, "[qcqp_to_socp_qp] no quadratic constraints; nothing to do.\n");
        return NULL;
    }

    int n_orig = orig->num_variables;
    int m_orig = orig->num_constraints;
    int K = orig->num_quadratic_constraints;
    const bool is_std = (default_type == CONE_STANDARD_SOC);
    const double SQRT2 = 1.4142135623730951;

    int *block_k = (int *)safe_malloc(K * sizeof(int));
    int **block_cols = (int **)safe_malloc(K * sizeof(int *));
    double **block_sqrt = (double **)safe_malloc(K * sizeof(double *));
    int *block_flip = (int *)safe_calloc(K, sizeof(int));
    double *block_b = (double *)safe_malloc(K * sizeof(double));
    long total_v = 0;
    for (int i = 0; i < K; ++i)
    {
        CsrComponent *Q = orig->quadratic_constraint_matrices[i];
        int nnz = orig->quadratic_constraint_matrix_num_nonzeros[i];
        int *cols = (int *)safe_malloc((nnz > 0 ? nnz : 1) * sizeof(int));
        double *qjj = (double *)safe_malloc((nnz > 0 ? nnz : 1) * sizeof(double));
        int sign = 0;
        int k = extract_diag_signed(Q, n_orig, nnz, cols, qjj, &sign);
        if (k < 0)
        {
            fprintf(stderr,
                    "[qcqp_to_socp_qp] Q_%d is non-diagonal or has mixed signs; "
                    "diagonal PSD (or all-NSD with >= sense) required.\n",
                    i);
            free(cols);
            free(qjj);
            for (int j = 0; j < i; ++j)
            {
                free(block_cols[j]);
                free(block_sqrt[j]);
            }
            free(block_k);
            free(block_cols);
            free(block_sqrt);
            free(block_flip);
            free(block_b);
            return NULL;
        }

        int row = orig->quadratic_constraint_row_indices[i];
        double lhs = orig->constraint_lower_bound[row];
        double rhs = orig->constraint_upper_bound[row];
        int flip = 0;
        double b_eff = 0.0;
        if (sign >= 0)
        {
            if (isfinite(lhs) || !isfinite(rhs))
            {
                fprintf(stderr,
                        "[qcqp_to_socp_qp] QC row %d (Q PSD) requires one-sided <= "
                        "(lhs=-inf, rhs finite); got lhs=%.3g rhs=%.3g.\n",
                        row,
                        lhs,
                        rhs);
                free(cols);
                free(qjj);
                for (int j = 0; j < i; ++j)
                {
                    free(block_cols[j]);
                    free(block_sqrt[j]);
                }
                free(block_k);
                free(block_cols);
                free(block_sqrt);
                free(block_flip);
                free(block_b);
                return NULL;
            }
            b_eff = rhs;
        }
        else
        {
            if (!isfinite(lhs) || isfinite(rhs))
            {
                fprintf(stderr,
                        "[qcqp_to_socp_qp] QC row %d (Q NSD) requires one-sided >= "
                        "(lhs finite, rhs=+inf); got lhs=%.3g rhs=%.3g.\n",
                        row,
                        lhs,
                        rhs);
                free(cols);
                free(qjj);
                for (int j = 0; j < i; ++j)
                {
                    free(block_cols[j]);
                    free(block_sqrt[j]);
                }
                free(block_k);
                free(block_cols);
                free(block_sqrt);
                free(block_flip);
                free(block_b);
                return NULL;
            }
            flip = 1;
            b_eff = -lhs;
        }

        for (int m = 0; m < k; ++m)
            qjj[m] = sqrt(2.0 * qjj[m]);
        block_k[i] = k;
        block_cols[i] = cols;
        block_sqrt[i] = qjj;
        block_flip[i] = flip;
        block_b[i] = b_eff;
        total_v += k;
    }

    const CsrComponent *A_orig_pre = orig->constraint_matrix;
    char *blk_pin = (char *)safe_calloc(K, sizeof(char));
    int num_pin = 0;
    for (int i = 0; i < K; ++i)
    {
        int row = orig->quadratic_constraint_row_indices[i];
        if (A_orig_pre->row_ptr[row + 1] - A_orig_pre->row_ptr[row] == 0)
        {
            blk_pin[i] = 1;
            num_pin++;
        }
    }

    long n_ext = (long)n_orig + total_v + 2L * K;
    long m_ext = (long)m_orig + total_v + K;
    long extras_per_block = is_std ? 2L : 1L;
    long nnz_ext = (long)orig->constraint_matrix_num_nonzeros + extras_per_block * (K - num_pin) + 2L * total_v +
        extras_per_block * (K - num_pin);
    if (n_ext > INT32_MAX || m_ext > INT32_MAX || nnz_ext > INT32_MAX)
    {
        fprintf(stderr, "[qcqp_to_socp_qp] extended problem size overflows int32.\n");
        goto fail_free_blocks;
    }

    qp_problem_t *out = (qp_problem_t *)safe_calloc(1, sizeof(qp_problem_t));
    out->num_variables = (int)n_ext;
    out->num_constraints = (int)m_ext;
    out->constraint_matrix_num_nonzeros = (int)nnz_ext;
    out->objective_constant = orig->objective_constant;

    out->objective_vector = (double *)safe_calloc(n_ext, sizeof(double));
    out->variable_lower_bound = (double *)safe_malloc(n_ext * sizeof(double));
    out->variable_upper_bound = (double *)safe_malloc(n_ext * sizeof(double));
    memcpy(out->objective_vector, orig->objective_vector, n_orig * sizeof(double));
    memcpy(out->variable_lower_bound, orig->variable_lower_bound, n_orig * sizeof(double));
    memcpy(out->variable_upper_bound, orig->variable_upper_bound, n_orig * sizeof(double));

    out->cones.num_cones = K;
    out->cones.start_idx = (int *)safe_malloc(K * sizeof(int));
    out->cones.v_dim = (int *)safe_malloc(K * sizeof(int));
    out->cones.type = (cone_type_t *)safe_malloc(K * sizeof(cone_type_t));
    out->cones.power_alpha = NULL;
    out->num_original_variables = n_orig;
    {
        long idx = n_orig;
        for (int i = 0; i < K; ++i)
        {
            int k = block_k[i];
            out->cones.start_idx[i] = (int)idx;
            out->cones.v_dim[i] = k;
            out->cones.type[i] = default_type;
            for (int m = 0; m < k; ++m)
            {
                out->variable_lower_bound[idx] = -INFINITY;
                out->variable_upper_bound[idx] = INFINITY;
                idx++;
            }
            out->variable_lower_bound[idx] = -INFINITY;
            out->variable_upper_bound[idx] = INFINITY;
            idx++;
            out->variable_lower_bound[idx] = -INFINITY;
            out->variable_upper_bound[idx] = INFINITY;
            idx++;
        }
    }

    out->constraint_lower_bound = (double *)safe_malloc(m_ext * sizeof(double));
    out->constraint_upper_bound = (double *)safe_malloc(m_ext * sizeof(double));
    memcpy(out->constraint_lower_bound, orig->constraint_lower_bound, m_orig * sizeof(double));
    memcpy(out->constraint_upper_bound, orig->constraint_upper_bound, m_orig * sizeof(double));
    for (int i = 0; i < K; ++i)
    {
        int row = orig->quadratic_constraint_row_indices[i];
        double rhs = blk_pin[i] ? 0.0 : (is_std ? block_b[i] * SQRT2 : block_b[i]);
        out->constraint_lower_bound[row] = rhs;
        out->constraint_upper_bound[row] = rhs;
    }
    for (long r = m_orig; r < m_orig + total_v; ++r)
    {
        out->constraint_lower_bound[r] = 0.0;
        out->constraint_upper_bound[r] = 0.0;
    }
    {
        double last_rhs = is_std ? SQRT2 : 1.0;
        long r = m_orig + total_v;
        for (int i = 0; i < K; ++i, ++r)
        {
            double v = blk_pin[i] ? 0.0 : last_rhs;
            out->constraint_lower_bound[r] = v;
            out->constraint_upper_bound[r] = v;
        }
    }

    int *qc_row_to_block = (int *)safe_malloc(m_orig * sizeof(int));
    for (int r = 0; r < m_orig; ++r)
        qc_row_to_block[r] = -1;
    for (int i = 0; i < K; ++i)
        qc_row_to_block[orig->quadratic_constraint_row_indices[i]] = i;

    out->constraint_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
    int *row_ptr_ext = (int *)safe_calloc(m_ext + 1, sizeof(int));
    int *col_ind_ext = (int *)safe_malloc(nnz_ext * sizeof(int));
    double *val_ext = (double *)safe_malloc(nnz_ext * sizeof(double));

    const CsrComponent *A_orig = orig->constraint_matrix;
    for (int r = 0; r < m_orig; ++r)
    {
        int orig_nnz = A_orig->row_ptr[r + 1] - A_orig->row_ptr[r];
        int blk_r = qc_row_to_block[r];
        int extra = (blk_r >= 0 && !blk_pin[blk_r]) ? (is_std ? 2 : 1) : 0;
        row_ptr_ext[r + 1] = orig_nnz + extra;
    }
    long extra_row = m_orig;
    for (int i = 0; i < K; ++i)
    {
        for (int m = 0; m < block_k[i]; ++m)
        {
            row_ptr_ext[extra_row + 1] = 2;
            extra_row++;
        }
    }
    for (int i = 0; i < K; ++i)
    {
        row_ptr_ext[extra_row + 1] = blk_pin[i] ? 0 : (is_std ? 2 : 1);
        extra_row++;
    }
    for (long r = 1; r <= m_ext; ++r)
        row_ptr_ext[r] += row_ptr_ext[r - 1];

    for (int r = 0; r < m_orig; ++r)
    {
        int dst = row_ptr_ext[r];
        int s = A_orig->row_ptr[r];
        int e = A_orig->row_ptr[r + 1];
        int blk = qc_row_to_block[r];
        double scale = (blk >= 0 && block_flip[blk]) ? -1.0 : 1.0;
        double xscale = (is_std && blk >= 0) ? scale * SQRT2 : scale;
        for (int k = s; k < e; ++k)
        {
            col_ind_ext[dst] = A_orig->col_ind[k];
            val_ext[dst] = xscale * A_orig->val[k];
            dst++;
        }
        if (blk >= 0 && !blk_pin[blk])
        {
            int aux0 = out->cones.start_idx[blk] + out->cones.v_dim[blk];
            col_ind_ext[dst] = aux0;
            val_ext[dst] = 1.0;
            dst++;
            if (is_std)
            {
                col_ind_ext[dst] = aux0 + 1;
                val_ext[dst] = 1.0;
                dst++;
            }
        }
    }
    extra_row = m_orig;
    for (int i = 0; i < K; ++i)
    {
        int v_start = out->cones.start_idx[i];
        for (int m = 0; m < block_k[i]; ++m)
        {
            int dst = row_ptr_ext[extra_row];
            col_ind_ext[dst] = block_cols[i][m];
            val_ext[dst] = -block_sqrt[i][m];
            dst++;
            col_ind_ext[dst] = v_start + m;
            val_ext[dst] = 1.0;
            extra_row++;
        }
    }
    for (int i = 0; i < K; ++i)
    {
        if (blk_pin[i])
        {
            extra_row++;
            continue;
        }
        int aux0 = out->cones.start_idx[i] + out->cones.v_dim[i];
        int dst = row_ptr_ext[extra_row];
        if (is_std)
        {
            col_ind_ext[dst] = aux0;
            val_ext[dst] = -1.0;
            dst++;
            col_ind_ext[dst] = aux0 + 1;
            val_ext[dst] = 1.0;
        }
        else
        {
            col_ind_ext[dst] = aux0 + 1;
            val_ext[dst] = 1.0;
        }
        extra_row++;
    }

    out->constraint_matrix->row_ptr = row_ptr_ext;
    out->constraint_matrix->col_ind = col_ind_ext;
    out->constraint_matrix->val = val_ext;

    out->num_rank_lowrank_obj = orig->num_rank_lowrank_obj;
    out->objective_sparse_matrix_num_nonzeros = orig->objective_sparse_matrix_num_nonzeros;
    out->objective_lowrank_matrix_num_nonzeros = orig->objective_lowrank_matrix_num_nonzeros;
    out->objective_lowrank_middle_matrix_num_nonzeros = orig->objective_lowrank_middle_matrix_num_nonzeros;

    if (orig->objective_sparse_matrix)
    {
        int nz = orig->objective_sparse_matrix_num_nonzeros;
        int total_rows = (int)n_ext;
        out->objective_sparse_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
        out->objective_sparse_matrix->row_ptr = (int *)safe_malloc((size_t)(total_rows + 1) * sizeof(int));
        memcpy(out->objective_sparse_matrix->row_ptr,
               orig->objective_sparse_matrix->row_ptr,
               (size_t)(n_orig + 1) * sizeof(int));
        int last = orig->objective_sparse_matrix->row_ptr[n_orig];
        for (int r = n_orig + 1; r <= total_rows; ++r)
            out->objective_sparse_matrix->row_ptr[r] = last;
        if (nz > 0)
        {
            out->objective_sparse_matrix->col_ind = (int *)safe_malloc((size_t)nz * sizeof(int));
            out->objective_sparse_matrix->val = (double *)safe_malloc((size_t)nz * sizeof(double));
            memcpy(out->objective_sparse_matrix->col_ind,
                   orig->objective_sparse_matrix->col_ind,
                   (size_t)nz * sizeof(int));
            memcpy(out->objective_sparse_matrix->val, orig->objective_sparse_matrix->val, (size_t)nz * sizeof(double));
        }
    }

    if (orig->objective_lowrank_matrix)
    {
        int nr = orig->num_rank_lowrank_obj;
        int nz = orig->objective_lowrank_matrix_num_nonzeros;
        out->objective_lowrank_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
        out->objective_lowrank_matrix->row_ptr = (int *)safe_malloc((size_t)(nr + 1) * sizeof(int));
        memcpy(out->objective_lowrank_matrix->row_ptr,
               orig->objective_lowrank_matrix->row_ptr,
               (size_t)(nr + 1) * sizeof(int));
        if (nz > 0)
        {
            out->objective_lowrank_matrix->col_ind = (int *)safe_malloc((size_t)nz * sizeof(int));
            out->objective_lowrank_matrix->val = (double *)safe_malloc((size_t)nz * sizeof(double));
            memcpy(out->objective_lowrank_matrix->col_ind,
                   orig->objective_lowrank_matrix->col_ind,
                   (size_t)nz * sizeof(int));
            memcpy(
                out->objective_lowrank_matrix->val, orig->objective_lowrank_matrix->val, (size_t)nz * sizeof(double));
        }
    }

    if (orig->objective_lowrank_middle_matrix)
    {
        int nr = orig->num_rank_lowrank_obj;
        int nz = orig->objective_lowrank_middle_matrix_num_nonzeros;
        out->objective_lowrank_middle_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
        out->objective_lowrank_middle_matrix->row_ptr = (int *)safe_malloc((size_t)(nr + 1) * sizeof(int));
        memcpy(out->objective_lowrank_middle_matrix->row_ptr,
               orig->objective_lowrank_middle_matrix->row_ptr,
               (size_t)(nr + 1) * sizeof(int));
        if (nz > 0)
        {
            out->objective_lowrank_middle_matrix->col_ind = (int *)safe_malloc((size_t)nz * sizeof(int));
            out->objective_lowrank_middle_matrix->val = (double *)safe_malloc((size_t)nz * sizeof(double));
            memcpy(out->objective_lowrank_middle_matrix->col_ind,
                   orig->objective_lowrank_middle_matrix->col_ind,
                   (size_t)nz * sizeof(int));
            memcpy(out->objective_lowrank_middle_matrix->val,
                   orig->objective_lowrank_middle_matrix->val,
                   (size_t)nz * sizeof(double));
        }
    }

    out->num_quadratic_constraints = 0;
    out->quadratic_constraint_row_indices = NULL;
    out->quadratic_constraint_matrices = NULL;
    out->quadratic_constraint_matrix_num_nonzeros = NULL;

    out->primal_start = NULL;
    out->dual_start = NULL;

    if (num_pin > 0)
    {
        out->cones.is_fixed = (char *)safe_calloc(n_ext, sizeof(char));
        out->primal_start = (double *)safe_calloc(n_ext, sizeof(double));
        for (int i = 0; i < K; ++i)
        {
            if (!blk_pin[i])
                continue;
            int s_slot = out->cones.start_idx[i] + out->cones.v_dim[i];
            int t_slot = s_slot + 1;
            out->cones.is_fixed[s_slot] = 1;
            out->cones.is_fixed[t_slot] = 1;
            if (is_std)
            {
                out->primal_start[s_slot] = (block_b[i] - 1.0) * SQRT2 * 0.5;
                out->primal_start[t_slot] = (block_b[i] + 1.0) * SQRT2 * 0.5;
            }
            else
            {
                out->primal_start[s_slot] = block_b[i];
                out->primal_start[t_slot] = 1.0;
            }
        }
    }

    free(blk_pin);
    free(qc_row_to_block);
    for (int i = 0; i < K; ++i)
    {
        free(block_cols[i]);
        free(block_sqrt[i]);
    }
    free(block_k);
    free(block_cols);
    free(block_sqrt);
    free(block_flip);
    free(block_b);

    return out;

fail_free_blocks:
    for (int j = 0; j < K; ++j)
    {
        free(block_cols[j]);
        free(block_sqrt[j]);
    }
    free(block_k);
    free(block_cols);
    free(block_sqrt);
    free(block_flip);
    free(block_b);
    free(blk_pin);
    return NULL;
}
