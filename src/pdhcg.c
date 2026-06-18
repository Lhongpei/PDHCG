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
#include "solver.h"
#include "utils.h"
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PDHCG_COMPILE_DISTRIBUTED
#include "distributed_solver.h"
#endif

volatile sig_atomic_t g_pdhcg_cancel_request = 0;

qp_problem_t *create_qp_problem(const double *objective_c,
                                const matrix_desc_t *Q_desc,
                                const matrix_desc_t *R_desc,
                                const matrix_desc_t *D_desc,
                                const matrix_desc_t *A_desc,
                                const double *con_lb,
                                const double *con_ub,
                                const double *var_lb,
                                const double *var_ub,
                                const double *objective_constant)
{
    qp_problem_t *prob = (qp_problem_t *)safe_malloc(sizeof(qp_problem_t));
    prob->primal_start = NULL;
    prob->dual_start = NULL;

    int n = 0;
    int m = 0;

    if (A_desc)
    {
        n = A_desc->n;
        m = A_desc->m;
    }
    else if (Q_desc)
    {
        n = Q_desc->n;
    }
    else if (R_desc)
    {
        n = R_desc->n;
    }
    else
    {
        fprintf(stderr,
                "[interface] Error: At least one matrix (A, Q, or R) must "
                "be provided for non-trivil optimization problem.\n");
        free(prob);
        return NULL;
    }

    if (n == 0 && (Q_desc || R_desc || A_desc))
    {
        fprintf(stderr,
                "[interface] Warning: Matrix dimensions seem to be zero or "
                "inconsistent.\n");
    }

    prob->num_variables = n;
    prob->num_constraints = m;

    prob->constraint_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
    if (A_desc)
    {
        switch (A_desc->fmt)
        {
            case matrix_dense:
                dense_to_csr(A_desc,
                             &prob->constraint_matrix->row_ptr,
                             &prob->constraint_matrix->col_ind,
                             &prob->constraint_matrix->val,
                             &prob->constraint_matrix_num_nonzeros);
                break;
            case matrix_csc:
            {
                int *row_ptr = NULL, *col_ind = NULL;
                double *vals = NULL;
                int nnz = 0;
                if (csc_to_csr(A_desc, &row_ptr, &col_ind, &vals, &nnz) != 0)
                {
                    fprintf(stderr, "[interface] A matrix CSC->CSR failed.\n");
                    free(prob);
                    return NULL;
                }
                prob->constraint_matrix_num_nonzeros = nnz;
                prob->constraint_matrix->row_ptr = row_ptr;
                prob->constraint_matrix->col_ind = col_ind;
                prob->constraint_matrix->val = vals;
                break;
            }
            case matrix_coo:
            {
                int *row_ptr = NULL, *col_ind = NULL;
                double *vals = NULL;
                int nnz = 0;
                if (coo_to_csr(A_desc, &row_ptr, &col_ind, &vals, &nnz) != 0)
                {
                    fprintf(stderr, "[interface] A matrix COO->CSR failed.\n");
                    free(prob);
                    return NULL;
                }
                prob->constraint_matrix_num_nonzeros = nnz;
                prob->constraint_matrix->row_ptr = row_ptr;
                prob->constraint_matrix->col_ind = col_ind;
                prob->constraint_matrix->val = vals;
                break;
            }
            case matrix_csr:
                prob->constraint_matrix_num_nonzeros = A_desc->data.csr.nnz;
                prob->constraint_matrix->row_ptr = (int *)safe_malloc((size_t)(A_desc->m + 1) * sizeof(int));
                prob->constraint_matrix->col_ind = (int *)safe_malloc((size_t)A_desc->data.csr.nnz * sizeof(int));
                prob->constraint_matrix->val = (double *)safe_malloc((size_t)A_desc->data.csr.nnz * sizeof(double));
                memcpy(
                    prob->constraint_matrix->row_ptr, A_desc->data.csr.row_ptr, (size_t)(A_desc->m + 1) * sizeof(int));
                memcpy(prob->constraint_matrix->col_ind,
                       A_desc->data.csr.col_ind,
                       (size_t)A_desc->data.csr.nnz * sizeof(int));
                memcpy(
                    prob->constraint_matrix->val, A_desc->data.csr.vals, (size_t)A_desc->data.csr.nnz * sizeof(double));
                break;
            default:
                fprintf(stderr, "[interface] A matrix: unsupported format %d.\n", A_desc->fmt);
                free(prob);
                return NULL;
        }
    }
    else
    {
        prob->constraint_matrix_num_nonzeros = 0;
        prob->constraint_matrix->row_ptr = (int *)safe_calloc(m + 1, sizeof(int));
        prob->constraint_matrix->col_ind = NULL;
        prob->constraint_matrix->val = NULL;
    }

    prob->objective_sparse_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
    if (Q_desc)
    {
        switch (Q_desc->fmt)
        {
            case matrix_dense:
                dense_to_csr(Q_desc,
                             &prob->objective_sparse_matrix->row_ptr,
                             &prob->objective_sparse_matrix->col_ind,
                             &prob->objective_sparse_matrix->val,
                             &prob->objective_sparse_matrix_num_nonzeros);
                break;
            case matrix_csc:
            {
                int *row_ptr = NULL, *col_ind = NULL;
                double *vals = NULL;
                int nnz = 0;
                if (csc_to_csr(Q_desc, &row_ptr, &col_ind, &vals, &nnz) != 0)
                {
                    fprintf(stderr, "[interface] Q matrix CSC->CSR failed.\n");
                    free(prob);
                    return NULL;
                }
                prob->objective_sparse_matrix_num_nonzeros = nnz;
                prob->objective_sparse_matrix->row_ptr = row_ptr;
                prob->objective_sparse_matrix->col_ind = col_ind;
                prob->objective_sparse_matrix->val = vals;
                break;
            }
            case matrix_coo:
            {
                int *row_ptr = NULL, *col_ind = NULL;
                double *vals = NULL;
                int nnz = 0;
                if (coo_to_csr(Q_desc, &row_ptr, &col_ind, &vals, &nnz) != 0)
                {
                    fprintf(stderr, "[interface] Q matrix COO->CSR failed.\n");
                    free(prob);
                    return NULL;
                }
                prob->objective_sparse_matrix_num_nonzeros = nnz;
                prob->objective_sparse_matrix->row_ptr = row_ptr;
                prob->objective_sparse_matrix->col_ind = col_ind;
                prob->objective_sparse_matrix->val = vals;
                break;
            }
            case matrix_csr:
                prob->objective_sparse_matrix_num_nonzeros = Q_desc->data.csr.nnz;
                prob->objective_sparse_matrix->row_ptr = (int *)safe_malloc((size_t)(Q_desc->m + 1) * sizeof(int));
                prob->objective_sparse_matrix->col_ind = (int *)safe_malloc((size_t)Q_desc->data.csr.nnz * sizeof(int));
                prob->objective_sparse_matrix->val =
                    (double *)safe_malloc((size_t)Q_desc->data.csr.nnz * sizeof(double));
                memcpy(prob->objective_sparse_matrix->row_ptr,
                       Q_desc->data.csr.row_ptr,
                       (size_t)(Q_desc->m + 1) * sizeof(int));
                memcpy(prob->objective_sparse_matrix->col_ind,
                       Q_desc->data.csr.col_ind,
                       (size_t)Q_desc->data.csr.nnz * sizeof(int));
                memcpy(prob->objective_sparse_matrix->val,
                       Q_desc->data.csr.vals,
                       (size_t)Q_desc->data.csr.nnz * sizeof(double));
                break;
            default:
                fprintf(stderr, "[interface] Q matrix: unsupported format %d.\n", Q_desc->fmt);
                free(prob);
                return NULL;
        }
    }
    else
    {
        prob->objective_sparse_matrix->row_ptr = (int *)safe_calloc(n + 1, sizeof(int));
        prob->objective_sparse_matrix->col_ind = NULL;
        prob->objective_sparse_matrix->val = NULL;
        prob->objective_sparse_matrix_num_nonzeros = 0;
    }

    prob->objective_lowrank_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
    prob->num_rank_lowrank_obj = 0;

    if (R_desc)
    {
        prob->num_rank_lowrank_obj = R_desc->m;
        switch (R_desc->fmt)
        {
            case matrix_dense:
                dense_to_csr(R_desc,
                             &prob->objective_lowrank_matrix->row_ptr,
                             &prob->objective_lowrank_matrix->col_ind,
                             &prob->objective_lowrank_matrix->val,
                             &prob->objective_lowrank_matrix_num_nonzeros);
                break;
            case matrix_csc:
            {
                int *row_ptr = NULL, *col_ind = NULL;
                double *vals = NULL;
                int nnz = 0;
                if (csc_to_csr(R_desc, &row_ptr, &col_ind, &vals, &nnz) != 0)
                {
                    fprintf(stderr, "[interface] R matrix CSC->CSR failed.\n");
                    free(prob);
                    return NULL;
                }
                prob->objective_lowrank_matrix_num_nonzeros = nnz;
                prob->objective_lowrank_matrix->row_ptr = row_ptr;
                prob->objective_lowrank_matrix->col_ind = col_ind;
                prob->objective_lowrank_matrix->val = vals;
                break;
            }
            case matrix_coo:
            {
                int *row_ptr = NULL, *col_ind = NULL;
                double *vals = NULL;
                int nnz = 0;
                if (coo_to_csr(R_desc, &row_ptr, &col_ind, &vals, &nnz) != 0)
                {
                    fprintf(stderr, "[interface] R matrix COO->CSR failed.\n");
                    free(prob);
                    return NULL;
                }
                prob->objective_lowrank_matrix_num_nonzeros = nnz;
                prob->objective_lowrank_matrix->row_ptr = row_ptr;
                prob->objective_lowrank_matrix->col_ind = col_ind;
                prob->objective_lowrank_matrix->val = vals;
                break;
            }
            case matrix_csr:
                prob->objective_lowrank_matrix_num_nonzeros = R_desc->data.csr.nnz;
                prob->objective_lowrank_matrix->row_ptr = (int *)safe_malloc((size_t)(R_desc->m + 1) * sizeof(int));
                prob->objective_lowrank_matrix->col_ind =
                    (int *)safe_malloc((size_t)R_desc->data.csr.nnz * sizeof(int));
                prob->objective_lowrank_matrix->val =
                    (double *)safe_malloc((size_t)R_desc->data.csr.nnz * sizeof(double));
                memcpy(prob->objective_lowrank_matrix->row_ptr,
                       R_desc->data.csr.row_ptr,
                       (size_t)(R_desc->m + 1) * sizeof(int));
                memcpy(prob->objective_lowrank_matrix->col_ind,
                       R_desc->data.csr.col_ind,
                       (size_t)R_desc->data.csr.nnz * sizeof(int));
                memcpy(prob->objective_lowrank_matrix->val,
                       R_desc->data.csr.vals,
                       (size_t)R_desc->data.csr.nnz * sizeof(double));
                break;
            default:
                fprintf(stderr, "[interface] R matrix: unsupported format %d.\n", R_desc->fmt);
                free(prob);
                return NULL;
        }
    }
    else
    {
        prob->objective_lowrank_matrix->row_ptr = (int *)safe_calloc(1, sizeof(int));
        prob->objective_lowrank_matrix->col_ind = NULL;
        prob->objective_lowrank_matrix->val = NULL;
        prob->objective_lowrank_matrix_num_nonzeros = 0;
    }

    prob->objective_lowrank_middle_matrix = NULL;
    prob->objective_lowrank_middle_matrix_num_nonzeros = 0;
    if (D_desc)
    {
        int k = prob->num_rank_lowrank_obj;
        if (k <= 0)
        {
            fprintf(stderr, "[interface] D matrix ignored: problem has no low-rank component.\n");
        }
        else if (D_desc->m != k || D_desc->n != k)
        {
            fprintf(stderr, "[interface] D matrix shape (%d, %d) must be (%d, %d).\n", D_desc->m, D_desc->n, k, k);
            qp_problem_free(prob);
            return NULL;
        }
        else
        {
            prob->objective_lowrank_middle_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
            int *rp = NULL, *ci = NULL;
            double *vv = NULL;
            int nnz = 0;
            int rc = 0;
            switch (D_desc->fmt)
            {
                case matrix_dense:
                    rc = dense_to_csr(D_desc, &rp, &ci, &vv, &nnz);
                    break;
                case matrix_csc:
                    rc = csc_to_csr(D_desc, &rp, &ci, &vv, &nnz);
                    break;
                case matrix_coo:
                    rc = coo_to_csr(D_desc, &rp, &ci, &vv, &nnz);
                    break;
                case matrix_csr:
                    nnz = D_desc->data.csr.nnz;
                    rp = (int *)safe_malloc((size_t)(k + 1) * sizeof(int));
                    ci = (int *)safe_malloc((size_t)nnz * sizeof(int));
                    vv = (double *)safe_malloc((size_t)nnz * sizeof(double));
                    memcpy(rp, D_desc->data.csr.row_ptr, (size_t)(k + 1) * sizeof(int));
                    memcpy(ci, D_desc->data.csr.col_ind, (size_t)nnz * sizeof(int));
                    memcpy(vv, D_desc->data.csr.vals, (size_t)nnz * sizeof(double));
                    break;
                default:
                    rc = -1;
                    fprintf(stderr, "[interface] D matrix: unsupported format %d.\n", D_desc->fmt);
                    break;
            }
            if (rc != 0)
            {
                free(rp);
                free(ci);
                free(vv);
                qp_problem_free(prob);
                return NULL;
            }
            prob->objective_lowrank_middle_matrix->row_ptr = rp;
            prob->objective_lowrank_middle_matrix->col_ind = ci;
            prob->objective_lowrank_middle_matrix->val = vv;
            prob->objective_lowrank_middle_matrix_num_nonzeros = nnz;
        }
    }

    prob->objective_constant = objective_constant ? *objective_constant : 0.0;
    fill_or_copy(&prob->objective_vector, prob->num_variables, objective_c, 0.0);
    fill_or_copy(&prob->variable_lower_bound, prob->num_variables, var_lb, -INFINITY);
    fill_or_copy(&prob->variable_upper_bound, prob->num_variables, var_ub, INFINITY);
    fill_or_copy(&prob->constraint_lower_bound, prob->num_constraints, con_lb, -INFINITY);
    fill_or_copy(&prob->constraint_upper_bound, prob->num_constraints, con_ub, INFINITY);

    prob->num_quadratic_constraints = 0;
    prob->quadratic_constraint_row_indices = NULL;
    prob->quadratic_constraint_matrices = NULL;
    prob->quadratic_constraint_matrix_num_nonzeros = NULL;

    prob->cones.num_cones = 0;
    prob->cones.start_idx = NULL;
    prob->cones.v_dim = NULL;
    prob->cones.type = NULL;
    prob->cones.is_fixed = NULL;
    prob->num_original_variables = 0;

    return prob;
}

qp_problem_t *create_conic_problem(const double *objective_c,
                                   const matrix_desc_t *Q_desc,
                                   const matrix_desc_t *R_desc,
                                   const matrix_desc_t *D_desc,
                                   const matrix_desc_t *A_desc,
                                   const double *con_lb,
                                   const double *con_ub,
                                   const double *var_lb,
                                   const double *var_ub,
                                   const double *objective_constant,
                                   int num_cones,
                                   const cone_spec_t *cones)
{
    qp_problem_t *prob = create_qp_problem(
        objective_c, Q_desc, R_desc, D_desc, A_desc, con_lb, con_ub, var_lb, var_ub, objective_constant);
    if (!prob)
        return NULL;

    if (num_cones <= 0 || cones == NULL)
        return prob;

    int n_vars = prob->num_variables;
    int n_orig = n_vars;
    char *in_cone = (char *)safe_calloc(n_vars, sizeof(char));
    for (int i = 0; i < num_cones; ++i)
    {
        int s = cones[i].start_idx;
        int len = (cones[i].type == CONE_EXPONENTIAL) ? 3 : (cones[i].v_dim + 2);
        if (s < 0 || s + len > n_vars)
        {
            fprintf(stderr,
                    "[create_conic_problem] cone %d out of range: start=%d len=%d num_variables=%d\n",
                    i,
                    s,
                    len,
                    n_vars);
            free(in_cone);
            qp_problem_free(prob);
            return NULL;
        }
        for (int j = s; j < s + len; ++j)
            in_cone[j] = 1;
        if (s < n_orig)
            n_orig = s;
    }

    free(in_cone);

    prob->cones.num_cones = num_cones;
    prob->cones.start_idx = (int *)safe_malloc(num_cones * sizeof(int));
    prob->cones.v_dim = (int *)safe_malloc(num_cones * sizeof(int));
    prob->cones.type = (cone_type_t *)safe_malloc(num_cones * sizeof(cone_type_t));
    prob->cones.is_fixed = NULL;
    int any_fix = 0;
    for (int i = 0; i < num_cones; ++i)
    {
        prob->cones.start_idx[i] = cones[i].start_idx;
        prob->cones.v_dim[i] = (cones[i].type == CONE_EXPONENTIAL) ? 1 : cones[i].v_dim;
        prob->cones.type[i] = cones[i].type;
        if (cones[i].is_fixed)
            any_fix = 1;
    }
    if (any_fix)
    {
        prob->cones.is_fixed = (char *)safe_calloc(n_vars, sizeof(char));
        for (int i = 0; i < num_cones; ++i)
        {
            if (!cones[i].is_fixed)
                continue;
            int s = cones[i].start_idx;
            int len = (cones[i].type == CONE_EXPONENTIAL) ? 3 : (cones[i].v_dim + 2);
            for (int j = 0; j < len; ++j)
                prob->cones.is_fixed[s + j] = cones[i].is_fixed[j] ? 1 : 0;
        }
    }
    prob->num_original_variables = n_orig;

    return prob;
}

void pdhcg_result_free(pdhcg_result_t *results)
{
    if (results == NULL)
    {
        return;
    }

    free(results->primal_solution);
    free(results->dual_solution);
    free(results);
}
void csr_component_free(CsrComponent *csr)
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
    csr_component_free(prob->objective_lowrank_matrix);
    csr_component_free(prob->constraint_matrix);
    free(prob->variable_lower_bound);
    free(prob->variable_upper_bound);
    free(prob->objective_vector);
    free(prob->constraint_lower_bound);
    free(prob->constraint_upper_bound);
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
    free(prob->cones.start_idx);
    free(prob->cones.v_dim);
    free(prob->cones.type);
    free(prob->cones.is_fixed);
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

    if (prob->primal_start)
    {
        free(prob->primal_start);
        prob->primal_start = NULL;
    }
    if (prob->dual_start)
    {
        free(prob->dual_start);
        prob->dual_start = NULL;
    }

    if (primal)
    {
        prob->primal_start = (double *)safe_malloc(n * sizeof(double));
        memcpy(prob->primal_start, primal, n * sizeof(double));
    }
    if (dual)
    {
        prob->dual_start = (double *)safe_malloc(m * sizeof(double));
        memcpy(prob->dual_start, dual, m * sizeof(double));
    }
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
    int len = (prob->cones.type[cone_idx] == CONE_EXPONENTIAL) ? 3 : (prob->cones.v_dim[cone_idx] + 2);
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

    if (!prob->cones.is_fixed)
        prob->cones.is_fixed = (char *)safe_calloc(prob->num_variables, sizeof(char));
    prob->cones.is_fixed[idx] = 1;

    if (!prob->primal_start)
        prob->primal_start = (double *)safe_calloc(prob->num_variables, sizeof(double));
    prob->primal_start[idx] = value;
    return 0;
}

pdhcg_result_t *solve_qp_problem(const qp_problem_t *prob, const pdhg_parameters_t *params)
{
    if (!prob)
    {
        fprintf(stderr, "[interface] solve_qp_problem: invalid arguments.\n");
        return NULL;
    }

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

#ifdef PDHCG_COMPILE_DISTRIBUTED
pdhcg_result_t *solve_qp_problem_distributed(const pdhg_parameters_t *params, const qp_problem_t *original_problem)
{
    return distributed_optimize(params, original_problem);
}
#endif
