/*
 * PDHCG-II PSQP Presolve Wrapper
 */

#ifdef PSQP_AVAILABLE

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "PSQP_API.h"
#include "PSQP_sol.h"
#include "PSQP_status.h"
#include "presolve_wrapper.h"
#include "utils.h"

#ifndef PSQP_VERSION
#define PSQP_VERSION "unknown"
#endif

const char *pdhcg_get_presolve_status_str(int status)
{
    switch (status)
    {
        case UNCHANGED:
            return "UNCHANGED";
        case REDUCED:
            return "REDUCED";
        case INFEASIBLE:
            return "INFEASIBLE";
        case UNBNDORINFEAS:
            return "INFEASIBLE_OR_UNBOUNDED";
        default:
            return "UNKNOWN_STATUS";
    }
}

static qp_problem_t *convert_psqp_to_pdhcg(PresolvedProblem *reduced_prob, double original_obj_constant)
{
    if (!reduced_prob)
        return NULL;

    qp_problem_t *pdhcg_prob = (qp_problem_t *)safe_calloc(1, sizeof(qp_problem_t));

    pdhcg_prob->objective_constant = original_obj_constant + reduced_prob->obj_offset;
    pdhcg_prob->objective_vector = reduced_prob->c;

    pdhcg_prob->constraint_lower_bound = reduced_prob->lhs;
    pdhcg_prob->constraint_upper_bound = reduced_prob->rhs;
    pdhcg_prob->variable_lower_bound = reduced_prob->lbs;
    pdhcg_prob->variable_upper_bound = reduced_prob->ubs;

    pdhcg_prob->num_variables = (int)reduced_prob->n;
    pdhcg_prob->num_constraints = (int)reduced_prob->m;
    pdhcg_prob->constraint_matrix_num_nonzeros = (int)reduced_prob->nnz;

    if (reduced_prob->nnz > 0 && reduced_prob->Ap)
    {
        pdhcg_prob->constraint_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
        pdhcg_prob->constraint_matrix->row_ptr = reduced_prob->Ap;
        pdhcg_prob->constraint_matrix->col_ind = reduced_prob->Ai;
        pdhcg_prob->constraint_matrix->val = reduced_prob->Ax;
    }

    if (reduced_prob->Qnnz > 0 && reduced_prob->Qp)
    {
        pdhcg_prob->objective_sparse_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
        pdhcg_prob->objective_sparse_matrix->row_ptr = reduced_prob->Qp;
        pdhcg_prob->objective_sparse_matrix->col_ind = reduced_prob->Qi;
        pdhcg_prob->objective_sparse_matrix->val = reduced_prob->Qx;
        pdhcg_prob->objective_sparse_matrix_num_nonzeros = (int)reduced_prob->Qnnz;
    }

    if (reduced_prob->Rnnz > 0 && reduced_prob->Rp)
    {
        pdhcg_prob->objective_lowrank_matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
        pdhcg_prob->objective_lowrank_matrix->row_ptr = reduced_prob->Rp;
        pdhcg_prob->objective_lowrank_matrix->col_ind = reduced_prob->Ri;
        pdhcg_prob->objective_lowrank_matrix->val = reduced_prob->Rx;
        pdhcg_prob->objective_lowrank_matrix_num_nonzeros = (int)reduced_prob->Rnnz;
        pdhcg_prob->num_rank_lowrank_obj = (int)reduced_prob->k;
    }

    return pdhcg_prob;
}

static void free_converted_problem(qp_problem_t *prob)
{
    if (!prob)
        return;

    // Only free the wrapper structs. The internal arrays are freed by free_presolver()
    if (prob->constraint_matrix)
        free(prob->constraint_matrix);
    if (prob->objective_sparse_matrix)
        free(prob->objective_sparse_matrix);

    free(prob);
}

pdhcg_presolve_info_t *pdhcg_presolve(const qp_problem_t *original_prob, const pdhg_parameters_t *params)
{
    if (!original_prob)
        return NULL;

    if (original_prob->num_constraints == 0)
    {
        if (params->verbose > 1)
        {
            printf("Note: Problem has no constraints, skipping presolve.\n");
        }
        return NULL;
    }

    clock_t start_time = clock();

    pdhcg_presolve_info_t *info = (pdhcg_presolve_info_t *)safe_calloc(1, sizeof(pdhcg_presolve_info_t));
    if (!info)
        return NULL;

    info->settings = default_settings();
    ((Settings *)info->settings)->verbose = true;
    ((Settings *)info->settings)->dual_fix = false;

    bool has_q = (original_prob->objective_sparse_matrix != NULL);
    bool has_r = (original_prob->objective_lowrank_matrix != NULL);

    Presolver *presolver = NULL;
    size_t m = (size_t)original_prob->num_constraints;
    size_t n = (size_t)original_prob->num_variables;
    size_t nnz = (size_t)original_prob->constraint_matrix_num_nonzeros;

    if (has_r)
    {
        size_t Qnnz = (has_q && original_prob->objective_sparse_matrix)
            ? (size_t)original_prob->objective_sparse_matrix_num_nonzeros
            : 0;
        size_t Rnnz = (size_t)original_prob->objective_lowrank_matrix_num_nonzeros;
        size_t k_rank = (size_t)original_prob->num_rank_lowrank_obj;

        presolver = new_qp_presolver_qr(
            original_prob->constraint_matrix ? original_prob->constraint_matrix->val : NULL,
            original_prob->constraint_matrix ? original_prob->constraint_matrix->col_ind : NULL,
            original_prob->constraint_matrix ? original_prob->constraint_matrix->row_ptr : NULL,
            m,
            n,
            nnz,
            original_prob->constraint_lower_bound,
            original_prob->constraint_upper_bound,
            original_prob->variable_lower_bound,
            original_prob->variable_upper_bound,
            original_prob->objective_vector,
            (has_q && original_prob->objective_sparse_matrix) ? original_prob->objective_sparse_matrix->val : NULL,
            (has_q && original_prob->objective_sparse_matrix) ? original_prob->objective_sparse_matrix->col_ind : NULL,
            (has_q && original_prob->objective_sparse_matrix) ? original_prob->objective_sparse_matrix->row_ptr : NULL,
            Qnnz,
            original_prob->objective_lowrank_matrix->val,
            original_prob->objective_lowrank_matrix->col_ind,
            original_prob->objective_lowrank_matrix->row_ptr,
            Rnnz,
            k_rank,
            info->settings);
    }
    else if (has_q)
    {
        size_t Qnnz = (size_t)original_prob->objective_sparse_matrix_num_nonzeros;
        presolver =
            new_qp_presolver_qr(original_prob->constraint_matrix ? original_prob->constraint_matrix->val : NULL,
                                original_prob->constraint_matrix ? original_prob->constraint_matrix->col_ind : NULL,
                                original_prob->constraint_matrix ? original_prob->constraint_matrix->row_ptr : NULL,
                                m,
                                n,
                                nnz,
                                original_prob->constraint_lower_bound,
                                original_prob->constraint_upper_bound,
                                original_prob->variable_lower_bound,
                                original_prob->variable_upper_bound,
                                original_prob->objective_vector,
                                original_prob->objective_sparse_matrix->val,
                                original_prob->objective_sparse_matrix->col_ind,
                                original_prob->objective_sparse_matrix->row_ptr,
                                Qnnz,
                                NULL, // no low-rank component
                                NULL,
                                NULL,
                                0,
                                0,
                                info->settings);
    }
    else
    {
        presolver = new_presolver(original_prob->constraint_matrix ? original_prob->constraint_matrix->val : NULL,
                                  original_prob->constraint_matrix ? original_prob->constraint_matrix->col_ind : NULL,
                                  original_prob->constraint_matrix ? original_prob->constraint_matrix->row_ptr : NULL,
                                  m,
                                  n,
                                  nnz,
                                  original_prob->constraint_lower_bound,
                                  original_prob->constraint_upper_bound,
                                  original_prob->variable_lower_bound,
                                  original_prob->variable_upper_bound,
                                  original_prob->objective_vector,
                                  info->settings);
    }

    if (!presolver)
    {
        free_settings(info->settings);
        free(info);
        return NULL;
    }
    info->presolver = presolver;

    PresolveStatus status = run_presolver(presolver);
    info->presolve_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    info->presolve_status = (int)status;

    if (params->verbose > 1)
    {
        printf("\nRunning presolver (PSQP %s)...\n", PSQP_VERSION);
        printf("  %-15s : %s\n", "status", pdhcg_get_presolve_status_str(status));
        printf("  %-15s : %.3g sec\n", "presolve time", info->presolve_time);
        if (presolver->reduced_prob)
        {
            printf("  %-15s : %zu rows, %zu columns, %zu nonzeros\n",
                   "reduced problem",
                   presolver->reduced_prob->m,
                   presolver->reduced_prob->n,
                   presolver->reduced_prob->nnz);
        }
    }

    if ((status & INFEASIBLE) || (status & UNBNDORINFEAS) ||
        (presolver->reduced_prob && presolver->reduced_prob->n == 0))
    {
        info->problem_solved_during_presolve = true;
        info->reduced_problem = NULL;
    }
    else
    {
        info->problem_solved_during_presolve = false;
        info->reduced_problem = convert_psqp_to_pdhcg(presolver->reduced_prob, original_prob->objective_constant);
    }

    return info;
}

pdhcg_result_t *pdhcg_create_result_from_presolve(const pdhcg_presolve_info_t *info, const qp_problem_t *original_prob)
{
    if (!info || !info->presolver)
        return NULL;

    Presolver *presolver = (Presolver *)info->presolver;
    pdhcg_result_t *result = (pdhcg_result_t *)safe_calloc(1, sizeof(pdhcg_result_t));

    result->num_variables = original_prob->num_variables;
    result->num_constraints = original_prob->num_constraints;
    result->num_nonzeros = original_prob->constraint_matrix_num_nonzeros;

    if (presolver->reduced_prob)
    {
        result->num_reduced_variables = (int)presolver->reduced_prob->n;
        result->num_reduced_constraints = (int)presolver->reduced_prob->m;
        result->num_reduced_nonzeros = (int)presolver->reduced_prob->nnz;
    }

    result->presolve_status = info->presolve_status;
    result->presolve_time = info->presolve_time;

    if (info->presolve_status == INFEASIBLE)
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
    }
    else if (info->presolve_status == UNBNDORINFEAS)
    {
        result->termination_reason = TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED;
        result->absolute_primal_residual = INFINITY;
        result->relative_primal_residual = INFINITY;
        result->absolute_dual_residual = INFINITY;
        result->relative_dual_residual = INFINITY;
        result->primal_objective_value = INFINITY;
        result->dual_objective_value = -INFINITY;
        result->objective_gap = INFINITY;
        result->relative_objective_gap = INFINITY;
    }
    else if (presolver->reduced_prob && presolver->reduced_prob->n == 0)
    {
        result->termination_reason = TERMINATION_REASON_OPTIMAL;
        // Delegate cleanly to postsolve just like LP wrapper does
        pdhcg_postsolve(info, result, original_prob);
        return result;
    }
    else
    {
        result->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    }

    if (result->num_variables > 0)
    {
        result->primal_solution = (double *)safe_calloc(result->num_variables, sizeof(double));
        result->reduced_cost = (double *)safe_calloc(result->num_variables, sizeof(double));
    }
    if (result->num_constraints > 0)
    {
        result->dual_solution = (double *)safe_calloc(result->num_constraints, sizeof(double));
    }

    return result;
}

void pdhcg_postsolve(const pdhcg_presolve_info_t *info, pdhcg_result_t *result, const qp_problem_t *original_prob)
{
    if (!info || !info->presolver || !result)
        return;

    Presolver *presolver = (Presolver *)info->presolver;

    postsolve(presolver, result->primal_solution, result->dual_solution, result->reduced_cost);

    double *full_primal = (double *)safe_calloc(original_prob->num_variables, sizeof(double));
    double *full_dual = (double *)safe_calloc(original_prob->num_constraints, sizeof(double));
    double *full_rc = (double *)safe_calloc(original_prob->num_variables, sizeof(double));

    if (presolver->sol)
    {
        if (presolver->sol->x)
            memcpy(full_primal, presolver->sol->x, original_prob->num_variables * sizeof(double));
        if (presolver->sol->y)
            memcpy(full_dual, presolver->sol->y, original_prob->num_constraints * sizeof(double));
        if (presolver->sol->z)
            memcpy(full_rc, presolver->sol->z, original_prob->num_variables * sizeof(double));
    }

    if (result->primal_solution)
        free(result->primal_solution);
    if (result->dual_solution)
        free(result->dual_solution);
    if (result->reduced_cost)
        free(result->reduced_cost);

    result->primal_solution = full_primal;
    result->dual_solution = full_dual;
    result->reduced_cost = full_rc;

    for (int i = 0; i < original_prob->num_variables; i++)
    {
        if (!isfinite(original_prob->variable_lower_bound[i]))
        {
            result->reduced_cost[i] = fmin(result->reduced_cost[i], 0.0);
        }
        if (!isfinite(original_prob->variable_upper_bound[i]))
        {
            result->reduced_cost[i] = fmax(result->reduced_cost[i], 0.0);
        }
    }

    if (presolver->reduced_prob && presolver->reduced_prob->n == 0)
    {
        double obj = original_prob->objective_constant + presolver->reduced_prob->obj_offset;
        result->primal_objective_value = obj;
        result->dual_objective_value = obj;
    }

    if (presolver->reduced_prob)
    {
        result->num_reduced_variables = (int)presolver->reduced_prob->n;
        result->num_reduced_constraints = (int)presolver->reduced_prob->m;
        result->num_reduced_nonzeros = (int)presolver->reduced_prob->nnz;
    }
    result->presolve_status = info->presolve_status;
    result->presolve_time = info->presolve_time;
}

void pdhcg_presolve_info_free(pdhcg_presolve_info_t *info)
{
    if (!info)
        return;

    if (info->reduced_problem)
    {
        free_converted_problem(info->reduced_problem);
    }

    if (info->presolver)
    {
        free_presolver((Presolver *)info->presolver);
    }

    if (info->settings)
    {
        free_settings((Settings *)info->settings);
    }

    free(info);
}

const char *pdhcg_presolve_version(void)
{
    return "PSQP " PSQP_VERSION;
}

int pdhcg_presolve_available(void)
{
    return 1;
}

#else /* PSQP_AVAILABLE */

#include "presolve_wrapper.h"
#include <stdio.h>
#include <stdlib.h>

const char *pdhcg_get_presolve_status_str(int status)
{
    (void)status;
    return "PSQP_NOT_AVAILABLE";
}

pdhcg_presolve_info_t *pdhcg_presolve(const qp_problem_t *original_prob, const pdhg_parameters_t *params)
{
    (void)original_prob;
    (void)params;
    fprintf(stderr, "Warning: PSQP not available, presolving disabled.\n");
    return NULL;
}

pdhcg_result_t *pdhcg_create_result_from_presolve(const pdhcg_presolve_info_t *info, const qp_problem_t *original_prob)
{
    (void)info;
    (void)original_prob;
    return NULL;
}

void pdhcg_postsolve(const pdhcg_presolve_info_t *info, pdhcg_result_t *result, const qp_problem_t *original_prob)
{
    (void)info;
    (void)result;
    (void)original_prob;
}

void pdhcg_presolve_info_free(pdhcg_presolve_info_t *info)
{
    (void)info;
}

const char *pdhcg_presolve_version(void)
{
    return "PSQP not available";
}

int pdhcg_presolve_available(void)
{
    return 0;
}

#endif /* PSQP_AVAILABLE */
