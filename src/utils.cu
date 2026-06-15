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
#include "pdhcg_kernels.cuh"
#include "solver_state.h"
#include "spmv_backend.h"
#include "utils.h"
#include <math.h>
#include <random>
#include <signal.h>

#ifndef PDHCG_VERSION
#define PDHCG_VERSION "unknown"
#endif

double get_uniform_random()
{
    thread_local std::mt19937 gen(1);
    thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);

    return dist(gen);
}

double get_normal_random()
{
    thread_local std::mt19937 gen(1);
    thread_local std::normal_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

void *safe_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        perror("Fatal error: malloc failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_calloc(size_t num, size_t size)
{
    void *ptr = calloc(num, size);
    if (ptr == NULL)
    {
        perror("Fatal error: calloc failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_realloc(void *ptr, size_t new_size)
{
    if (new_size == 0)
    {
        free(ptr);
        return NULL;
    }
    void *tmp = realloc(ptr, new_size);
    if (!tmp)
    {
        perror("Fatal error: realloc failed");
        exit(EXIT_FAILURE);
    }
    return tmp;
}

qp_problem_t *create_problem_with_dummy_constraint(const qp_problem_t *prob)
{
    if (prob == NULL)
        return NULL;

    qp_problem_t *new_prob = deepcopy_problem(prob);

    new_prob->num_constraints = 1;
    new_prob->constraint_matrix_num_nonzeros = 1;

    if (new_prob->constraint_matrix == NULL)
    {
        new_prob->constraint_matrix = (CsrComponent *)malloc(sizeof(CsrComponent));
    }

    new_prob->constraint_matrix->row_ptr = (int *)malloc(2 * sizeof(int));
    new_prob->constraint_matrix->row_ptr[0] = 0;
    new_prob->constraint_matrix->row_ptr[1] = 1;

    new_prob->constraint_matrix->col_ind = (int *)malloc(1 * sizeof(int));
    new_prob->constraint_matrix->col_ind[0] = 0;

    new_prob->constraint_matrix->val = (double *)malloc(1 * sizeof(double));
    new_prob->constraint_matrix->val[0] = 1.0;

    if (new_prob->constraint_lower_bound)
        free(new_prob->constraint_lower_bound);
    if (new_prob->constraint_upper_bound)
        free(new_prob->constraint_upper_bound);

    new_prob->constraint_lower_bound = (double *)malloc(1 * sizeof(double));
    new_prob->constraint_lower_bound[0] = -INFINITY;

    new_prob->constraint_upper_bound = (double *)malloc(1 * sizeof(double));
    new_prob->constraint_upper_bound[0] = INFINITY;

    if (new_prob->dual_start != NULL)
    {
        free(new_prob->dual_start);
    }
    new_prob->dual_start = (double *)calloc(1, sizeof(double));

    return new_prob;
}

void compute_interaction_and_movement(pdhg_solver_state_t *state, double *interaction, double *movement)
{
    double dual_norm, primal_norm, cross_term;

    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_norm));
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_norm));
    *movement = 0.5 * (primal_norm * primal_norm * state->primal_weight + dual_norm * dual_norm / state->primal_weight);

    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            state->dual_product,
                            1,
                            state->delta_primal_solution,
                            1,
                            &cross_term));
    *interaction = fabs(cross_term);
}

const char *termination_reason_to_string(termination_reason_t reason)
{
    switch (reason)
    {
        case TERMINATION_REASON_OPTIMAL:
            return "OPTIMAL";
        case TERMINATION_REASON_PRIMAL_INFEASIBLE:
            return "PRIMAL_INFEASIBLE";
        case TERMINATION_REASON_DUAL_INFEASIBLE:
            return "DUAL_INFEASIBLE";
        case TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED:
            return "INFEASIBLE_OR_UNBOUNDED";
        case TERMINATION_REASON_TIME_LIMIT:
            return "TIME_LIMIT";
        case TERMINATION_REASON_ITERATION_LIMIT:
            return "ITERATION_LIMIT";
        case TERMINATION_REASON_UNSPECIFIED:
            return "UNSPECIFIED";
        case TERMINATION_REASON_USER_INTERRUPT:
            return "USER_INTERRUPT";
        case TERMINATION_REASON_FEAS_POLISH_SUCCESS:
            return "FEAS_POLISH_SUCCESS";
        default:
            return "UNKNOWN";
    }
}

const char *problem_type_to_string(problem_type_t type)
{
    switch (type)
    {
        case LP:
            return "LP";
        case CONVEX_QP:
            return "CONVEX_QP";
        case NONCONVEX_QP:
            return "NONCONVEX_QP";
        case CONVEX_QCQP:
            return "CONVEX_QCQP";
        default:
            return "UNKNOWN_PROBLEM_TYPE";
    }
}

const char *quad_obj_type_to_string(quad_obj_type_t type)
{
    switch (type)
    {
        case PDHCG_NON_Q:
            return "No Q (Linear)";
        case PDHCG_DIAG_Q:
            return "Diagonal Q";
        case PDHCG_SPARSE_Q:
            return "Sparse Q";
        case PDHCG_LOW_RANK_Q:
            return "Low Rank Q";
        case PDHCG_LOW_RANK_PLUS_SPARSE_Q:
            return "Low Rank + Sparse Q";
        default:
            return "UNKNOWN_Q_TYPE";
    }
}

bool optimality_criteria_met(const pdhg_solver_state_t *state, double rel_opt_tol, double rel_feas_tol)
{
    return state->relative_dual_residual < rel_feas_tol && state->relative_primal_residual < rel_feas_tol &&
        state->relative_objective_gap < rel_opt_tol;
}

bool primal_infeasibility_criteria_met(const pdhg_solver_state_t *state, double eps)
{
    if (state->dual_ray_objective <= 0.0)
    {
        return false;
    }
    return state->max_dual_ray_infeasibility / state->dual_ray_objective <= eps;
}

bool dual_infeasibility_criteria_met(const pdhg_solver_state_t *state, double eps)
{
    if (state->primal_ray_linear_objective >= 0.0)
    {
        return false;
    }
    return state->max_primal_ray_infeasibility / (-state->primal_ray_linear_objective) <= eps;
}

extern volatile sig_atomic_t g_pdhcg_cancel_request;

void check_termination_criteria(pdhg_solver_state_t *solver_state, const termination_criteria_t *criteria)
{
    if (g_pdhcg_cancel_request)
    {
        solver_state->termination_reason = TERMINATION_REASON_USER_INTERRUPT;
        return;
    }

    if (optimality_criteria_met(solver_state, criteria->eps_optimal_relative, criteria->eps_feasible_relative))
    {
        solver_state->termination_reason = TERMINATION_REASON_OPTIMAL;
        return;
    }
    if (primal_infeasibility_criteria_met(solver_state, criteria->eps_infeasible))
    {
        solver_state->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
        return;
    }
    if (dual_infeasibility_criteria_met(solver_state, criteria->eps_infeasible))
    {
        solver_state->termination_reason = TERMINATION_REASON_DUAL_INFEASIBLE;
        return;
    }
    if (solver_state->total_count >= criteria->iteration_limit)
    {
        solver_state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        return;
    }
    if (solver_state->cumulative_time_sec >= criteria->time_sec_limit)
    {
        solver_state->termination_reason = TERMINATION_REASON_TIME_LIMIT;
        return;
    }
}

bool should_do_adaptive_restart(pdhg_solver_state_t *solver_state,
                                const restart_parameters_t *restart_params,
                                int termination_evaluation_frequency)
{
    bool do_restart = false;
    if (solver_state->total_count == termination_evaluation_frequency)
    {
        do_restart = true;
    }
    else if (solver_state->total_count > termination_evaluation_frequency)
    {
        if (solver_state->fixed_point_error <=
            restart_params->sufficient_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            do_restart = true;
        }
        if (solver_state->fixed_point_error <=
            restart_params->necessary_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            if (solver_state->fixed_point_error > solver_state->last_trial_fixed_point_error)
            {
                do_restart = true;
            }
        }
        if (solver_state->inner_count >= restart_params->artificial_restart_threshold * solver_state->total_count)
        {
            do_restart = true;
        }
    }
    solver_state->last_trial_fixed_point_error = solver_state->fixed_point_error;
    return do_restart;
}

void set_default_parameters(pdhg_parameters_t *params)
{
    params->l_inf_ruiz_iterations = 10;
    params->has_pock_chambolle_alpha = true;
    params->pock_chambolle_alpha = 1.0;
    params->bound_objective_rescaling = true;
    params->verbose = 1;
    params->termination_evaluation_frequency = 200;
    params->feasibility_polishing = false;
    params->reflection_coefficient = 1.0;
    params->presolve = false;

    params->sv_max_iter = 5000;
    params->sv_tol = 1e-4;

    params->termination_criteria.eps_optimal_relative = 1e-4;
    params->termination_criteria.eps_feasible_relative = 1e-4;
    params->termination_criteria.eps_infeasible = 1e-10;
    params->termination_criteria.time_sec_limit = 3600.0;
    params->termination_criteria.iteration_limit = INT32_MAX;
    params->termination_criteria.eps_feas_polish_relative = 1e-6;

    params->restart_params.artificial_restart_threshold = 0.36;
    params->restart_params.sufficient_reduction_for_restart = 0.2;
    params->restart_params.necessary_reduction_for_restart = 0.8;
    params->restart_params.k_p = 0.99;
    params->restart_params.k_i = 0.01;
    params->restart_params.k_d = 0.0;
    params->restart_params.i_smooth = 0.3;

    params->optimality_norm = NORM_TYPE_L_INF;

    params->inner_solver_parameters.iteration_limit = 1000;
    params->inner_solver_parameters.initial_tolerance = 1e-3;
    params->inner_solver_parameters.min_tolerance = 1e-9;

    params->grid_size.decided = false;
    params->partition_method = NNZ_BALANCE_PARTITION;
    params->permute_method = BLOCK_RANDOM_PERMUTATION;

    params->diag_jacobi_precond = true;
    params->soc_formulation = SOC_ROTATED;
}

#define PRINT_DIFF_INT(name, current, default_val)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((current) != (default_val))                                                                                \
        {                                                                                                              \
            printf("  %-18s : %d\n", name, current);                                                                   \
        }                                                                                                              \
    } while (0)

#define PRINT_DIFF_DBL(name, current, default_val)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (fabs((current) - (default_val)) > 1e-9)                                                                    \
        {                                                                                                              \
            printf("  %-18s : %.1e\n", name, (double)(current));                                                       \
        }                                                                                                              \
    } while (0)

#define PRINT_DIFF_BOOL(name, current, default_val)                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((current) != (default_val))                                                                                \
        {                                                                                                              \
            printf("  %-18s : %s\n", name, (current) ? "on" : "off");                                                  \
        }                                                                                                              \
    } while (0)

void print_initial_info(const pdhg_parameters_t *params, const qp_problem_t *problem)
{
    pdhg_parameters_t default_params;
    set_default_parameters(&default_params);
    if (params->verbose < 2)
    {
        return;
    }

    const int total_width = 96;
    const char *line = "---------------------------------------------------------"
                       "------------------------------------------";

    printf("%s\n", line);

    auto print_centered = [total_width](const char *text)
    {
        int len = strlen(text);
        int padding = (total_width + len) / 2;
        printf("%*s\n", padding, text);
    };

    print_centered("PDHCG-II");
    print_centered("A GPU-Accelerated First-Order Solver for Convex QPs");
    print_centered("(c) Hongpei Li, 2026");
    print_centered("Contact: ishongpeili@gmail.com");
    printf("\n");

    printf("%s\n", line);

    printf("problem: %d rows, %d columns, %d nonzeros\n",
           problem->num_constraints,
           problem->num_variables,
           problem->constraint_matrix_num_nonzeros);
    if (problem->num_quadratic_constraints > 0)
    {
        long total_q_nnz = 0;
        for (int i = 0; i < problem->num_quadratic_constraints; ++i)
        {
            total_q_nnz += problem->quadratic_constraint_matrix_num_nonzeros[i];
        }
        printf("         + %d quadratic constraint(s), %ld Q-nnz total\n",
               problem->num_quadratic_constraints,
               total_q_nnz);
    }

    printf("settings:\n");
    printf("  iter_limit         : %d\n", params->termination_criteria.iteration_limit);
    printf("  time_limit         : %.2f sec\n", params->termination_criteria.time_sec_limit);
    printf("  eps_opt            : %.1e\n", params->termination_criteria.eps_optimal_relative);
    printf("  eps_feas           : %.1e\n", params->termination_criteria.eps_feasible_relative);
    printf("  eps_infeas_detect  : %.1e\n", params->termination_criteria.eps_infeasible);
    printf("  spmv backend       : %s\n", pdhcg_use_spmvop_by_default() ? "SpMVOp" : "SpMV");
    if (params->optimality_norm != default_params.optimality_norm)
    {
        printf("  optimality_norm    : %s\n", params->optimality_norm == NORM_TYPE_L_INF ? "L_inf" : "L2");
    }

    PRINT_DIFF_INT("l_inf_ruiz_iter", params->l_inf_ruiz_iterations, default_params.l_inf_ruiz_iterations);
    PRINT_DIFF_DBL("pock_chambolle_alpha", params->pock_chambolle_alpha, default_params.pock_chambolle_alpha);
    PRINT_DIFF_BOOL(
        "has_pock_chambolle_alpha", params->has_pock_chambolle_alpha, default_params.has_pock_chambolle_alpha);
    PRINT_DIFF_BOOL("bound_obj_rescaling", params->bound_objective_rescaling, default_params.bound_objective_rescaling);
    PRINT_DIFF_INT("sv_max_iter", params->sv_max_iter, default_params.sv_max_iter);
    PRINT_DIFF_DBL("sv_tol", params->sv_tol, default_params.sv_tol);
    PRINT_DIFF_INT(
        "evaluation_freq", params->termination_evaluation_frequency, default_params.termination_evaluation_frequency);
    PRINT_DIFF_BOOL("feasibility_polishing", params->feasibility_polishing, default_params.feasibility_polishing);
    PRINT_DIFF_DBL("eps_feas_polish_relative",
                   params->termination_criteria.eps_feas_polish_relative,
                   default_params.termination_criteria.eps_feas_polish_relative);
}

#undef PRINT_DIFF_INT
#undef PRINT_DIFF_DBL
#undef PRINT_DIFF_BOOL

void pdhg_final_log(const pdhcg_result_t *result, const pdhg_parameters_t *params)
{
    if (params->verbose >= 2)
    {
        printf("-------------------------------------------------------------------"
               "----------"
               "----------------------\n");
    }
    if (params->verbose < 1)
        return;
    printf("Solution Summary\n");
    printf("  Status             : %s\n", termination_reason_to_string(result->termination_reason));
    printf("  Solve time         : %.3g sec\n", result->cumulative_time_sec);
    printf("  Iterations         : %d\n", result->total_count);
    if (result->total_inner_count > 0)
    {
        printf("  Inner Iterations   : %d\n", result->total_inner_count);
    }
    printf("  Primal objective   : %.10g\n", result->primal_objective_value);
    printf("  Dual objective     : %.10g\n", result->dual_objective_value);
    printf("  Objective gap      : %.3e\n", result->relative_objective_gap);
    printf("  Primal infeas      : %.3e\n", result->relative_primal_residual);
    printf("  Dual infeas        : %.3e\n", result->relative_dual_residual);
}

void display_iteration_stats(const pdhg_solver_state_t *state, int verbose)
{
    if (verbose < 2)
    {
        return;
    }
    if (state->total_count % get_print_frequency(state->total_count) == 0)
    {
        printf("%6d %7d %.1e | %8.1e  %8.1e | %.1e %.1e %.1e | %.1e %.1e %.1e \n",
               state->total_count,
               state->inner_solver->total_count,
               state->cumulative_time_sec,
               state->primal_objective_value,
               state->dual_objective_value,
               state->absolute_primal_residual,
               state->absolute_dual_residual,
               state->objective_gap,
               state->relative_primal_residual,
               state->relative_dual_residual,
               state->relative_objective_gap);
    }
}

int get_print_frequency(int iter)
{
    int step = 10;
    long long threshold = 1000;

    while (iter >= threshold)
    {
        step *= 10;
        threshold *= 10;
    }
    return step;
}

double get_vector_inf_norm(cublasHandle_t handle, int n, const double *x_d)
{
    if (n <= 0)
        return 0.0;

    int index = 1;

    CUBLAS_CHECK(cublasIdamax(handle, n, x_d, 1, &index));

    if (index < 1 || index > n)
        return 0.0;

    double max_val = 0.0;
    CUDA_CHECK(cudaMemcpy(&max_val, x_d + (index - 1), sizeof(double), cudaMemcpyDeviceToHost));

    if (!isfinite(max_val))
        return INFINITY;

    return fabs(max_val);
}

double get_vector_sum(cublasHandle_t handle, int n, double *ones_d, const double *x_d)
{
    if (n <= 0)
        return 0.0;

    double sum;
    CUBLAS_CHECK(cublasDdot(handle, n, x_d, 1, ones_d, 1, &sum));
    return sum;
}

// helper function to allocate and fill or copy an array
void fill_or_copy(double **dst, int n, const double *src, double fill_val)
{
    *dst = (double *)safe_malloc((size_t)n * sizeof(double));
    if (src)
        memcpy(*dst, src, (size_t)n * sizeof(double));
    else
        for (int i = 0; i < n; ++i)
            (*dst)[i] = fill_val;
}

// convert dense → CSR
int dense_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind, double **vals, int *nnz_out)
{
    int m = desc->m, n = desc->n;
    double tol = (desc->zero_tolerance > 0) ? desc->zero_tolerance : 1e-12;

    // count nnz
    int nnz = 0;
    for (int i = 0; i < m * n; ++i)
    {
        if (fabs(desc->data.dense.A[i]) > tol)
            ++nnz;
    }

    // allocate
    *row_ptr = (int *)safe_malloc((size_t)(m + 1) * sizeof(int));
    *col_ind = (int *)safe_malloc((size_t)nnz * sizeof(int));
    *vals = (double *)safe_malloc((size_t)nnz * sizeof(double));

    // fill
    int nz = 0;
    for (int i = 0; i < m; ++i)
    {
        (*row_ptr)[i] = nz;
        for (int j = 0; j < n; ++j)
        {
            double v = desc->data.dense.A[i * n + j];
            if (fabs(v) > tol)
            {
                (*col_ind)[nz] = j;
                (*vals)[nz] = v;
                ++nz;
            }
        }
    }
    (*row_ptr)[m] = nz;
    *nnz_out = nz;
    return 0;
}

// convert CSC → CSR
int csc_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind, double **vals, int *nnz_out)
{
    const int m = desc->m, n = desc->n;
    const int *col_ptr = desc->data.csc.col_ptr;
    const int *row_ind = desc->data.csc.row_ind;
    const double *v = desc->data.csc.vals;

    const double tol = (desc->zero_tolerance > 0) ? desc->zero_tolerance : 0.0;

    // count entries per row
    *row_ptr = (int *)safe_malloc((size_t)(m + 1) * sizeof(int));
    for (int i = 0; i <= m; ++i)
        (*row_ptr)[i] = 0;

    // count nnz
    int eff_nnz = 0;
    for (int j = 0; j < n; ++j)
    {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
        {
            int ri = row_ind[k];
            if (ri < 0 || ri >= m)
            {
                fprintf(stderr, "[interface] CSC: row index out of range\n");
                return -1;
            }
            double val = v[k];
            if (tol > 0 && fabs(val) <= tol)
                continue;
            ++((*row_ptr)[ri + 1]);
            ++eff_nnz;
        }
    }

    // exclusive scan
    for (int i = 0; i < m; ++i)
        (*row_ptr)[i + 1] += (*row_ptr)[i];

    // allocate
    *col_ind = (int *)safe_malloc((size_t)eff_nnz * sizeof(int));
    *vals = (double *)safe_malloc((size_t)eff_nnz * sizeof(double));

    // next position to fill in each row
    int *next = (int *)safe_malloc((size_t)m * sizeof(int));
    for (int i = 0; i < m; ++i)
        next[i] = (*row_ptr)[i];

    // fill column indices and values
    for (int j = 0; j < n; ++j)
    {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
        {
            int ri = row_ind[k];
            double val = v[k];
            if (tol > 0 && fabs(val) <= tol)
                continue;
            int pos = next[ri]++;
            (*col_ind)[pos] = j;
            (*vals)[pos] = val;
        }
    }

    free(next);
    *nnz_out = eff_nnz;
    return 0;
}

// convert COO → CSR
int coo_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind, double **vals, int *nnz_out)
{
    const int m = desc->m, n = desc->n;
    const int nnz_in = desc->data.coo.nnz;
    const int *r = desc->data.coo.row_ind;
    const int *c = desc->data.coo.col_ind;
    const double *v = desc->data.coo.vals;
    const double tol = (desc->zero_tolerance > 0) ? desc->zero_tolerance : 0.0;

    // count nnz
    int nnz = 0;
    if (tol > 0)
    {
        for (int k = 0; k < nnz_in; ++k)
            if (fabs(v[k]) > tol)
                ++nnz;
    }
    else
    {
        nnz = nnz_in;
    }

    *row_ptr = (int *)safe_malloc((size_t)(m + 1) * sizeof(int));
    *col_ind = (int *)safe_malloc((size_t)nnz * sizeof(int));
    *vals = (double *)safe_malloc((size_t)nnz * sizeof(double));

    // count entries per row
    for (int i = 0; i <= m; ++i)
        (*row_ptr)[i] = 0;
    if (tol > 0)
    {
        for (int k = 0; k < nnz_in; ++k)
            if (fabs(v[k]) > tol)
            {
                int ri = r[k];
                if (ri < 0 || ri >= m)
                {
                    fprintf(stderr, "[interface] COO: row index out of range\n");
                    return -1;
                }
                ++((*row_ptr)[ri + 1]);
            }
    }
    else
    {
        for (int k = 0; k < nnz_in; ++k)
        {
            int ri = r[k];
            if (ri < 0 || ri >= m)
            {
                fprintf(stderr, "[interface] COO: row index out of range\n");
                return -1;
            }
            ++((*row_ptr)[ri + 1]);
        }
    }

    // exclusive scan
    for (int i = 0; i < m; ++i)
        (*row_ptr)[i + 1] += (*row_ptr)[i];

    // next position to fill in each row
    int *next = (int *)safe_malloc((size_t)m * sizeof(int));
    for (int i = 0; i < m; ++i)
        next[i] = (*row_ptr)[i];

    // fill column indices and values
    if (tol > 0)
    {
        for (int k = 0; k < nnz_in; ++k)
        {
            if (fabs(v[k]) <= tol)
                continue;
            int ri = r[k], cj = c[k];
            if (cj < 0 || cj >= n)
            {
                fprintf(stderr, "[interface] COO: col index out of range\n");
                free(next);
                return -1;
            }
            int pos = next[ri]++;
            (*col_ind)[pos] = cj;
            (*vals)[pos] = v[k];
        }
    }
    else
    {
        for (int k = 0; k < nnz_in; ++k)
        {
            int ri = r[k], cj = c[k];
            if (cj < 0 || cj >= n)
            {
                fprintf(stderr, "[interface] COO: col index out of range\n");
                free(next);
                return -1;
            }
            int pos = next[ri]++;
            (*col_ind)[pos] = cj;
            (*vals)[pos] = v[k];
        }
    }

    free(next);
    *nnz_out = nnz;
    return 0;
}

CsrComponent *deepcopy_csr_component(const CsrComponent *src, size_t num_rows, size_t nnz)
{
    if (!src)
        return NULL;
    if (!src->row_ptr && nnz == 0)
    {
        return NULL;
    }
    CsrComponent *copy = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));

    size_t row_ptr_size = (num_rows + 1) * sizeof(int);
    size_t col_ind_size = nnz * sizeof(int);
    size_t val_size = nnz * sizeof(double);

    copy->row_ptr = (int *)safe_calloc(num_rows + 1, sizeof(int));
    copy->col_ind = (int *)safe_malloc(col_ind_size);
    copy->val = (double *)safe_malloc(val_size);

    memcpy(copy->row_ptr, src->row_ptr, row_ptr_size);
    if (nnz > 0)
    {
        memcpy(copy->col_ind, src->col_ind, col_ind_size);
        memcpy(copy->val, src->val, val_size);
    }

    return copy;
}

quad_obj_type_t detect_q_type(const CsrComponent *sparse_component,
                              const CsrComponent *low_rank_component,
                              int num_rows_sparse,
                              int num_rows_low_rank)
{
    bool has_sparse = sparse_component && sparse_component->row_ptr && sparse_component->col_ind;
    bool has_low_rank = low_rank_component && low_rank_component->row_ptr && low_rank_component->col_ind;

    if (!has_sparse && !has_low_rank)
    {
        return PDHCG_NON_Q;
    }

    int nnz_sparse = has_sparse ? sparse_component->row_ptr[num_rows_sparse] - sparse_component->row_ptr[0] : 0;
    int nnz_low_rank =
        has_low_rank ? low_rank_component->row_ptr[num_rows_low_rank] - low_rank_component->row_ptr[0] : 0;

    if (nnz_sparse == 0 && nnz_low_rank == 0)
    {
        return PDHCG_NON_Q;
    }

    if (nnz_low_rank > 0)
    {
        if (nnz_sparse > 0)
        {
            return PDHCG_LOW_RANK_PLUS_SPARSE_Q;
        }
        else
        {
            return PDHCG_LOW_RANK_Q;
        }
    }
    else
    {
        for (int i = 0; i < num_rows_sparse; ++i)
        {
            int row_start = sparse_component->row_ptr[i];
            int row_end = sparse_component->row_ptr[i + 1];

            for (int k = row_start; k < row_end; ++k)
            {
                int j = sparse_component->col_ind[k];
                if (i != j)
                {
                    return PDHCG_SPARSE_Q;
                }
            }
        }
        return PDHCG_DIAG_Q;
    }
}

void ensure_objective_matrix_initialized(qp_problem_t *prob)
{
    if (!prob)
        return;
    if (prob->objective_sparse_matrix == NULL)
    {
        prob->objective_sparse_matrix = (CsrComponent *)safe_malloc(sizeof(CsrComponent));

        prob->objective_sparse_matrix->row_ptr = NULL;
        prob->objective_sparse_matrix->col_ind = NULL;
        prob->objective_sparse_matrix->val = NULL;
    }

    if (prob->objective_sparse_matrix->row_ptr == NULL)
    {
        prob->objective_sparse_matrix->row_ptr = (int *)safe_calloc(prob->num_variables + 1, sizeof(int));
    }
    if (prob->objective_lowrank_matrix == NULL)
    {
        prob->objective_lowrank_matrix = (CsrComponent *)safe_malloc(sizeof(CsrComponent));

        prob->objective_lowrank_matrix->row_ptr = NULL;
        prob->objective_lowrank_matrix->col_ind = NULL;
        prob->objective_lowrank_matrix->val = NULL;
    }

    if (prob->objective_lowrank_matrix->row_ptr == NULL)
    {
        prob->objective_lowrank_matrix->row_ptr = (int *)safe_calloc(prob->num_variables + 1, sizeof(int));
    }
}
