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

#include "pdhcg.h"
#include "pdhcg_types.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void *checked_calloc(size_t count, size_t size)
{
    void *ptr = calloc(count ? count : 1, size ? size : 1);
    if (!ptr)
    {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    return ptr;
}

static int read_exact(FILE *stream, void *ptr, size_t size, size_t count)
{
    return count == 0 || fread(ptr, size, count, stream) == count;
}

static const char *termination_name(termination_reason_t reason)
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
        case TERMINATION_REASON_USER_INTERRUPT:
            return "USER_INTERRUPT";
        case TERMINATION_REASON_FEAS_POLISH_SUCCESS:
            return "FEAS_POLISH_SUCCESS";
        default:
            return "UNSPECIFIED";
    }
}

static double monotonic_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + 1.0e-9 * (double)value.tv_nsec;
}

static void print_host_sanity(const qp_problem_t *problem)
{
    double max_row_violation = 0.0;
    double max_bound_violation = 0.0;
    double linear_objective = problem->objective_constant;
    const double *x = problem->primal_start;

    for (int row = 0; row < problem->num_constraints; ++row)
    {
        double activity = 0.0;
        if (x)
        {
            for (int p = problem->constraint_matrix->row_ptr[row]; p < problem->constraint_matrix->row_ptr[row + 1];
                 ++p)
            {
                activity += problem->constraint_matrix->val[p] * x[problem->constraint_matrix->col_ind[p]];
            }
        }
        double projected =
            fmax(problem->constraint_lower_bound[row], fmin(activity, problem->constraint_upper_bound[row]));
        max_row_violation = fmax(max_row_violation, fabs(activity - projected));
    }
    for (int col = 0; col < problem->num_variables; ++col)
    {
        double value = x ? x[col] : 0.0;
        double projected = fmax(problem->variable_lower_bound[col], fmin(value, problem->variable_upper_bound[col]));
        max_bound_violation = fmax(max_bound_violation, fabs(value - projected));
        linear_objective += problem->objective_vector[col] * value;
    }
    fprintf(stderr,
            "host sanity: row_violation=%.17g bound_violation=%.17g "
            "linear_objective=%.17g first_row=[%.17g,%.17g] last_row=[%.17g,%.17g]\n",
            max_row_violation,
            max_bound_violation,
            linear_objective,
            problem->constraint_lower_bound[0],
            problem->constraint_upper_bound[0],
            problem->constraint_lower_bound[problem->num_constraints - 1],
            problem->constraint_upper_bound[problem->num_constraints - 1]);
}

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 5)
    {
        fprintf(stderr, "usage: %s MODEL.bin EPS TIME_LIMIT [VERBOSE]\n", argv[0]);
        return 2;
    }

    const char *path = argv[1];
    double eps = strtod(argv[2], NULL);
    double time_limit = strtod(argv[3], NULL);
    int verbose = argc == 5 ? atoi(argv[4]) : 1;

    FILE *stream = fopen(path, "rb");
    if (!stream)
    {
        perror(path);
        return 1;
    }

    char magic[8];
    int32_t header[18];
    double objective_constant;
    if (!read_exact(stream, magic, 1, sizeof(magic)) || memcmp(magic, "PDHQCQ1", 7) != 0 ||
        !read_exact(stream, header, sizeof(int32_t), 18) || !read_exact(stream, &objective_constant, sizeof(double), 1))
    {
        fprintf(stderr, "invalid or truncated model header: %s\n", path);
        fclose(stream);
        return 1;
    }

    int n = header[1];
    int m = header[2];
    int a_nnz = header[3];
    int q_nnz = header[4];
    int num_cones = header[5];
    int has_fixed = header[13];
    int has_primal = header[14];
    if (n <= 0 || m < 0 || a_nnz < 0 || q_nnz < 0 || num_cones < 0)
    {
        fprintf(stderr, "unsupported model dimensions\n");
        fclose(stream);
        return 1;
    }

    double *c = checked_calloc((size_t)n, sizeof(double));
    double *lbx = checked_calloc((size_t)n, sizeof(double));
    double *ubx = checked_calloc((size_t)n, sizeof(double));
    double *lbc = checked_calloc((size_t)m, sizeof(double));
    double *ubc = checked_calloc((size_t)m, sizeof(double));
    int *a_row = checked_calloc((size_t)m + 1, sizeof(int));
    int *a_col = checked_calloc((size_t)a_nnz, sizeof(int));
    double *a_val = checked_calloc((size_t)a_nnz, sizeof(double));
    int *q_row = checked_calloc((size_t)n + 1, sizeof(int));
    int *q_col = checked_calloc((size_t)q_nnz, sizeof(int));
    double *q_val = checked_calloc((size_t)q_nnz, sizeof(double));
    int *cone_start = checked_calloc((size_t)num_cones, sizeof(int));
    int *cone_vdim = checked_calloc((size_t)num_cones, sizeof(int));
    cone_type_t *cone_type = checked_calloc((size_t)num_cones, sizeof(cone_type_t));
    char *fixed = has_fixed ? checked_calloc((size_t)n, sizeof(char)) : NULL;
    double *primal = has_primal ? checked_calloc((size_t)n, sizeof(double)) : NULL;

    int ok = read_exact(stream, c, sizeof(double), (size_t)n) && read_exact(stream, lbx, sizeof(double), (size_t)n) &&
        read_exact(stream, ubx, sizeof(double), (size_t)n) && read_exact(stream, lbc, sizeof(double), (size_t)m) &&
        read_exact(stream, ubc, sizeof(double), (size_t)m) &&
        read_exact(stream, a_row, sizeof(int32_t), (size_t)m + 1) &&
        read_exact(stream, a_col, sizeof(int32_t), (size_t)a_nnz) &&
        read_exact(stream, a_val, sizeof(double), (size_t)a_nnz) &&
        read_exact(stream, q_row, sizeof(int32_t), (size_t)n + 1) &&
        read_exact(stream, q_col, sizeof(int32_t), (size_t)q_nnz) &&
        read_exact(stream, q_val, sizeof(double), (size_t)q_nnz) &&
        read_exact(stream, cone_start, sizeof(int32_t), (size_t)num_cones) &&
        read_exact(stream, cone_vdim, sizeof(int32_t), (size_t)num_cones) &&
        read_exact(stream, cone_type, sizeof(int32_t), (size_t)num_cones) &&
        (!has_fixed || read_exact(stream, fixed, sizeof(char), (size_t)n)) &&
        (!has_primal || read_exact(stream, primal, sizeof(double), (size_t)n));
    fclose(stream);
    if (!ok)
    {
        fprintf(stderr, "truncated model body: %s\n", path);
        return 1;
    }
    if (getenv("PDHCG_PROJECT_BOX_START"))
    {
        if (!primal)
            primal = checked_calloc((size_t)n, sizeof(double));
        for (int i = 0; i < n; ++i)
            primal[i] = fmax(lbx[i], fmin(primal[i], ubx[i]));
    }

    matrix_desc_t A;
    memset(&A, 0, sizeof(A));
    A.m = m;
    A.n = n;
    A.fmt = matrix_csr;
    A.data.csr.nnz = a_nnz;
    A.data.csr.row_ptr = a_row;
    A.data.csr.col_ind = a_col;
    A.data.csr.vals = a_val;

    matrix_desc_t Q;
    memset(&Q, 0, sizeof(Q));
    Q.m = n;
    Q.n = n;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = q_nnz;
    Q.data.csr.row_ptr = q_row;
    Q.data.csr.col_ind = q_col;
    Q.data.csr.vals = q_val;

    cone_spec_t *specs = checked_calloc((size_t)num_cones, sizeof(cone_spec_t));
    for (int i = 0; i < num_cones; ++i)
    {
        specs[i].type = cone_type[i];
        specs[i].start_idx = cone_start[i];
        specs[i].v_dim = cone_vdim[i];
        specs[i].is_fixed = (fixed && !getenv("PDHCG_IGNORE_FIXED_MASK")) ? fixed + cone_start[i] : NULL;
    }

    int effective_num_cones = getenv("PDHCG_IGNORE_CONES") ? 0 : num_cones;
    qp_problem_t *problem = create_qp_problem(c,
                                              q_nnz > 0 ? &Q : NULL,
                                              NULL,
                                              NULL,
                                              &A,
                                              lbc,
                                              ubc,
                                              lbx,
                                              ubx,
                                              &objective_constant,
                                              effective_num_cones,
                                              specs,
                                              0,
                                              NULL,
                                              NULL);
    if (!problem)
    {
        fprintf(stderr, "create_qp_problem failed\n");
        return 1;
    }
    if (primal && !getenv("PDHCG_IGNORE_START"))
        set_start_values(problem, primal, NULL);
    if (verbose >= 2)
        print_host_sanity(problem);

    free(c);
    free(lbx);
    free(ubx);
    free(lbc);
    free(ubc);
    free(a_row);
    free(a_col);
    free(a_val);
    free(q_row);
    free(q_col);
    free(q_val);
    free(cone_start);
    free(cone_vdim);
    free(cone_type);
    free(fixed);
    free(primal);
    free(specs);

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = verbose;
    params.termination_criteria.eps_feasible_relative = eps;
    params.termination_criteria.eps_optimal_relative = eps;
    params.termination_criteria.time_sec_limit = time_limit;
    if (getenv("PDHCG_NO_BOUND_OBJ_RESCALING"))
        params.bound_objective_rescaling = false;
    if (getenv("PDHCG_NO_SCALING"))
    {
        params.l_inf_ruiz_iterations = 0;
        params.has_pock_chambolle_alpha = false;
        params.curtis_reid_iterations = 0;
        params.bound_objective_rescaling = false;
    }

    double wall_start = monotonic_seconds();
    pdhcg_result_t *result = solve_qp_problem(problem, &params);
    double wall_time = monotonic_seconds() - wall_start;
    if (!result)
    {
        fprintf(stderr, "solve_qp_problem failed\n");
        qp_problem_free(problem);
        return 1;
    }

    printf("{\"status\":\"%s\",\"runtime_sec\":%.17g,\"wall_time_sec\":%.17g,"
           "\"iterations\":%d,\"inner_iterations\":%d,"
           "\"primal_objective\":%.17g,\"dual_objective\":%.17g,"
           "\"absolute_primal_residual\":%.17g,\"absolute_dual_residual\":%.17g,"
           "\"absolute_objective_gap\":%.17g,"
           "\"relative_primal_residual\":%.17g,\"relative_dual_residual\":%.17g,"
           "\"relative_objective_gap\":%.17g,\"n\":%d,\"m\":%d,\"A_nnz\":%d,"
           "\"Q_nnz\":%d,"
           "\"num_cones\":%d}\n",
           termination_name(result->termination_reason),
           result->cumulative_time_sec,
           wall_time,
           result->total_count,
           result->total_inner_count,
           result->primal_objective_value,
           result->dual_objective_value,
           result->absolute_primal_residual,
           result->absolute_dual_residual,
           result->objective_gap,
           result->relative_primal_residual,
           result->relative_dual_residual,
           result->relative_objective_gap,
           problem->num_variables,
           problem->num_constraints,
           problem->constraint_matrix_num_nonzeros,
           problem->objective_sparse_matrix_num_nonzeros,
           problem->cones.num_cones);
    fflush(stdout);

    pdhcg_result_free(result);
    qp_problem_free(problem);
    return 0;
}
