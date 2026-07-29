#include "pdhcg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef PDHCG_COMPILE_DISTRIBUTED
#include <mpi.h>

static qp_problem_t *make_problem(void)
{
    static const int row_ptr[] = {0, 1, 2, 3};
    static const int col_ind[] = {1, 2, 3};
    static const double values[] = {1.0, 1.0, 1.0};
    static const double objective[] = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    static const double var_lb[] = {0.0, -INFINITY, -INFINITY, -INFINITY, -INFINITY, 0.0};
    static const double var_ub[] = {0.0, INFINITY, INFINITY, INFINITY, INFINITY, 0.0};
    static const double rhs[] = {3.0, 4.0, 0.0};
    const cone_spec_t cone = {
        .type = CONE_STANDARD_SOC,
        .start_idx = 1,
        .v_dim = 2,
        .alpha = 0.0,
        .is_fixed = NULL,
    };
    matrix_desc_t A = {0};
    A.m = 3;
    A.n = 6;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 3;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;
    return create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, var_lb, var_ub, NULL, 1, &cone);
}

static qp_problem_t *make_fixed_rsoc_problem(void)
{
    static const int row_ptr[] = {0, 0};
    static const int col_ind[] = {0};
    static const double values[] = {0.0};
    static const double objective[] = {-1.0, 0.0, 0.0};
    static const double zero[] = {0.0};
    const cone_spec_t cone = {
        .type = CONE_ROTATED_SOC,
        .start_idx = 0,
        .v_dim = 1,
        .alpha = 0.0,
        .is_fixed = NULL,
    };
    matrix_desc_t A = {0};
    A.m = 1;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 0;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;
    qp_problem_t *problem = create_qp_problem(objective, NULL, NULL, NULL, &A, zero, zero, NULL, NULL, NULL, 1, &cone);
    if (!problem || set_cone_fixed(problem, 0, 1, 1.0) != 0 || set_cone_fixed(problem, 0, 2, 1.0) != 0)
    {
        qp_problem_free(problem);
        return NULL;
    }
    return problem;
}

static qp_problem_t *make_atomic_soc_problem(void)
{
    static const int row_ptr[] = {0, 1, 2, 3};
    static const int col_ind[] = {3, 4, 5};
    static const double values[] = {1.0, 1.0, 1.0};
    static const double objective[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    static const double var_lb[] = {0.0, 0.0, 0.0, -INFINITY, -INFINITY, -INFINITY, -INFINITY, 0.0, 0.0, 0.0};
    static const double var_ub[] = {0.0, 0.0, 0.0, INFINITY, INFINITY, INFINITY, INFINITY, 0.0, 0.0, 0.0};
    static const double rhs[] = {3.0, 4.0, 0.0};
    const cone_spec_t cone = {
        .type = CONE_STANDARD_SOC,
        .start_idx = 3,
        .v_dim = 2,
        .alpha = 0.0,
        .is_fixed = NULL,
    };
    matrix_desc_t A = {0};
    A.m = 3;
    A.n = 10;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 3;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;
    return create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, var_lb, var_ub, NULL, 1, &cone);
}

static qp_problem_t *make_exponential_problem(void)
{
    static const int row_ptr[] = {0, 1, 2, 3, 4};
    static const int col_ind[] = {0, 3, 6, 9};
    static const double values[] = {1.0, 1.0, 1.0, 1.0};
    static const double objective[] = {
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
    };
    static const double rhs[] = {0.0, 0.0, 0.0, 0.0};
    cone_spec_t cones[4];
    for (int cone = 0; cone < 4; ++cone)
    {
        cones[cone].type = CONE_EXPONENTIAL;
        cones[cone].start_idx = 3 * cone;
        cones[cone].v_dim = 1;
        cones[cone].alpha = 0.0;
        cones[cone].is_fixed = NULL;
    }

    matrix_desc_t A = {0};
    A.m = 4;
    A.n = 12;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 4;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;
    qp_problem_t *problem = create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, NULL, NULL, NULL, 4, cones);
    if (!problem)
        return NULL;
    for (int cone = 0; cone < 4; ++cone)
    {
        if (set_cone_fixed(problem, cone, 1, 1.0) != 0)
        {
            qp_problem_free(problem);
            return NULL;
        }
    }
    return problem;
}

static qp_problem_t *make_large_soc_problem(int v_dim)
{
    int n = v_dim + 2;
    int *row_ptr = (int *)malloc((size_t)(v_dim + 1) * sizeof(int));
    int *col_ind = (int *)malloc((size_t)v_dim * sizeof(int));
    double *values = (double *)malloc((size_t)v_dim * sizeof(double));
    double *objective = (double *)calloc((size_t)n, sizeof(double));
    double *rhs = (double *)malloc((size_t)v_dim * sizeof(double));
    if (!row_ptr || !col_ind || !values || !objective || !rhs)
    {
        free(row_ptr);
        free(col_ind);
        free(values);
        free(objective);
        free(rhs);
        return NULL;
    }

    double value = 1.0 / sqrt((double)v_dim);
    for (int i = 0; i < v_dim; ++i)
    {
        row_ptr[i] = i;
        col_ind[i] = i;
        values[i] = 1.0;
        rhs[i] = value;
    }
    row_ptr[v_dim] = v_dim;
    objective[n - 1] = 1.0;

    cone_spec_t cone = {
        .type = CONE_STANDARD_SOC,
        .start_idx = 0,
        .v_dim = v_dim,
        .alpha = 0.0,
        .is_fixed = NULL,
    };
    matrix_desc_t A = {0};
    A.m = v_dim;
    A.n = n;
    A.fmt = matrix_csr;
    A.data.csr.nnz = v_dim;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;
    qp_problem_t *problem = create_qp_problem(objective, NULL, NULL, NULL, &A, rhs, rhs, NULL, NULL, NULL, 1, &cone);

    free(row_ptr);
    free(col_ind);
    free(values);
    free(objective);
    free(rhs);
    return problem;
}

static qp_problem_t *make_qcqp_problem(void)
{
    static const int row_ptr[] = {0, 0};
    static const int col_ind[] = {0};
    static const double values[] = {0.0};
    static const double objective[] = {-1.0};
    static const double con_lb[] = {-INFINITY};
    static const double con_ub[] = {1.0};
    matrix_desc_t A = {0};
    A.m = 1;
    A.n = 1;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 0;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;
    qp_problem_t *problem =
        create_qp_problem(objective, NULL, NULL, NULL, &A, con_lb, con_ub, NULL, NULL, NULL, 0, NULL);
    if (!problem)
        return NULL;

    problem->num_quadratic_constraints = 1;
    problem->quadratic_constraint_row_indices = (int *)calloc(1, sizeof(int));
    problem->quadratic_constraint_matrix_num_nonzeros = (int *)calloc(1, sizeof(int));
    problem->quadratic_constraint_matrices = (CsrComponent **)calloc(1, sizeof(CsrComponent *));
    CsrComponent *Q = (CsrComponent *)calloc(1, sizeof(CsrComponent));
    if (!problem->quadratic_constraint_row_indices || !problem->quadratic_constraint_matrix_num_nonzeros ||
        !problem->quadratic_constraint_matrices || !Q)
    {
        free(Q);
        qp_problem_free(problem);
        return NULL;
    }
    Q->row_ptr = (int *)malloc(2 * sizeof(int));
    Q->col_ind = (int *)malloc(sizeof(int));
    Q->val = (double *)malloc(sizeof(double));
    if (!Q->row_ptr || !Q->col_ind || !Q->val)
    {
        free(Q->row_ptr);
        free(Q->col_ind);
        free(Q->val);
        free(Q);
        problem->quadratic_constraint_matrices[0] = NULL;
        qp_problem_free(problem);
        return NULL;
    }
    Q->row_ptr[0] = 0;
    Q->row_ptr[1] = 1;
    Q->col_ind[0] = 0;
    Q->val[0] = 1.0;
    problem->quadratic_constraint_row_indices[0] = 0;
    problem->quadratic_constraint_matrix_num_nonzeros[0] = 1;
    problem->quadratic_constraint_matrices[0] = Q;
    return problem;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2 || size % 2 != 0)
    {
        if (rank == 0)
            fprintf(stderr, "test_distributed_conic requires an even number of MPI ranks\n");
        MPI_Finalize();
        return 77;
    }

    qp_problem_t *problem = rank == 0 ? make_problem() : NULL;
    pdhg_parameters_t parameters;
    set_default_parameters(&parameters);
    parameters.verbose = getenv("PDHCG_TEST_VERBOSE") ? 1 : 0;
    parameters.grid_size.decided = true;
    parameters.grid_size.row_dims = size / 2;
    parameters.grid_size.col_dims = 2;
    parameters.partition_method = UNIFORM_PARTITION;
    parameters.permute_method = FULL_RANDOM_PERMUTATION;
    parameters.curtis_reid_iterations = 0;
    parameters.l_inf_ruiz_iterations = 0;
    parameters.has_pock_chambolle_alpha = false;
    parameters.bound_objective_rescaling = false;
    parameters.sv_max_iter = 50;
    parameters.sv_tol = 1e-3;
    parameters.termination_evaluation_frequency = 20;
    parameters.termination_criteria.eps_optimal_relative = 1e-7;
    parameters.termination_criteria.eps_feasible_relative = 1e-7;
    parameters.termination_criteria.iteration_limit = 1000000;
    const char *time_limit = getenv("PDHCG_TEST_TIME_LIMIT");
    parameters.termination_criteria.time_sec_limit = time_limit ? atof(time_limit) : 30.0;

    pdhcg_result_t *result = solve_qp_problem_distributed(&parameters, problem);
    int failed = 0;
    if (rank == 0)
    {
        if (!result || result->termination_reason != TERMINATION_REASON_OPTIMAL ||
            fabs(result->primal_solution[0]) > 2e-4 || fabs(result->primal_solution[1] - 3.0) > 2e-4 ||
            fabs(result->primal_solution[2] - 4.0) > 2e-4 || fabs(result->primal_solution[3]) > 2e-4 ||
            fabs(result->primal_solution[4] - 5.0) > 2e-4 || fabs(result->primal_solution[5]) > 2e-4)
        {
            fprintf(stderr, "distributed SOC solve returned an incorrect solution\n");
            failed = 1;
        }
        pdhcg_result_free(result);
        qp_problem_free(problem);
    }

    problem = rank == 0 ? make_atomic_soc_problem() : NULL;
    parameters.permute_method = NO_PERMUTATION;
    result = solve_qp_problem_distributed(&parameters, problem);
    if (rank == 0)
    {
        if (!result || result->termination_reason != TERMINATION_REASON_OPTIMAL ||
            fabs(result->primal_solution[3] - 3.0) > 2e-4 || fabs(result->primal_solution[4] - 4.0) > 2e-4 ||
            fabs(result->primal_solution[5]) > 2e-4 || fabs(result->primal_solution[6] - 5.0) > 2e-4)
        {
            fprintf(stderr, "distributed atomic SOC solve returned an incorrect solution\n");
            failed = 1;
        }
        pdhcg_result_free(result);
        qp_problem_free(problem);
    }

    problem = rank == 0 ? make_fixed_rsoc_problem() : NULL;
    parameters.permute_method = NO_PERMUTATION;
    result = solve_qp_problem_distributed(&parameters, problem);
    if (rank == 0)
    {
        const double expected = sqrt(2.0);
        if (!result || result->termination_reason != TERMINATION_REASON_OPTIMAL ||
            fabs(result->primal_solution[0] - expected) > 2e-4 || fabs(result->primal_solution[1] - 1.0) > 1e-12 ||
            fabs(result->primal_solution[2] - 1.0) > 1e-12)
        {
            fprintf(stderr, "distributed fixed-endpoint RSOC solve returned an incorrect solution\n");
            failed = 1;
        }
        pdhcg_result_free(result);
        qp_problem_free(problem);
    }

    problem = rank == 0 ? make_exponential_problem() : NULL;
    parameters.permute_method = FULL_RANDOM_PERMUTATION;
    result = solve_qp_problem_distributed(&parameters, problem);
    if (rank == 0)
    {
        double max_error = 0.0;
        double max_cone_violation = 0.0;
        if (result)
        {
            for (int cone = 0; cone < 4; ++cone)
            {
                double z = result->primal_solution[3 * cone];
                double y = result->primal_solution[3 * cone + 1];
                double t = result->primal_solution[3 * cone + 2];
                max_error = fmax(max_error, fabs(z));
                max_error = fmax(max_error, fabs(y - 1.0));
                max_error = fmax(max_error, fabs(t - 1.0));
                max_cone_violation = fmax(max_cone_violation, y * exp(z / y) - t);
            }
        }
        if (!result || result->termination_reason != TERMINATION_REASON_OPTIMAL || max_error > 2e-4 ||
            max_cone_violation > 2e-6)
        {
            fprintf(stderr,
                    "distributed exponential-cone solve returned an incorrect solution "
                    "(status=%d, max_error=%.3e, max_violation=%.3e)\n",
                    result ? (int)result->termination_reason : -1,
                    max_error,
                    max_cone_violation);
            if (result)
            {
                for (int cone = 0; cone < 4; ++cone)
                    fprintf(stderr,
                            "  cone %d: z=%.9g y=%.9g t=%.9g\n",
                            cone,
                            result->primal_solution[3 * cone],
                            result->primal_solution[3 * cone + 1],
                            result->primal_solution[3 * cone + 2]);
            }
            failed = 1;
        }
        pdhcg_result_free(result);
        qp_problem_free(problem);
    }

    problem = rank == 0 ? make_qcqp_problem() : NULL;
    parameters.permute_method = NO_PERMUTATION;
    result = solve_qp_problem_distributed(&parameters, problem);
    if (rank == 0)
    {
        if (!result || result->termination_reason != TERMINATION_REASON_OPTIMAL || result->num_variables != 1 ||
            fabs(result->primal_solution[0] - 1.0) > 2e-4)
        {
            fprintf(stderr, "distributed QCQP reformulation returned an incorrect solution\n");
            failed = 1;
        }
        pdhcg_result_free(result);
        qp_problem_free(problem);
    }

    const int large_v_dim = 1025;
    problem = rank == 0 ? make_large_soc_problem(large_v_dim) : NULL;
    parameters.permute_method = NO_PERMUTATION;
    result = solve_qp_problem_distributed(&parameters, problem);
    if (rank == 0)
    {
        double expected_v = 1.0 / sqrt((double)large_v_dim);
        double max_v_error = 0.0;
        if (result)
        {
            for (int i = 0; i < large_v_dim; ++i)
                max_v_error = fmax(max_v_error, fabs(result->primal_solution[i] - expected_v));
        }
        if (!result || result->termination_reason != TERMINATION_REASON_OPTIMAL || max_v_error > 2e-4 ||
            fabs(result->primal_solution[large_v_dim]) > 2e-4 ||
            fabs(result->primal_solution[large_v_dim + 1] - 1.0) > 2e-4)
        {
            fprintf(stderr, "distributed large SOC solve returned an incorrect solution\n");
            failed = 1;
        }
        pdhcg_result_free(result);
        qp_problem_free(problem);
    }

    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return failed;
}

#else
int main(void)
{
    return 0;
}
#endif
