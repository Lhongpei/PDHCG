#include "pdhcg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef PDHCG_COMPILE_DISTRIBUTED
#include <mpi.h>

static qp_problem_t *make_variable_psd_problem(void)
{
    const double sqrt_two = 1.41421356237309504880;
    static const int row_ptr[] = {0, 1, 2};
    static const int col_ind[] = {1, 2};
    static const double values[] = {1.0, 1.0};
    double rhs[] = {1.0, 2.0 * sqrt_two};
    static const double objective[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    static const double var_lb[] = {0.0, -INFINITY, -INFINITY, -INFINITY, 0.0, 0.0};
    static const double var_ub[] = {0.0, INFINITY, INFINITY, INFINITY, 0.0, 0.0};
    matrix_desc_t A = {0};
    A.m = 2;
    A.n = 6;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 2;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;
    cone_spec_t cone = {.type = CONE_PSD, .start_idx = 1, .v_dim = 2};
    return create_qp_problem(
        objective, NULL, NULL, NULL, &A, rhs, rhs, var_lb, var_ub, NULL, 1, &cone, NULL, NULL, 0, NULL);
}

static qp_problem_t *make_affine_psd_problem(void)
{
    const double sqrt_two = 1.41421356237309504880;
    static const int a_row_ptr[] = {0, 0};
    static const double scalar_rhs[] = {0.0};
    static const int f_row_ptr[] = {0, 1, 1, 1};
    static const int f_col_ind[] = {0};
    static const double f_values[] = {1.0};
    double offset[] = {0.0, sqrt_two, 1.0};
    static const double objective[] = {1.0};
    matrix_desc_t A = {0};
    matrix_desc_t F = {0};
    A.m = 1;
    A.n = 1;
    A.fmt = matrix_csr;
    A.data.csr.row_ptr = a_row_ptr;
    F.m = 3;
    F.n = 1;
    F.fmt = matrix_csr;
    F.data.csr.nnz = 1;
    F.data.csr.row_ptr = f_row_ptr;
    F.data.csr.col_ind = f_col_ind;
    F.data.csr.vals = f_values;
    cone_spec_t cone = {.type = CONE_PSD, .start_idx = 0, .v_dim = 2};
    return create_qp_problem(
        objective, NULL, NULL, NULL, &A, scalar_rhs, scalar_rhs, NULL, NULL, NULL, 0, NULL, &F, offset, 1, &cone);
}

static void configure(pdhg_parameters_t *params, int row_dims, int col_dims)
{
    set_default_parameters(params);
    params->verbose = 0;
    params->grid_size.decided = true;
    params->grid_size.row_dims = row_dims;
    params->grid_size.col_dims = col_dims;
    params->partition_method = UNIFORM_PARTITION;
    params->permute_method = NO_PERMUTATION;
    params->presolve = false;
    params->termination_evaluation_frequency = 10;
    params->termination_criteria.eps_optimal_relative = 1e-7;
    params->termination_criteria.eps_feasible_relative = 1e-7;
    params->termination_criteria.time_sec_limit = 30.0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2)
    {
        if (rank == 0)
            fprintf(stderr, "test_distributed_psd requires exactly two MPI ranks\n");
        MPI_Finalize();
        return 77;
    }

    int failed = 0;
    pdhg_parameters_t params;
    configure(&params, 1, 2);
    qp_problem_t *problem = rank == 0 ? make_variable_psd_problem() : NULL;
    pdhcg_result_t *result = solve_qp_problem_distributed(&params, problem);
    if (rank == 0)
    {
        const double sqrt_two = 1.41421356237309504880;
        if (!result || result->termination_reason != TERMINATION_REASON_OPTIMAL ||
            fabs(result->primal_solution[1] - 1.0) > 8e-4 || fabs(result->primal_solution[2] - 2.0 * sqrt_two) > 8e-4 ||
            fabs(result->primal_solution[3] - 4.0) > 1e-3)
        {
            fprintf(stderr, "distributed variable PSD solve returned an incorrect solution\n");
            failed = 1;
        }
        pdhcg_result_free(result);
        qp_problem_free(problem);
    }

    configure(&params, 2, 1);
    problem = rank == 0 ? make_affine_psd_problem() : NULL;
    result = solve_qp_problem_distributed(&params, problem);
    if (rank == 0)
    {
        if (!result || result->termination_reason != TERMINATION_REASON_OPTIMAL ||
            fabs(result->primal_solution[0] - 1.0) > 1e-3 || result->relative_primal_residual > 3e-6 ||
            result->relative_dual_residual > 3e-6)
        {
            fprintf(stderr, "distributed affine PSD solve returned an incorrect solution\n");
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
