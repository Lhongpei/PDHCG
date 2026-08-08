#include "pdhcg.h"
#include "permute.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition);                            \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

int main(void)
{
    const int n = 16;
    static const int row_ptr[] = {0, 0, 0, 0, 0, 0, 0, 0};
    static const int col_ind[] = {0};
    static const double values[] = {0.0};
    double objective[n];
    for (int i = 0; i < n; ++i)
        objective[i] = (double)i;

    matrix_desc_t A = {0};
    A.m = 7;
    A.n = n;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 0;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;
    const double constraint_lower[] = {0.0, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    const double constraint_upper[] = {0.0, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY};
    static const double affine_offset[] = {0.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0};
    const char fixed0[] = {0, 0, 1};
    const char fixed1[] = {1, 0, 0};
    const char fixed2[] = {0, 1, 0};
    const char fixed3[] = {0, 0, 1, 1};
    const cone_spec_t cones[] = {
        {.type = CONE_STANDARD_SOC, .start_idx = 1, .v_dim = 1, .power_alpha = 0.0, .is_fixed = fixed0},
        {.type = CONE_POWER, .start_idx = 5, .v_dim = 1, .power_alpha = 0.3, .is_fixed = fixed1},
        {.type = CONE_EXPONENTIAL, .start_idx = 9, .v_dim = 1, .power_alpha = 0.0, .is_fixed = fixed2},
        {.type = CONE_ROTATED_SOC, .start_idx = 12, .v_dim = 2, .power_alpha = 0.0, .is_fixed = fixed3},
    };
    const cone_spec_t affine_cones[] = {
        {.type = CONE_STANDARD_SOC, .start_idx = 1, .v_dim = 1},
        {.type = CONE_EXPONENTIAL, .start_idx = 4, .v_dim = 1},
    };

    qp_problem_t *problem = create_qp_problem(objective,
                                              NULL,
                                              NULL,
                                              NULL,
                                              &A,
                                              constraint_lower,
                                              constraint_upper,
                                              NULL,
                                              NULL,
                                              NULL,
                                              4,
                                              cones,
                                              2,
                                              affine_cones,
                                              affine_offset);
    CHECK(problem != NULL);

    int permutation[n];
    int row_permutation[] = {0, 1, 2, 3, 4, 5, 6};
    srand(7);
    generate_cone_aware_permutation(problem, FULL_RANDOM_PERMUTATION, 1, permutation);
    CHECK(validate_cone_permutation(problem, permutation));

    qp_problem_t *permuted = permute_problem_return_new(problem, row_permutation, permutation);
    CHECK(permuted != NULL);
    for (int cone = 0; cone < 4; ++cone)
    {
        int old_start = problem->cones.start_idx[cone];
        int new_start = permuted->cones.start_idx[cone];
        int length = (problem->cones.type[cone] == CONE_EXPONENTIAL || problem->cones.type[cone] == CONE_POWER)
            ? 3
            : problem->cones.v_dim[cone] + 2;
        CHECK(permuted->cones.type[cone] == problem->cones.type[cone]);
        if (problem->cones.type[cone] == CONE_POWER)
            CHECK(permuted->cones.power_alpha[cone] == problem->cones.power_alpha[cone]);
        for (int slot = 0; slot < length; ++slot)
        {
            CHECK(permuted->objective_vector[new_start + slot] == objective[old_start + slot]);
            CHECK(permuted->cones.is_fixed[new_start + slot] == problem->cones.is_fixed[old_start + slot]);
        }
    }

    int block_permutation[n];
    srand(11);
    generate_cone_aware_permutation(problem, BLOCK_RANDOM_PERMUTATION, 2, block_permutation);
    CHECK(validate_cone_permutation(problem, block_permutation));

    int row_cone_permutation[7];
    int identity_columns[n];
    for (int col = 0; col < n; ++col)
        identity_columns[col] = col;
    srand(13);
    generate_affine_cone_aware_row_permutation(problem, FULL_RANDOM_PERMUTATION, 1, row_cone_permutation);
    CHECK(validate_affine_cone_row_permutation(problem, row_cone_permutation));
    qp_problem_t *affine_permuted = permute_problem_return_new(problem, row_cone_permutation, identity_columns);
    CHECK(affine_permuted != NULL);
    for (int cone = 0; cone < 2; ++cone)
    {
        int old_start = problem->affine_cones.start_idx[cone];
        int new_start = affine_permuted->affine_cones.start_idx[cone];
        for (int slot = 0; slot < 3; ++slot)
            CHECK(affine_permuted->affine_cone_offset[new_start + slot] ==
                  problem->affine_cone_offset[old_start + slot]);
    }
    int invalid_rows[] = {1, 0, 2, 3, 4, 5, 6};
    CHECK(!validate_affine_cone_row_permutation(problem, invalid_rows));
    CHECK(!permute_problem(problem, invalid_rows, identity_columns));
    int duplicate_columns[n];
    memcpy(duplicate_columns, identity_columns, sizeof(duplicate_columns));
    duplicate_columns[1] = duplicate_columns[0];
    CHECK(!validate_cone_permutation(problem, duplicate_columns));
    qp_problem_free(affine_permuted);

    qp_problem_free(permuted);
    qp_problem_free(problem);

    const cone_spec_t overlapping[] = {
        {.type = CONE_POWER, .start_idx = 0, .v_dim = 1, .power_alpha = 0.3, .is_fixed = NULL},
        {.type = CONE_EXPONENTIAL, .start_idx = 2, .v_dim = 1, .power_alpha = 0.0, .is_fixed = NULL},
    };
    problem = create_qp_problem(objective,
                                NULL,
                                NULL,
                                NULL,
                                &A,
                                constraint_lower,
                                constraint_upper,
                                NULL,
                                NULL,
                                NULL,
                                2,
                                overlapping,
                                0,
                                NULL,
                                NULL);
    CHECK(problem == NULL);

    {
        static const int combined_row_ptr[] = {0, 1, 2, 2, 2, 3, 3, 3};
        static const int combined_col_ind[] = {0, 0, 0};
        static const double combined_values[] = {1.0, 1.0, 1.0};
        static const double combined_lower[] = {0.0, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
        static const double combined_upper[] = {INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY};
        static const double affine_offset[] = {0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0};
        static const cone_spec_t row_cones[] = {
            {.type = CONE_EXPONENTIAL, .start_idx = 1, .v_dim = 1},
            {.type = CONE_EXPONENTIAL, .start_idx = 4, .v_dim = 1},
        };
        double linear_objective = -1.0;
        matrix_desc_t combined_matrix = {0};
        combined_matrix.m = 7;
        combined_matrix.n = 1;
        combined_matrix.fmt = matrix_csr;
        combined_matrix.data.csr.nnz = 3;
        combined_matrix.data.csr.row_ptr = combined_row_ptr;
        combined_matrix.data.csr.col_ind = combined_col_ind;
        combined_matrix.data.csr.vals = combined_values;

        problem = create_qp_problem(&linear_objective,
                                    NULL,
                                    NULL,
                                    NULL,
                                    &combined_matrix,
                                    combined_lower,
                                    combined_upper,
                                    NULL,
                                    NULL,
                                    NULL,
                                    0,
                                    NULL,
                                    2,
                                    row_cones,
                                    affine_offset);
        CHECK(problem != NULL);
        int interleaved_rows[] = {1, 2, 3, 0, 4, 5, 6};
        int identity_column[] = {0};
        CHECK(validate_affine_cone_row_permutation(problem, interleaved_rows));
        CHECK(permute_problem(problem, interleaved_rows, identity_column));
        CHECK(problem->affine_cones.start_idx[0] == 0);
        CHECK(problem->affine_cones.start_idx[1] == 4);
        CHECK(problem->constraint_lower_bound[3] == 0.0);
        CHECK(isinf(problem->constraint_upper_bound[3]));

        pdhg_parameters_t parameters;
        set_default_parameters(&parameters);
        parameters.verbose = 0;
        parameters.presolve = false;
        parameters.termination_evaluation_frequency = 10;
        parameters.termination_criteria.eps_optimal_relative = 1e-7;
        parameters.termination_criteria.eps_feasible_relative = 1e-7;
        parameters.termination_criteria.iteration_limit = 1000000;
        parameters.termination_criteria.time_sec_limit = 30.0;
        pdhcg_result_t *result = solve_qp_problem(problem, &parameters);
        CHECK(result != NULL);
        CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL);
        CHECK(fabs(result->primal_solution[0] - log(2.0)) <= 2e-4);
        pdhcg_result_free(result);
        qp_problem_free(problem);
    }
    return 0;
}
