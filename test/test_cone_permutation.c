#include "pdhcg.h"
#include "permute.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
    static const int row_ptr[] = {0, 0};
    static const int col_ind[] = {0};
    static const double values[] = {0.0};
    double objective[n];
    for (int i = 0; i < n; ++i)
        objective[i] = (double)i;

    matrix_desc_t A = {0};
    A.m = 1;
    A.n = n;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 0;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;
    const double constraint[] = {0.0};
    const char fixed0[] = {0, 0, 1};
    const char fixed1[] = {1, 0, 0};
    const char fixed2[] = {0, 1, 0};
    const char fixed3[] = {0, 0, 1, 1};
    const cone_spec_t cones[] = {
        {.type = CONE_STANDARD_SOC, .start_idx = 1, .v_dim = 1, .alpha = 0.0, .is_fixed = fixed0},
        {.type = CONE_POWER, .start_idx = 5, .v_dim = 1, .alpha = 0.3, .is_fixed = fixed1},
        {.type = CONE_EXPONENTIAL, .start_idx = 9, .v_dim = 1, .alpha = 0.0, .is_fixed = fixed2},
        {.type = CONE_ROTATED_SOC, .start_idx = 12, .v_dim = 2, .alpha = 0.0, .is_fixed = fixed3},
    };

    qp_problem_t *problem =
        create_qp_problem(objective, NULL, NULL, NULL, &A, constraint, constraint, NULL, NULL, NULL, 4, cones);
    CHECK(problem != NULL);

    int permutation[n];
    int row_permutation[] = {0};
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

    qp_problem_free(permuted);
    qp_problem_free(problem);

    const cone_spec_t overlapping[] = {
        {.type = CONE_POWER, .start_idx = 0, .v_dim = 1, .alpha = 0.3, .is_fixed = NULL},
        {.type = CONE_EXPONENTIAL, .start_idx = 2, .v_dim = 1, .alpha = 0.0, .is_fixed = NULL},
    };
    problem =
        create_qp_problem(objective, NULL, NULL, NULL, &A, constraint, constraint, NULL, NULL, NULL, 2, overlapping);
    CHECK(problem == NULL);
    return 0;
}
