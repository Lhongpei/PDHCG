#include "pdhcg.h"

#include <stdio.h>

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition);                            \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

static matrix_desc_t empty_csr(int rows, int columns, const int *row_ptr)
{
    matrix_desc_t matrix = {0};
    matrix.m = rows;
    matrix.n = columns;
    matrix.fmt = matrix_csr;
    matrix.data.csr.row_ptr = row_ptr;
    return matrix;
}

int main(void)
{
    static const int a_row_ptr[] = {0, 0};
    static const int q2_row_ptr[] = {0, 0, 0};
    static const int r_row_ptr[] = {0, 0};
    matrix_desc_t A = empty_csr(1, 3, a_row_ptr);
    matrix_desc_t Q2 = empty_csr(2, 2, q2_row_ptr);
    matrix_desc_t R2 = empty_csr(1, 2, r_row_ptr);

    qp_problem_t *problem =
        create_qp_problem(NULL, &Q2, NULL, NULL, &A, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL, 0, NULL);
    CHECK(problem == NULL);

    problem = create_qp_problem(NULL, NULL, &R2, NULL, &A, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL, 0, NULL);
    CHECK(problem == NULL);

    matrix_desc_t R3 = empty_csr(1, 3, r_row_ptr);
    matrix_desc_t D2 = empty_csr(2, 2, q2_row_ptr);
    problem = create_qp_problem(NULL, NULL, &R3, &D2, &A, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL, 0, NULL);
    CHECK(problem == NULL);

    static const int invalid_row_ptr[] = {0, 1};
    static const int invalid_column[] = {3};
    static const double one[] = {1.0};
    matrix_desc_t invalid_A = empty_csr(1, 3, invalid_row_ptr);
    invalid_A.data.csr.nnz = 1;
    invalid_A.data.csr.col_ind = invalid_column;
    invalid_A.data.csr.vals = one;
    problem = create_qp_problem(
        NULL, NULL, NULL, NULL, &invalid_A, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL, 0, NULL);
    CHECK(problem == NULL);

    problem =
        create_qp_problem(NULL, NULL, NULL, NULL, &A, NULL, NULL, NULL, NULL, NULL, -1, NULL, NULL, NULL, 0, NULL);
    CHECK(problem == NULL);

    static const char fixed[] = {0, 1, 0};
    cone_spec_t cone = {
        .type = CONE_EXPONENTIAL,
        .start_idx = 0,
        .v_dim = 1,
        .is_fixed = fixed,
    };
    problem =
        create_qp_problem(NULL, NULL, NULL, NULL, &A, NULL, NULL, NULL, NULL, NULL, 1, &cone, NULL, NULL, 0, NULL);
    CHECK(problem != NULL);
    CHECK(problem->cones.fixed_mask_size == problem->num_variables);
    CHECK(problem->cones.is_fixed != NULL && problem->cones.is_fixed[1] == 1);
    qp_problem_free(problem);

    static const int f3_row_ptr[] = {0, 0, 0, 0};
    static const int f4_row_ptr[] = {0, 0, 0, 0, 0};
    static const double affine_offset[] = {1.0, 0.0, 0.0};
    cone_spec_t affine_cone = {
        .type = CONE_STANDARD_SOC,
        .start_idx = 0,
        .v_dim = 1,
    };

    matrix_desc_t wrong_width_F = empty_csr(3, 2, f3_row_ptr);
    problem = create_qp_problem(NULL,
                                NULL,
                                NULL,
                                NULL,
                                &A,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                0,
                                NULL,
                                &wrong_width_F,
                                affine_offset,
                                1,
                                &affine_cone);
    CHECK(problem == NULL);

    matrix_desc_t uncovered_F = empty_csr(4, 3, f4_row_ptr);
    problem = create_qp_problem(
        NULL, NULL, NULL, NULL, &A, NULL, NULL, NULL, NULL, NULL, 0, NULL, &uncovered_F, NULL, 1, &affine_cone);
    CHECK(problem == NULL);

    problem = create_qp_problem(
        NULL, NULL, NULL, NULL, &A, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL, affine_offset, 1, &affine_cone);
    CHECK(problem == NULL);

    return 0;
}
