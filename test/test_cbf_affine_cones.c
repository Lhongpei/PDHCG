#include "cbf_parser.h"
#include "pdhcg.h"

#include <math.h>
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

static int write_test_cbf(const char *path)
{
    static const char model[] = "VER\n"
                                "1\n\n"
                                "OBJSENSE\n"
                                "MIN\n\n"
                                "VAR\n"
                                "1 1\n"
                                "F 1\n\n"
                                "CON\n"
                                "4 2\n"
                                "Q 3\n"
                                "L+ 1\n\n"
                                "OBJACOORD\n"
                                "1\n"
                                "0 1\n\n"
                                "ACOORD\n"
                                "2\n"
                                "0 0 1\n"
                                "3 0 1\n\n"
                                "BCOORD\n"
                                "1\n"
                                "1 1\n";

    FILE *file = fopen(path, "w");
    if (!file)
        return 0;
    return fputs(model, file) >= 0 && fclose(file) == 0;
}

int main(void)
{
    const char *path = "test_cbf_affine_cones_tmp.cbf";
    CHECK(write_test_cbf(path));
    qp_problem_t *problem = read_cbf_file(path);
    remove(path);
    CHECK(problem != NULL);

    CHECK(problem->num_variables == 1);
    CHECK(problem->num_constraints == 4);
    CHECK(problem->cones.num_cones == 0);
    CHECK(problem->affine_cones.num_cones == 1);
    CHECK(problem->affine_cones.type[0] == CONE_STANDARD_SOC);
    CHECK(problem->affine_cones.start_idx[0] == 0);
    CHECK(problem->affine_cones.v_dim[0] == 1);

    /* CBF Q is (z,v,w), while the runtime order is (v,w,z). */
    CHECK(problem->constraint_matrix_num_nonzeros == 2);
    CHECK(problem->constraint_matrix->row_ptr[0] == 0);
    CHECK(problem->constraint_matrix->row_ptr[1] == 0);
    CHECK(problem->constraint_matrix->row_ptr[2] == 0);
    CHECK(problem->constraint_matrix->row_ptr[3] == 1);
    CHECK(problem->constraint_matrix->row_ptr[4] == 2);
    CHECK(problem->constraint_matrix->col_ind[0] == 0);
    CHECK(problem->constraint_matrix->val[0] == 1.0);
    CHECK(problem->constraint_matrix->col_ind[1] == 0);
    CHECK(problem->constraint_matrix->val[1] == 1.0);
    CHECK(problem->constraint_lower_bound[3] == 0.0);
    CHECK(isinf(problem->constraint_upper_bound[3]) && problem->constraint_upper_bound[3] > 0.0);
    CHECK(problem->affine_cone_offset[0] == 1.0);
    CHECK(problem->affine_cone_offset[1] == 0.0);
    CHECK(problem->affine_cone_offset[2] == 0.0);
    CHECK(problem->affine_cone_offset[3] == 0.0);

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
    CHECK(fabs(result->primal_solution[0] - 1.0) <= 2e-4);
    CHECK(fabs(result->primal_objective_value - result->dual_objective_value) <= 2e-4);

    pdhcg_result_free(result);
    qp_problem_free(problem);
    return 0;
}
