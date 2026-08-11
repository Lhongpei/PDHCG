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
                                "3 1\n"
                                "Q 3\n\n"
                                "CON\n"
                                "1 1\n"
                                "L= 1\n\n"
                                "OBJACOORD\n"
                                "1\n"
                                "1 -1\n\n"
                                "ACOORD\n"
                                "1\n"
                                "0 0 -1\n\n"
                                "BCOORD\n"
                                "1\n"
                                "0 1\n";

    FILE *file = fopen(path, "w");
    if (!file)
        return 0;
    int ok = fputs(model, file) >= 0 && fclose(file) == 0;
    return ok;
}

int main(void)
{
    const char *path = "test_cbf_fixed_slots_tmp.cbf";
    CHECK(write_test_cbf(path));

    qp_problem_t *problem = read_cbf_file(path);
    remove(path);
    CHECK(problem != NULL);
    CHECK(problem->num_variables == 3);
    CHECK(problem->num_constraints == 1);
    CHECK(problem->cones.num_cones == 1);
    CHECK(problem->cones.type[0] == CONE_STANDARD_SOC);
    CHECK(problem->cones.start_idx[0] == 0);
    CHECK(problem->cones.v_dim[0] == 1);

    /* CBF Q ordering is (z, v, w); internal ordering is (v, w, z). */
    CHECK(problem->constraint_matrix_num_nonzeros == 1);
    CHECK(problem->constraint_matrix->row_ptr[0] == 0);
    CHECK(problem->constraint_matrix->row_ptr[1] == 1);
    CHECK(problem->constraint_matrix->col_ind[0] == 2);
    CHECK(problem->constraint_matrix->val[0] == -1.0);
    CHECK(problem->cones.is_fixed && problem->cones.is_fixed[2]);
    CHECK(problem->primal_start && problem->primal_start[2] == 1.0);
    CHECK(isinf(problem->variable_lower_bound[2]) && problem->variable_lower_bound[2] < 0.0);
    CHECK(isinf(problem->variable_upper_bound[2]) && problem->variable_upper_bound[2] > 0.0);
    CHECK(problem->constraint_lower_bound[0] == -1.0);
    CHECK(problem->constraint_upper_bound[0] == -1.0);
    CHECK(problem->affine_cone_offset[0] == 0.0);

    pdhg_parameters_t parameters;
    set_default_parameters(&parameters);
    parameters.verbose = 0;
    parameters.termination_criteria.eps_optimal_relative = 1e-7;
    parameters.termination_criteria.eps_feasible_relative = 1e-7;
    parameters.termination_criteria.iteration_limit = 100000;
    parameters.termination_criteria.time_sec_limit = 30.0;

    pdhcg_result_t *result = solve_qp_problem(problem, &parameters);
    CHECK(result != NULL);
    CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL);
    CHECK(fabs(result->primal_solution[0] - 1.0) <= 1e-5);
    CHECK(fabs(result->primal_solution[1]) <= 1e-5);
    CHECK(fabs(result->primal_solution[2] - 1.0) <= 1e-12);
    CHECK(fabs(result->primal_objective_value + 1.0) <= 1e-5);

    pdhcg_result_free(result);
    qp_problem_free(problem);
    return 0;
}
