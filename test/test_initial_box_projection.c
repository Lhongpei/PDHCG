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
#include <math.h>
#include <stdio.h>

static int run_case(const double *primal_start, double expected)
{
    const int row_ptr[] = {0, 1};
    const int col_ind[] = {0};
    const double values[] = {1.0};
    const double objective[] = {0.0};
    const double con_lb[] = {0.0};
    const double con_ub[] = {10.0};
    const double var_lb[] = {1.0};
    const double var_ub[] = {2.0};
    matrix_desc_t A = {0};
    pdhg_parameters_t params;
    pdhcg_result_t *result;

    A.m = 1;
    A.n = 1;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 1;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;

    qp_problem_t *problem = create_qp_problem(
        objective, NULL, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 0, NULL, 0, NULL, NULL);
    if (!problem)
        return 1;

    if (primal_start)
        set_start_values(problem, primal_start, NULL);

    set_default_parameters(&params);
    params.presolve = false;
    params.verbose = 0;
    result = solve_qp_problem(problem, &params);

    int failed = !result || result->termination_reason != TERMINATION_REASON_OPTIMAL || result->total_count != 0 ||
        fabs(result->primal_solution[0] - expected) > 1e-12;

    if (failed)
    {
        fprintf(stderr,
                "initial box projection failed: expected x=%.17g, got status=%d iter=%d x=%.17g\n",
                expected,
                result ? (int)result->termination_reason : -1,
                result ? result->total_count : -1,
                result ? result->primal_solution[0] : NAN);
    }

    pdhcg_result_free(result);
    qp_problem_free(problem);
    return failed;
}

int main(void)
{
    const double upper_infeasible_start[] = {3.0};
    int failed = 0;

    failed |= run_case(NULL, 1.0);
    failed |= run_case(upper_infeasible_start, 2.0);
    return failed ? 1 : 0;
}
