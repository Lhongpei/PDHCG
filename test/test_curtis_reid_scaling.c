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
#include "preconditioner.h"
#include "solver_state.h"
#include <math.h>
#include <stdio.h>

static int failures = 0;

static void check_close(const char *name, double actual, double expected, double tolerance)
{
    const double scale = fmax(1.0, fmax(fabs(actual), fabs(expected)));
    if (!isfinite(actual) || fabs(actual - expected) > tolerance * scale)
    {
        fprintf(stderr, "%s: got %.17g, expected %.17g\n", name, actual, expected);
        ++failures;
    }
}

static qp_problem_t *make_plain_problem(void)
{
    static const int row_ptr[] = {0, 2, 4};
    static const int col_ind[] = {0, 1, 0, 1};
    static const double values[] = {1e-6, 1e-3, 1e3, 1e6};
    static const double objective[] = {3.0, -4.0};
    static const double con_lb[] = {2.0, -5.0};
    static const double con_ub[] = {4.0, 8.0};
    static const double var_lb[] = {-2.0, -3.0};
    static const double var_ub[] = {7.0, 9.0};
    matrix_desc_t A = {0};
    A.m = 2;
    A.n = 2;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 4;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;
    return create_qp_problem(objective, NULL, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 0, NULL);
}

static qp_problem_t *make_cone_problem(void)
{
    static const int row_ptr[] = {0, 4};
    static const int col_ind[] = {0, 1, 2, 3};
    static const double values[] = {162754.79141900392, 1.0, 7.38905609893065, 54.598150033144236};
    static const double objective[] = {0.0, 0.0, 0.0, 0.0};
    static const double con_lb[] = {0.0};
    static const double con_ub[] = {0.0};
    const cone_spec_t cone = {
        .type = CONE_STANDARD_SOC,
        .start_idx = 1,
        .v_dim = 1,
        .alpha = 0.0,
        .is_fixed = NULL,
    };
    matrix_desc_t A = {0};
    A.m = 1;
    A.n = 4;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 4;
    A.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = col_ind;
    A.data.csr.vals = values;
    return create_qp_problem(objective, NULL, NULL, NULL, &A, con_lb, con_ub, NULL, NULL, NULL, 1, &cone);
}

static pdhg_parameters_t curtis_reid_only_parameters(void)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.curtis_reid_iterations = 20;
    params.l_inf_ruiz_iterations = 0;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = false;
    return params;
}

static void test_plain_scaling(void)
{
    qp_problem_t *problem = make_plain_problem();
    if (!problem)
    {
        fprintf(stderr, "failed to create plain scaling problem\n");
        ++failures;
        return;
    }

    pdhg_parameters_t params = curtis_reid_only_parameters();
    rescale_info_t *info = rescale_problem(&params, problem);
    if (!info)
    {
        fprintf(stderr, "plain Curtis-Reid scaling returned NULL\n");
        ++failures;
        qp_problem_free(problem);
        return;
    }

    for (int row = 0; row < problem->num_constraints; ++row)
    {
        for (int nz = problem->constraint_matrix->row_ptr[row]; nz < problem->constraint_matrix->row_ptr[row + 1]; ++nz)
        {
            const int col = problem->constraint_matrix->col_ind[nz];
            const double expected =
                problem->constraint_matrix->val[nz] / (info->con_rescale[row] * info->var_rescale[col]);
            check_close("scaled A equivalence", info->scaled_problem->constraint_matrix->val[nz], expected, 1e-12);
            check_close(
                "Curtis-Reid unit magnitude", fabs(info->scaled_problem->constraint_matrix->val[nz]), 1.0, 1e-12);
        }
        check_close("constraint lower bound",
                    info->scaled_problem->constraint_lower_bound[row],
                    problem->constraint_lower_bound[row] / info->con_rescale[row],
                    1e-12);
        check_close("constraint upper bound",
                    info->scaled_problem->constraint_upper_bound[row],
                    problem->constraint_upper_bound[row] / info->con_rescale[row],
                    1e-12);
    }
    for (int col = 0; col < problem->num_variables; ++col)
    {
        check_close("objective scaling",
                    info->scaled_problem->objective_vector[col],
                    problem->objective_vector[col] / info->var_rescale[col],
                    1e-12);
        check_close("variable lower bound",
                    info->scaled_problem->variable_lower_bound[col],
                    problem->variable_lower_bound[col] * info->var_rescale[col],
                    1e-12);
        check_close("variable upper bound",
                    info->scaled_problem->variable_upper_bound[col],
                    problem->variable_upper_bound[col] * info->var_rescale[col],
                    1e-12);
    }

    rescale_info_free(info);
    qp_problem_free(problem);
}

static void test_cone_block_scaling(void)
{
    qp_problem_t *problem = make_cone_problem();
    if (!problem)
    {
        fprintf(stderr, "failed to create cone scaling problem\n");
        ++failures;
        return;
    }

    pdhg_parameters_t params = curtis_reid_only_parameters();
    rescale_info_t *uniform = rescale_problem(&params, problem);
    if (!uniform)
    {
        fprintf(stderr, "uniform cone Curtis-Reid scaling returned NULL\n");
        ++failures;
        qp_problem_free(problem);
        return;
    }
    check_close("uniform row scale", uniform->con_rescale[0], exp(4.5), 1e-12);
    check_close("uniform non-cone scale", uniform->var_rescale[0], exp(7.5), 1e-12);
    check_close("uniform cone block minimizer", uniform->var_rescale[1], exp(-2.5), 1e-12);
    check_close("uniform cone scale slot 2", uniform->var_rescale[2], uniform->var_rescale[1], 1e-14);
    check_close("uniform cone scale slot 3", uniform->var_rescale[3], uniform->var_rescale[1], 1e-14);

    params.heterogeneous_cone_scaling = true;
    rescale_info_t *heterogeneous = rescale_problem(&params, problem);
    if (!heterogeneous)
    {
        fprintf(stderr, "heterogeneous cone Curtis-Reid scaling returned NULL\n");
        ++failures;
    }
    else
    {
        for (int nz = 0; nz < problem->constraint_matrix_num_nonzeros; ++nz)
        {
            check_close("heterogeneous cone unit magnitude",
                        fabs(heterogeneous->scaled_problem->constraint_matrix->val[nz]),
                        1.0,
                        1e-12);
        }
        if (heterogeneous->var_rescale[1] == heterogeneous->var_rescale[3])
        {
            fprintf(stderr, "heterogeneous cone scaling unexpectedly tied all slots\n");
            ++failures;
        }
        rescale_info_free(heterogeneous);
    }

    rescale_info_free(uniform);
    qp_problem_free(problem);
}

int main(void)
{
    pdhg_parameters_t defaults;
    set_default_parameters(&defaults);
    if (defaults.curtis_reid_iterations != 0)
    {
        fprintf(stderr, "default Curtis-Reid iterations: got %d, expected 0\n", defaults.curtis_reid_iterations);
        ++failures;
    }

    test_plain_scaling();
    test_cone_block_scaling();
    return failures == 0 ? 0 : 1;
}
