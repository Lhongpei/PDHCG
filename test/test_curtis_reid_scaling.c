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
#include <stdlib.h>

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
    return create_qp_problem(
        objective, NULL, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 0, NULL, NULL, NULL, 0, NULL);
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
        .power_alpha = 0.0,
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
    return create_qp_problem(
        objective, NULL, NULL, NULL, &A, con_lb, con_ub, NULL, NULL, NULL, 1, &cone, NULL, NULL, 0, NULL);
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
    rescale_info_t *cone_preserving = rescale_problem(&params, problem);
    if (!cone_preserving)
    {
        fprintf(stderr, "cone-preserving Curtis-Reid scaling returned NULL\n");
        ++failures;
        qp_problem_free(problem);
        return;
    }
    check_close("cone-preserving row scale", cone_preserving->con_rescale[0], exp(4.5), 1e-12);
    check_close("cone-preserving non-cone scale", cone_preserving->var_rescale[0], exp(7.5), 1e-12);
    check_close("cone-preserving block minimizer", cone_preserving->var_rescale[1], exp(-2.5), 1e-12);
    check_close(
        "cone-preserving scale slot 2", cone_preserving->var_rescale[2], cone_preserving->var_rescale[1], 1e-14);
    check_close(
        "cone-preserving scale slot 3", cone_preserving->var_rescale[3], cone_preserving->var_rescale[1], 1e-14);

    params.use_cone_preserving_scaling = false;
    rescale_info_t *coordinatewise = rescale_problem(&params, problem);
    if (!coordinatewise)
    {
        fprintf(stderr, "coordinate-wise cone Curtis-Reid scaling returned NULL\n");
        ++failures;
    }
    else
    {
        for (int nz = 0; nz < problem->constraint_matrix_num_nonzeros; ++nz)
        {
            check_close("coordinate-wise cone unit magnitude",
                        fabs(coordinatewise->scaled_problem->constraint_matrix->val[nz]),
                        1.0,
                        1e-12);
        }
        if (coordinatewise->var_rescale[1] == coordinatewise->var_rescale[3])
        {
            fprintf(stderr, "coordinate-wise cone scaling unexpectedly tied all slots\n");
            ++failures;
        }
        rescale_info_free(coordinatewise);
    }

    rescale_info_free(cone_preserving);
    qp_problem_free(problem);
}

static qp_problem_t *make_phase_taper_problem(int length, int affine)
{
    int *row_ptr = (int *)malloc((size_t)(length + 1) * sizeof(int));
    int *col_ind = (int *)malloc((size_t)length * sizeof(int));
    double *values = (double *)malloc((size_t)length * sizeof(double));
    double *objective = (double *)calloc((size_t)length, sizeof(double));
    if (!row_ptr || !col_ind || !values || !objective)
    {
        free(row_ptr);
        free(col_ind);
        free(values);
        free(objective);
        return NULL;
    }
    for (int index = 0; index < length; ++index)
    {
        row_ptr[index] = index;
        col_ind[index] = index;
        values[index] = (double)(index + 1) * (double)(index + 1);
    }
    row_ptr[length] = length;

    matrix_desc_t diagonal = {0};
    diagonal.m = length;
    diagonal.n = length;
    diagonal.fmt = matrix_csr;
    diagonal.data.csr.nnz = length;
    diagonal.data.csr.row_ptr = row_ptr;
    diagonal.data.csr.col_ind = col_ind;
    diagonal.data.csr.vals = values;
    cone_spec_t cone = {
        .type = CONE_STANDARD_SOC,
        .start_idx = 0,
        .v_dim = length - 2,
    };

    qp_problem_t *problem = NULL;
    if (!affine)
    {
        problem = create_qp_problem(
            objective, NULL, NULL, NULL, &diagonal, NULL, NULL, NULL, NULL, NULL, 1, &cone, NULL, NULL, 0, NULL);
    }
    else
    {
        problem = create_qp_problem(
            objective, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL, &diagonal, NULL, 1, &cone);
    }

    free(row_ptr);
    free(col_ind);
    free(values);
    free(objective);
    return problem;
}

static pdhg_parameters_t phase_taper_parameters(int ruiz)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.curtis_reid_iterations = 0;
    params.l_inf_ruiz_iterations = ruiz ? 1 : 0;
    params.has_pock_chambolle_alpha = !ruiz;
    params.pock_chambolle_alpha = 1.0;
    params.bound_objective_rescaling = false;
    params.use_cone_preserving_scaling = true;
    return params;
}

static void test_phase_taper_case(int length, int affine, int ruiz)
{
    qp_problem_t *problem = make_phase_taper_problem(length, affine);
    if (!problem)
    {
        fprintf(
            stderr, "failed to create %s phase-taper problem of length %d\n", affine ? "affine" : "variable", length);
        ++failures;
        return;
    }

    pdhg_parameters_t params = phase_taper_parameters(ruiz);
    rescale_info_t *info = rescale_problem(&params, problem);
    if (!info)
    {
        fprintf(stderr, "%s phase-taper scaling returned NULL\n", ruiz ? "Ruiz" : "Pock-Chambolle");
        ++failures;
        qp_problem_free(problem);
        return;
    }

    double sum_sq = (double)length * (double)(length + 1) * (double)(2 * length + 1) / 6.0;
    double rms = sqrt(sum_sq / (double)length);
    double block_max = (double)length;
    double expected = ruiz ? (length <= 8 ? block_max : rms) : (length <= 8 ? rms : sqrt(block_max * rms));
    int start = affine ? problem->affine_cones.start_idx[0] : problem->cones.start_idx[0];
    const double *scaling = affine ? info->con_rescale : info->var_rescale;
    for (int index = start; index < start + length; ++index)
        check_close("phase-taper cone scale", scaling[index], expected, 1e-12);

    rescale_info_free(info);
    qp_problem_free(problem);
}

static void test_phase_taper_scaling(void)
{
    for (int length = 8; length <= 9; ++length)
    {
        for (int affine = 0; affine <= 1; ++affine)
        {
            test_phase_taper_case(length, affine, 1);
            test_phase_taper_case(length, affine, 0);
        }
    }
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
    if (!defaults.use_cone_preserving_scaling)
    {
        fprintf(stderr, "cone-preserving scaling must be enabled by default\n");
        ++failures;
    }

    test_plain_scaling();
    test_cone_block_scaling();
    test_phase_taper_scaling();
    return failures == 0 ? 0 : 1;
}
