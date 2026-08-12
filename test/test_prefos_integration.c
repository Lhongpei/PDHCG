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

#include "pdhcg_types.h"
#include "presolve_wrapper.h"

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

static CsrComponent *create_csr(int rows, int nnz)
{
    CsrComponent *matrix = (CsrComponent *)calloc(1, sizeof(CsrComponent));
    if (!matrix)
        return NULL;
    matrix->row_ptr = (int *)calloc((size_t)rows + 1, sizeof(int));
    matrix->col_ind = nnz > 0 ? (int *)calloc((size_t)nnz, sizeof(int)) : NULL;
    matrix->val = nnz > 0 ? (double *)calloc((size_t)nnz, sizeof(double)) : NULL;
    if (!matrix->row_ptr || (nnz > 0 && (!matrix->col_ind || !matrix->val)))
    {
        free(matrix->row_ptr);
        free(matrix->col_ind);
        free(matrix->val);
        free(matrix);
        return NULL;
    }
    return matrix;
}

static void free_csr(CsrComponent *matrix)
{
    if (!matrix)
        return;
    free(matrix->row_ptr);
    free(matrix->col_ind);
    free(matrix->val);
    free(matrix);
}

static pdhg_parameters_t quiet_parameters(void)
{
    pdhg_parameters_t parameters;
    memset(&parameters, 0, sizeof(parameters));
    parameters.verbose = 0;
    parameters.termination_criteria.eps_feasible_relative = 1e-8;
    return parameters;
}

static int test_all_fixed_postsolve(void)
{
    qp_problem_t problem;
    pdhg_parameters_t parameters = quiet_parameters();
    pdhcg_presolve_info_t *info;
    pdhcg_result_t *result;
    double lower[] = {0.0, 0.0};
    double upper[] = {1e30, 0.0};
    double objective[] = {1.0, 2.0};
    double row_lower[] = {1.0};
    double row_upper[] = {1.0};
    double row_offset[] = {0.0};

    memset(&problem, 0, sizeof(problem));
    problem.num_variables = 2;
    problem.num_constraints = 1;
    problem.constraint_matrix_num_nonzeros = 2;
    problem.constraint_matrix = create_csr(1, 2);
    CHECK(problem.constraint_matrix != NULL);
    problem.constraint_matrix->row_ptr[1] = 2;
    problem.constraint_matrix->col_ind[0] = 0;
    problem.constraint_matrix->col_ind[1] = 1;
    problem.constraint_matrix->val[0] = 1.0;
    problem.constraint_matrix->val[1] = 1.0;
    problem.variable_lower_bound = lower;
    problem.variable_upper_bound = upper;
    problem.objective_vector = objective;
    problem.constraint_lower_bound = row_lower;
    problem.constraint_upper_bound = row_upper;
    problem.affine_cone_offset = row_offset;

    info = pdhcg_presolve(&problem, &parameters);
    CHECK(info != NULL);
    CHECK(info->presolve_status == PDHCG_PRESOLVE_STATUS_REDUCED);
    if (info->problem_solved_during_presolve)
    {
        result = pdhcg_create_result_from_presolve(info, &problem);
        CHECK(result != NULL);
        CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL);
        CHECK(fabs(result->primal_objective_value - 1.0) <= 1e-12);
    }
    else
    {
        CHECK(info->reduced_problem != NULL);
        CHECK(info->reduced_problem->num_variables == 1);
        result = (pdhcg_result_t *)calloc(1, sizeof(pdhcg_result_t));
        CHECK(result != NULL);
        result->primal_solution = (double *)calloc((size_t)info->reduced_problem->num_variables, sizeof(double));
        result->dual_solution = (double *)calloc((size_t)info->reduced_problem->num_constraints, sizeof(double));
        result->reduced_cost = (double *)calloc((size_t)info->reduced_problem->num_variables, sizeof(double));
        CHECK(result->primal_solution && result->dual_solution && result->reduced_cost);
        result->primal_solution[0] = 1.0;
        result->dual_solution[0] = 1.0;
        CHECK(pdhcg_postsolve(info, result, &problem));
    }
    CHECK(result->primal_solution != NULL);
    CHECK(fabs(result->primal_solution[0] - 1.0) <= 1e-12);
    CHECK(fabs(result->primal_solution[1]) <= 1e-12);

    free(result->primal_solution);
    free(result->dual_solution);
    free(result->reduced_cost);
    free(result);
    pdhcg_presolve_info_free(info);
    free_csr(problem.constraint_matrix);
    return 0;
}

static int test_soc_layout_and_postsolve(void)
{
    qp_problem_t problem;
    pdhg_parameters_t parameters = quiet_parameters();
    pdhcg_presolve_info_t *info;
    pdhcg_result_t result;
    double lower[] = {2.0, -INFINITY, -INFINITY, -INFINITY};
    double upper[] = {2.0, INFINITY, INFINITY, INFINITY};
    double objective[] = {0.0, 0.0, 0.0, 0.0};
    int cone_start[] = {1};
    int cone_v_dim[] = {1};
    cone_type_t cone_type[] = {CONE_STANDARD_SOC};

    memset(&problem, 0, sizeof(problem));
    problem.num_variables = 4;
    problem.num_constraints = 0;
    problem.constraint_matrix = create_csr(0, 0);
    CHECK(problem.constraint_matrix != NULL);
    problem.variable_lower_bound = lower;
    problem.variable_upper_bound = upper;
    problem.objective_vector = objective;
    problem.cones.num_cones = 1;
    problem.cones.start_idx = cone_start;
    problem.cones.v_dim = cone_v_dim;
    problem.cones.type = cone_type;

    info = pdhcg_presolve(&problem, &parameters);
    CHECK(info != NULL);
    CHECK(info->presolve_status == PDHCG_PRESOLVE_STATUS_REDUCED);
    CHECK(!info->problem_solved_during_presolve);
    CHECK(info->reduced_problem != NULL);
    CHECK(info->reduced_problem->num_variables == 3);
    CHECK(info->reduced_problem->cones.num_cones == 1);
    CHECK(info->reduced_problem->cones.start_idx[0] == 0);
    CHECK(info->reduced_problem->cones.v_dim[0] == 1);
    CHECK(info->reduced_problem->cones.type[0] == CONE_STANDARD_SOC);

    memset(&result, 0, sizeof(result));
    result.primal_solution = (double *)calloc(3, sizeof(double));
    result.reduced_cost = (double *)calloc(3, sizeof(double));
    CHECK(result.primal_solution && result.reduced_cost);
    result.primal_solution[2] = 1.0;
    CHECK(pdhcg_postsolve(info, &result, &problem));
    CHECK(result.num_variables == 4);
    CHECK(fabs(result.primal_solution[0] - 2.0) <= 1e-12);
    CHECK(fabs(result.primal_solution[1]) <= 1e-12);
    CHECK(fabs(result.primal_solution[2]) <= 1e-12);
    CHECK(fabs(result.primal_solution[3] - 1.0) <= 1e-12);

    free(result.primal_solution);
    free(result.dual_solution);
    free(result.reduced_cost);
    pdhcg_presolve_info_free(info);
    free_csr(problem.constraint_matrix);
    return 0;
}

static int test_power_layout_and_alpha(void)
{
    qp_problem_t problem;
    pdhg_parameters_t parameters = quiet_parameters();
    pdhcg_presolve_info_t *info;
    double lower[] = {2.0, -INFINITY, -INFINITY, -INFINITY};
    double upper[] = {2.0, INFINITY, INFINITY, INFINITY};
    double objective[] = {0.0, 0.0, 0.0, 0.0};
    int cone_start[] = {1};
    int cone_v_dim[] = {1};
    cone_type_t cone_type[] = {CONE_POWER};
    double power_alpha[] = {0.37};

    memset(&problem, 0, sizeof(problem));
    problem.num_variables = 4;
    problem.constraint_matrix = create_csr(0, 0);
    CHECK(problem.constraint_matrix != NULL);
    problem.variable_lower_bound = lower;
    problem.variable_upper_bound = upper;
    problem.objective_vector = objective;
    problem.cones.num_cones = 1;
    problem.cones.start_idx = cone_start;
    problem.cones.v_dim = cone_v_dim;
    problem.cones.type = cone_type;
    problem.cones.power_alpha = power_alpha;

    info = pdhcg_presolve(&problem, &parameters);
    CHECK(info != NULL);
    CHECK(info->presolve_status == PDHCG_PRESOLVE_STATUS_REDUCED);
    CHECK(!info->problem_solved_during_presolve);
    CHECK(info->reduced_problem != NULL);
    CHECK(info->reduced_problem->num_variables == 3);
    CHECK(info->reduced_problem->cones.num_cones == 1);
    CHECK(info->reduced_problem->cones.start_idx[0] == 0);
    CHECK(info->reduced_problem->cones.v_dim[0] == 1);
    CHECK(info->reduced_problem->cones.type[0] == CONE_POWER);
    CHECK(info->reduced_problem->cones.power_alpha != NULL);
    CHECK(fabs(info->reduced_problem->cones.power_alpha[0] - power_alpha[0]) <= 1e-15);

    pdhcg_presolve_info_free(info);
    free_csr(problem.constraint_matrix);
    return 0;
}

static int test_psd_layout_and_postsolve(void)
{
    qp_problem_t problem;
    pdhg_parameters_t parameters = quiet_parameters();
    pdhcg_presolve_info_t *info;
    pdhcg_result_t result;
    double lower[] = {2.0, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    double upper[] = {2.0, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY};
    double objective[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int cone_start[] = {1};
    int cone_v_dim[] = {3};
    cone_type_t cone_type[] = {CONE_PSD};
    const double identity_svec[] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
    int i;

    memset(&problem, 0, sizeof(problem));
    problem.num_variables = 7;
    problem.constraint_matrix = create_csr(0, 0);
    CHECK(problem.constraint_matrix != NULL);
    problem.variable_lower_bound = lower;
    problem.variable_upper_bound = upper;
    problem.objective_vector = objective;
    problem.cones.num_cones = 1;
    problem.cones.start_idx = cone_start;
    problem.cones.v_dim = cone_v_dim;
    problem.cones.type = cone_type;

    info = pdhcg_presolve(&problem, &parameters);
    CHECK(info != NULL);
    CHECK(info->presolve_status == PDHCG_PRESOLVE_STATUS_REDUCED);
    CHECK(!info->problem_solved_during_presolve);
    CHECK(info->reduced_problem != NULL);
    CHECK(info->reduced_problem->num_variables == 6);
    CHECK(info->reduced_problem->cones.num_cones == 1);
    CHECK(info->reduced_problem->cones.start_idx[0] == 0);
    CHECK(info->reduced_problem->cones.v_dim[0] == 3);
    CHECK(info->reduced_problem->cones.type[0] == CONE_PSD);

    memset(&result, 0, sizeof(result));
    result.primal_solution = (double *)calloc(6, sizeof(double));
    result.reduced_cost = (double *)calloc(6, sizeof(double));
    CHECK(result.primal_solution && result.reduced_cost);
    memcpy(result.primal_solution, identity_svec, sizeof(identity_svec));
    CHECK(pdhcg_postsolve(info, &result, &problem));
    CHECK(result.num_variables == 7);
    CHECK(fabs(result.primal_solution[0] - 2.0) <= 1e-12);
    for (i = 0; i < 6; ++i)
        CHECK(fabs(result.primal_solution[i + 1] - identity_svec[i]) <= 1e-12);

    free(result.primal_solution);
    free(result.dual_solution);
    free(result.reduced_cost);
    pdhcg_presolve_info_free(info);
    free_csr(problem.constraint_matrix);
    return 0;
}

static int test_diagonal_middle_matrix(void)
{
    qp_problem_t problem;
    pdhg_parameters_t parameters = quiet_parameters();
    pdhcg_presolve_info_t *info;
    double lower[] = {0.0, 0.0};
    double upper[] = {0.0, INFINITY};
    double objective[] = {0.0, 2.0};

    memset(&problem, 0, sizeof(problem));
    problem.num_variables = 2;
    problem.constraint_matrix = create_csr(0, 0);
    problem.objective_sparse_matrix = create_csr(2, 1);
    problem.objective_lowrank_matrix = create_csr(1, 1);
    problem.objective_lowrank_middle_matrix = create_csr(1, 1);
    CHECK(problem.constraint_matrix && problem.objective_sparse_matrix && problem.objective_lowrank_matrix &&
          problem.objective_lowrank_middle_matrix);
    problem.objective_sparse_matrix->row_ptr[1] = 1;
    problem.objective_sparse_matrix->row_ptr[2] = 1;
    problem.objective_sparse_matrix->col_ind[0] = 0;
    problem.objective_sparse_matrix->val[0] = 1.0;
    problem.objective_sparse_matrix_num_nonzeros = 1;
    problem.objective_lowrank_matrix->row_ptr[1] = 1;
    problem.objective_lowrank_matrix->col_ind[0] = 1;
    problem.objective_lowrank_matrix->val[0] = 1.0;
    problem.objective_lowrank_matrix_num_nonzeros = 1;
    problem.num_rank_lowrank_obj = 1;
    problem.objective_lowrank_middle_matrix->row_ptr[1] = 1;
    problem.objective_lowrank_middle_matrix->col_ind[0] = 0;
    problem.objective_lowrank_middle_matrix->val[0] = 2.0;
    problem.objective_lowrank_middle_matrix_num_nonzeros = 1;
    problem.variable_lower_bound = lower;
    problem.variable_upper_bound = upper;
    problem.objective_vector = objective;

    info = pdhcg_presolve(&problem, &parameters);
    CHECK(info != NULL);
    CHECK(info->presolve_status == PDHCG_PRESOLVE_STATUS_REDUCED);
    CHECK(info->reduced_problem != NULL);
    CHECK(info->reduced_problem->num_variables == 1);
    CHECK(info->reduced_problem->num_rank_lowrank_obj == 1);
    CHECK(info->reduced_problem->objective_lowrank_middle_matrix != NULL);
    CHECK(info->reduced_problem->objective_lowrank_middle_matrix_num_nonzeros == 1);
    CHECK(info->reduced_problem->objective_lowrank_middle_matrix->col_ind[0] == 0);
    CHECK(fabs(info->reduced_problem->objective_lowrank_middle_matrix->val[0] - 2.0) <= 1e-12);
    pdhcg_presolve_info_free(info);
    free_csr(problem.constraint_matrix);
    free_csr(problem.objective_sparse_matrix);
    free_csr(problem.objective_lowrank_matrix);
    free_csr(problem.objective_lowrank_middle_matrix);
    return 0;
}

int main(void)
{
    if (!pdhcg_presolve_available())
    {
        printf("PreFOS integration is disabled; skipping.\n");
        return 0;
    }
    CHECK(strncmp(pdhcg_presolve_version(), "PreFOS ", 7) == 0);
    printf("all-fixed postsolve...\n");
    fflush(stdout);
    if (test_all_fixed_postsolve())
        return 1;
    printf("SOC layout and postsolve...\n");
    fflush(stdout);
    if (test_soc_layout_and_postsolve())
        return 1;
    printf("Power-cone layout and alpha...\n");
    fflush(stdout);
    if (test_power_layout_and_alpha())
        return 1;
    printf("PSD layout and postsolve...\n");
    fflush(stdout);
    if (test_psd_layout_and_postsolve())
        return 1;
    printf("diagonal middle matrix...\n");
    fflush(stdout);
    if (test_diagonal_middle_matrix())
        return 1;
    printf("PreFOS integration tests passed.\n");
    return 0;
}
