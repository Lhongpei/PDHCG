/*
Copyright 2025 Haihao Lu
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

#include "preconditioner.h"
#include "cone_utils.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define SCALING_EPSILON 1e-12
#define CURTIS_REID_MIN_ABS 1e-300
#define CURTIS_REID_LOG_SCALE_LIMIT 69.07755278982137
#define PHASE_TAPER_CONE_THRESHOLD 8

typedef enum
{
    CONE_SCALING_RUIZ,
    CONE_SCALING_POCK_CHAMBOLLE,
} cone_scaling_phase_t;

/*
 * Cone-block aggregation follows HPR-SOCP's :phase_taper strategy:
 * https://github.com/PolyU-IOR/HPR-SOCP
 */
static void apply_cone_preserving_scaling(double *scaling, const cone_blocks_t *cones, cone_scaling_phase_t phase)
{
    for (int block = 0; block < cones->num_cones; ++block)
    {
        int start = cones->start_idx[block];
        int length = cone_block_length(cones, block);
        double block_max = 0.0;
        double sum_sq = 0.0;
        for (int index = start; index < start + length; ++index)
        {
            block_max = fmax(block_max, scaling[index]);
            sum_sq += scaling[index] * scaling[index];
        }

        double rms = sqrt(sum_sq / (double)length);
        double block_scale = phase == CONE_SCALING_RUIZ
            ? (length <= PHASE_TAPER_CONE_THRESHOLD ? block_max : rms)
            : (length <= PHASE_TAPER_CONE_THRESHOLD ? rms : sqrt(block_max * rms));
        for (int index = start; index < start + length; ++index)
            scaling[index] = block_scale;
    }
}

static double curtis_reid_exp_clamped(double value)
{
    if (isnan(value))
        return 1.0;
    if (value > CURTIS_REID_LOG_SCALE_LIMIT)
        value = CURTIS_REID_LOG_SCALE_LIMIT;
    else if (value < -CURTIS_REID_LOG_SCALE_LIMIT)
        value = -CURTIS_REID_LOG_SCALE_LIMIT;
    return exp(value);
}

static void scale_problem(qp_problem_t *problem, const double *con_rescale, const double *var_rescale);
static void pin_scaled_fixed_cone_bounds(qp_problem_t *scaled_problem,
                                         const qp_problem_t *source_problem,
                                         const rescale_info_t *rescale_info);
static void curtis_reid_rescaling(qp_problem_t *problem,
                                  int num_iters,
                                  bool use_cone_preserving_scaling,
                                  double *cum_con_rescale,
                                  double *cum_var_rescale);
static void ruiz_rescaling(qp_problem_t *problem,
                           int num_iters,
                           bool use_cone_preserving_scaling,
                           double *cum_con_rescale,
                           double *cum_var_rescale);
static void pock_chambolle_rescaling(qp_problem_t *problem,
                                     double alpha,
                                     bool use_cone_preserving_scaling,
                                     double *cum_con_rescale,
                                     double *cum_var_rescale);

qp_problem_t *deepcopy_problem(const qp_problem_t *prob)
{
    qp_problem_t *new_prob = (qp_problem_t *)safe_calloc(1, sizeof(qp_problem_t));

    new_prob->num_variables = prob->num_variables;
    new_prob->num_constraints = prob->num_constraints;
    new_prob->constraint_matrix_num_nonzeros = prob->constraint_matrix_num_nonzeros;
    new_prob->objective_sparse_matrix_num_nonzeros = prob->objective_sparse_matrix_num_nonzeros;
    new_prob->objective_lowrank_matrix_num_nonzeros = prob->objective_lowrank_matrix_num_nonzeros;
    new_prob->objective_constant = prob->objective_constant;
    new_prob->num_rank_lowrank_obj = prob->num_rank_lowrank_obj;

    size_t var_bytes = prob->num_variables * sizeof(double);
    size_t con_bytes = prob->num_constraints * sizeof(double);

    new_prob->variable_lower_bound = safe_malloc(var_bytes);
    new_prob->variable_upper_bound = safe_malloc(var_bytes);
    new_prob->objective_vector = safe_malloc(var_bytes);
    new_prob->constraint_lower_bound = safe_malloc(con_bytes);
    new_prob->constraint_upper_bound = safe_malloc(con_bytes);
    new_prob->affine_cone_offset = safe_malloc(con_bytes);

    memcpy(new_prob->variable_lower_bound, prob->variable_lower_bound, var_bytes);
    memcpy(new_prob->variable_upper_bound, prob->variable_upper_bound, var_bytes);
    memcpy(new_prob->objective_vector, prob->objective_vector, var_bytes);
    memcpy(new_prob->constraint_lower_bound, prob->constraint_lower_bound, con_bytes);
    memcpy(new_prob->constraint_upper_bound, prob->constraint_upper_bound, con_bytes);
    memcpy(new_prob->affine_cone_offset, prob->affine_cone_offset, con_bytes);
    new_prob->constraint_matrix =
        deepcopy_csr_component(prob->constraint_matrix, prob->num_constraints, prob->constraint_matrix_num_nonzeros);
    new_prob->objective_sparse_matrix = deepcopy_csr_component(
        prob->objective_sparse_matrix, prob->num_variables, prob->objective_sparse_matrix_num_nonzeros);
    new_prob->objective_lowrank_matrix = deepcopy_csr_component(
        prob->objective_lowrank_matrix, prob->num_rank_lowrank_obj, prob->objective_lowrank_matrix_num_nonzeros);

    new_prob->objective_lowrank_middle_matrix =
        deepcopy_csr_component(prob->objective_lowrank_middle_matrix,
                               prob->num_rank_lowrank_obj,
                               prob->objective_lowrank_middle_matrix_num_nonzeros);
    new_prob->objective_lowrank_middle_matrix_num_nonzeros = prob->objective_lowrank_middle_matrix_num_nonzeros;

    if (prob->primal_start)
    {
        new_prob->primal_start = safe_malloc(var_bytes);
        memcpy(new_prob->primal_start, prob->primal_start, var_bytes);
    }
    if (prob->dual_start)
    {
        new_prob->dual_start = safe_malloc(con_bytes);
        memcpy(new_prob->dual_start, prob->dual_start, con_bytes);
    }

    new_prob->num_quadratic_constraints = prob->num_quadratic_constraints;
    if (prob->num_quadratic_constraints > 0)
    {
        int K = prob->num_quadratic_constraints;
        new_prob->quadratic_constraint_row_indices = safe_malloc(K * sizeof(int));
        new_prob->quadratic_constraint_matrix_num_nonzeros = safe_malloc(K * sizeof(int));
        new_prob->quadratic_constraint_matrices = safe_malloc(K * sizeof(CsrComponent *));
        memcpy(new_prob->quadratic_constraint_row_indices, prob->quadratic_constraint_row_indices, K * sizeof(int));
        memcpy(new_prob->quadratic_constraint_matrix_num_nonzeros,
               prob->quadratic_constraint_matrix_num_nonzeros,
               K * sizeof(int));
        for (int i = 0; i < K; ++i)
        {
            new_prob->quadratic_constraint_matrices[i] =
                deepcopy_csr_component(prob->quadratic_constraint_matrices[i],
                                       prob->num_variables,
                                       prob->quadratic_constraint_matrix_num_nonzeros[i]);
        }
    }

    new_prob->num_original_variables = prob->num_original_variables;
    cone_blocks_clone(&new_prob->cones, &prob->cones);
    cone_blocks_clone(&new_prob->affine_cones, &prob->affine_cones);

    return new_prob;
}

static void scale_problem(qp_problem_t *problem, const double *constraint_rescaling, const double *variable_rescaling)
{
    for (int i = 0; i < problem->num_variables; ++i)
    {
        problem->objective_vector[i] /= variable_rescaling[i];
        problem->variable_upper_bound[i] *= variable_rescaling[i];
        problem->variable_lower_bound[i] *= variable_rescaling[i];
    }
    for (int i = 0; i < problem->num_constraints; ++i)
    {
        problem->constraint_lower_bound[i] /= constraint_rescaling[i];
        problem->constraint_upper_bound[i] /= constraint_rescaling[i];
        problem->affine_cone_offset[i] /= constraint_rescaling[i];
    }

    for (int row = 0; row < problem->num_constraints; ++row)
    {
        for (int nz_idx = problem->constraint_matrix->row_ptr[row];
             nz_idx < problem->constraint_matrix->row_ptr[row + 1];
             ++nz_idx)
        {
            int col = problem->constraint_matrix->col_ind[nz_idx];
            problem->constraint_matrix->val[nz_idx] /= (constraint_rescaling[row] * variable_rescaling[col]);
        }
    }

    if (problem->objective_sparse_matrix)
    {
        for (int q_row = 0; q_row < problem->num_variables; ++q_row)
        {
            for (int nz_idx = problem->objective_sparse_matrix->row_ptr[q_row];
                 nz_idx < problem->objective_sparse_matrix->row_ptr[q_row + 1];
                 ++nz_idx)
            {
                int q_col = problem->objective_sparse_matrix->col_ind[nz_idx];
                problem->objective_sparse_matrix->val[nz_idx] /=
                    (variable_rescaling[q_row] * variable_rescaling[q_col]);
            }
        }
    }
    if (problem->objective_lowrank_matrix_num_nonzeros > 0)
    {
        for (int r = 0; r < problem->num_rank_lowrank_obj; ++r)
        {
            for (int nz_idx = problem->objective_lowrank_matrix->row_ptr[r];
                 nz_idx < problem->objective_lowrank_matrix->row_ptr[r + 1];
                 ++nz_idx)
            {
                int col = problem->objective_lowrank_matrix->col_ind[nz_idx];
                problem->objective_lowrank_matrix->val[nz_idx] /= variable_rescaling[col];
            }
        }
    }
}

static void pin_scaled_fixed_cone_bounds(qp_problem_t *scaled_problem,
                                         const qp_problem_t *source_problem,
                                         const rescale_info_t *rescale_info)
{
    if (!source_problem->cones.is_fixed)
        return;

    for (int variable = 0; variable < source_problem->num_variables; ++variable)
    {
        if (!source_problem->cones.is_fixed[variable])
            continue;

        double value = source_problem->primal_start ? source_problem->primal_start[variable] : 0.0;
        double scaled_value = value * rescale_info->var_rescale[variable] * rescale_info->con_bound_rescale;
        scaled_problem->variable_lower_bound[variable] = scaled_value;
        scaled_problem->variable_upper_bound[variable] = scaled_value;
    }
}

/*
 * A. R. Curtis and J. K. Reid, "On the Automatic Scaling of Matrices
 * for Gaussian Elimination", IMA J. Appl. Math. 10(1), 118-124 (1972).
 */
static void curtis_reid_rescaling(qp_problem_t *problem,
                                  int num_iterations,
                                  bool use_cone_preserving_scaling,
                                  double *cum_constraint_rescaling,
                                  double *cum_variable_rescaling)
{
    const int num_cons = problem->num_constraints;
    const int num_vars = problem->num_variables;
    const int num_nonzeros = problem->constraint_matrix_num_nonzeros;
    double *con_rescale = safe_malloc((size_t)num_cons * sizeof(double));
    double *var_rescale = safe_malloc((size_t)num_vars * sizeof(double));

    for (int row = 0; row < num_cons; ++row)
        con_rescale[row] = 1.0;
    for (int col = 0; col < num_vars; ++col)
        var_rescale[col] = 1.0;

    if (num_iterations > 0 && num_cons > 0 && num_vars > 0 && num_nonzeros > 0)
    {
        const CsrComponent *matrix = problem->constraint_matrix;
        double *row_log_scale = safe_calloc((size_t)num_cons, sizeof(double));
        double *col_log_scale = safe_calloc((size_t)num_vars, sizeof(double));
        double *row_log_abs_sum = safe_calloc((size_t)num_cons, sizeof(double));
        double *col_log_abs_sum = safe_calloc((size_t)num_vars, sizeof(double));
        double *col_sum = safe_calloc((size_t)num_vars, sizeof(double));
        int *col_count = safe_calloc((size_t)num_vars, sizeof(int));

        for (int row = 0; row < num_cons; ++row)
        {
            for (int nz = matrix->row_ptr[row]; nz < matrix->row_ptr[row + 1]; ++nz)
            {
                const int col = matrix->col_ind[nz];
                const double log_abs = log(fmax(fabs(matrix->val[nz]), CURTIS_REID_MIN_ABS));
                row_log_abs_sum[row] += log_abs;
                col_log_abs_sum[col] += log_abs;
                ++col_count[col];
            }
        }

        /*
         * Minimize sum_(i,j) in nz(A) (log|A_ij| - r_i - c_j)^2 by
         * alternating exact row and column least-squares updates.
         */
        for (int iter = 0; iter < num_iterations; ++iter)
        {
            for (int row = 0; row < num_cons; ++row)
            {
                const int begin = matrix->row_ptr[row];
                const int end = matrix->row_ptr[row + 1];
                double sum = row_log_abs_sum[row];
                for (int nz = begin; nz < end; ++nz)
                    sum -= col_log_scale[matrix->col_ind[nz]];
                row_log_scale[row] = (end > begin) ? sum / (double)(end - begin) : 0.0;
            }

            if (use_cone_preserving_scaling)
            {
                for (int block = 0; block < problem->affine_cones.num_cones; ++block)
                {
                    int start = problem->affine_cones.start_idx[block];
                    int length = cone_block_length(&problem->affine_cones, block);
                    double block_sum = 0.0;
                    int block_count = 0;
                    for (int row = start; row < start + length; ++row)
                    {
                        int begin = matrix->row_ptr[row];
                        int end = matrix->row_ptr[row + 1];
                        block_sum += row_log_abs_sum[row];
                        block_count += end - begin;
                        for (int nz = begin; nz < end; ++nz)
                            block_sum -= col_log_scale[matrix->col_ind[nz]];
                    }
                    double block_log_scale = block_count > 0 ? block_sum / (double)block_count : 0.0;
                    for (int row = start; row < start + length; ++row)
                        row_log_scale[row] = block_log_scale;
                }
            }

            memcpy(col_sum, col_log_abs_sum, (size_t)num_vars * sizeof(double));
            for (int row = 0; row < num_cons; ++row)
            {
                for (int nz = matrix->row_ptr[row]; nz < matrix->row_ptr[row + 1]; ++nz)
                    col_sum[matrix->col_ind[nz]] -= row_log_scale[row];
            }
            for (int col = 0; col < num_vars; ++col)
                col_log_scale[col] = col_count[col] > 0 ? col_sum[col] / (double)col_count[col] : 0.0;

            if (use_cone_preserving_scaling)
            {
                /*
                 * Adding c_j = c_B for all j in cone block B gives the exact
                 * block minimizer below. With cone-preserving scaling disabled,
                 * the independent column minimizers above are retained.
                 */
                for (int block = 0; block < problem->cones.num_cones; ++block)
                {
                    const int start = problem->cones.start_idx[block];
                    const int length = cone_block_length(&problem->cones, block);
                    double block_sum = 0.0;
                    int block_count = 0;
                    for (int col = start; col < start + length; ++col)
                    {
                        block_sum += col_sum[col];
                        block_count += col_count[col];
                    }
                    const double block_log_scale = block_count > 0 ? block_sum / (double)block_count : 0.0;
                    for (int col = start; col < start + length; ++col)
                        col_log_scale[col] = block_log_scale;
                }
            }
        }

        for (int row = 0; row < num_cons; ++row)
            con_rescale[row] = curtis_reid_exp_clamped(row_log_scale[row]);
        for (int col = 0; col < num_vars; ++col)
            var_rescale[col] = curtis_reid_exp_clamped(col_log_scale[col]);

        free(row_log_scale);
        free(col_log_scale);
        free(row_log_abs_sum);
        free(col_log_abs_sum);
        free(col_sum);
        free(col_count);
    }

    scale_problem(problem, con_rescale, var_rescale);

    for (int row = 0; row < num_cons; ++row)
        cum_constraint_rescaling[row] *= con_rescale[row];
    for (int col = 0; col < num_vars; ++col)
        cum_variable_rescaling[col] *= var_rescale[col];

    free(con_rescale);
    free(var_rescale);
}

static void ruiz_rescaling(qp_problem_t *problem,
                           int num_iterations,
                           bool use_cone_preserving_scaling,
                           double *cum_constraint_rescaling,
                           double *cum_variable_rescaling)
{
    int num_cons = problem->num_constraints;
    int num_vars = problem->num_variables;
    double *con_rescale = safe_malloc(num_cons * sizeof(double));
    double *var_rescale = safe_malloc(num_vars * sizeof(double));

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        for (int i = 0; i < num_vars; ++i)
            var_rescale[i] = 0.0;
        for (int i = 0; i < num_cons; ++i)
            con_rescale[i] = 0.0;

        for (int row = 0; row < num_cons; ++row)
        {
            for (int nz_idx = problem->constraint_matrix->row_ptr[row];
                 nz_idx < problem->constraint_matrix->row_ptr[row + 1];
                 ++nz_idx)
            {
                int col = problem->constraint_matrix->col_ind[nz_idx];
                if (col < 0 || col >= num_vars)
                {
                    fprintf(stderr,
                            "Error: Invalid column index %d at nz_idx %d for row %d. "
                            "Must be in [0, %d).\n",
                            col,
                            nz_idx,
                            row,
                            num_vars);
                }
                double val = fabs(problem->constraint_matrix->val[nz_idx]);
                if (val > var_rescale[col])
                    var_rescale[col] = val;
                if (val > con_rescale[row])
                    con_rescale[row] = val;
            }
        }
        for (int i = 0; i < num_vars; ++i)
            var_rescale[i] = (var_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(var_rescale[i]);
        for (int i = 0; i < num_cons; ++i)
            con_rescale[i] = (con_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(con_rescale[i]);

        if (use_cone_preserving_scaling)
        {
            apply_cone_preserving_scaling(var_rescale, &problem->cones, CONE_SCALING_RUIZ);
            apply_cone_preserving_scaling(con_rescale, &problem->affine_cones, CONE_SCALING_RUIZ);
        }

        scale_problem(problem, con_rescale, var_rescale);
        for (int i = 0; i < num_vars; ++i)
            cum_variable_rescaling[i] *= var_rescale[i];
        for (int i = 0; i < num_cons; ++i)
            cum_constraint_rescaling[i] *= con_rescale[i];
    }
    free(con_rescale);
    free(var_rescale);
}

static void pock_chambolle_rescaling(qp_problem_t *problem,
                                     double alpha,
                                     bool use_cone_preserving_scaling,
                                     double *cum_constraint_rescaling,
                                     double *cum_variable_rescaling)
{
    int num_cons = problem->num_constraints;
    int num_vars = problem->num_variables;
    double *con_rescale = safe_calloc(num_cons, sizeof(double));
    double *var_rescale = safe_calloc(num_vars, sizeof(double));

    for (int row = 0; row < num_cons; ++row)
    {
        for (int nz_idx = problem->constraint_matrix->row_ptr[row];
             nz_idx < problem->constraint_matrix->row_ptr[row + 1];
             ++nz_idx)
        {
            int col = problem->constraint_matrix->col_ind[nz_idx];
            double val = fabs(problem->constraint_matrix->val[nz_idx]);
            var_rescale[col] += pow(val, 2.0 - alpha);
            con_rescale[row] += pow(val, alpha);
        }
    }

    for (int i = 0; i < num_vars; ++i)
        var_rescale[i] = (var_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(var_rescale[i]);
    for (int i = 0; i < num_cons; ++i)
        con_rescale[i] = (con_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(con_rescale[i]);

    if (use_cone_preserving_scaling)
    {
        apply_cone_preserving_scaling(var_rescale, &problem->cones, CONE_SCALING_POCK_CHAMBOLLE);
        apply_cone_preserving_scaling(con_rescale, &problem->affine_cones, CONE_SCALING_POCK_CHAMBOLLE);
    }

    scale_problem(problem, con_rescale, var_rescale);
    for (int i = 0; i < num_vars; ++i)
        cum_variable_rescaling[i] *= var_rescale[i];
    for (int i = 0; i < num_cons; ++i)
        cum_constraint_rescaling[i] *= con_rescale[i];

    free(con_rescale);
    free(var_rescale);
}

static void bound_obj_rescaling(qp_problem_t *problem, rescale_info_t *rescale_info)
{
    double b_norm_sq = 0.0;
    for (int i = 0; i < problem->num_constraints; ++i)
    {
        if (isfinite(problem->constraint_lower_bound[i]) &&
            (problem->constraint_lower_bound[i] != problem->constraint_upper_bound[i]))
        {
            b_norm_sq += problem->constraint_lower_bound[i] * problem->constraint_lower_bound[i];
        }
        if (isfinite(problem->constraint_upper_bound[i]))
        {
            b_norm_sq += problem->constraint_upper_bound[i] * problem->constraint_upper_bound[i];
        }
    }
    for (int i = 0; i < problem->num_constraints; ++i)
        b_norm_sq += problem->affine_cone_offset[i] * problem->affine_cone_offset[i];
    double c_norm_sq = 0.0;
    for (int i = 0; i < problem->num_variables; ++i)
    {
        c_norm_sq += problem->objective_vector[i] * problem->objective_vector[i];
    }
    rescale_info->con_bound_rescale = 1.0 / (sqrt(b_norm_sq) + 1.0);
    rescale_info->obj_vec_rescale = 1.0 / (sqrt(c_norm_sq) + 1.0);

    for (int i = 0; i < problem->num_constraints; ++i)
    {
        problem->constraint_lower_bound[i] *= rescale_info->con_bound_rescale;
        problem->constraint_upper_bound[i] *= rescale_info->con_bound_rescale;
        problem->affine_cone_offset[i] *= rescale_info->con_bound_rescale;
    }
    for (int i = 0; i < problem->num_variables; ++i)
    {
        problem->variable_lower_bound[i] *= rescale_info->con_bound_rescale;
        problem->variable_upper_bound[i] *= rescale_info->con_bound_rescale;
        problem->objective_vector[i] *= rescale_info->obj_vec_rescale;
    }
    for (int nnz_idx = 0; nnz_idx < problem->objective_sparse_matrix_num_nonzeros; ++nnz_idx)
    {
        problem->objective_sparse_matrix->val[nnz_idx] *=
            rescale_info->obj_vec_rescale / rescale_info->con_bound_rescale;
    }
    if (problem->objective_lowrank_matrix_num_nonzeros > 0)
    {
        double R_scale_factor = sqrt(rescale_info->obj_vec_rescale / rescale_info->con_bound_rescale);
        for (int nnz_idx = 0; nnz_idx < problem->objective_lowrank_matrix_num_nonzeros; ++nnz_idx)
        {
            problem->objective_lowrank_matrix->val[nnz_idx] *= R_scale_factor;
        }
    }
}

rescale_info_t *rescale_problem(const pdhg_parameters_t *params, const qp_problem_t *working_problem)
{
    clock_t start_rescaling = clock();
    rescale_info_t *rescale_info = (rescale_info_t *)safe_calloc(1, sizeof(rescale_info_t));
    rescale_info->scaled_problem = deepcopy_problem(working_problem);
    if (rescale_info->scaled_problem == NULL)
    {
        fprintf(stderr, "Failed to create a copy of the problem. Aborting rescale.\n");
        return NULL;
    }
    int num_cons = working_problem->num_constraints;
    int num_vars = working_problem->num_variables;

    rescale_info->con_rescale = safe_malloc(num_cons * sizeof(double));
    rescale_info->var_rescale = safe_malloc(num_vars * sizeof(double));

    for (int i = 0; i < num_cons; ++i)
        rescale_info->con_rescale[i] = 1.0;
    for (int i = 0; i < num_vars; ++i)
        rescale_info->var_rescale[i] = 1.0;

    bool use_cone_preserving_scaling = params->use_cone_preserving_scaling;
    if (params->curtis_reid_iterations > 0)
    {
        curtis_reid_rescaling(rescale_info->scaled_problem,
                              params->curtis_reid_iterations,
                              use_cone_preserving_scaling,
                              rescale_info->con_rescale,
                              rescale_info->var_rescale);
    }
    if (params->l_inf_ruiz_iterations > 0)
    {
        ruiz_rescaling(rescale_info->scaled_problem,
                       params->l_inf_ruiz_iterations,
                       use_cone_preserving_scaling,
                       rescale_info->con_rescale,
                       rescale_info->var_rescale);
    }
    if (params->has_pock_chambolle_alpha)
    {
        pock_chambolle_rescaling(rescale_info->scaled_problem,
                                 params->pock_chambolle_alpha,
                                 use_cone_preserving_scaling,
                                 rescale_info->con_rescale,
                                 rescale_info->var_rescale);
    }
    if (params->bound_objective_rescaling)
    {
        bound_obj_rescaling(rescale_info->scaled_problem, rescale_info);
    }
    else
    {
        rescale_info->con_bound_rescale = 1.0;
        rescale_info->obj_vec_rescale = 1.0;
    }
    pin_scaled_fixed_cone_bounds(rescale_info->scaled_problem, working_problem, rescale_info);
    rescale_info->processed_problem = preprocess_qp_problem(rescale_info->scaled_problem);
    rescale_info->rescaling_time_sec = (double)(clock() - start_rescaling) / CLOCKS_PER_SEC;
    return rescale_info;
}

processed_qp_problem_t *preprocess_qp_problem(const qp_problem_t *raw_problem)
{
    if (raw_problem == NULL)
        return NULL;

    processed_qp_problem_t *processed = (processed_qp_problem_t *)safe_calloc(1, sizeof(processed_qp_problem_t));

    processed->num_variables = raw_problem->num_variables;
    processed->num_constraints = raw_problem->num_constraints;
    processed->num_rank_lowrank_obj = raw_problem->num_rank_lowrank_obj;
    processed->objective_constant = raw_problem->objective_constant;

    processed->constraint_matrix_num_nonzeros = raw_problem->constraint_matrix_num_nonzeros;
    processed->objective_sparse_matrix_num_nonzeros = raw_problem->objective_sparse_matrix_num_nonzeros;
    processed->objective_lowrank_matrix_num_nonzeros = raw_problem->objective_lowrank_matrix_num_nonzeros;

    processed->variable_lower_bound = raw_problem->variable_lower_bound;
    processed->variable_upper_bound = raw_problem->variable_upper_bound;
    processed->objective_vector = raw_problem->objective_vector;
    processed->constraint_lower_bound = raw_problem->constraint_lower_bound;
    processed->constraint_upper_bound = raw_problem->constraint_upper_bound;
    processed->primal_start = raw_problem->primal_start;
    processed->dual_start = raw_problem->dual_start;
    processed->constraint_matrix = raw_problem->constraint_matrix;
    processed->objective_sparse_matrix = raw_problem->objective_sparse_matrix;
    processed->objective_lowrank_matrix = raw_problem->objective_lowrank_matrix;

    processed->objective_lowrank_middle_kind = PDHCG_D_NONE;
    processed->objective_lowrank_middle_diag = NULL;
    processed->objective_lowrank_middle_dense = NULL;
    {
        CsrComponent *D_csr = raw_problem->objective_lowrank_middle_matrix;
        int D_nnz = raw_problem->objective_lowrank_middle_matrix_num_nonzeros;
        int k = raw_problem->num_rank_lowrank_obj;
        if (D_csr && D_csr->row_ptr && D_nnz > 0 && k > 0)
        {
            bool diag_only = true;
            for (int r = 0; r < k && diag_only; ++r)
            {
                for (int j = D_csr->row_ptr[r]; j < D_csr->row_ptr[r + 1]; ++j)
                {
                    if (D_csr->col_ind[j] != r)
                    {
                        diag_only = false;
                        break;
                    }
                }
            }

            if (diag_only)
            {
                processed->objective_lowrank_middle_kind = PDHCG_D_DIAG;
                processed->objective_lowrank_middle_diag = (double *)safe_calloc(k, sizeof(double));
                for (int r = 0; r < k; ++r)
                {
                    for (int j = D_csr->row_ptr[r]; j < D_csr->row_ptr[r + 1]; ++j)
                    {
                        processed->objective_lowrank_middle_diag[r] = D_csr->val[j];
                    }
                }
            }
            else
            {
                processed->objective_lowrank_middle_kind = PDHCG_D_DENSE;
                size_t rank2 = (size_t)k * (size_t)k;
                processed->objective_lowrank_middle_dense = (double *)safe_calloc(rank2, sizeof(double));
                for (int r = 0; r < k; ++r)
                {
                    for (int j = D_csr->row_ptr[r]; j < D_csr->row_ptr[r + 1]; ++j)
                    {
                        int c = D_csr->col_ind[j];
                        processed->objective_lowrank_middle_dense[(size_t)r * (size_t)k + (size_t)c] = D_csr->val[j];
                    }
                }
            }
        }
    }

    if ((!raw_problem->objective_sparse_matrix || raw_problem->objective_sparse_matrix_num_nonzeros == 0) &&
        (!raw_problem->objective_lowrank_matrix || raw_problem->objective_lowrank_matrix_num_nonzeros == 0))
    {
        processed->quad_type = PDHCG_NON_Q;
    }
    else
    {
        processed->quad_type = detect_q_type(raw_problem->objective_sparse_matrix,
                                             raw_problem->objective_lowrank_matrix,
                                             raw_problem->num_variables,
                                             raw_problem->num_rank_lowrank_obj);
    }

    processed->diagonal_quad_objective = NULL;
    if (processed->quad_type == PDHCG_DIAG_Q)
    {
        int n = processed->num_variables;
        processed->diagonal_quad_objective = (double *)safe_calloc(n, sizeof(double));

        CsrComponent *csr = processed->objective_sparse_matrix;
        if (csr && csr->row_ptr && csr->col_ind && csr->val)
        {
            for (int i = 0; i < n; ++i)
            {
                for (int k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; ++k)
                {
                    int col = csr->col_ind[k];
                    if (col < n)
                    {
                        processed->diagonal_quad_objective[col] = csr->val[k] + 1e-12;
                    }
                }
            }
        }
    }

    return processed;
}

void free_processed_qp_problem(processed_qp_problem_t *processed)
{
    if (processed == NULL)
        return;

    if (processed->diagonal_quad_objective != NULL)
    {
        free(processed->diagonal_quad_objective);
        processed->diagonal_quad_objective = NULL;
    }

    free(processed->objective_lowrank_middle_diag);
    processed->objective_lowrank_middle_diag = NULL;
    free(processed->objective_lowrank_middle_dense);
    processed->objective_lowrank_middle_dense = NULL;

    free(processed);
}
