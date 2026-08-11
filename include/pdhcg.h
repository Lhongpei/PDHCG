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

#pragma once

#include "pdhcg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* Pass NULL for any optional matrix descriptor (defaults: Q=0, R=0, D=I).
       A models scalar rows con_lb <= A*x <= con_ub. affine_cone_matrix_desc
       models F*x + affine_cone_offset in K; affine cone indices refer to rows
       of F, which must be fully covered by the supplied cone blocks. */
    qp_problem_t *create_qp_problem(const double *objective_c,
                                    const matrix_desc_t *Q_desc,
                                    const matrix_desc_t *R_desc,
                                    const matrix_desc_t *D_desc,
                                    const matrix_desc_t *A_desc,
                                    const double *con_lb,
                                    const double *con_ub,
                                    const double *var_lb,
                                    const double *var_ub,
                                    const double *objective_constant,
                                    int num_var_cones,
                                    const cone_spec_t *var_cones,
                                    const matrix_desc_t *affine_cone_matrix_desc,
                                    const double *affine_cone_offset,
                                    int num_affine_cones,
                                    const cone_spec_t *affine_cones);

    /* dual has rows(A) + rows(F) entries ordered as [dual_A, dual_F]. */
    void set_start_values(qp_problem_t *prob, const double *primal, const double *dual);

    int set_cone_fixed(qp_problem_t *prob, int cone_idx, int slot, double value);

    qp_problem_t *qcqp_to_socp_qp(const qp_problem_t *orig_qcqp, cone_type_t default_type);

    // solve the LP problem using PDHG
    pdhcg_result_t *solve_qp_problem(const qp_problem_t *prob, const pdhg_parameters_t *params);

    // solve the QP problem using distributed multi-GPU PDHG
    pdhcg_result_t *solve_qp_problem_distributed(const pdhg_parameters_t *params, const qp_problem_t *original_problem);

    // parameter
    void set_default_parameters(pdhg_parameters_t *params);

    void pdhcg_result_free(pdhcg_result_t *results);

    void qp_problem_free(qp_problem_t *prob);

#ifdef __cplusplus
} // extern "C"
#endif
