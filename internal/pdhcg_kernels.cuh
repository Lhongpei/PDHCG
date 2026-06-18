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

#ifndef PDHCG_KERNELS_CUH
#define PDHCG_KERNELS_CUH

#include <cuda_runtime.h>
#include <math.h>

#ifdef __cplusplus
extern "C"
{
#endif
    // ======================================================================
    // Utility Operations
    // ======================================================================

    __global__ void
    element_wise_mul_kernel(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int n);

    __global__ void element_wise_mul_inplace_kernel(double *__restrict__ x, const double *__restrict__ d, int n);

    __global__ void compute_diag_q_obj_and_grad_kernel(const double *__restrict__ Q_diag,
                                                       const double *__restrict__ x,
                                                       const double *__restrict__ c,
                                                       double *__restrict__ Qx,
                                                       double *__restrict__ obj_grad,
                                                       int n);

    // ======================================================================
    // Advanced Metrics & Reduced Costs
    // ======================================================================

    __global__ void compute_and_rescale_reduced_cost_kernel(double *reduced_cost,
                                                            const double *objective,
                                                            const double *dual_product,
                                                            const double *variable_rescaling,
                                                            const double objective_vector_rescaling,
                                                            const double constraint_bound_rescaling,
                                                            int n_vars);

    // ======================================================================
    // Primal Updates
    // ======================================================================

    __global__ void compute_lp_next_pdhg_primal_solution_kernel(const double *current_primal,
                                                                double *reflected_primal,
                                                                const double *dual_product,
                                                                const double *objective,
                                                                const double *var_lb,
                                                                const double *var_ub,
                                                                int n,
                                                                double step_size);

    __global__ void compute_lp_next_pdhg_primal_solution_major_kernel(const double *current_primal,
                                                                      double *pdhg_primal,
                                                                      double *reflected_primal,
                                                                      const double *dual_product,
                                                                      const double *objective,
                                                                      const double *var_lb,
                                                                      const double *var_ub,
                                                                      int n,
                                                                      double step_size,
                                                                      double *dual_slack);

    __global__ void compute_diagonal_q_next_pdhg_primal_solution_kernel(const double *current_primal,
                                                                        double *reflected_primal,
                                                                        double *objective_product,
                                                                        const double *dual_product,
                                                                        const double *objective,
                                                                        const double *var_lb,
                                                                        const double *var_ub,
                                                                        int n,
                                                                        double step_size);

    __global__ void compute_diagonal_q_next_pdhg_primal_solution_major_kernel(const double *current_primal,
                                                                              double *pdhg_primal,
                                                                              double *reflected_primal,
                                                                              double *objective_product,
                                                                              const double *dual_product,
                                                                              const double *objective,
                                                                              const double *var_lb,
                                                                              const double *var_ub,
                                                                              int n,
                                                                              double step_size);

    // ======================================================================
    // Dual Updates
    // ======================================================================

    __global__ void compute_next_pdhg_dual_solution_kernel(const double *current_dual,
                                                           double *reflected_dual,
                                                           const double *primal_product,
                                                           const double *const_lb,
                                                           const double *const_ub,
                                                           int n,
                                                           double step_size);

    __global__ void compute_next_pdhg_dual_solution_major_kernel(const double *current_dual,
                                                                 double *pdhg_dual,
                                                                 double *reflected_dual,
                                                                 const double *primal_product,
                                                                 const double *const_lb,
                                                                 const double *const_ub,
                                                                 int n,
                                                                 double step_size);

    // ======================================================================
    // Halpern & Solution Management
    // ======================================================================

    __global__ void halpern_update_kernel(const double *initial_primal,
                                          double *current_primal,
                                          const double *reflected_primal,
                                          const double *initial_dual,
                                          double *current_dual,
                                          const double *reflected_dual,
                                          int n_vars,
                                          int n_cons,
                                          double weight,
                                          double reflection_coeff);

    __global__ void rescale_solution_kernel(double *primal_solution,
                                            double *dual_solution,
                                            const double *variable_rescaling,
                                            const double *constraint_rescaling,
                                            const double objective_vector_rescaling,
                                            const double constraint_bound_rescaling,
                                            int n_vars,
                                            int n_cons);

    __global__ void compute_delta_solution_kernel(const double *initial_primal,
                                                  const double *pdhg_primal,
                                                  double *delta_primal,
                                                  const double *initial_dual,
                                                  const double *pdhg_dual,
                                                  double *delta_dual,
                                                  int n_vars,
                                                  int n_cons);

    // ======================================================================
    // Primal Inner Solver (Gradient Descent & Barzilai-Borwein)
    // ======================================================================

    __global__ void primal_gradient_descent_kernel(const double *dual_product,
                                                   const double *current_primal_solution,
                                                   double *reflected_primal,
                                                   const double *objective_vector,
                                                   const double *objective_product,
                                                   const double *var_lb,
                                                   const double *var_ub,
                                                   const double stepsize,
                                                   const int n_vars);

    __global__ void primal_gradient_descent_kernel_major(const double *dual_product,
                                                         const double *current_primal_solution,
                                                         double *reflected_primal,
                                                         double *pdhg_primal_solution,
                                                         const double *objective_vector,
                                                         const double *objective_product,
                                                         const double *var_lb,
                                                         const double *var_ub,
                                                         const double stepsize,
                                                         const int n_vars);

    __global__ void compute_bb_alpha_safeguard_kernel(const double *d_norm_gtg, const double *d_tmp, double *d_alpha);

    __global__ void compute_bb_alpha_M_kernel(const double *d_stMs, const double *d_tmp, double *d_alpha);

    __global__ void scalar_sqrt_copy_kernel(const double *src, double *dst);

    __global__ void
    compute_csr_diag_kernel(const int *row_ptr, const int *col_ind, const double *val, double *diag, int num_rows);

    __global__ void compute_csr_row_sq_norm_kernel(const int *row_ptr, const double *val, double *out, int num_rows);

    __global__ void compute_csr_row_sq_norm_weighted_kernel(
        const int *row_ptr, const int *col_ind, const double *val, const double *weights, double *out, int num_rows);

    __global__ void compute_csr_row_quad_form_dense_kernel(const int *row_ptr,
                                                           const int *col_ind,
                                                           const double *val,
                                                           const double *D_dense,
                                                           int rank,
                                                           double *out,
                                                           int num_rows);

    __global__ void refresh_inner_precond_kernel(
        const double *diag_h_static, double inv_tau, double *m_diag, double *m_inv, int n_vars);

    __global__ void primal_gradient_descent_kernel_bb_init(const double *dual_product,
                                                           double *gradient,
                                                           double *direction,
                                                           const double *current_primal_solution,
                                                           double *pdhg_primal_solution,
                                                           const double *objective_vector,
                                                           const double *objective_product,
                                                           const double *var_lb,
                                                           const double *var_ub,
                                                           const double stepsize,
                                                           const int n_vars);

    __global__ void primal_bb_update_gradient_kernel(double *pdhg_primal_solution,
                                                     const double *current_primal_solution,
                                                     const double *objective_vector,
                                                     const double *dual_product,
                                                     const double *objective_product,
                                                     double *gradient,
                                                     double *delta_gradient,
                                                     const double inv_step_size,
                                                     const int n_vars);

    __global__ void primal_bb_update_direction_kernel(double *pdhg_primal_solution,
                                                      const double *gradient,
                                                      double *direction,
                                                      const double *var_lb,
                                                      const double *var_ub,
                                                      const double *d_alpha,
                                                      const int n_vars);

    __global__ void primal_gradient_descent_kernel_bb_init_precond(const double *dual_product,
                                                                   double *gradient,
                                                                   double *direction,
                                                                   const double *current_primal_solution,
                                                                   double *pdhg_primal_solution,
                                                                   const double *objective_vector,
                                                                   const double *objective_product,
                                                                   const double *var_lb,
                                                                   const double *var_ub,
                                                                   const double *m_inv,
                                                                   const double stepsize,
                                                                   const int n_vars);

    __global__ void primal_bb_update_direction_kernel_precond(double *pdhg_primal_solution,
                                                              const double *gradient,
                                                              double *direction,
                                                              const double *var_lb,
                                                              const double *var_ub,
                                                              const double *m_inv,
                                                              const double *d_alpha,
                                                              const int n_vars);

    __global__ void primal_bb_final_kernel(const double *current_primal_solution,
                                           const double *pdhg_primal_solution,
                                           double *reflected_primal_solution,
                                           const int n_vars);

    // ======================================================================
    // Residuals & Infeasibility Metrics
    // ======================================================================

    __global__ void compute_lp_residual_kernel(double *primal_residual,
                                               const double *primal_product,
                                               const double *constraint_lower_bound,
                                               const double *constraint_upper_bound,
                                               const double *dual_solution,
                                               double *dual_residual,
                                               const double *dual_product,
                                               const double *dual_slack,
                                               const double *objective_vector,
                                               const double *constraint_rescaling,
                                               const double *variable_rescaling,
                                               double *dual_obj_contribution,
                                               const double *const_lb_finite,
                                               const double *const_ub_finite,
                                               int num_constraints,
                                               int num_variables);

    __global__ void compute_qp_residual_kernel(double *primal_residual,
                                               const double *primal_product,
                                               const double *primal_obj_product,
                                               const double *primal_solution,
                                               const double *constraint_lower_bound,
                                               const double *constraint_upper_bound,
                                               const double *variable_lower_bound,
                                               const double *variable_upper_bound,
                                               const double *dual_solution,
                                               double *dual_residual,
                                               const double *dual_product,
                                               double *dual_slack,
                                               const double *objective_vector,
                                               const double *constraint_rescaling,
                                               const double *variable_rescaling,
                                               double *dual_obj_contribution,
                                               const double *const_lb_finite,
                                               const double *const_ub_finite,
                                               const double step_size,
                                               int num_constraints,
                                               int num_variables);

    __global__ void recover_primal_obj_dual_product(double *dual_product,
                                                    double *primal_obj_product,
                                                    const double *variable_rescaling,
                                                    int num_variables);

    __global__ void primal_infeasibility_project_kernel(double *primal_ray_estimate,
                                                        const double *variable_lower_bound,
                                                        const double *variable_upper_bound,
                                                        int num_variables);

    __global__ void dual_infeasibility_project_kernel(double *dual_ray_estimate,
                                                      const double *constraint_lower_bound,
                                                      const double *constraint_upper_bound,
                                                      int num_constraints);

    __global__ void compute_primal_infeasibility_kernel(const double *primal_product,
                                                        const double *const_lb,
                                                        const double *const_ub,
                                                        int num_constraints,
                                                        double *primal_infeasibility,
                                                        const double *constraint_rescaling);

    __global__ void compute_dual_infeasibility_kernel(const double *dual_product,
                                                      const double *var_lb,
                                                      const double *var_ub,
                                                      int num_variables,
                                                      double *dual_infeasibility,
                                                      const double *variable_rescaling);

    __global__ void
    dual_solution_dual_objective_contribution_kernel(const double *constraint_lower_bound_finite_val,
                                                     const double *constraint_upper_bound_finite_val,
                                                     const double *dual_solution,
                                                     int num_constraints,
                                                     double *dual_objective_dual_solution_contribution_array);

    __global__ void
    dual_objective_dual_slack_contribution_array_kernel(const double *dual_slack,
                                                        double *dual_objective_dual_slack_contribution_array,
                                                        const double *variable_lower_bound_finite_val,
                                                        const double *variable_upper_bound_finite_val,
                                                        int num_variables);

    __global__ void compute_and_rescale_reduced_cost_qp_kernel(double *__restrict__ reduced_cost,
                                                               const double *__restrict__ objective,
                                                               const double *__restrict__ quadratic_product,
                                                               const double *__restrict__ dual_product,
                                                               const double *__restrict__ variable_rescaling,
                                                               const double objective_vector_rescaling,
                                                               const double constraint_bound_rescaling,
                                                               const double *__restrict__ variable_lower_bound,
                                                               const double *__restrict__ variable_upper_bound,
                                                               int n_vars);

    __global__ void project_rotated_soc_kernel(double *__restrict__ primal_solution,
                                               const double *__restrict__ variable_rescaling,
                                               double *__restrict__ warm_start,
                                               const int *__restrict__ start_idx,
                                               const int *__restrict__ v_dim,
                                               int num_blocks);

    __global__ void project_rotated_soc_warp_kernel(double *__restrict__ primal_solution,
                                                    const double *__restrict__ variable_rescaling,
                                                    double *__restrict__ warm_start,
                                                    const int *__restrict__ start_idx,
                                                    const int *__restrict__ v_dim,
                                                    int num_blocks);

    __global__ void project_standard_soc_kernel(double *__restrict__ primal_solution,
                                                const double *__restrict__ variable_rescaling,
                                                double *__restrict__ warm_start,
                                                const int *__restrict__ start_idx,
                                                const int *__restrict__ v_dim,
                                                int num_blocks);

    __global__ void project_standard_soc_warp_kernel(double *__restrict__ primal_solution,
                                                     const double *__restrict__ variable_rescaling,
                                                     double *__restrict__ warm_start,
                                                     const int *__restrict__ start_idx,
                                                     const int *__restrict__ v_dim,
                                                     int num_blocks);

    __global__ void compute_cone_dual_residual_kernel(double *__restrict__ dual_residual,
                                                      const double *__restrict__ objective_vector,
                                                      const double *__restrict__ dual_product,
                                                      const double *__restrict__ variable_rescaling,
                                                      double *__restrict__ warm_start,
                                                      const int *__restrict__ start_idx,
                                                      const int *__restrict__ v_dim,
                                                      int num_blocks);

    __global__ void compute_cone_dual_residual_warp_kernel(double *__restrict__ dual_residual,
                                                           const double *__restrict__ objective_vector,
                                                           const double *__restrict__ dual_product,
                                                           const double *__restrict__ variable_rescaling,
                                                           double *__restrict__ warm_start,
                                                           const int *__restrict__ start_idx,
                                                           const int *__restrict__ v_dim,
                                                           int num_blocks);

    __global__ void compute_cone_dual_residual_standard_warp_kernel(double *__restrict__ dual_residual,
                                                                    const double *__restrict__ objective_vector,
                                                                    const double *__restrict__ dual_product,
                                                                    const double *__restrict__ variable_rescaling,
                                                                    double *__restrict__ warm_start,
                                                                    const int *__restrict__ start_idx,
                                                                    const int *__restrict__ v_dim,
                                                                    int num_blocks);

    __global__ void project_exp_cone_kernel(double *__restrict__ primal_solution,
                                            const double *__restrict__ variable_rescaling,
                                            double *__restrict__ warm_start,
                                            const int *__restrict__ start_idx,
                                            const int *__restrict__ v_dim,
                                            const char *__restrict__ is_fixed,
                                            int num_blocks);

    __global__ void compute_cone_dual_residual_exp_kernel(double *__restrict__ dual_residual,
                                                          const double *__restrict__ objective_vector,
                                                          const double *__restrict__ dual_product,
                                                          const double *__restrict__ variable_rescaling,
                                                          double *__restrict__ warm_start,
                                                          const int *__restrict__ start_idx,
                                                          const int *__restrict__ v_dim,
                                                          const char *__restrict__ is_fixed,
                                                          int num_blocks);

    __global__ void compute_cone_dual_residual_standard_kernel(double *__restrict__ dual_residual,
                                                               const double *__restrict__ objective_vector,
                                                               const double *__restrict__ dual_product,
                                                               const double *__restrict__ variable_rescaling,
                                                               double *__restrict__ warm_start,
                                                               const int *__restrict__ start_idx,
                                                               const int *__restrict__ v_dim,
                                                               int num_blocks);

    __global__ void set_cone_dual_slack_kernel(double *__restrict__ dual_slack,
                                               const double *__restrict__ objective_vector,
                                               const double *__restrict__ dual_product,
                                               const int *__restrict__ start_idx,
                                               const int *__restrict__ v_dim,
                                               int num_blocks);

    __global__ void recompute_reflected_at_cone_kernel(double *__restrict__ reflected_primal,
                                                       const double *__restrict__ pdhg_primal,
                                                       const double *__restrict__ current_primal,
                                                       const int *__restrict__ start_idx,
                                                       const int *__restrict__ v_dim,
                                                       int num_blocks);

    __global__ void project_rotated_soc_diag_q_kernel(double *__restrict__ pdhg_primal,
                                                      double *__restrict__ reflected_primal,
                                                      const double *__restrict__ current_primal,
                                                      const double *__restrict__ variable_rescaling,
                                                      const double *__restrict__ Q_diag,
                                                      double tau,
                                                      double *__restrict__ warm_start,
                                                      const int *__restrict__ start_idx,
                                                      const int *__restrict__ v_dim,
                                                      int num_blocks);

    __global__ void project_standard_soc_diag_q_kernel(double *__restrict__ pdhg_primal,
                                                       double *__restrict__ reflected_primal,
                                                       const double *__restrict__ current_primal,
                                                       const double *__restrict__ variable_rescaling,
                                                       const double *__restrict__ Q_diag,
                                                       double tau,
                                                       double *__restrict__ warm_start,
                                                       const int *__restrict__ start_idx,
                                                       const int *__restrict__ v_dim,
                                                       int num_blocks);

    __global__ void project_exp_cone_diag_q_kernel(double *__restrict__ pdhg_primal,
                                                   double *__restrict__ reflected_primal,
                                                   const double *__restrict__ current_primal,
                                                   const double *__restrict__ variable_rescaling,
                                                   const double *__restrict__ Q_diag,
                                                   double tau,
                                                   double *__restrict__ warm_start,
                                                   const int *__restrict__ start_idx,
                                                   const int *__restrict__ v_dim,
                                                   const char *__restrict__ is_fixed,
                                                   int num_blocks);

#ifdef __cplusplus
}
#endif

#endif // PDHCG_KERNELS_CUH
