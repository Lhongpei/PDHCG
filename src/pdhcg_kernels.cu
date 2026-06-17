/*
Copyright 2025-2026 Haihao Lu
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

#include "pdhcg_kernels.cuh"
#include <cuda_runtime.h>
#include <math.h>
__global__ void compute_and_rescale_reduced_cost_kernel(double *reduced_cost,
                                                        const double *objective,
                                                        const double *dual_product,
                                                        const double *variable_rescaling,
                                                        const double objective_vector_rescaling,
                                                        const double constraint_bound_rescaling,
                                                        int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        reduced_cost[i] = (objective[i] - dual_product[i]) * variable_rescaling[i] / objective_vector_rescaling;
    }
}

__global__ void
element_wise_mul_kernel(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int n)
{
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        C[idx] = A[idx] * B[idx];
    }
}
__global__ void compute_lp_next_pdhg_primal_solution_kernel(const double *current_primal,
                                                            double *reflected_primal,
                                                            const double *dual_product,
                                                            const double *objective,
                                                            const double *var_lb,
                                                            const double *var_ub,
                                                            int n,
                                                            double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double current_primal_i = current_primal[i];
        double temp = current_primal_i - step_size * (objective[i] - dual_product[i]);
        double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
        reflected_primal[i] = 2.0 * temp_proj - current_primal_i;
    }
}

__global__ void compute_lp_next_pdhg_primal_solution_major_kernel(const double *current_primal,
                                                                  double *pdhg_primal,
                                                                  double *reflected_primal,
                                                                  const double *dual_product,
                                                                  const double *objective,
                                                                  const double *var_lb,
                                                                  const double *var_ub,
                                                                  int n,
                                                                  double step_size,
                                                                  double *dual_slack)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double current_primal_i = current_primal[i];
        double temp = current_primal_i - step_size * (objective[i] - dual_product[i]);
        double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
        reflected_primal[i] = 2.0 * temp_proj - current_primal_i;
        pdhg_primal[i] = temp_proj;
        dual_slack[i] = (temp_proj - temp) / step_size;
    }
}

__global__ void compute_diagonal_q_next_pdhg_primal_solution_major_kernel(const double *current_primal,
                                                                          double *pdhg_primal,
                                                                          double *reflected_primal,
                                                                          double *objective_product,
                                                                          const double *dual_product,
                                                                          const double *objective,
                                                                          const double *var_lb,
                                                                          const double *var_ub,
                                                                          int n,
                                                                          double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double current_primal_i = current_primal[i];
        double temp = (current_primal_i - step_size * (objective[i] - dual_product[i])) /
            (1.0 + step_size * objective_product[i]);
        double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
        reflected_primal[i] = 2.0 * temp_proj - current_primal_i;
        pdhg_primal[i] = temp_proj;
    }
}

__global__ void compute_diagonal_q_next_pdhg_primal_solution_kernel(const double *current_primal,
                                                                    double *reflected_primal,
                                                                    double *objective_product,
                                                                    const double *dual_product,
                                                                    const double *objective,
                                                                    const double *var_lb,
                                                                    const double *var_ub,
                                                                    int n,
                                                                    double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double current_primal_i = current_primal[i];
        double temp = (current_primal_i - step_size * (objective[i] - dual_product[i])) /
            (1.0 + step_size * objective_product[i]);
        double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
        reflected_primal[i] = 2.0 * temp_proj - current_primal_i;
    }
}

__global__ void compute_next_pdhg_dual_solution_kernel(const double *current_dual,
                                                       double *reflected_dual,
                                                       const double *primal_product,
                                                       const double *const_lb,
                                                       const double *const_ub,
                                                       int n,
                                                       double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_dual[i] / step_size - primal_product[i];
        double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
        reflected_dual[i] = 2.0 * (temp - temp_proj) * step_size - current_dual[i];
    }
}

__global__ void compute_next_pdhg_dual_solution_major_kernel(const double *current_dual,
                                                             double *pdhg_dual,
                                                             double *reflected_dual,
                                                             const double *primal_product,
                                                             const double *const_lb,
                                                             const double *const_ub,
                                                             int n,
                                                             double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_dual[i] / step_size - primal_product[i];
        double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
        pdhg_dual[i] = (temp - temp_proj) * step_size;
        reflected_dual[i] = 2.0 * pdhg_dual[i] - current_dual[i];
    }
}

__global__ void halpern_update_kernel(const double *initial_primal,
                                      double *current_primal,
                                      const double *reflected_primal,
                                      const double *initial_dual,
                                      double *current_dual,
                                      const double *reflected_dual,
                                      int n_vars,
                                      int n_cons,
                                      double weight,
                                      double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double reflected = reflection_coeff * reflected_primal[i] + (1.0 - reflection_coeff) * current_primal[i];
        current_primal[i] = weight * reflected + (1.0 - weight) * initial_primal[i];
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        double reflected = reflection_coeff * reflected_dual[idx] + (1.0 - reflection_coeff) * current_dual[idx];
        current_dual[idx] = weight * reflected + (1.0 - weight) * initial_dual[idx];
    }
}

__global__ void rescale_solution_kernel(double *primal_solution,
                                        double *dual_solution,
                                        const double *variable_rescaling,
                                        const double *constraint_rescaling,
                                        const double objective_vector_rescaling,
                                        const double constraint_bound_rescaling,
                                        int n_vars,
                                        int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        primal_solution[i] = primal_solution[i] / variable_rescaling[i] / constraint_bound_rescaling;
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        dual_solution[idx] = dual_solution[idx] / constraint_rescaling[idx] / objective_vector_rescaling;
    }
}

__global__ void compute_delta_solution_kernel(const double *initial_primal,
                                              const double *pdhg_primal,
                                              double *delta_primal,
                                              const double *initial_dual,
                                              const double *pdhg_dual,
                                              double *delta_dual,
                                              int n_vars,
                                              int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        delta_primal[i] = pdhg_primal[i] - initial_primal[i];
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        delta_dual[idx] = pdhg_dual[idx] - initial_dual[idx];
    }
}
__global__ void primal_gradient_descent_kernel(const double *dual_product,
                                               const double *current_primal_solution,
                                               double *reflected_primal,
                                               const double *objective_vector,
                                               const double *objective_product,
                                               const double *var_lb,
                                               const double *var_ub,
                                               const double stepsize,
                                               const int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double current_grad = objective_product[i] + objective_vector[i] - dual_product[i];
        double current_primal_sol = current_primal_solution[i];
        double next_primal_sol = current_primal_sol - stepsize * current_grad;
        next_primal_sol = fmax(var_lb[i], fmin(next_primal_sol, var_ub[i]));
        reflected_primal[i] = 2.0 * next_primal_sol - current_primal_sol;
    }
}

__global__ void primal_gradient_descent_kernel_major(const double *dual_product,
                                                     const double *current_primal_solution,
                                                     double *reflected_primal,
                                                     double *pdhg_primal_solution,
                                                     const double *objective_vector,
                                                     const double *objective_product,
                                                     const double *var_lb,
                                                     const double *var_ub,
                                                     const double stepsize,
                                                     const int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double current_grad = objective_product[i] + objective_vector[i] - dual_product[i];
        double current_primal_sol = current_primal_solution[i];
        double next_primal_sol = current_primal_sol - stepsize * current_grad;
        next_primal_sol = fmax(var_lb[i], fmin(next_primal_sol, var_ub[i]));
        pdhg_primal_solution[i] = next_primal_sol;
        reflected_primal[i] = 2.0 * next_primal_sol - current_primal_sol;
    }
}
__global__ void compute_bb_alpha_safeguard_kernel(const double *d_norm_gtg, const double *d_tmp, double *d_alpha)
{
    *d_alpha = (*d_norm_gtg * *d_norm_gtg) / *d_tmp;
}

__global__ void compute_bb_alpha_M_kernel(const double *d_stMs, const double *d_tmp, double *d_alpha)
{
    *d_alpha = *d_stMs / *d_tmp;
}

__global__ void scalar_sqrt_copy_kernel(const double *src, double *dst)
{
    *dst = sqrt(*src);
}

__global__ void
compute_csr_diag_kernel(const int *row_ptr, const int *col_ind, const double *val, double *diag, int num_rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows)
    {
        double sum = 0.0;
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        for (int k = start; k < end; ++k)
        {
            if (col_ind[k] == i)
                sum += val[k];
        }
        diag[i] = sum;
    }
}

__global__ void compute_csr_row_sq_norm_kernel(const int *row_ptr, const double *val, double *out, int num_rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows)
    {
        double sum = 0.0;
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        for (int k = start; k < end; ++k)
        {
            double v = val[k];
            sum += v * v;
        }
        out[i] = sum;
    }
}

__global__ void element_wise_mul_inplace_kernel(double *__restrict__ x, const double *__restrict__ d, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        x[i] *= d[i];
    }
}

__global__ void compute_csr_row_sq_norm_weighted_kernel(
    const int *row_ptr, const int *col_ind, const double *val, const double *weights, double *out, int num_rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows)
    {
        double sum = 0.0;
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        for (int k = start; k < end; ++k)
        {
            double v = val[k];
            sum += weights[col_ind[k]] * v * v;
        }
        out[i] = sum;
    }
}

__global__ void compute_csr_row_quad_form_dense_kernel(const int *row_ptr,
                                                       const int *col_ind,
                                                       const double *val,
                                                       const double *D_dense,
                                                       int rank,
                                                       double *out,
                                                       int num_rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows)
    {
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        double sum = 0.0;
        for (int ka = start; ka < end; ++ka)
        {
            int ra = col_ind[ka];
            double va = val[ka];
            const double *Drow = D_dense + (size_t)ra * (size_t)rank;
            for (int kb = start; kb < end; ++kb)
            {
                sum += va * val[kb] * Drow[col_ind[kb]];
            }
        }
        out[i] = sum;
    }
}

__global__ void
refresh_inner_precond_kernel(const double *diag_h_static, double inv_tau, double *m_diag, double *m_inv, int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double m = diag_h_static[i] + inv_tau;
        if (m <= 0.0)
            m = 1.0;
        m_diag[i] = m;
        m_inv[i] = 1.0 / m;
    }
}

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
                                                       const int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double current_grad = objective_product[i] + objective_vector[i] - dual_product[i];
        double current_primal_sol = current_primal_solution[i];
        double next_primal_sol = current_primal_sol - stepsize * current_grad;
        next_primal_sol = fmax(var_lb[i], fmin(next_primal_sol, var_ub[i]));
        pdhg_primal_solution[i] = next_primal_sol;
        gradient[i] = current_grad;
        direction[i] = next_primal_sol - current_primal_sol;
    }
}

__global__ void primal_bb_update_gradient_kernel(double *pdhg_primal_solution,
                                                 const double *current_primal_solution,
                                                 const double *objective_vector,
                                                 const double *dual_product,
                                                 const double *objective_product,
                                                 double *gradient,
                                                 double *delta_gradient,
                                                 const double inv_step_size,
                                                 const int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double last_gradient = gradient[i];
        double current_gradient = objective_product[i] + objective_vector[i] - dual_product[i] +
            inv_step_size * (pdhg_primal_solution[i] - current_primal_solution[i]);
        delta_gradient[i] = current_gradient - last_gradient;
        gradient[i] = current_gradient;
    }
}

__global__ void primal_bb_update_direction_kernel(double *pdhg_primal_solution,
                                                  const double *gradient,
                                                  double *direction,
                                                  const double *var_lb,
                                                  const double *var_ub,
                                                  const double *d_alpha,
                                                  const int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double alpha = *d_alpha;
        double cur_sol = pdhg_primal_solution[i];
        double next_sol = cur_sol - alpha * gradient[i];
        next_sol = fmax(var_lb[i], fmin(next_sol, var_ub[i]));
        direction[i] = next_sol - cur_sol;
        pdhg_primal_solution[i] = next_sol;
    }
}

__global__ void primal_bb_final_kernel(const double *current_primal_solution,
                                       const double *pdhg_primal_solution,
                                       double *reflected_primal_solution,
                                       const int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double cur_sol = pdhg_primal_solution[i];
        double last_sol = current_primal_solution[i];
        reflected_primal_solution[i] = 2.0 * cur_sol - last_sol;
    }
}

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
                                                               const int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double current_grad = objective_product[i] + objective_vector[i] - dual_product[i];
        double current_primal_sol = current_primal_solution[i];
        double next_primal_sol = current_primal_sol - stepsize * m_inv[i] * current_grad;
        next_primal_sol = fmax(var_lb[i], fmin(next_primal_sol, var_ub[i]));
        pdhg_primal_solution[i] = next_primal_sol;
        gradient[i] = current_grad;
        direction[i] = next_primal_sol - current_primal_sol;
    }
}

__global__ void primal_bb_update_direction_kernel_precond(double *pdhg_primal_solution,
                                                          const double *gradient,
                                                          double *direction,
                                                          const double *var_lb,
                                                          const double *var_ub,
                                                          const double *m_inv,
                                                          const double *d_alpha,
                                                          const int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double alpha = *d_alpha;
        double cur_sol = pdhg_primal_solution[i];
        double next_sol = cur_sol - alpha * m_inv[i] * gradient[i];
        next_sol = fmax(var_lb[i], fmin(next_sol, var_ub[i]));
        direction[i] = next_sol - cur_sol;
        pdhg_primal_solution[i] = next_sol;
    }
}

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
                                           int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {
        double clamped_val = fmax(constraint_lower_bound[i], fmin(primal_product[i], constraint_upper_bound[i]));
        primal_residual[i] = (primal_product[i] - clamped_val) * constraint_rescaling[i];

        dual_obj_contribution[i] =
            fmax(dual_solution[i], 0.0) * const_lb_finite[i] + fmin(dual_solution[i], 0.0) * const_ub_finite[i];
    }
    else if (i < num_constraints + num_variables)
    {
        int idx = i - num_constraints;
        dual_residual[idx] = (objective_vector[idx] - dual_product[idx] - dual_slack[idx]) * variable_rescaling[idx];
    }
}

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
                                           int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {
        double clamped_val = fmax(constraint_lower_bound[i], fmin(primal_product[i], constraint_upper_bound[i]));
        primal_residual[i] = (primal_product[i] - clamped_val) * constraint_rescaling[i];

        dual_obj_contribution[i] =
            fmax(dual_solution[i], 0.0) * const_lb_finite[i] + fmin(dual_solution[i], 0.0) * const_ub_finite[i];
    }
    else if (i < num_constraints + num_variables)
    {
        int idx = i - num_constraints;
        double gradient = primal_obj_product[idx] + objective_vector[idx] - dual_product[idx];
        double tmp = primal_solution[idx] - step_size * gradient;
        double proj_tmp = fmax(variable_lower_bound[idx], fmin(variable_upper_bound[idx], tmp));
        double dual_slack_idx = (proj_tmp - tmp) / step_size;
        dual_residual[idx] = (gradient - dual_slack_idx) * variable_rescaling[idx];
        dual_slack[idx] = dual_slack_idx;
    }
}

__global__ void recover_primal_obj_dual_product(double *dual_product,
                                                double *primal_obj_product,
                                                const double *variable_rescaling,
                                                int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_variables)
    {
        dual_product[i] = dual_product[i] * variable_rescaling[i];
        primal_obj_product[i] = primal_obj_product[i] * variable_rescaling[i];
    }
}

__global__ void primal_infeasibility_project_kernel(double *primal_ray_estimate,
                                                    const double *variable_lower_bound,
                                                    const double *variable_upper_bound,
                                                    int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_variables)
    {
        if (isfinite(variable_lower_bound[i]))
        {
            primal_ray_estimate[i] = fmax(primal_ray_estimate[i], 0.0);
        }
        if (isfinite(variable_upper_bound[i]))
        {
            primal_ray_estimate[i] = fmin(primal_ray_estimate[i], 0.0);
        }
    }
}

__global__ void dual_infeasibility_project_kernel(double *dual_ray_estimate,
                                                  const double *constraint_lower_bound,
                                                  const double *constraint_upper_bound,
                                                  int num_constraints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_constraints)
    {
        if (!isfinite(constraint_lower_bound[i]))
        {
            dual_ray_estimate[i] = fmin(dual_ray_estimate[i], 0.0);
        }
        if (!isfinite(constraint_upper_bound[i]))
        {
            dual_ray_estimate[i] = fmax(dual_ray_estimate[i], 0.0);
        }
    }
}

__global__ void compute_primal_infeasibility_kernel(const double *primal_product,
                                                    const double *const_lb,
                                                    const double *const_ub,
                                                    int num_constraints,
                                                    double *primal_infeasibility,
                                                    const double *constraint_rescaling)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_constraints)
    {
        double pp_val = primal_product[i];
        primal_infeasibility[i] =
            (fmax(0.0, -pp_val) * isfinite(const_lb[i]) + fmax(0.0, pp_val) * isfinite(const_ub[i])) *
            constraint_rescaling[i];
    }
}

__global__ void compute_dual_infeasibility_kernel(const double *dual_product,
                                                  const double *var_lb,
                                                  const double *var_ub,
                                                  int num_variables,
                                                  double *dual_infeasibility,
                                                  const double *variable_rescaling)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_variables)
    {
        double dp_val = -dual_product[i];
        dual_infeasibility[i] = (fmax(0.0, dp_val) * !isfinite(var_lb[i]) - fmin(0.0, dp_val) * !isfinite(var_ub[i])) *
            variable_rescaling[i];
    }
}

__global__ void
dual_solution_dual_objective_contribution_kernel(const double *constraint_lower_bound_finite_val,
                                                 const double *constraint_upper_bound_finite_val,
                                                 const double *dual_solution,
                                                 int num_constraints,
                                                 double *dual_objective_dual_solution_contribution_array)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {
        dual_objective_dual_solution_contribution_array[i] =
            fmax(dual_solution[i], 0.0) * constraint_lower_bound_finite_val[i] +
            fmin(dual_solution[i], 0.0) * constraint_upper_bound_finite_val[i];
    }
}

__global__ void
dual_objective_dual_slack_contribution_array_kernel(const double *dual_slack,
                                                    double *dual_objective_dual_slack_contribution_array,
                                                    const double *variable_lower_bound_finite_val,
                                                    const double *variable_upper_bound_finite_val,
                                                    int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_variables)
    {
        dual_objective_dual_slack_contribution_array[i] =
            fmax(-dual_slack[i], 0.0) * variable_lower_bound_finite_val[i] +
            fmin(-dual_slack[i], 0.0) * variable_upper_bound_finite_val[i];
    }
}

__global__ void compute_and_rescale_reduced_cost_qp_kernel(double *__restrict__ reduced_cost,
                                                           const double *__restrict__ objective,
                                                           const double *__restrict__ quadratic_product,
                                                           const double *__restrict__ dual_product,
                                                           const double *__restrict__ variable_rescaling,
                                                           const double objective_vector_rescaling,
                                                           const double constraint_bound_rescaling,
                                                           const double *__restrict__ variable_lower_bound,
                                                           const double *__restrict__ variable_upper_bound,
                                                           int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double grad_i = objective[i];
        if (quadratic_product != NULL)
        {
            grad_i += quadratic_product[i];
        }

        double rc = (grad_i - dual_product[i]) * variable_rescaling[i] / objective_vector_rescaling;

        if (!isfinite(variable_lower_bound[i]))
        {
            rc = fmin(rc, 0.0);
        }
        if (!isfinite(variable_upper_bound[i]))
        {
            rc = fmax(rc, 0.0);
        }

        reduced_cost[i] = rc;
    }
}

__global__ void project_rotated_soc_kernel(double *__restrict__ primal_solution,
                                           const double *__restrict__ variable_rescaling,
                                           double *__restrict__ warm_start,
                                           const int *__restrict__ start_idx,
                                           const int *__restrict__ v_dim,
                                           int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    const double INV_SQRT2 = 0.7071067811865475;

    int start = start_idx[blk];
    int k = v_dim[blk];
    double *v = primal_solution + start;
    double *sptr = primal_solution + start + k;
    double *tptr = primal_solution + start + k + 1;

    double s = *sptr;
    double t = *tptr;
    double w = (s - t) * INV_SQRT2;
    double z = (s + t) * INV_SQRT2;

    double d_s = variable_rescaling[start + k];
    double d_t = variable_rescaling[start + k + 1];
    double d_st = sqrt(d_s * d_t);

    bool diag_uniform = true;
    for (int m = 0; m < k && diag_uniform; ++m)
    {
        if (variable_rescaling[start + m] != d_st)
            diag_uniform = false;
    }

    if (diag_uniform)
    {
        double sumsq = w * w;
        for (int m = 0; m < k; ++m)
            sumsq += v[m] * v[m];
        double r = sqrt(sumsq);
        if (r <= z)
            return;
        if (r <= -z)
        {
            for (int m = 0; m < k; ++m)
                v[m] = 0.0;
            *sptr = 0.0;
            *tptr = 0.0;
            return;
        }
        double scale = (z + r) / (2.0 * r);
        for (int m = 0; m < k; ++m)
            v[m] *= scale;
        double w_new = scale * w;
        double z_new = scale * r;
        *sptr = (z_new + w_new) * INV_SQRT2;
        *tptr = (z_new - w_new) * INV_SQRT2;
        return;
    }

    double r_inv_sq = w * w;
    double r_pos_sq = w * w;
    for (int m = 0; m < k; ++m)
    {
        double dh = variable_rescaling[start + m] / d_st;
        double v_m = v[m];
        r_inv_sq += (v_m / dh) * (v_m / dh);
        r_pos_sq += (v_m * dh) * (v_m * dh);
    }
    double r_inv = sqrt(r_inv_sq);
    if (r_inv <= z)
        return;
    double r_pos = sqrt(r_pos_sq);
    if (r_pos <= -z)
    {
        for (int m = 0; m < k; ++m)
            v[m] = 0.0;
        *sptr = 0.0;
        *tptr = 0.0;
        return;
    }

    double lo, hi;
    bool z_pos = (z > 0.0);
    if (z_pos)
    {
        lo = 0.0;
        hi = 0.5 - 1e-14;
    }
    else
    {
        lo = 0.5 + 1e-14;
        hi = 1.0;
        for (int doubling = 0; doubling < 60; ++doubling)
        {
            double sum_hi = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double dh = variable_rescaling[start + m] / d_st;
                double dh2 = dh * dh;
                double tt = v[m] * dh / (dh2 + 2.0 * hi);
                sum_hi += tt * tt;
            }
            double tw_hi = w / (1.0 + 2.0 * hi);
            sum_hi += tw_hi * tw_hi;
            double zt_hi = z / (1.0 - 2.0 * hi);
            double f_hi = sum_hi - zt_hi * zt_hi;
            if (f_hi > 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_lam = warm_start[blk];
    if (warm_lam > lo && warm_lam < hi)
    {
        double sum_w = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double dh = variable_rescaling[start + m] / d_st;
            double dh2 = dh * dh;
            double tt = v[m] * dh / (dh2 + 2.0 * warm_lam);
            sum_w += tt * tt;
        }
        double tw = w / (1.0 + 2.0 * warm_lam);
        sum_w += tw * tw;
        double zt = z / (1.0 - 2.0 * warm_lam);
        double f = sum_w - zt * zt;
        if (fabs(f) < 1e-12)
        {
            double w_new = w / (1.0 + 2.0 * warm_lam);
            double z_new = z / (1.0 - 2.0 * warm_lam);
            for (int m = 0; m < k; ++m)
            {
                double dh = variable_rescaling[start + m] / d_st;
                double dh2 = dh * dh;
                v[m] = v[m] * dh2 / (dh2 + 2.0 * warm_lam);
            }
            *sptr = (z_new + w_new) * INV_SQRT2;
            *tptr = (z_new - w_new) * INV_SQRT2;
            return;
        }
        if (z_pos)
        {
            if (f > 0.0)
                lo = warm_lam;
            else
                hi = warm_lam;
        }
        else
        {
            if (f > 0.0)
                hi = warm_lam;
            else
                lo = warm_lam;
        }
    }

    for (int it = 0; it < 60; ++it)
    {
        double lam = 0.5 * (lo + hi);
        double sum = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double dh = variable_rescaling[start + m] / d_st;
            double dh2 = dh * dh;
            double tt = v[m] * dh / (dh2 + 2.0 * lam);
            sum += tt * tt;
        }
        double tw = w / (1.0 + 2.0 * lam);
        sum += tw * tw;
        double zt = z / (1.0 - 2.0 * lam);
        double f = sum - zt * zt;
        if (z_pos)
        {
            if (f > 0.0)
                lo = lam;
            else
                hi = lam;
        }
        else
        {
            if (f > 0.0)
                hi = lam;
            else
                lo = lam;
        }
        if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
            break;
    }
    double lam = 0.5 * (lo + hi);
    warm_start[blk] = lam;

    double w_new = w / (1.0 + 2.0 * lam);
    double z_new = z / (1.0 - 2.0 * lam);
    for (int m = 0; m < k; ++m)
    {
        double dh = variable_rescaling[start + m] / d_st;
        double dh2 = dh * dh;
        v[m] = v[m] * dh2 / (dh2 + 2.0 * lam);
    }
    *sptr = (z_new + w_new) * INV_SQRT2;
    *tptr = (z_new - w_new) * INV_SQRT2;
}

__global__ void compute_cone_dual_residual_kernel(double *__restrict__ dual_residual,
                                                  const double *__restrict__ objective_vector,
                                                  const double *__restrict__ dual_product,
                                                  const double *__restrict__ variable_rescaling,
                                                  double *__restrict__ warm_start,
                                                  const int *__restrict__ start_idx,
                                                  const int *__restrict__ v_dim,
                                                  int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    const double INV_SQRT2 = 0.7071067811865475;
    int start = start_idx[blk];
    int k = v_dim[blk];

    double r_s = objective_vector[start + k] - dual_product[start + k];
    double r_t = objective_vector[start + k + 1] - dual_product[start + k + 1];
    double r_w = (r_s - r_t) * INV_SQRT2;
    double r_z = (r_s + r_t) * INV_SQRT2;

    double d_s = variable_rescaling[start + k];
    double d_t = variable_rescaling[start + k + 1];
    double d_st = sqrt(d_s * d_t);

    bool diag_uniform = true;
    for (int m = 0; m < k && diag_uniform; ++m)
    {
        if (variable_rescaling[start + m] != d_st)
            diag_uniform = false;
    }

    if (diag_uniform)
    {
        double sumsq = r_w * r_w;
        for (int m = 0; m < k; ++m)
        {
            double v_m = objective_vector[start + m] - dual_product[start + m];
            sumsq += v_m * v_m;
        }
        double r_norm = sqrt(sumsq);

        double v_factor, p_s, p_t;
        if (r_norm <= r_z)
        {
            v_factor = 0.0;
            p_s = r_s;
            p_t = r_t;
        }
        else if (r_norm <= -r_z)
        {
            v_factor = 1.0;
            p_s = 0.0;
            p_t = 0.0;
        }
        else
        {
            double scale = (r_z + r_norm) / (2.0 * r_norm);
            v_factor = 1.0 - scale;
            double w_new = scale * r_w;
            double z_new = scale * r_norm;
            p_s = (z_new + w_new) * INV_SQRT2;
            p_t = (z_new - w_new) * INV_SQRT2;
        }
        for (int m = 0; m < k; ++m)
        {
            double v_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = v_m * v_factor * variable_rescaling[start + m];
        }
        dual_residual[start + k] = (r_s - p_s) * variable_rescaling[start + k];
        dual_residual[start + k + 1] = (r_t - p_t) * variable_rescaling[start + k + 1];
        return;
    }

    double r_inv_sq = r_w * r_w;
    double r_pos_sq = r_w * r_w;
    for (int m = 0; m < k; ++m)
    {
        double e_m = d_st / variable_rescaling[start + m];
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        r_inv_sq += (rc_m / e_m) * (rc_m / e_m);
        r_pos_sq += (rc_m * e_m) * (rc_m * e_m);
    }
    double r_inv = sqrt(r_inv_sq);
    double r_pos = sqrt(r_pos_sq);

    if (r_inv <= r_z)
    {
        for (int m = 0; m < k; ++m)
            dual_residual[start + m] = 0.0;
        dual_residual[start + k] = 0.0;
        dual_residual[start + k + 1] = 0.0;
        return;
    }
    if (r_pos <= -r_z)
    {
        for (int m = 0; m < k; ++m)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = rc_m * variable_rescaling[start + m];
        }
        dual_residual[start + k] = r_s * variable_rescaling[start + k];
        dual_residual[start + k + 1] = r_t * variable_rescaling[start + k + 1];
        return;
    }

    double lo, hi;
    bool z_pos = (r_z > 0.0);
    if (z_pos)
    {
        lo = 0.0;
        hi = 0.5 - 1e-14;
    }
    else
    {
        lo = 0.5 + 1e-14;
        hi = 1.0;
        for (int doubling = 0; doubling < 60; ++doubling)
        {
            double sum_hi = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double e_m = d_st / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double tt = rc_m * e_m / (e_m2 + 2.0 * hi);
                sum_hi += tt * tt;
            }
            double tw_hi = r_w / (1.0 + 2.0 * hi);
            sum_hi += tw_hi * tw_hi;
            double zt_hi = r_z / (1.0 - 2.0 * hi);
            double f_hi = sum_hi - zt_hi * zt_hi;
            if (f_hi > 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_lam = warm_start[blk];
    if (warm_lam > lo && warm_lam < hi)
    {
        double sum_w = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double e_m = d_st / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double tt = rc_m * e_m / (e_m2 + 2.0 * warm_lam);
            sum_w += tt * tt;
        }
        double tw = r_w / (1.0 + 2.0 * warm_lam);
        sum_w += tw * tw;
        double zt = r_z / (1.0 - 2.0 * warm_lam);
        double f = sum_w - zt * zt;
        if (fabs(f) < 1e-12)
        {
            double p_w_w = r_w / (1.0 + 2.0 * warm_lam);
            double p_z_w = r_z / (1.0 - 2.0 * warm_lam);
            double p_s_w = (p_z_w + p_w_w) * INV_SQRT2;
            double p_t_w = (p_z_w - p_w_w) * INV_SQRT2;
            for (int m = 0; m < k; ++m)
            {
                double e_m = d_st / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double p_m = rc_m * e_m2 / (e_m2 + 2.0 * warm_lam);
                dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
            }
            dual_residual[start + k] = (r_s - p_s_w) * variable_rescaling[start + k];
            dual_residual[start + k + 1] = (r_t - p_t_w) * variable_rescaling[start + k + 1];
            return;
        }
        if (z_pos)
        {
            if (f > 0.0)
                lo = warm_lam;
            else
                hi = warm_lam;
        }
        else
        {
            if (f > 0.0)
                hi = warm_lam;
            else
                lo = warm_lam;
        }
    }

    for (int it = 0; it < 60; ++it)
    {
        double lam = 0.5 * (lo + hi);
        double sum = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double e_m = d_st / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double tt = rc_m * e_m / (e_m2 + 2.0 * lam);
            sum += tt * tt;
        }
        double tw = r_w / (1.0 + 2.0 * lam);
        sum += tw * tw;
        double zt = r_z / (1.0 - 2.0 * lam);
        double f = sum - zt * zt;
        if (z_pos)
        {
            if (f > 0.0)
                lo = lam;
            else
                hi = lam;
        }
        else
        {
            if (f > 0.0)
                hi = lam;
            else
                lo = lam;
        }
        if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
            break;
    }
    double lam = 0.5 * (lo + hi);
    warm_start[blk] = lam;

    double p_w = r_w / (1.0 + 2.0 * lam);
    double p_z = r_z / (1.0 - 2.0 * lam);
    double p_s = (p_z + p_w) * INV_SQRT2;
    double p_t = (p_z - p_w) * INV_SQRT2;

    for (int m = 0; m < k; ++m)
    {
        double e_m = d_st / variable_rescaling[start + m];
        double e_m2 = e_m * e_m;
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        double p_m = rc_m * e_m2 / (e_m2 + 2.0 * lam);
        dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
    }
    dual_residual[start + k] = (r_s - p_s) * variable_rescaling[start + k];
    dual_residual[start + k + 1] = (r_t - p_t) * variable_rescaling[start + k + 1];
}

__global__ void project_standard_soc_kernel(double *__restrict__ primal_solution,
                                            const double *__restrict__ variable_rescaling,
                                            double *__restrict__ warm_start,
                                            const int *__restrict__ start_idx,
                                            const int *__restrict__ v_dim,
                                            int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int start = start_idx[blk];
    int k = v_dim[blk];
    double *v = primal_solution + start;
    double *wptr = primal_solution + start + k;
    double *zptr = primal_solution + start + k + 1;

    double w = *wptr;
    double z = *zptr;

    double d_z = variable_rescaling[start + k + 1];
    double dhat_w = variable_rescaling[start + k] / d_z;
    double dhat_w2 = dhat_w * dhat_w;

    bool diag_uniform = (dhat_w == 1.0);
    for (int m = 0; m < k && diag_uniform; ++m)
    {
        if (variable_rescaling[start + m] != d_z)
            diag_uniform = false;
    }

    if (diag_uniform)
    {
        double sumsq = w * w;
        for (int m = 0; m < k; ++m)
            sumsq += v[m] * v[m];
        double r = sqrt(sumsq);
        if (r <= z)
            return;
        if (r <= -z)
        {
            for (int m = 0; m < k; ++m)
                v[m] = 0.0;
            *wptr = 0.0;
            *zptr = 0.0;
            return;
        }
        double scale = (z + r) / (2.0 * r);
        for (int m = 0; m < k; ++m)
            v[m] *= scale;
        *wptr = scale * w;
        *zptr = scale * r;
        return;
    }

    double r_inv_sq = (w / dhat_w) * (w / dhat_w);
    double r_pos_sq = (w * dhat_w) * (w * dhat_w);
    for (int m = 0; m < k; ++m)
    {
        double dh = variable_rescaling[start + m] / d_z;
        double v_m = v[m];
        r_inv_sq += (v_m / dh) * (v_m / dh);
        r_pos_sq += (v_m * dh) * (v_m * dh);
    }
    double r_inv = sqrt(r_inv_sq);
    if (r_inv <= z)
        return;
    double r_pos = sqrt(r_pos_sq);
    if (r_pos <= -z)
    {
        for (int m = 0; m < k; ++m)
            v[m] = 0.0;
        *wptr = 0.0;
        *zptr = 0.0;
        return;
    }

    double lo, hi;
    bool z_pos = (z > 0.0);
    if (z_pos)
    {
        lo = 0.0;
        hi = 0.5 - 1e-14;
    }
    else
    {
        lo = 0.5 + 1e-14;
        hi = 1.0;
        for (int doubling = 0; doubling < 60; ++doubling)
        {
            double sum_hi = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double dh = variable_rescaling[start + m] / d_z;
                double dh2 = dh * dh;
                double t = v[m] * dh / (dh2 + 2.0 * hi);
                sum_hi += t * t;
            }
            double tw_hi = w * dhat_w / (dhat_w2 + 2.0 * hi);
            sum_hi += tw_hi * tw_hi;
            double zt_hi = z / (1.0 - 2.0 * hi);
            double f_hi = sum_hi - zt_hi * zt_hi;
            if (f_hi > 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_lam = warm_start[blk];
    if (warm_lam > lo && warm_lam < hi)
    {
        double sum_w = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double dh = variable_rescaling[start + m] / d_z;
            double dh2 = dh * dh;
            double t = v[m] * dh / (dh2 + 2.0 * warm_lam);
            sum_w += t * t;
        }
        double tw = w * dhat_w / (dhat_w2 + 2.0 * warm_lam);
        sum_w += tw * tw;
        double zt = z / (1.0 - 2.0 * warm_lam);
        double f = sum_w - zt * zt;
        if (fabs(f) < 1e-12)
        {
            *zptr = z / (1.0 - 2.0 * warm_lam);
            *wptr = w * dhat_w2 / (dhat_w2 + 2.0 * warm_lam);
            for (int m = 0; m < k; ++m)
            {
                double dh = variable_rescaling[start + m] / d_z;
                double dh2 = dh * dh;
                v[m] = v[m] * dh2 / (dh2 + 2.0 * warm_lam);
            }
            return;
        }
        if (z_pos)
        {
            if (f > 0.0)
                lo = warm_lam;
            else
                hi = warm_lam;
        }
        else
        {
            if (f > 0.0)
                hi = warm_lam;
            else
                lo = warm_lam;
        }
    }

    for (int it = 0; it < 60; ++it)
    {
        double lam = 0.5 * (lo + hi);
        double sum = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double dh = variable_rescaling[start + m] / d_z;
            double dh2 = dh * dh;
            double t = v[m] * dh / (dh2 + 2.0 * lam);
            sum += t * t;
        }
        double tw = w * dhat_w / (dhat_w2 + 2.0 * lam);
        sum += tw * tw;
        double zt = z / (1.0 - 2.0 * lam);
        double f = sum - zt * zt;
        if (z_pos)
        {
            if (f > 0.0)
                lo = lam;
            else
                hi = lam;
        }
        else
        {
            if (f > 0.0)
                hi = lam;
            else
                lo = lam;
        }
        if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
            break;
    }
    double lam = 0.5 * (lo + hi);
    warm_start[blk] = lam;

    *zptr = z / (1.0 - 2.0 * lam);
    *wptr = w * dhat_w2 / (dhat_w2 + 2.0 * lam);
    for (int m = 0; m < k; ++m)
    {
        double dh = variable_rescaling[start + m] / d_z;
        double dh2 = dh * dh;
        v[m] = v[m] * dh2 / (dh2 + 2.0 * lam);
    }
}

__global__ void compute_cone_dual_residual_standard_kernel(double *__restrict__ dual_residual,
                                                           const double *__restrict__ objective_vector,
                                                           const double *__restrict__ dual_product,
                                                           const double *__restrict__ variable_rescaling,
                                                           double *__restrict__ warm_start,
                                                           const int *__restrict__ start_idx,
                                                           const int *__restrict__ v_dim,
                                                           int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int start = start_idx[blk];
    int k = v_dim[blk];

    double r_w = objective_vector[start + k] - dual_product[start + k];
    double r_z = objective_vector[start + k + 1] - dual_product[start + k + 1];

    double d_z = variable_rescaling[start + k + 1];
    double e_w = d_z / variable_rescaling[start + k];
    double e_w2 = e_w * e_w;

    bool diag_uniform = (e_w == 1.0);
    for (int m = 0; m < k && diag_uniform; ++m)
    {
        if (variable_rescaling[start + m] != d_z)
            diag_uniform = false;
    }

    if (diag_uniform)
    {
        double sumsq = r_w * r_w;
        for (int m = 0; m < k; ++m)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            sumsq += rc_m * rc_m;
        }
        double r = sqrt(sumsq);
        double v_factor, p_w, p_z;
        if (r <= r_z)
        {
            v_factor = 0.0;
            p_w = r_w;
            p_z = r_z;
        }
        else if (r <= -r_z)
        {
            v_factor = 1.0;
            p_w = 0.0;
            p_z = 0.0;
        }
        else
        {
            double scale = (r_z + r) / (2.0 * r);
            v_factor = 1.0 - scale;
            p_w = scale * r_w;
            p_z = scale * r;
        }
        for (int m = 0; m < k; ++m)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = rc_m * v_factor * variable_rescaling[start + m];
        }
        dual_residual[start + k] = (r_w - p_w) * variable_rescaling[start + k];
        dual_residual[start + k + 1] = (r_z - p_z) * variable_rescaling[start + k + 1];
        return;
    }

    double r_inv_sq = (r_w / e_w) * (r_w / e_w);
    double r_pos_sq = (r_w * e_w) * (r_w * e_w);
    for (int m = 0; m < k; ++m)
    {
        double e_m = d_z / variable_rescaling[start + m];
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        r_inv_sq += (rc_m / e_m) * (rc_m / e_m);
        r_pos_sq += (rc_m * e_m) * (rc_m * e_m);
    }
    double r_inv = sqrt(r_inv_sq);
    double r_pos = sqrt(r_pos_sq);

    if (r_inv <= r_z)
    {
        for (int m = 0; m < k; ++m)
            dual_residual[start + m] = 0.0;
        dual_residual[start + k] = 0.0;
        dual_residual[start + k + 1] = 0.0;
        return;
    }
    if (r_pos <= -r_z)
    {
        for (int m = 0; m < k; ++m)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = rc_m * variable_rescaling[start + m];
        }
        dual_residual[start + k] = r_w * variable_rescaling[start + k];
        dual_residual[start + k + 1] = r_z * variable_rescaling[start + k + 1];
        return;
    }

    double lo, hi;
    bool z_pos = (r_z > 0.0);
    if (z_pos)
    {
        lo = 0.0;
        hi = 0.5 - 1e-14;
    }
    else
    {
        lo = 0.5 + 1e-14;
        hi = 1.0;
        for (int doubling = 0; doubling < 60; ++doubling)
        {
            double sum_hi = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double e_m = d_z / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double t = rc_m * e_m / (e_m2 + 2.0 * hi);
                sum_hi += t * t;
            }
            double tw_hi = r_w * e_w / (e_w2 + 2.0 * hi);
            sum_hi += tw_hi * tw_hi;
            double zt_hi = r_z / (1.0 - 2.0 * hi);
            double f_hi = sum_hi - zt_hi * zt_hi;
            if (f_hi > 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_lam = warm_start[blk];
    if (warm_lam > lo && warm_lam < hi)
    {
        double sum_w = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double e_m = d_z / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double t = rc_m * e_m / (e_m2 + 2.0 * warm_lam);
            sum_w += t * t;
        }
        double tw = r_w * e_w / (e_w2 + 2.0 * warm_lam);
        sum_w += tw * tw;
        double zt = r_z / (1.0 - 2.0 * warm_lam);
        double f = sum_w - zt * zt;
        if (fabs(f) < 1e-12)
        {
            double p_z_w = r_z / (1.0 - 2.0 * warm_lam);
            double p_w_w = r_w * e_w2 / (e_w2 + 2.0 * warm_lam);
            for (int m = 0; m < k; ++m)
            {
                double e_m = d_z / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double p_m = rc_m * e_m2 / (e_m2 + 2.0 * warm_lam);
                dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
            }
            dual_residual[start + k] = (r_w - p_w_w) * variable_rescaling[start + k];
            dual_residual[start + k + 1] = (r_z - p_z_w) * variable_rescaling[start + k + 1];
            return;
        }
        if (z_pos)
        {
            if (f > 0.0)
                lo = warm_lam;
            else
                hi = warm_lam;
        }
        else
        {
            if (f > 0.0)
                hi = warm_lam;
            else
                lo = warm_lam;
        }
    }

    for (int it = 0; it < 60; ++it)
    {
        double lam = 0.5 * (lo + hi);
        double sum = 0.0;
        for (int m = 0; m < k; ++m)
        {
            double e_m = d_z / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double t = rc_m * e_m / (e_m2 + 2.0 * lam);
            sum += t * t;
        }
        double tw = r_w * e_w / (e_w2 + 2.0 * lam);
        sum += tw * tw;
        double zt = r_z / (1.0 - 2.0 * lam);
        double f = sum - zt * zt;
        if (z_pos)
        {
            if (f > 0.0)
                lo = lam;
            else
                hi = lam;
        }
        else
        {
            if (f > 0.0)
                hi = lam;
            else
                lo = lam;
        }
        if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
            break;
    }
    double lam = 0.5 * (lo + hi);
    warm_start[blk] = lam;

    double p_z = r_z / (1.0 - 2.0 * lam);
    double p_w = r_w * e_w2 / (e_w2 + 2.0 * lam);

    for (int m = 0; m < k; ++m)
    {
        double e_m = d_z / variable_rescaling[start + m];
        double e_m2 = e_m * e_m;
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        double p_m = rc_m * e_m2 / (e_m2 + 2.0 * lam);
        dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
    }
    dual_residual[start + k] = (r_w - p_w) * variable_rescaling[start + k];
    dual_residual[start + k + 1] = (r_z - p_z) * variable_rescaling[start + k + 1];
}

__global__ void project_rotated_soc_warp_kernel(double *__restrict__ primal_solution,
                                                const double *__restrict__ variable_rescaling,
                                                double *__restrict__ warm_start,
                                                const int *__restrict__ start_idx,
                                                const int *__restrict__ v_dim,
                                                int num_blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk = tid >> 5;
    int lane = tid & 31;
    if (blk >= num_blocks)
        return;

    const double INV_SQRT2 = 0.7071067811865475;
    const unsigned MASK = 0xffffffffu;

    int start = start_idx[blk];
    int k = v_dim[blk];

    double s_val = primal_solution[start + k];
    double t_val = primal_solution[start + k + 1];
    double w = (s_val - t_val) * INV_SQRT2;
    double z = (s_val + t_val) * INV_SQRT2;

    double d_s = variable_rescaling[start + k];
    double d_t = variable_rescaling[start + k + 1];
    double d_st = sqrt(d_s * d_t);

    int my_diff = 0;
    for (int m = lane; m < k; m += 32)
    {
        if (variable_rescaling[start + m] != d_st)
            my_diff = 1;
    }
    for (int o = 16; o > 0; o >>= 1)
        my_diff |= __shfl_xor_sync(MASK, my_diff, o);

    if (my_diff == 0)
    {
        double my_sumsq = (lane == 0) ? w * w : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double v_m = primal_solution[start + m];
            my_sumsq += v_m * v_m;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sumsq += __shfl_xor_sync(MASK, my_sumsq, o);
        double r = sqrt(my_sumsq);
        if (r <= z)
            return;
        if (r <= -z)
        {
            for (int m = lane; m < k; m += 32)
                primal_solution[start + m] = 0.0;
            if (lane == 0)
            {
                primal_solution[start + k] = 0.0;
                primal_solution[start + k + 1] = 0.0;
            }
            return;
        }
        double scale = (z + r) / (2.0 * r);
        for (int m = lane; m < k; m += 32)
            primal_solution[start + m] *= scale;
        double w_new = scale * w;
        double z_new = scale * r;
        if (lane == 0)
        {
            primal_solution[start + k] = (z_new + w_new) * INV_SQRT2;
            primal_solution[start + k + 1] = (z_new - w_new) * INV_SQRT2;
        }
        return;
    }

    double my_inv = (lane == 0) ? w * w : 0.0;
    double my_pos = (lane == 0) ? w * w : 0.0;
    for (int m = lane; m < k; m += 32)
    {
        double dh = variable_rescaling[start + m] / d_st;
        double v_m = primal_solution[start + m];
        my_inv += (v_m / dh) * (v_m / dh);
        my_pos += (v_m * dh) * (v_m * dh);
    }
    for (int o = 16; o > 0; o >>= 1)
    {
        my_inv += __shfl_xor_sync(MASK, my_inv, o);
        my_pos += __shfl_xor_sync(MASK, my_pos, o);
    }
    double r_inv = sqrt(my_inv);
    if (r_inv <= z)
        return;
    double r_pos = sqrt(my_pos);
    if (r_pos <= -z)
    {
        for (int m = lane; m < k; m += 32)
            primal_solution[start + m] = 0.0;
        if (lane == 0)
        {
            primal_solution[start + k] = 0.0;
            primal_solution[start + k + 1] = 0.0;
        }
        return;
    }

    double lo, hi;
    bool z_pos = (z > 0.0);
    if (z_pos)
    {
        lo = 0.0;
        hi = 0.5 - 1e-14;
    }
    else
    {
        lo = 0.5 + 1e-14;
        hi = 1.0;
        for (int doubling = 0; doubling < 60; ++doubling)
        {
            double my_sum = (lane == 0) ? (w / (1.0 + 2.0 * hi)) * (w / (1.0 + 2.0 * hi)) : 0.0;
            for (int m = lane; m < k; m += 32)
            {
                double dh = variable_rescaling[start + m] / d_st;
                double dh2 = dh * dh;
                double tt = primal_solution[start + m] * dh / (dh2 + 2.0 * hi);
                my_sum += tt * tt;
            }
            for (int o = 16; o > 0; o >>= 1)
                my_sum += __shfl_xor_sync(MASK, my_sum, o);
            double zt_hi = z / (1.0 - 2.0 * hi);
            double f_hi = my_sum - zt_hi * zt_hi;
            if (f_hi > 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_lam = warm_start[blk];
    if (warm_lam > lo && warm_lam < hi)
    {
        double my_sum = (lane == 0) ? (w / (1.0 + 2.0 * warm_lam)) * (w / (1.0 + 2.0 * warm_lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double dh = variable_rescaling[start + m] / d_st;
            double dh2 = dh * dh;
            double tt = primal_solution[start + m] * dh / (dh2 + 2.0 * warm_lam);
            my_sum += tt * tt;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(MASK, my_sum, o);
        double zt = z / (1.0 - 2.0 * warm_lam);
        double f = my_sum - zt * zt;
        if (fabs(f) < 1e-12)
        {
            double w_new = w / (1.0 + 2.0 * warm_lam);
            double z_new = z / (1.0 - 2.0 * warm_lam);
            for (int m = lane; m < k; m += 32)
            {
                double dh = variable_rescaling[start + m] / d_st;
                double dh2 = dh * dh;
                primal_solution[start + m] = primal_solution[start + m] * dh2 / (dh2 + 2.0 * warm_lam);
            }
            if (lane == 0)
            {
                primal_solution[start + k] = (z_new + w_new) * INV_SQRT2;
                primal_solution[start + k + 1] = (z_new - w_new) * INV_SQRT2;
            }
            return;
        }
        if (z_pos)
        {
            if (f > 0.0)
                lo = warm_lam;
            else
                hi = warm_lam;
        }
        else
        {
            if (f > 0.0)
                hi = warm_lam;
            else
                lo = warm_lam;
        }
    }

    for (int it = 0; it < 60; ++it)
    {
        double lam = 0.5 * (lo + hi);
        double my_sum = (lane == 0) ? (w / (1.0 + 2.0 * lam)) * (w / (1.0 + 2.0 * lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double dh = variable_rescaling[start + m] / d_st;
            double dh2 = dh * dh;
            double tt = primal_solution[start + m] * dh / (dh2 + 2.0 * lam);
            my_sum += tt * tt;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(MASK, my_sum, o);
        double zt = z / (1.0 - 2.0 * lam);
        double f = my_sum - zt * zt;
        if (z_pos)
        {
            if (f > 0.0)
                lo = lam;
            else
                hi = lam;
        }
        else
        {
            if (f > 0.0)
                hi = lam;
            else
                lo = lam;
        }
        if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
            break;
    }
    double lam = 0.5 * (lo + hi);
    if (lane == 0)
        warm_start[blk] = lam;

    double w_new = w / (1.0 + 2.0 * lam);
    double z_new = z / (1.0 - 2.0 * lam);
    for (int m = lane; m < k; m += 32)
    {
        double dh = variable_rescaling[start + m] / d_st;
        double dh2 = dh * dh;
        primal_solution[start + m] = primal_solution[start + m] * dh2 / (dh2 + 2.0 * lam);
    }
    if (lane == 0)
    {
        primal_solution[start + k] = (z_new + w_new) * INV_SQRT2;
        primal_solution[start + k + 1] = (z_new - w_new) * INV_SQRT2;
    }
}

__global__ void compute_cone_dual_residual_warp_kernel(double *__restrict__ dual_residual,
                                                       const double *__restrict__ objective_vector,
                                                       const double *__restrict__ dual_product,
                                                       const double *__restrict__ variable_rescaling,
                                                       double *__restrict__ warm_start,
                                                       const int *__restrict__ start_idx,
                                                       const int *__restrict__ v_dim,
                                                       int num_blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk = tid >> 5;
    int lane = tid & 31;
    if (blk >= num_blocks)
        return;

    const double INV_SQRT2 = 0.7071067811865475;
    const unsigned MASK = 0xffffffffu;

    int start = start_idx[blk];
    int k = v_dim[blk];

    double r_s = objective_vector[start + k] - dual_product[start + k];
    double r_t = objective_vector[start + k + 1] - dual_product[start + k + 1];
    double r_w = (r_s - r_t) * INV_SQRT2;
    double r_z = (r_s + r_t) * INV_SQRT2;

    double d_s = variable_rescaling[start + k];
    double d_t = variable_rescaling[start + k + 1];
    double d_st = sqrt(d_s * d_t);

    int my_diff = 0;
    for (int m = lane; m < k; m += 32)
    {
        if (variable_rescaling[start + m] != d_st)
            my_diff = 1;
    }
    for (int o = 16; o > 0; o >>= 1)
        my_diff |= __shfl_xor_sync(MASK, my_diff, o);

    if (my_diff == 0)
    {
        double my_sumsq = (lane == 0) ? r_w * r_w : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            my_sumsq += rc_m * rc_m;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sumsq += __shfl_xor_sync(MASK, my_sumsq, o);
        double r_norm = sqrt(my_sumsq);

        double v_factor, p_s, p_t;
        if (r_norm <= r_z)
        {
            v_factor = 0.0;
            p_s = r_s;
            p_t = r_t;
        }
        else if (r_norm <= -r_z)
        {
            v_factor = 1.0;
            p_s = 0.0;
            p_t = 0.0;
        }
        else
        {
            double scale = (r_z + r_norm) / (2.0 * r_norm);
            v_factor = 1.0 - scale;
            double w_new = scale * r_w;
            double z_new = scale * r_norm;
            p_s = (z_new + w_new) * INV_SQRT2;
            p_t = (z_new - w_new) * INV_SQRT2;
        }

        for (int m = lane; m < k; m += 32)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = rc_m * v_factor * variable_rescaling[start + m];
        }
        if (lane == 0)
        {
            dual_residual[start + k] = (r_s - p_s) * variable_rescaling[start + k];
            dual_residual[start + k + 1] = (r_t - p_t) * variable_rescaling[start + k + 1];
        }
        return;
    }

    double my_inv = (lane == 0) ? r_w * r_w : 0.0;
    double my_pos = (lane == 0) ? r_w * r_w : 0.0;
    for (int m = lane; m < k; m += 32)
    {
        double e_m = d_st / variable_rescaling[start + m];
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        my_inv += (rc_m / e_m) * (rc_m / e_m);
        my_pos += (rc_m * e_m) * (rc_m * e_m);
    }
    for (int o = 16; o > 0; o >>= 1)
    {
        my_inv += __shfl_xor_sync(MASK, my_inv, o);
        my_pos += __shfl_xor_sync(MASK, my_pos, o);
    }
    double r_inv = sqrt(my_inv);
    double r_pos = sqrt(my_pos);

    if (r_inv <= r_z)
    {
        for (int m = lane; m < k; m += 32)
            dual_residual[start + m] = 0.0;
        if (lane == 0)
        {
            dual_residual[start + k] = 0.0;
            dual_residual[start + k + 1] = 0.0;
        }
        return;
    }
    if (r_pos <= -r_z)
    {
        for (int m = lane; m < k; m += 32)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = rc_m * variable_rescaling[start + m];
        }
        if (lane == 0)
        {
            dual_residual[start + k] = r_s * variable_rescaling[start + k];
            dual_residual[start + k + 1] = r_t * variable_rescaling[start + k + 1];
        }
        return;
    }

    double lo, hi;
    bool z_pos = (r_z > 0.0);
    if (z_pos)
    {
        lo = 0.0;
        hi = 0.5 - 1e-14;
    }
    else
    {
        lo = 0.5 + 1e-14;
        hi = 1.0;
        for (int doubling = 0; doubling < 60; ++doubling)
        {
            double my_sum = (lane == 0) ? (r_w / (1.0 + 2.0 * hi)) * (r_w / (1.0 + 2.0 * hi)) : 0.0;
            for (int m = lane; m < k; m += 32)
            {
                double e_m = d_st / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double tt = rc_m * e_m / (e_m2 + 2.0 * hi);
                my_sum += tt * tt;
            }
            for (int o = 16; o > 0; o >>= 1)
                my_sum += __shfl_xor_sync(MASK, my_sum, o);
            double zt_hi = r_z / (1.0 - 2.0 * hi);
            double f_hi = my_sum - zt_hi * zt_hi;
            if (f_hi > 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_lam = warm_start[blk];
    if (warm_lam > lo && warm_lam < hi)
    {
        double my_sum = (lane == 0) ? (r_w / (1.0 + 2.0 * warm_lam)) * (r_w / (1.0 + 2.0 * warm_lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double e_m = d_st / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double tt = rc_m * e_m / (e_m2 + 2.0 * warm_lam);
            my_sum += tt * tt;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(MASK, my_sum, o);
        double zt = r_z / (1.0 - 2.0 * warm_lam);
        double f = my_sum - zt * zt;
        if (fabs(f) < 1e-12)
        {
            double p_w_w = r_w / (1.0 + 2.0 * warm_lam);
            double p_z_w = r_z / (1.0 - 2.0 * warm_lam);
            double p_s_w = (p_z_w + p_w_w) * INV_SQRT2;
            double p_t_w = (p_z_w - p_w_w) * INV_SQRT2;
            for (int m = lane; m < k; m += 32)
            {
                double e_m = d_st / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double p_m = rc_m * e_m2 / (e_m2 + 2.0 * warm_lam);
                dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
            }
            if (lane == 0)
            {
                dual_residual[start + k] = (r_s - p_s_w) * variable_rescaling[start + k];
                dual_residual[start + k + 1] = (r_t - p_t_w) * variable_rescaling[start + k + 1];
            }
            return;
        }
        if (z_pos)
        {
            if (f > 0.0)
                lo = warm_lam;
            else
                hi = warm_lam;
        }
        else
        {
            if (f > 0.0)
                hi = warm_lam;
            else
                lo = warm_lam;
        }
    }

    for (int it = 0; it < 60; ++it)
    {
        double lam = 0.5 * (lo + hi);
        double my_sum = (lane == 0) ? (r_w / (1.0 + 2.0 * lam)) * (r_w / (1.0 + 2.0 * lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double e_m = d_st / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double tt = rc_m * e_m / (e_m2 + 2.0 * lam);
            my_sum += tt * tt;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(MASK, my_sum, o);
        double zt = r_z / (1.0 - 2.0 * lam);
        double f = my_sum - zt * zt;
        if (z_pos)
        {
            if (f > 0.0)
                lo = lam;
            else
                hi = lam;
        }
        else
        {
            if (f > 0.0)
                hi = lam;
            else
                lo = lam;
        }
        if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
            break;
    }
    double lam = 0.5 * (lo + hi);
    if (lane == 0)
        warm_start[blk] = lam;

    double p_w = r_w / (1.0 + 2.0 * lam);
    double p_z = r_z / (1.0 - 2.0 * lam);
    double p_s = (p_z + p_w) * INV_SQRT2;
    double p_t = (p_z - p_w) * INV_SQRT2;

    for (int m = lane; m < k; m += 32)
    {
        double e_m = d_st / variable_rescaling[start + m];
        double e_m2 = e_m * e_m;
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        double p_m = rc_m * e_m2 / (e_m2 + 2.0 * lam);
        dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
    }
    if (lane == 0)
    {
        dual_residual[start + k] = (r_s - p_s) * variable_rescaling[start + k];
        dual_residual[start + k + 1] = (r_t - p_t) * variable_rescaling[start + k + 1];
    }
}

__global__ void project_standard_soc_warp_kernel(double *__restrict__ primal_solution,
                                                 const double *__restrict__ variable_rescaling,
                                                 double *__restrict__ warm_start,
                                                 const int *__restrict__ start_idx,
                                                 const int *__restrict__ v_dim,
                                                 int num_blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk = tid >> 5;
    int lane = tid & 31;
    if (blk >= num_blocks)
        return;

    const unsigned MASK = 0xffffffffu;

    int start = start_idx[blk];
    int k = v_dim[blk];

    double w = primal_solution[start + k];
    double z = primal_solution[start + k + 1];

    double d_z = variable_rescaling[start + k + 1];
    double dhat_w = variable_rescaling[start + k] / d_z;
    double dhat_w2 = dhat_w * dhat_w;

    int my_diff = (lane == 0 && dhat_w != 1.0) ? 1 : 0;
    for (int m = lane; m < k; m += 32)
    {
        if (variable_rescaling[start + m] != d_z)
            my_diff = 1;
    }
    for (int o = 16; o > 0; o >>= 1)
        my_diff |= __shfl_xor_sync(MASK, my_diff, o);

    if (my_diff == 0)
    {
        double my_sumsq = (lane == 0) ? w * w : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double v_m = primal_solution[start + m];
            my_sumsq += v_m * v_m;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sumsq += __shfl_xor_sync(MASK, my_sumsq, o);
        double r = sqrt(my_sumsq);
        if (r <= z)
            return;
        if (r <= -z)
        {
            for (int m = lane; m < k; m += 32)
                primal_solution[start + m] = 0.0;
            if (lane == 0)
            {
                primal_solution[start + k] = 0.0;
                primal_solution[start + k + 1] = 0.0;
            }
            return;
        }
        double scale = (z + r) / (2.0 * r);
        for (int m = lane; m < k; m += 32)
            primal_solution[start + m] *= scale;
        if (lane == 0)
        {
            primal_solution[start + k] = scale * w;
            primal_solution[start + k + 1] = scale * r;
        }
        return;
    }

    double my_inv = (lane == 0) ? (w / dhat_w) * (w / dhat_w) : 0.0;
    double my_pos = (lane == 0) ? (w * dhat_w) * (w * dhat_w) : 0.0;
    for (int m = lane; m < k; m += 32)
    {
        double dh = variable_rescaling[start + m] / d_z;
        double v_m = primal_solution[start + m];
        my_inv += (v_m / dh) * (v_m / dh);
        my_pos += (v_m * dh) * (v_m * dh);
    }
    for (int o = 16; o > 0; o >>= 1)
    {
        my_inv += __shfl_xor_sync(MASK, my_inv, o);
        my_pos += __shfl_xor_sync(MASK, my_pos, o);
    }
    double r_inv = sqrt(my_inv);
    if (r_inv <= z)
        return;
    double r_pos = sqrt(my_pos);
    if (r_pos <= -z)
    {
        for (int m = lane; m < k; m += 32)
            primal_solution[start + m] = 0.0;
        if (lane == 0)
        {
            primal_solution[start + k] = 0.0;
            primal_solution[start + k + 1] = 0.0;
        }
        return;
    }

    double lo, hi;
    bool z_pos = (z > 0.0);
    if (z_pos)
    {
        lo = 0.0;
        hi = 0.5 - 1e-14;
    }
    else
    {
        lo = 0.5 + 1e-14;
        hi = 1.0;
        for (int doubling = 0; doubling < 60; ++doubling)
        {
            double my_sum =
                (lane == 0) ? (w * dhat_w / (dhat_w2 + 2.0 * hi)) * (w * dhat_w / (dhat_w2 + 2.0 * hi)) : 0.0;
            for (int m = lane; m < k; m += 32)
            {
                double dh = variable_rescaling[start + m] / d_z;
                double dh2 = dh * dh;
                double tt = primal_solution[start + m] * dh / (dh2 + 2.0 * hi);
                my_sum += tt * tt;
            }
            for (int o = 16; o > 0; o >>= 1)
                my_sum += __shfl_xor_sync(MASK, my_sum, o);
            double zt_hi = z / (1.0 - 2.0 * hi);
            double f_hi = my_sum - zt_hi * zt_hi;
            if (f_hi > 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_lam = warm_start[blk];
    if (warm_lam > lo && warm_lam < hi)
    {
        double my_sum =
            (lane == 0) ? (w * dhat_w / (dhat_w2 + 2.0 * warm_lam)) * (w * dhat_w / (dhat_w2 + 2.0 * warm_lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double dh = variable_rescaling[start + m] / d_z;
            double dh2 = dh * dh;
            double tt = primal_solution[start + m] * dh / (dh2 + 2.0 * warm_lam);
            my_sum += tt * tt;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(MASK, my_sum, o);
        double zt = z / (1.0 - 2.0 * warm_lam);
        double f = my_sum - zt * zt;
        if (fabs(f) < 1e-12)
        {
            for (int m = lane; m < k; m += 32)
            {
                double dh = variable_rescaling[start + m] / d_z;
                double dh2 = dh * dh;
                primal_solution[start + m] = primal_solution[start + m] * dh2 / (dh2 + 2.0 * warm_lam);
            }
            if (lane == 0)
            {
                primal_solution[start + k + 1] = z / (1.0 - 2.0 * warm_lam);
                primal_solution[start + k] = w * dhat_w2 / (dhat_w2 + 2.0 * warm_lam);
            }
            return;
        }
        if (z_pos)
        {
            if (f > 0.0)
                lo = warm_lam;
            else
                hi = warm_lam;
        }
        else
        {
            if (f > 0.0)
                hi = warm_lam;
            else
                lo = warm_lam;
        }
    }

    for (int it = 0; it < 60; ++it)
    {
        double lam = 0.5 * (lo + hi);
        double my_sum = (lane == 0) ? (w * dhat_w / (dhat_w2 + 2.0 * lam)) * (w * dhat_w / (dhat_w2 + 2.0 * lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double dh = variable_rescaling[start + m] / d_z;
            double dh2 = dh * dh;
            double tt = primal_solution[start + m] * dh / (dh2 + 2.0 * lam);
            my_sum += tt * tt;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(MASK, my_sum, o);
        double zt = z / (1.0 - 2.0 * lam);
        double f = my_sum - zt * zt;
        if (z_pos)
        {
            if (f > 0.0)
                lo = lam;
            else
                hi = lam;
        }
        else
        {
            if (f > 0.0)
                hi = lam;
            else
                lo = lam;
        }
        if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
            break;
    }
    double lam = 0.5 * (lo + hi);
    if (lane == 0)
        warm_start[blk] = lam;

    for (int m = lane; m < k; m += 32)
    {
        double dh = variable_rescaling[start + m] / d_z;
        double dh2 = dh * dh;
        primal_solution[start + m] = primal_solution[start + m] * dh2 / (dh2 + 2.0 * lam);
    }
    if (lane == 0)
    {
        primal_solution[start + k + 1] = z / (1.0 - 2.0 * lam);
        primal_solution[start + k] = w * dhat_w2 / (dhat_w2 + 2.0 * lam);
    }
}

__global__ void compute_cone_dual_residual_standard_warp_kernel(double *__restrict__ dual_residual,
                                                                const double *__restrict__ objective_vector,
                                                                const double *__restrict__ dual_product,
                                                                const double *__restrict__ variable_rescaling,
                                                                double *__restrict__ warm_start,
                                                                const int *__restrict__ start_idx,
                                                                const int *__restrict__ v_dim,
                                                                int num_blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk = tid >> 5;
    int lane = tid & 31;
    if (blk >= num_blocks)
        return;

    const unsigned MASK = 0xffffffffu;

    int start = start_idx[blk];
    int k = v_dim[blk];

    double r_w = objective_vector[start + k] - dual_product[start + k];
    double r_z = objective_vector[start + k + 1] - dual_product[start + k + 1];

    double d_z = variable_rescaling[start + k + 1];
    double e_w = d_z / variable_rescaling[start + k];
    double e_w2 = e_w * e_w;

    int my_diff = (lane == 0 && e_w != 1.0) ? 1 : 0;
    for (int m = lane; m < k; m += 32)
    {
        if (variable_rescaling[start + m] != d_z)
            my_diff = 1;
    }
    for (int o = 16; o > 0; o >>= 1)
        my_diff |= __shfl_xor_sync(MASK, my_diff, o);

    if (my_diff == 0)
    {
        double my_sumsq = (lane == 0) ? r_w * r_w : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            my_sumsq += rc_m * rc_m;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sumsq += __shfl_xor_sync(MASK, my_sumsq, o);
        double r = sqrt(my_sumsq);
        double v_factor, p_w, p_z;
        if (r <= r_z)
        {
            v_factor = 0.0;
            p_w = r_w;
            p_z = r_z;
        }
        else if (r <= -r_z)
        {
            v_factor = 1.0;
            p_w = 0.0;
            p_z = 0.0;
        }
        else
        {
            double scale = (r_z + r) / (2.0 * r);
            v_factor = 1.0 - scale;
            p_w = scale * r_w;
            p_z = scale * r;
        }
        for (int m = lane; m < k; m += 32)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = rc_m * v_factor * variable_rescaling[start + m];
        }
        if (lane == 0)
        {
            dual_residual[start + k] = (r_w - p_w) * variable_rescaling[start + k];
            dual_residual[start + k + 1] = (r_z - p_z) * variable_rescaling[start + k + 1];
        }
        return;
    }

    double my_inv = (lane == 0) ? (r_w / e_w) * (r_w / e_w) : 0.0;
    double my_pos = (lane == 0) ? (r_w * e_w) * (r_w * e_w) : 0.0;
    for (int m = lane; m < k; m += 32)
    {
        double e_m = d_z / variable_rescaling[start + m];
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        my_inv += (rc_m / e_m) * (rc_m / e_m);
        my_pos += (rc_m * e_m) * (rc_m * e_m);
    }
    for (int o = 16; o > 0; o >>= 1)
    {
        my_inv += __shfl_xor_sync(MASK, my_inv, o);
        my_pos += __shfl_xor_sync(MASK, my_pos, o);
    }
    double r_inv = sqrt(my_inv);
    double r_pos = sqrt(my_pos);

    if (r_inv <= r_z)
    {
        for (int m = lane; m < k; m += 32)
            dual_residual[start + m] = 0.0;
        if (lane == 0)
        {
            dual_residual[start + k] = 0.0;
            dual_residual[start + k + 1] = 0.0;
        }
        return;
    }
    if (r_pos <= -r_z)
    {
        for (int m = lane; m < k; m += 32)
        {
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            dual_residual[start + m] = rc_m * variable_rescaling[start + m];
        }
        if (lane == 0)
        {
            dual_residual[start + k] = r_w * variable_rescaling[start + k];
            dual_residual[start + k + 1] = r_z * variable_rescaling[start + k + 1];
        }
        return;
    }

    double lo, hi;
    bool z_pos = (r_z > 0.0);
    if (z_pos)
    {
        lo = 0.0;
        hi = 0.5 - 1e-14;
    }
    else
    {
        lo = 0.5 + 1e-14;
        hi = 1.0;
        for (int doubling = 0; doubling < 60; ++doubling)
        {
            double my_sum = (lane == 0) ? (r_w * e_w / (e_w2 + 2.0 * hi)) * (r_w * e_w / (e_w2 + 2.0 * hi)) : 0.0;
            for (int m = lane; m < k; m += 32)
            {
                double e_m = d_z / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double tt = rc_m * e_m / (e_m2 + 2.0 * hi);
                my_sum += tt * tt;
            }
            for (int o = 16; o > 0; o >>= 1)
                my_sum += __shfl_xor_sync(MASK, my_sum, o);
            double zt_hi = r_z / (1.0 - 2.0 * hi);
            double f_hi = my_sum - zt_hi * zt_hi;
            if (f_hi > 0.0)
                break;
            lo = hi;
            hi *= 2.0;
        }
    }

    double warm_lam = warm_start[blk];
    if (warm_lam > lo && warm_lam < hi)
    {
        double my_sum =
            (lane == 0) ? (r_w * e_w / (e_w2 + 2.0 * warm_lam)) * (r_w * e_w / (e_w2 + 2.0 * warm_lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double e_m = d_z / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double tt = rc_m * e_m / (e_m2 + 2.0 * warm_lam);
            my_sum += tt * tt;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(MASK, my_sum, o);
        double zt = r_z / (1.0 - 2.0 * warm_lam);
        double f = my_sum - zt * zt;
        if (fabs(f) < 1e-12)
        {
            double p_z_w = r_z / (1.0 - 2.0 * warm_lam);
            double p_w_w = r_w * e_w2 / (e_w2 + 2.0 * warm_lam);
            for (int m = lane; m < k; m += 32)
            {
                double e_m = d_z / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double p_m = rc_m * e_m2 / (e_m2 + 2.0 * warm_lam);
                dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
            }
            if (lane == 0)
            {
                dual_residual[start + k] = (r_w - p_w_w) * variable_rescaling[start + k];
                dual_residual[start + k + 1] = (r_z - p_z_w) * variable_rescaling[start + k + 1];
            }
            return;
        }
        if (z_pos)
        {
            if (f > 0.0)
                lo = warm_lam;
            else
                hi = warm_lam;
        }
        else
        {
            if (f > 0.0)
                hi = warm_lam;
            else
                lo = warm_lam;
        }
    }

    for (int it = 0; it < 60; ++it)
    {
        double lam = 0.5 * (lo + hi);
        double my_sum = (lane == 0) ? (r_w * e_w / (e_w2 + 2.0 * lam)) * (r_w * e_w / (e_w2 + 2.0 * lam)) : 0.0;
        for (int m = lane; m < k; m += 32)
        {
            double e_m = d_z / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double tt = rc_m * e_m / (e_m2 + 2.0 * lam);
            my_sum += tt * tt;
        }
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(MASK, my_sum, o);
        double zt = r_z / (1.0 - 2.0 * lam);
        double f = my_sum - zt * zt;
        if (z_pos)
        {
            if (f > 0.0)
                lo = lam;
            else
                hi = lam;
        }
        else
        {
            if (f > 0.0)
                hi = lam;
            else
                lo = lam;
        }
        if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
            break;
    }
    double lam = 0.5 * (lo + hi);
    if (lane == 0)
        warm_start[blk] = lam;

    double p_z = r_z / (1.0 - 2.0 * lam);
    double p_w = r_w * e_w2 / (e_w2 + 2.0 * lam);

    for (int m = lane; m < k; m += 32)
    {
        double e_m = d_z / variable_rescaling[start + m];
        double e_m2 = e_m * e_m;
        double rc_m = objective_vector[start + m] - dual_product[start + m];
        double p_m = rc_m * e_m2 / (e_m2 + 2.0 * lam);
        dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
    }
    if (lane == 0)
    {
        dual_residual[start + k] = (r_w - p_w) * variable_rescaling[start + k];
        dual_residual[start + k + 1] = (r_z - p_z) * variable_rescaling[start + k + 1];
    }
}

/* Project (r1, r2, r3) onto D * K_exp with D = diag(d1, d2, d3).
   Reduces to standard projection onto K_exp when d1=d2=d3=1.
   Algorithm: change variables u = D^{-1} x to convert to weighted projection onto K_exp.
   With weights d_i^2, KKT gives a 1D Newton equation in rho = u_1/u_2 with parameters
   alpha = (d3/d1)^2, beta = (d3/d2)^2. */
__device__ static inline void project_exp_cone_point(
    double r1, double r2, double r3, double d1, double d2, double d3, double *xo, double *yo, double *zo)
{
    const double E_CONST = 2.718281828459045;
    double rr1 = r1 / d1, rr2 = r2 / d2, rr3 = r3 / d3;

    /* Case 1: r in D K_exp  <=>  D^{-1} r in K_exp. */
    if (rr2 > 0.0)
    {
        double ratio = rr1 / rr2;
        if (ratio < 700.0 && rr2 * exp(ratio) <= rr3)
        {
            *xo = r1;
            *yo = r2;
            *zo = r3;
            return;
        }
    }
    else if (rr2 == 0.0 && rr1 <= 0.0 && rr3 >= 0.0)
    {
        *xo = r1;
        *yo = r2;
        *zo = r3;
        return;
    }

    /* Case 2: r in polar = -D^{-1} K_exp^*.
       -D r in K_exp^*: -d1 r1 < 0 (so r1>0), and  -(-d1 r1) exp(-d2 r2/(-d1 r1)) <= e * (-d3 r3)
                       i.e.  d1 r1 * exp(d2 r2/(d1 r1)) <= -e d3 r3. */
    if (r1 > 0.0)
    {
        double ratio = (d2 * r2) / (d1 * r1);
        if (ratio < 700.0 && d1 * r1 * exp(ratio) + E_CONST * d3 * r3 <= 0.0)
        {
            *xo = 0.0;
            *yo = 0.0;
            *zo = 0.0;
            return;
        }
    }
    else if (r1 == 0.0 && r2 <= 0.0 && r3 <= 0.0)
    {
        *xo = 0.0;
        *yo = 0.0;
        *zo = 0.0;
        return;
    }

    /* Case 3: recession edge.  D K_exp contains {(c1, 0, c3) : c1 <= 0, c3 >= 0}
       (no scaling needed since y=0). Project x and z components, zero out y. */
    if (rr1 <= 0.0 && rr2 <= 0.0)
    {
        *xo = r1;
        *yo = 0.0;
        *zo = (rr3 < 0.0) ? 0.0 : r3;
        return;
    }

    /* Case 4: Newton iteration on f(rho)=0 with rho = u_1/u_2 (u = D^{-1} x). */
    double alpha = (d3 / d1) * (d3 / d1);
    double beta = (d3 / d2) * (d3 / d2);
    double rho = 0.0;
    bool diverged = false;
    for (int it = 0; it < 100; ++it)
    {
        if (rho >= 299.0 || rho <= -299.0)
        {
            diverged = true;
            break;
        }
        double e_rho = exp(rho);
        double e_2rho = e_rho * e_rho;
        double one_m_rho = 1.0 - rho;

        double a_term = alpha - beta * rho * one_m_rho;
        double b_term = beta * rr1 * one_m_rho - alpha * rr2;
        double f = rr1 - rr2 * rho + rr3 * e_rho * a_term + e_2rho * b_term;

        double da_drho = -beta * (1.0 - 2.0 * rho);
        double db_drho = -beta * rr1;
        double df = -rr2 + rr3 * e_rho * (a_term + da_drho) + e_2rho * (2.0 * b_term + db_drho);

        if (fabs(df) < 1e-300)
            break;
        double step = f / df;
        /* Damp aggressive steps to prevent runaway divergence. */
        if (step > 10.0)
            step = 10.0;
        if (step < -10.0)
            step = -10.0;
        rho -= step;
        if (fabs(step) < 1e-13 * (1.0 + fabs(rho)))
            break;
    }

    double e_rho = exp(rho);
    double denom = rho + alpha * e_rho * e_rho;
    double u2 = (rr1 + alpha * rr3 * e_rho) / denom;

    /* Robustness: if Newton failed to find a valid cone point, fall back to
       projection onto the recession edge {(x, 0, z) : x <= 0, z >= 0}. */
    if (diverged || !isfinite(u2) || u2 <= 0.0)
    {
        *xo = (r1 < 0.0) ? r1 : 0.0;
        *yo = 0.0;
        *zo = (r3 > 0.0) ? r3 : 0.0;
        return;
    }

    double u1 = rho * u2;
    double u3 = u2 * e_rho;

    *xo = d1 * u1;
    *yo = d2 * u2;
    *zo = d3 * u3;
}

/* Project (rz0, rt0) onto the y-fixed cross-section of D K_exp with D = diag(d_r, d_y, d_t),
   y slot held at scaled value ry. In scaled coordinates the constraint is
       d_t * y_eff * exp((rz/d_r) / y_eff) <= rt,    y_eff = ry / d_y > 0.
   Parameterize the boundary by u = exp((rz/d_r) / y_eff) > 0:
       rz_s = d_r * y_eff * log(u),   rt_s = d_t * y_eff * u.
   The KKT stationarity for distance^2 = (rz - rz0)^2 + (rt - rt0)^2 gives
       f(u) = d_t^2 * y_eff * u^2 - d_t * rt0 * u + d_r^2 * y_eff * log(u) - d_r * rz0 = 0.
   We bracket [u_lo, u_hi] with f(u_lo) < 0, f(u_hi) > 0 and run Newton with bisection
   fallback. */
__device__ static inline void
project_2d_exp_persp(double rz0, double ry, double rt0, double d_r, double d_y, double d_t, double *rzo, double *rto)
{
    if (d_r <= 0.0 || d_y <= 0.0 || d_t <= 0.0)
    {
        *rzo = rz0;
        *rto = rt0;
        return;
    }
    double y_eff = ry / d_y;
    if (y_eff <= 0.0)
    {
        /* Degenerate: K_exp interior needs y > 0. Project to recession edge. */
        *rzo = (rz0 < 0.0) ? rz0 : 0.0;
        *rto = (rt0 > 0.0) ? rt0 : 0.0;
        return;
    }

    /* In-cone test. */
    double arg = (rz0 / d_r) / y_eff;
    if (arg < 700.0)
    {
        double rhs = y_eff * d_t * exp(arg);
        if (rhs <= rt0)
        {
            *rzo = rz0;
            *rto = rt0;
            return;
        }
    }

    double a = d_t * d_t * y_eff; /* > 0 */
    double b = d_t * rt0;         /* any sign */
    double c = d_r * d_r * y_eff; /* > 0 */
    double e = d_r * rz0;         /* any sign */

    /* Bracket. f(u_lo) < 0, f(u_hi) > 0. */
    double u_lo = 1e-30;
    double u_hi = 1.0;
    for (int g = 0; g < 200; ++g)
    {
        double lu = log(u_hi);
        double f_hi = a * u_hi * u_hi - b * u_hi + c * lu - e;
        if (isfinite(f_hi) && f_hi > 0.0)
            break;
        u_hi *= 4.0;
        if (u_hi > 1e150)
            break;
    }
    for (int g = 0; g < 200; ++g)
    {
        double lu = log(u_lo);
        double f_lo = a * u_lo * u_lo - b * u_lo + c * lu - e;
        if (isfinite(f_lo) && f_lo < 0.0)
            break;
        u_lo *= 0.25;
        if (u_lo < 1e-300)
            break;
    }
    if (u_lo >= u_hi)
    {
        *rzo = rz0;
        *rto = rt0;
        return;
    }

    /* Geometric-mean init keeps log(u) well-conditioned. */
    double u = exp(0.5 * (log(u_lo) + log(u_hi)));

    for (int it = 0; it < 80; ++it)
    {
        double lu = log(u);
        double f = a * u * u - b * u + c * lu - e;
        double df = 2.0 * a * u - b + c / u;
        if (f > 0.0)
            u_hi = u;
        else
            u_lo = u;
        double u_new;
        if (df > 1e-300 && isfinite(df) && isfinite(f))
        {
            u_new = u - f / df;
            if (!isfinite(u_new) || u_new <= u_lo || u_new >= u_hi)
                u_new = exp(0.5 * (log(u_lo) + log(u_hi)));
        }
        else
        {
            u_new = exp(0.5 * (log(u_lo) + log(u_hi)));
        }
        if (fabs(u_new - u) < 1e-14 * (1.0 + fabs(u_new)))
        {
            u = u_new;
            break;
        }
        u = u_new;
    }
    *rzo = d_r * y_eff * log(u);
    *rto = d_t * y_eff * u;
}

__global__ void project_exp_cone_kernel(double *__restrict__ primal_solution,
                                        const double *__restrict__ variable_rescaling,
                                        double *__restrict__ warm_start,
                                        const int *__restrict__ start_idx,
                                        const int *__restrict__ v_dim,
                                        const char *__restrict__ is_fixed,
                                        int num_blocks)
{
    (void)warm_start;
    (void)v_dim;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int s_idx = start_idx[blk];
    double r1 = primal_solution[s_idx + 0];
    double r2 = primal_solution[s_idx + 1];
    double r3 = primal_solution[s_idx + 2];

    double d1 = variable_rescaling[s_idx + 0];
    double d2 = variable_rescaling[s_idx + 1];
    double d3 = variable_rescaling[s_idx + 2];

    /* If the y slot is fixed, do a 2D cross-section projection at y = r2 (constant). */
    if (is_fixed && is_fixed[s_idx + 1])
    {
        double zo, to;
        project_2d_exp_persp(r1, r2, r3, d1, d2, d3, &zo, &to);
        primal_solution[s_idx + 0] = zo;
        /* primal_solution[s_idx + 1] left untouched (= r2). */
        primal_solution[s_idx + 2] = to;
        return;
    }

    double xo, yo, zo;
    project_exp_cone_point(r1, r2, r3, d1, d2, d3, &xo, &yo, &zo);

    primal_solution[s_idx + 0] = xo;
    primal_solution[s_idx + 1] = yo;
    primal_solution[s_idx + 2] = zo;
}

__global__ void compute_cone_dual_residual_exp_kernel(double *__restrict__ dual_residual,
                                                      const double *__restrict__ objective_vector,
                                                      const double *__restrict__ dual_product,
                                                      const double *__restrict__ variable_rescaling,
                                                      double *__restrict__ warm_start,
                                                      const int *__restrict__ start_idx,
                                                      const int *__restrict__ v_dim,
                                                      const char *__restrict__ is_fixed,
                                                      int num_blocks)
{
    (void)warm_start;
    (void)v_dim;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;

    int s_idx = start_idx[blk];
    double r1 = objective_vector[s_idx + 0] - dual_product[s_idx + 0];
    double r2 = objective_vector[s_idx + 1] - dual_product[s_idx + 1];
    double r3 = objective_vector[s_idx + 2] - dual_product[s_idx + 2];

    /* y fixed: dual cross-section of K_exp^* projected onto (z, t) plane is {u <= 0, w >= 0}
       (since y free in K_exp^* allows any feasible (u, w) with u <= 0, w >= 0). Residual
       outside this set = (max(r_z, 0), 0, min(r_t, 0)). */
    if (is_fixed && is_fixed[s_idx + 1])
    {
        dual_residual[s_idx + 0] = ((r1 > 0.0) ? r1 : 0.0) * variable_rescaling[s_idx + 0];
        dual_residual[s_idx + 1] = 0.0;
        dual_residual[s_idx + 2] = ((r3 < 0.0) ? r3 : 0.0) * variable_rescaling[s_idx + 2];
        return;
    }

    /* Inverse scaling for dual cone projection (K_exp^* lives in dual space). */
    double d1 = 1.0 / variable_rescaling[s_idx + 0];
    double d2 = 1.0 / variable_rescaling[s_idx + 1];
    double d3 = 1.0 / variable_rescaling[s_idx + 2];

    /* dist(r, K_exp^*) = || -proj_{K_exp}(-r) || via Moreau identity. */
    double xo, yo, zo;
    project_exp_cone_point(-r1, -r2, -r3, d1, d2, d3, &xo, &yo, &zo);

    dual_residual[s_idx + 0] = -xo * variable_rescaling[s_idx + 0];
    dual_residual[s_idx + 1] = -yo * variable_rescaling[s_idx + 1];
    dual_residual[s_idx + 2] = -zo * variable_rescaling[s_idx + 2];
}

__global__ void set_cone_dual_slack_kernel(double *__restrict__ dual_slack,
                                           const double *__restrict__ objective_vector,
                                           const double *__restrict__ dual_product,
                                           const int *__restrict__ start_idx,
                                           const int *__restrict__ v_dim,
                                           int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;
    int start = start_idx[blk];
    int k = v_dim[blk];
    for (int m = 0; m < k + 2; ++m)
    {
        int idx = start + m;
        dual_slack[idx] = objective_vector[idx] - dual_product[idx];
    }
}

__global__ void recompute_reflected_at_cone_kernel(double *__restrict__ reflected_primal,
                                                   const double *__restrict__ pdhg_primal,
                                                   const double *__restrict__ current_primal,
                                                   const int *__restrict__ start_idx,
                                                   const int *__restrict__ v_dim,
                                                   int num_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks)
        return;
    int start = start_idx[blk];
    int k = v_dim[blk];
    for (int m = 0; m < k + 2; ++m)
    {
        int idx = start + m;
        reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
    }
}
