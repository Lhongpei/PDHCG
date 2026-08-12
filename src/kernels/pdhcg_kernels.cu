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

#include "pdhcg_kernels.h"
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

__global__ void
vector_sub_kernel(double *__restrict__ direction, const double *__restrict__ a, const double *__restrict__ b, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        direction[i] = a[i] - b[i];
    }
}

__global__ void
vector_add_kernel(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ out, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        out[i] = a[i] + b[i];
    }
}

__global__ void project_primal_onto_bounds_kernel(double *__restrict__ primal_solution,
                                                  const double *__restrict__ variable_lower_bound,
                                                  const double *__restrict__ variable_upper_bound,
                                                  int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_variables)
    {
        primal_solution[i] = fmax(variable_lower_bound[i], fmin(primal_solution[i], variable_upper_bound[i]));
    }
}

__global__ void prepare_projected_gradient_point_kernel(double *__restrict__ projected_point,
                                                        const double *__restrict__ primal_solution,
                                                        const double *__restrict__ effective_objective,
                                                        const double *__restrict__ dual_product,
                                                        const double *__restrict__ variable_lower_bound,
                                                        const double *__restrict__ variable_upper_bound,
                                                        double step_size,
                                                        int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_variables)
    {
        double gradient = effective_objective[i] - dual_product[i];
        double point = primal_solution[i] - step_size * gradient;
        projected_point[i] = fmax(variable_lower_bound[i], fmin(point, variable_upper_bound[i]));
    }
}

__global__ void augment_projected_gradient_residual_kernel(double *__restrict__ dual_residual,
                                                           const double *__restrict__ primal_solution,
                                                           const double *__restrict__ projected_point,
                                                           const double *__restrict__ variable_rescaling,
                                                           double step_size,
                                                           int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_variables)
    {
        double residual = (primal_solution[i] - projected_point[i]) / step_size * variable_rescaling[i];
        if (!isfinite(residual))
            residual = copysign(INFINITY, residual);
        if (fabs(residual) > fabs(dual_residual[i]))
            dual_residual[i] = residual;
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

__device__ static inline double
next_constraint_dual(double current_dual, double primal_value, double lower_bound, double upper_bound, double step_size)
{
    double projected_value = fmax(lower_bound, fmin(primal_value - current_dual / step_size, upper_bound));
    return current_dual - step_size * primal_value + step_size * projected_value;
}

__global__ void compute_next_pdhg_dual_solution_kernel(const double *current_dual,
                                                       double *reflected_dual,
                                                       const double *primal_product,
                                                       const double *affine_cone_offset,
                                                       const double *constraint_lower_bound,
                                                       const double *constraint_upper_bound,
                                                       int n,
                                                       double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double current = current_dual[i];
        double value = primal_product[i] + affine_cone_offset[i];
        double next =
            next_constraint_dual(current, value, constraint_lower_bound[i], constraint_upper_bound[i], step_size);
        reflected_dual[i] = 2.0 * next - current;
    }
}

__global__ void compute_next_pdhg_dual_solution_major_kernel(const double *current_dual,
                                                             double *pdhg_dual,
                                                             double *reflected_dual,
                                                             const double *primal_product,
                                                             const double *affine_cone_offset,
                                                             const double *constraint_lower_bound,
                                                             const double *constraint_upper_bound,
                                                             int n,
                                                             double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double current = current_dual[i];
        double value = primal_product[i] + affine_cone_offset[i];
        double next =
            next_constraint_dual(current, value, constraint_lower_bound[i], constraint_upper_bound[i], step_size);
        pdhg_dual[i] = next;
        reflected_dual[i] = 2.0 * next - current;
    }
}

__global__ void prepare_constraint_dual_update_kernel(const double *current_dual,
                                                      const double *primal_product,
                                                      const double *affine_cone_offset,
                                                      const double *constraint_lower_bound,
                                                      const double *constraint_upper_bound,
                                                      double *projected_constraint_value,
                                                      int n,
                                                      double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double value = primal_product[i] + affine_cone_offset[i] - current_dual[i] / step_size;
        projected_constraint_value[i] = fmax(constraint_lower_bound[i], fmin(value, constraint_upper_bound[i]));
    }
}

__global__ void finish_constraint_dual_update_kernel(const double *current_dual,
                                                     const double *primal_product,
                                                     const double *affine_cone_offset,
                                                     const double *projected_constraint_value,
                                                     double *pdhg_dual,
                                                     double *reflected_dual,
                                                     int n,
                                                     double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double value = primal_product[i] + affine_cone_offset[i];
        double next_dual = current_dual[i] - step_size * value + step_size * projected_constraint_value[i];
        if (pdhg_dual)
            pdhg_dual[i] = next_dual;
        reflected_dual[i] = 2.0 * next_dual - current_dual[i];
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
                                           const double *affine_cone_offset,
                                           const double *constraint_lower_bound,
                                           const double *constraint_upper_bound,
                                           const double *dual_solution,
                                           double *dual_residual,
                                           const double *dual_product,
                                           const double *dual_slack,
                                           const double *objective_vector,
                                           const double *constraint_rescaling,
                                           const double *variable_rescaling,
                                           double *affine_dual_membership,
                                           double *dual_obj_contribution,
                                           const double *const_lb_finite,
                                           const double *const_ub_finite,
                                           bool defer_constraint_projection,
                                           int num_constraints,
                                           int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {
        double value = primal_product[i] + affine_cone_offset[i];
        double projected_value = fmax(constraint_lower_bound[i], fmin(value, constraint_upper_bound[i]));
        if (defer_constraint_projection)
        {
            primal_residual[i] = projected_value;
            affine_dual_membership[i] = 0.0;
        }
        else
        {
            primal_residual[i] = (value - projected_value) * constraint_rescaling[i];
        }

        dual_obj_contribution[i] = fmax(dual_solution[i], 0.0) * const_lb_finite[i] +
            fmin(dual_solution[i], 0.0) * const_ub_finite[i] - affine_cone_offset[i] * dual_solution[i];
    }
    else if (i < num_constraints + num_variables)
    {
        int idx = i - num_constraints;
        dual_residual[idx] = (objective_vector[idx] - dual_product[idx] - dual_slack[idx]) * variable_rescaling[idx];
    }
}

__global__ void compute_qp_residual_kernel(double *primal_residual,
                                           const double *primal_product,
                                           const double *affine_cone_offset,
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
                                           double *affine_dual_membership,
                                           double *dual_obj_contribution,
                                           const double *const_lb_finite,
                                           const double *const_ub_finite,
                                           const double step_size,
                                           bool defer_constraint_projection,
                                           int num_constraints,
                                           int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {
        double value = primal_product[i] + affine_cone_offset[i];
        double projected_value = fmax(constraint_lower_bound[i], fmin(value, constraint_upper_bound[i]));
        if (defer_constraint_projection)
        {
            primal_residual[i] = projected_value;
            affine_dual_membership[i] = 0.0;
        }
        else
        {
            primal_residual[i] = (value - projected_value) * constraint_rescaling[i];
        }

        dual_obj_contribution[i] = fmax(dual_solution[i], 0.0) * const_lb_finite[i] +
            fmin(dual_solution[i], 0.0) * const_ub_finite[i] - affine_cone_offset[i] * dual_solution[i];
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
                                                 const double *affine_cone_offset,
                                                 const double *dual_solution,
                                                 int num_constraints,
                                                 double *dual_objective_dual_solution_contribution_array)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {
        dual_objective_dual_solution_contribution_array[i] =
            fmax(dual_solution[i], 0.0) * constraint_lower_bound_finite_val[i] +
            fmin(dual_solution[i], 0.0) * constraint_upper_bound_finite_val[i] -
            affine_cone_offset[i] * dual_solution[i];
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
