#pragma once
#pragma once
#include <cuda_runtime.h>

template <typename real>
void launch_utility_csr(
    int m,
    const real* d_x_val,
    const real* d_u_val,
    const int* d_row_ptr,
    real*       d_utility,
    const real power);

template <typename real>
void launch_objective_csr(
    int m,
    const real* d_x_val,
    const real* d_u_val,
    const real* d_w,
    const int* d_row_ptr,
    const int* d_col_indice,
    const real power,
    real* d_objective,
    const real* d_p,
    const real pho,
    const real* d_x_old_val);

template <typename real>
void launch_gradient_csr(
    int m,
    const real* d_x_val,
    const real* d_u_val,
    const real* d_w,
    const int* d_row_ptr,
    const int* d_col_indice,
    const real power,
    const real* d_p,
    const real pho,
    const real* d_x_old_val,
    real* d_utility_no_power,
    real* d_gradient);