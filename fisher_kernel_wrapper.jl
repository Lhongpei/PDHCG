using CUDA
using CUDA: CuPtr
using CUDA: CuArray
using CUDA: CuModule, CuFunction
using CUDA: @sync
using CUDA: cudacall
KERNEL_DIR = joinpath(@__DIR__, "src/kernel/compiled_kernel/")
osgm_band_path = joinpath(KERNEL_DIR, "fisher_kernel.ptx")
const fisher_mod = CuModule(read(osgm_band_path))

# double 版本
const launch_gradient_csr_double = CuFunction(fisher_mod,
    "_Z19gradient_csr_kernelILi256EdEviPKT0_S2_S2_PKiS4_S0_S2_S0_S2_S2_PS0_")
const launch_objective_csr_double = CuFunction(fisher_mod,
    "_Z20objective_csr_kernelILi256EdEviPKT0_S2_S2_PKiS4_S0_PS0_S2_S0_S2_")
const launch_utility_csr_double = CuFunction(fisher_mod,
    "_Z18utility_csr_kernelILi256EdEviPKT0_S2_PKiPS0_S0_")

# ------------------------------------------------------------------
# 3. wrapper 函数
# ------------------------------------------------------------------
"""
    launch_gradient_csr!(
        m::Int,
        d_x_val::CuArray{Float64,1},
        d_u_val::CuArray{Float64,1},
        d_w::CuArray{Float64,1},
        d_row_ptr::CuArray{Int32,1},
        d_col_ind::CuArray{Int32,1},
        power::Float64,
        d_p::CuArray{Float64,1},
        pho::Float64,
        d_x_old_val::CuArray{Float64,1},
        d_utility_no_power::CuArray{Float64,1},
        d_gradient::CuArray{Float64,1}
    )

在 GPU 上并行计算 CSR 格式稀疏矩阵的梯度。
"""
function launch_gradient_csr!(
    m::Int,
    d_x_val::CuArray{Float64,1},
    d_u_val::CuArray{Float64,1},
    d_w::CuArray{Float64,1},
    d_row_ptr::CuArray{Int32,1},
    d_col_ind::CuArray{Int32,1},
    power::Float64,
    d_p::CuArray{Float64,1},
    pho::Float64,
    d_x_old_val::CuArray{Float64,1},
    d_utility_no_power::CuArray{Float64,1},
    d_gradient::CuArray{Float64,1}
)
    @sync cudacall(
        launch_gradient_csr_double,
        (Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble},
         CuPtr{Cint}, CuPtr{Cint}, Cdouble,
         CuPtr{Cdouble}, Cdouble,
         CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}),
        m,
        d_x_val, d_u_val, d_w,
        d_row_ptr, d_col_ind, power,
        d_p, pho,
        d_x_old_val, d_utility_no_power, d_gradient
    )
end

"""
    launch_objective_csr!(
        m::Int,
        d_x_val::CuArray{Float64,1},
        d_u_val::CuArray{Float64,1},
        d_w::CuArray{Float64,1},
        d_row_ptr::CuArray{Int32,1},
        d_col_ind::CuArray{Int32,1},
        power::Float64,
        d_objective::CuArray{Float64,1},
        d_p::CuArray{Float64,1},
        pho::Float64,
        d_x_old_val::CuArray{Float64,1}
    )

在 GPU 上并行计算 CSR 格式稀疏矩阵的目标函数值。
"""
function launch_objective_csr!(
    m::Int,
    d_x_val::CuArray{Float64,1},
    d_u_val::CuArray{Float64,1},
    d_w::CuArray{Float64,1},
    d_row_ptr::CuArray{Int32,1},
    d_col_ind::CuArray{Int32,1},
    power::Float64,
    d_objective::CuArray{Float64,1},
    d_p::CuArray{Float64,1},
    pho::Float64,
    d_x_old_val::CuArray{Float64,1}
)
    @sync cudacall(
        launch_objective_csr_double,
        (Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble},
         CuPtr{Cint}, CuPtr{Cint}, Cdouble,
         CuPtr{Cdouble}, CuPtr{Cdouble}, Cdouble,
         CuPtr{Cdouble}),
        m,
        d_x_val, d_u_val, d_w,
        d_row_ptr, d_col_ind, power,
        d_objective, d_p, pho,
        d_x_old_val
    )
end

"""
    launch_utility_csr!(
        m::Int,
        d_x_val::CuArray{Float64,1},
        d_u_val::CuArray{Float64,1},
        d_row_ptr::CuArray{Int32,1},
        d_utility::CuArray{Float64,1},
        power::Float64
    )

在 GPU 上并行计算 CSR 格式稀疏矩阵的 utility 值（每行求和）。
"""
function launch_utility_csr!(
    m::Int,
    d_x_val::CuArray{Float64,1},
    d_u_val::CuArray{Float64,1},
    d_row_ptr::CuArray{Int32,1},
    d_utility::CuArray{Float64,1},
    power::Float64
)
    @sync cudacall(
        launch_utility_csr_double,
        (Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cdouble}, Cdouble),
        m, d_x_val, d_u_val, d_row_ptr, d_utility, power
    )
end

# # ------------------------------------------------------------------
# # 4. 导出接口
# # ------------------------------------------------------------------
# export launch_gradient_csr!, launch_objective_csr!, launch_utility_csr!