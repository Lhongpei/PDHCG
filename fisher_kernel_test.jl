using StatsBase
using LBFGSB
using LinearAlgebra
using SparseArrays
using Printf
function generate_problem(m, n, nnz)
    # 随机生成权重 w 和效用矩阵 u
    w = rand(m)  # 随机生成 m 个权重
    u = rand(nnz) .+ 1e-5 # 随机生成 nnz 个效用值
    x = rand(nnz) # This 'x' is not returned, it's just a placeholder, maybe x0 is intended?
    
    # Use SparseArrays for colval and rowptr if they represent a sparse matrix structure
    # However, your current structure (CSR-like) is fine for direct indexing.
    rowptr = Vector{Int64}(undef, m + 1)
    colval = Vector{Int64}(undef, nnz)
    
    # 分配非零元素到各行（至少每行1个）
    min_per_row = div(nnz, m)
    extra = nnz - min_per_row * m
    
    rowptr[1] = 1
    # Ensure rowptr is populated correctly and efficiently
    @inbounds for i in 1:m
        rowptr[i+1] = rowptr[i] + min_per_row + (i <= extra ? 1 : 0)
    end

    for i in 1:m
        start_idx = rowptr[i]
        end_idx = rowptr[i+1] - 1
        num_cols_in_row = end_idx - start_idx + 1
        
        # Optimization: If num_cols_in_row is small compared to n, this is fine.
        # If num_cols_in_row is large, and you need unique elements,
        # consider a Fisher-Yates shuffle variant or a Set-based approach
        # if the sampling without replacement is a bottleneck.
        # For typical sparse matrix generation, randperm(n)[1:k] is common.
        
        # Using @inbounds where safe
        @inbounds begin
            # Generate unique columns for this row
            cols = StatsBase.sample(1:n, num_cols_in_row, replace=false) # More explicit for unique sampling
            sort!(cols) # Sorting is necessary for your sparse structure if you iterate efficiently
            colval[start_idx:end_idx] .= cols # Use .= for broadcasting assignment
        end
    end

    b = 0.25 * m * ones(n) # Better: fill(0.25*m, n) for direct allocation with value
    
    # u_sum_dim_1 can be pre-allocated and calculated more efficiently
    u_sum_dim_1 = zeros(m)
    @inbounds for i in 1:m
        start_idx = rowptr[i]
        end_idx = rowptr[i+1] - 1
        # No need for inner loop, just sum the slice if it's contiguous
        # Check if slice access is efficient, it generally is for contiguous arrays
        u_sum_dim_1[i] = sum(u[start_idx:end_idx])
    end

    vec_val = b ./ u_sum_dim_1 # Direct element-wise division between vectors
    
    x0 = similar(x) # 'x' from line 4. Ensure `x0` has the same type as `x` for consistency.
    @inbounds for i in 1:m
        start_idx = rowptr[i]
        end_idx = rowptr[i+1] - 1
        for j in start_idx:end_idx
            x0[j] = vec_val[i] * u[j]
        end
    end
    return x0, w, u, b, colval, rowptr
end

using LinearAlgebra # For norm

# This function calculates the gradient of the original objective function,
# i.e., -w_i * log(U_i(x)) term's gradient w.r.t. x_j
function calculate_objective_gradient_part(x_val, w_val, u_val, row_ptr, m_len, power)
    utility_raw = cal_utility(x_val, u_val, nothing, row_ptr, m_len, power) # col_indice not needed for utility_raw calculation
    grad_obj_part = similar(x_val) # nnz length

    @inbounds for i in 1:m_len
        start_idx = row_ptr[i]
        end_idx = row_ptr[i+1] - 1
        
        utility_denom = max(1e-12, utility_raw[i])
        
        for k_idx in start_idx:end_idx
            grad_obj_part[k_idx] = -w_val[i] * u_val[k_idx] * x_val[k_idx]^(power - 1) / utility_denom
        end
    end
    return grad_obj_part
end


function cal_utility(x_val, u_val, col_indice, row_ptr, m_len, power) # Renamed x_len to m_len for clarity
    # 计算效用
    # x has same structure as u, so its length is nnz, not m.
    # The 'utility' array should have length 'm', as it's computed per row.
    utility = zeros(m_len) # Pre-allocate utility with correct length m_len

    @inbounds for i in 1:m_len # Loop over rows (m)
        start_idx = row_ptr[i]
        end_idx = row_ptr[i + 1] - 1
        current_utility_sum = 0.0 # Use a local sum for better accumulation precision and speed
        for j in start_idx:end_idx
            current_utility_sum += x_val[j]^power * u_val[j]
        end
        utility[i] = current_utility_sum
    end
    return utility
end

function objective(x_val, w_val, u_val, col_indice, row_ptr, m_len, power, p, rho, x_old_val)
    utility_raw = cal_utility(x_val, u_val, col_indice, row_ptr, m_len, power)
    utility_scaled = utility_raw .^ (1.0 / power) # Element-wise power for vector
    
    obj = 0.0
    # Using @inbounds for performance
    @inbounds for i in 1:m_len
        p_tot = 0.0
        x_delta = 0.0
        start_idx = row_ptr[i]
        end_idx = row_ptr[i + 1] - 1
        
        # Loop for p_tot and x_delta can be more efficient if col_indice is sorted per row
        # (which it is in generate_problem)
        for k_idx in start_idx:end_idx # Renamed j to k_idx to avoid confusion with outer loop i
            col_idx = col_indice[k_idx]
            p_tot += x_val[k_idx] * p[col_idx]
            x_delta += (x_val[k_idx] - x_old_val[k_idx])^2
        end
        
        # Numerical Stability: log(max(1e-8, utility_scaled[i]))
        # utility_scaled[i] should be positive. This is a good safeguard.
        obj += -w_val[i] * log(max(1e-12, utility_scaled[i])) + p_tot + (1/(2*rho)) * x_delta
    end
    return obj
end

function gradient(x_val, w_val, u_val, col_indice, row_ptr, m_len, power, p, rho, x_old_val)
    utility_raw = cal_utility(x_val, u_val, col_indice, row_ptr, m_len, power)
    grad = similar(x_val) # grad has length nnz, same as x_val
    inv_rho = 1.0 / rho

    @inbounds for i in 1:m_len
        start_idx = row_ptr[i]
        end_idx = row_ptr[i + 1] - 1
        
        utility_denom = max(1e-12, utility_raw[i]) # Safeguard
        
        for k_idx in start_idx:end_idx
            # This calculation is for the k_idx-th component of the gradient vector `grad`
            # which corresponds to the k_idx-th element of `x_val`.
            grad[k_idx] = -w_val[i] * u_val[k_idx] * x_val[k_idx]^(power - 1) / utility_denom +
                          p[col_indice[k_idx]] + # p[col_indice[k_idx]] is the relevant term for x_val[k_idx]
                          (x_val[k_idx] - x_old_val[k_idx]) * inv_rho
        end
    end
    return grad
end

#set random seed
using Random
Random.seed!(1234) # Set a fixed seed for reproducibility
# Parameters that are constant across iterations should be `const`
const POWER = 1.0
const RHO = 3.0
const SIGMA = 0.002
const MAX_ITER = 200000
const TOL = 1e-6
const LBFGS_M = 17 # m parameter for L-BFGS-B, not to be confused with problem dimension m

m_dim = 10 # Renamed from `m` to avoid confusion with `LBFGS_M`
n_dim = 10 # Renamed from `n`
nnz_val = 100# Renamed from `nonz`

# Initialize p outside the loop, as it's updated iteratively
p = zeros(n_dim) # ones(n) .* 0.0 is less direct
p_old = similar(p) # Not strictly needed if p is updated in place.

# Generate problem data
x0, w, u_val, b, col_indice, row_ptr = generate_problem(m_dim, n_dim, nnz_val)
# Note: 'u' is used for the utility values, renamed to u_val to avoid conflict with `u` (upper bounds).
println("size of ptr", size(row_ptr))
println("size of col_indice", size(col_indice))
println("nnz = ", length(x0))   # GPU 端长度
print("x0: ", x0)
println("w: ", w)
println("u_val: ", u_val)
println("b: ", b)
x0 .= 1.0
u_val .= 1.0 # Ensure u_val is positive and non-zero
w .= 1.0 # Ensure w is positive and non-zero
p .= 0.5 # Ensure p is positive and non-zero
# x0 .+= 1.0
# x0  .= max.(x0, 1e-5) # Ensure x0 is positive and non-zero
x_old = zeros(length(x0))# Initialize x_old with x0
x_old .= 0.5
x = copy(x0) # Current iterate for x, initialized with x0


h_utility = cal_utility(x, u_val, col_indice, row_ptr, m_dim, POWER)
println("Utility norm: ", norm(h_utility))
println("CPU Utility: ", h_utility)

obj = objective(
    x, w, u_val, col_indice, row_ptr, m_dim, POWER, p, RHO, x_old)
println("Objective value: ", obj)
gradient_val = gradient(
    x, w, u_val, col_indice, row_ptr, m_dim, POWER, p, RHO, x_old)
println("Gradient norm: ", gradient_val)