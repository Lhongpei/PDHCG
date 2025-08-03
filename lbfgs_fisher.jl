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
    
    # 随机生成列索引（确保不重复）
    # This part can be slow if n is very large and (end_idx - start_idx + 1) is also large,
    # as randperm(n) creates a full permutation.
    # If the number of elements per row is small relative to n, consider sampling directly.
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

function check_kkt_conditions(x_val, w_val, u_val, b_val, col_indice, row_ptr, m_len, n_len, power, p_val, l_bounds, u_bounds, tol)
    println("\n--- KKT Conditions Check ---")

    # 1. Primal Feasibility (Box Constraints)
    box_violation = 0.0
    for j in eachindex(x_val)
        if x_val[j] < l_bounds[j] - tol || x_val[j] > u_bounds[j] + tol
            box_violation += max(0, l_bounds[j] - x_val[j], x_val[j] - u_bounds[j])
        end
    end
    println("1. Primal Feasibility (Box Constraints): ", @sprintf("%.2e", box_violation), (box_violation < tol ? " (OK)" : " (VIOLATED)"))

    # 2. Primal Feasibility (Equality Constraints)
    sum_x_fea = zeros(n_len)
    @inbounds for i in 1:m_len
        start_idx = row_ptr[i]
        end_idx = row_ptr[i+1] - 1
        for k_idx in start_idx:end_idx
            sum_x_fea[col_indice[k_idx]] += x_val[k_idx]
        end
    end
    equality_violation = norm(sum_x_fea .- 1.0) # Assuming constraints are sum(x) per column = 1
    println("2. Primal Feasibility (Equality Constraints): ", @sprintf("%.2e", equality_violation), (equality_violation < tol ? " (OK)" : " (VIOLATED)"))

    # 3. Stationarity (Gradient condition, accounting for box constraints)
    # This is the most complex part. We'll use the gradient of the original objective + p terms
    # and then check against the active bounds.
    
    # Calculate the gradient of the objective part
    grad_obj_part = calculate_objective_gradient_part(x_val, w_val, u_val, row_ptr, m_len, power)
    
    # KKT_grad_val_j = grad_obj_part[j] + p_val[col_indice[j]]
    # This KKT_grad_val_j should be:
    # - approximately 0 if x_j is strictly between bounds
    # - >= -tol if x_j is at lower bound (0)
    # - <= tol if x_j is at upper bound (1)

    kkt_stationarity_violation = 0.0
    active_lower_count = 0
    active_upper_count = 0
    free_variable_grad_norm = 0.0
    free_variable_grad_count = 0
    
    for j in eachindex(x_val)
        grad_term_j = grad_obj_part[j] + p_val[col_indice[j]] # This is the gradient of L_full w.r.t x_j
                                                              # assuming p_val is lambda
        
        if x_val[j] < l_bounds[j] + tol && abs(l_bounds[j] - x_val[j]) < tol # x_j is at or very near lower bound
            # Expected: grad_term_j >= -tol
            if grad_term_j < -tol
                kkt_stationarity_violation += abs(grad_term_j) # Violation if points inwards negatively
                active_lower_count += 1
            end
        elseif x_val[j] > u_bounds[j] - tol && abs(u_bounds[j] - x_val[j]) < tol # x_j is at or very near upper bound
            # Expected: grad_term_j <= tol
            if grad_term_j > tol
                kkt_stationarity_violation += abs(grad_term_j) # Violation if points inwards positively
                active_upper_count += 1
            end
        else # x_j is strictly between bounds (free variable)
            # Expected: grad_term_j approx 0
            free_variable_grad_norm += grad_term_j^2
            free_variable_grad_count += 1
            if abs(grad_term_j) > tol
                kkt_stationarity_violation += abs(grad_term_j)
            end
        end
    end
    
    # Add the norm of free variable gradients
    if free_variable_grad_count > 0
        free_variable_grad_norm = sqrt(free_variable_grad_norm)
    end

    println("3. Stationarity (Combined - KKT Gradient Check): ", @sprintf("%.2e", kkt_stationarity_violation), (kkt_stationarity_violation < tol ? " (OK)" : " (VIOLATED)"))
    println("   - Free variable gradient norm: ", @sprintf("%.2e", free_variable_grad_norm))
    println("   - Active lower bound count: ", active_lower_count)
    println("   - Active upper bound count: ", active_upper_count)

    # 4. Dual Feasibility (for box constraints) and Complementary Slackness
    # These are harder to check directly without explicitly computing the multipliers.
    # L-BFGS-B handles these implicitly. If the L-BFGS-B termination criteria are met (pgtol small),
    # these are usually satisfied for the inner problem.
    # For the overall ADMM, p (our lambda) has no sign constraints as it's for equality constraints.

    # # We can indicate that these are implicitly handled or require more complex calculation
    # println("4. Dual Feasibility (Box Constraints) & Complementary Slackness: Implicitly handled by L-BFGS-B. (Assumed OK if L-BFGS-B converges)")
    
    # Overall KKT satisfaction (loosely)
    overall_kkt_ok = (box_violation < tol && equality_violation < tol && kkt_stationarity_violation < tol)
    println("Overall KKT Check: ", overall_kkt_ok ? "PASSED" : "FAILED")
    println("--------------------\n")

    return overall_kkt_ok, box_violation, equality_violation, kkt_stationarity_violation
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
const RHO = 0.002
const SIGMA = 0.002
const MAX_ITER = 200000
const TOL = 1e-6
const LBFGS_M = 17 # m parameter for L-BFGS-B, not to be confused with problem dimension m

m_dim = 100 # Renamed from `m` to avoid confusion with `LBFGS_M`
n_dim = 100 # Renamed from `n`
nnz_val = 10000# Renamed from `nonz`

# Initialize p outside the loop, as it's updated iteratively
p = zeros(n_dim) # ones(n) .* 0.0 is less direct
p_old = similar(p) # Not strictly needed if p is updated in place.

# Generate problem data
x0, w, u_val, b, col_indice, row_ptr = generate_problem(m_dim, n_dim, nnz_val)
# Note: 'u' is used for the utility values, renamed to u_val to avoid conflict with `u` (upper bounds).

x_old = similar(x0) # Previous iterate for x
x = copy(x0) # Current iterate for x, initialized with x0

# Bounds for x (nnz elements)
l_bounds = fill(1e-17, nnz_val)  # Lower bounds for x
u_bounds = fill(1.0, nnz_val)  # Upper bounds for x
# L_BFGS_B bounds format: 3xN matrix [nbd; lb; ub]
# nbd: 0 = no bound, 1 = lower, 2 = upper, 3 = both
bounds = zeros(3, nnz_val)
bounds[1, :] .= 1 # Indicate both lower and upper bounds
bounds[2, :] .= l_bounds
bounds[3, :] .= u_bounds
x.+= 0.5 # Initialize x to the middle of the bounds
x_old .= x # Initialize x_old to the same value as x
optimizer = L_BFGS_B(nnz_val, LBFGS_M) # Use nnz_val for the problem dimension

println("Initial Objective: ", objective(x, w, u_val, col_indice, row_ptr, m_dim, POWER, p, RHO, x_old))
println("Initial Gradient Norm: ", norm(gradient(x, w, u_val, col_indice, row_ptr, m_dim, POWER, p, RHO, x_old)))

println("Starting ADMM/Proximal Loop...")
total_optim_time = 0.0

start_tot_time = time() # Start total time tracking
for iter in 1:MAX_ITER
    global total_optim_time
    # 1. Store current x as x_old for the current iteration's objective/gradient
    copyto!(x_old, x)

    # Define the objective and gradient functions for L-BFGS-B.
    # Crucially, these functions capture the *current* state of p, rho, x_old, etc.
    # This is where closures are useful.
    f_inner(x_current) = objective(x_current, w, u_val, col_indice, row_ptr, m_dim, POWER, p, RHO, x_old)
    
    function g_inner!(storage, x_current)
        storage .= gradient(x_current, w, u_val, col_indice, row_ptr, m_dim, POWER, p, RHO, x_old)
    end
    
    # Run the L-BFGS-B optimization for the inner problem
    # @time will print allocation and time for *this* block.
    # To get cumulative time, sum it up.
    optim_start_time = time()
    fout, x_new = optimizer(f_inner, g_inner!, x, bounds, m=LBFGS_M, factr=1e12, pgtol=1e-9, iprint=-1, maxiter=10000)
    # iprint=-1 to suppress verbose output from L-BFGS-B, makes logs cleaner for outer loop
    optim_end_time = time()
    total_optim_time += (optim_end_time - optim_start_time)
    
    # Update x with the result from L-BFGS-B
    copyto!(x, x_new) # Make sure x is updated with the new optimal x_new

    # Calculate current gradient and feasibility violation for reporting
    current_grad = gradient(x, w, u_val, col_indice, row_ptr, m_dim, POWER, p, RHO, x_old)
    grad_norm = norm(current_grad)

    sum_x_fea = zeros(n_dim) # Re-initialize for each iteration
    # Calculate sum_x_fea using the updated x
    @inbounds for i in 1:m_dim
        start_idx = row_ptr[i]
        end_idx = row_ptr[i + 1] - 1
        for k_idx in start_idx:end_idx
            sum_x_fea[col_indice[k_idx]] += x[k_idx]
        end
    end
    feasibility_violation = norm(sum_x_fea .- 1.0) # Sum of x per column should be 1

    sum_x_for_p_update = zeros(n_dim)
    @inbounds for i in 1:m_dim
        start_idx = row_ptr[i]
        end_idx = row_ptr[i + 1] - 1
        for k_idx in start_idx:end_idx
            sum_x_for_p_update[col_indice[k_idx]] += (2 * x[k_idx] - x_old[k_idx])
        end
    end

    p .+= SIGMA .* (sum_x_for_p_update .- 1) # Assuming constraint is sum(x_j for fixed column) = 1

    # println("Iteration: $iter, Objective: $fout, Gradient Norm: $(grad_norm), Feasibility Violation: $(feasibility_violation)")

    # Check KKT conditions
    if iter % 10 == 0 # Check KKT conditions every 10 iterations
        println("Checking KKT conditions at iteration $iter...")
        overall_kkt_ok, box_violation, equality_violation, kkt_stationarity_violation = check_kkt_conditions(
            x, w, u_val, b, col_indice, row_ptr, m_dim, n_dim, POWER, p, l_bounds, u_bounds, TOL
        )
        if overall_kkt_ok
            println("KKT conditions violated at iteration $iter. Stopping optimization.")
            break
        end
    end

end
end_tot_time = time() # End total time tracking
println("Total primal optimization time: $(total_optim_time) seconds.")
println("Total optimization time: $(end_tot_time - start_tot_time) seconds.")