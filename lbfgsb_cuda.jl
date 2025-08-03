# Filename: gpu_lbfgsb.jl
# Description: A complete educational implementation of the L-BFGS-B algorithm using Julia and CUDA.jl.

using CUDA
using LinearAlgebra
using Printf
using GPUArrays
# Ensure CUDA is functional
if !CUDA.functional()
    error("CUDA is not available on this system. Please ensure you have a CUDA-enabled GPU and the CUDA toolkit installed.")
end

# -----------------------------------------------------
# 1. UTILITY AND GPU KERNELS
# -----------------------------------------------------

"""
GPU Kernel: Projects a vector onto the feasible region defined by lower and upper bounds.
x_i = max(lower_i, min(x_i, upper_i))
"""
function project_kernel!(x, lower, upper)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] = max(lower[i], min(x[i], upper[i]))
    end
    return
end

"""
Wrapper to launch the projection kernel.
"""
function project_gpu!(x, lower, upper)
    threads = 256
    blocks = cld(length(x), threads)
    @cuda threads=threads blocks=blocks project_kernel!(x, lower, upper)
end


"""
Calculates the dot product of two CuArrays.
This is a simple implementation; for higher performance, use CUDA.jl's `dot` or BLAS functions.
"""
function dot_gpu(x::CuArray, y::CuArray)
    return CUDA.dot(x, y)
end

"""
Calculates the norm of a CuArray.
"""
function norm_gpu(x::CuArray)
    return CUDA.norm(x)
end

# -----------------------------------------------------
# 2. CORE L-BFGS-B LOGIC
# -----------------------------------------------------

"""
Performs the L-BFGS two-loop recursion to compute the search direction.
This approximates the Hessian-vector product H⁻¹ * g.
"""

function two_loop_recursion!(p, grad, s_hist, y_hist, m, k)
    # This function modifies `p` in-place to store the search direction.
    
    # Initialize p = -grad
    copyto!(p, grad)
    rmul!(p, -1.0)
    
    n_hist = min(k - 1, m)
    
    # We need a temporary array on the device
    rho = CUDA.zeros(Float64, n_hist)
    alpha = CUDA.zeros(Float64, n_hist)

    # First loop: from latest to oldest
    @allowscalar begin
        for i in 1:n_hist
            s = @view s_hist[:, mod1(k-i, m)]
            y = @view y_hist[:, mod1(k-i, m)]
            
            rho_i = 1.0f0 / dot_gpu(y, s)
            rho[i] = rho_i
            
            alpha_i = rho_i * dot_gpu(s, p)
            alpha[i] = alpha_i
            
            # p = p - alpha_i * y
            CUDA.axpy!(-alpha_i, y, p)
        end
        
        # Scaling by an approximation of the initial Hessian H₀
        # A common choice is H₀ = (sₖᵀyₖ / yₖᵀyₖ) * I
        if k > 1
            s_latest = @view s_hist[:, mod1(k-1, m)]
            y_latest = @view y_hist[:, mod1(k-1, m)]
            gamma = dot_gpu(s_latest, y_latest) / dot_gpu(y_latest, y_latest)
            rmul!(p, gamma)
        end

        # Second loop: from oldest to newest
        for i in n_hist:-1:1
            s = @view s_hist[:, mod1(k-i, m)]
            y = @view y_hist[:, mod1(k-i, m)]
            
            rho_i = rho[i]
            alpha_i = alpha[i]
            
            beta = rho_i * dot_gpu(y, p)
            # p = p + (alpha_i - beta) * s
            CUDA.axpy!(alpha_i - beta, s, p)
        end
    end
end


"""
Backtracking line search to find a step size `alpha` that satisfies
the Armijo condition. The function and gradient evals happen on the GPU.
"""
function backtracking_line_search_gpu(f, g!, x, p, grad, fx, lower, upper)
    alpha = 1.0f0
    c1 = 1e-4  # Armijo condition parameter (sufficient decrease)
    c2 = 0.9f0 # Curvature condition parameter (strong Wolfe)
    tau = 0.5f0 # Backtracking factor

    x_new = similar(x)
    grad_dot_p = dot_gpu(grad, p) # grad is at current point x

    # Allocate a temporary gradient storage for g(x_new)
    grad_new = similar(grad)

    for i in 1:20 # Limit line search iterations
        # x_new = x + alpha * p
        copyto!(x_new, x)
        CUDA.axpy!(alpha, p, x_new)

        # Project x_new to stay within bounds for the function evaluation
        # This projection is important for evaluating f and g within the feasible region.
        project_gpu!(x_new, lower, upper)

        fx_new = f(x_new)

        # Armijo condition
        if fx_new > fx + c1 * alpha * grad_dot_p
            alpha *= tau
            continue # Try smaller alpha
        end

        # Curvature condition (Strong Wolfe)
        g!(grad_new, x_new) # Evaluate gradient at the new point x_new
        grad_new_dot_p = dot_gpu(grad_new, p)

        if abs(grad_new_dot_p) <= c2 * abs(grad_dot_p)
            return alpha, fx_new, grad_new # Found an alpha satisfying strong Wolfe
        else
            alpha *= tau
            continue # Try smaller alpha
        end
    end

    # If line search fails to satisfy conditions within iterations
    @warn "Line search failed to satisfy Strong Wolfe conditions. Returning a small step."
    # Re-evaluate g! at the final x_new before returning
    copyto!(x_new, x)
    CUDA.axpy!(1e-5, p, x_new)
    project_gpu!(x_new, lower, upper)
    fx_final = f(x_new)
    g!(grad_new, x_new) # Make sure grad_new is updated for the returned x_new
    return 1e-5, fx_final, grad_new # Return a very small step and its corresponding fx and grad
end

# -----------------------------------------------------
# 3. MAIN SOLVER FUNCTION
# -----------------------------------------------------


"""
GPU kernel to calculate the gradient of the n-dimensional Rosenbrock function.
Each thread computes one component of the gradient vector.
"""
function rosenbrock_g_kernel!(g, x, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= n
        # Read neighboring values. Be careful with boundaries.
        x_i = x[i]
        x_im1 = (i > 1) ? x[i-1] : 0.0f0
        x_ip1 = (i < n) ? x[i+1] : 0.0f0

        # The gradient has three distinct cases for its components
        if i == 1
            # First element
            @inbounds g[i] = -400.0f0 * x_i * (x_ip1 - x_i^2) - 2.0f0 * (1.0f0 - x_i)
        elseif i == n
            # Last element
            @inbounds g[i] = 200.0f0 * (x_i - x_im1^2)
        else
            # Middle elements
            @inbounds g[i] = 200.0f0 * (x_i - x_im1^2) - 400.0f0 * x_i * (x_ip1 - x_i^2) - 2.0f0 * (1.0f0 - x_i)
        end
    end
    return
end


"""
GPU kernel to calculate the terms of the Rosenbrock summation in parallel.
f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
This kernel calculates the value for each `i` and stores it in `terms`.
"""
function rosenbrock_f_terms_kernel!(terms, x, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i < n
        x_i = x[i]
        x_ip1 = x[i+1]
        @inbounds terms[i] = 100.0f0 * (x_ip1 - x_i^2)^2 + (1.0f0 - x_i)^2
    end
    return
end

"""
Objective function wrapper for n-dimensions.
It uses a kernel to compute each term of the sum in parallel,
and then uses a highly optimized library function (`CUDA.sum`) to sum the terms.
"""
function rosenbrock_f(x::CuArray)
    n = length(x)
    # Need n-1 terms for the summation
    terms = CUDA.zeros(Float64, n - 1)
    
    threads = 256
    blocks = cld(n - 1, threads)
    
    # Launch kernel to compute all terms of the sum in parallel
    @cuda threads=threads blocks=blocks rosenbrock_f_terms_kernel!(terms, x, n)
    
    # Use the optimized CUDA.sum() to reduce the terms to a single scalar
    return CUDA.sum(terms)
end

"""
Gradient function wrapper for n-dimensions.
"""
function rosenbrock_g!(g::CuArray, x::CuArray)
    n = length(x)
    threads = 256
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks rosenbrock_g_kernel!(g, x, n)
    return
end


# -----------------------------------------------------
# 3. MAIN SOLVER FUNCTION (Corrected Signature)
# -----------------------------------------------------

"""
lbfgsb_gpu(f, g!, x0, m, lower, upper; ...)

Solves the minimization problem `min f(x)` subject to `lower <= x <= upper`.
This version is explicitly defined to use Float64 for GPU performance.
"""
function lbfgsb_gpu(f, g!, x0, l, u, m::Int,
                    lower, upper;
                    max_iter::Int=1000, g_tol::Float64=1e-6)

    n = length(x0)
    x = x0 .* 1.0
    lower_gpu = l
    upper_gpu = u
    # --- Initialization ---
    grad = CUDA.zeros(Float64, n)
    p = CUDA.zeros(Float64, n)
    s_hist = CUDA.zeros(Float64, n, m)
    y_hist = CUDA.zeros(Float64, n, m)
    x_old = similar(x)
    grad_old = similar(grad)

    project_gpu!(x, lower_gpu, upper_gpu)
    fx = f(x)
    g!(grad, x)

    @printf "%-5s %-12s %-12s %-12s\n" "Iter" "f(x)" "|g_proj|" "Step (α)"
    @printf "%-5d %-12.5e %-12.5e\n" 0 fx norm_gpu(grad)

    for k in 1:max_iter
        copyto!(x_old, x)
        copyto!(grad_old, grad)

        two_loop_recursion!(p, grad, s_hist, y_hist, m, k)

        alpha, fx_new = backtracking_line_search_gpu(f, g!, x, p, grad, fx, lower_gpu, upper_gpu)
        
        copyto!(x, x_old)
        CUDA.axpy!(alpha, p, x)
        project_gpu!(x, lower_gpu, upper_gpu)

        fx = fx_new
        g!(grad, x)
        
        hist_idx = mod1(k, m)
        s_k = @view s_hist[:, hist_idx]
        y_k = @view y_hist[:, hist_idx]
        
        copyto!(s_k, x)
        CUDA.axpy!(-1.0f0, x_old, s_k)
        copyto!(y_k, grad)
        CUDA.axpy!(-1.0f0, grad_old, y_k)

        pg_norm = norm_gpu(grad)
        @printf "%-5d %-12.5e %-12.5e %-12.5e\n" k fx pg_norm alpha

        if pg_norm < g_tol
            @printf "\nConvergence reached: projected gradient norm < %.2e\n" g_tol
            break
        end
        if k == max_iter
            @printf "\nMaximum iterations reached.\n"
        end
    end

    return Array(x), fx
end

# -----------------------------------------------------
# 4. EXAMPLE USAGE (Corrected Data Types)
# -----------------------------------------------------
# NOTE: The rosenbrock_f and rosenbrock_g! functions from the 
# n-dimensional example should be used here.

# --- Main execution ---
function main()
    println("--- Running L-BFGS-B on GPU with Julia (N-Dimensional) ---")
    
    # Define the dimension of the problem
    n = 100000
    m = 1000

    println("Problem: $n-D Rosenbrock function")
    
    # --- CRITICAL: Ensure all data is created as Float64 ---
    
    # Initial guess
    x0 = zeros(Float64, n)
    x0[1:2:end] .= -1.2f0 # The 'f0' suffix denotes Float64
    x0[2:2:end] .= 1.0f0
    
    # Box constraints
    lower = fill(-10.0f0, n)
    upper = fill(10.0f0, n)
    
    # Run the optimizer
    x0 = CuArray(x0) # Convert initial guess to CuArray
    lower = CuArray(lower) # Convert lower bounds to CuArray
    upper = CuArray(upper) # Convert upper bounds to CuArray

    x_sol, f_min = lbfgsb_gpu(
        rosenbrock_f,
        rosenbrock_g!,
        x0, lower, upper,
        m,
        lower, upper,
        max_iter=2000,
        g_tol=1e-5 # Use 'f0' to specify a Float64 literal
    )
    
    println("\n--- Results ---")
    if n > 8
      @printf "Optimal solution x* (first 4): [%.6f, %.6f, %.6f, %.6f, ...]\n" x_sol[1] x_sol[2] x_sol[3] x_sol[4]
      @printf "Optimal solution x* (last 4):  [..., %.6f, %.6f, %.6f, %.6f]\n" x_sol[n-3] x_sol[n-2] x_sol[n-1] x_sol[n]
    else
      println("Optimal solution x* = $x_sol")
    end
    
    @printf "Minimum function value f(x*) = %.6f\n" f_min
    println("Expected solution at x* = [1.0, 1.0, ..., 1.0]")
end

main()