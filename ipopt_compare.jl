using JuMP
using Ipopt
using LinearAlgebra # For dot product
using LBFGSB
# --- Problem Definition ---
function rosenbrock_problem_ipopt(n = 1500)
# 1. Objective Function (Rosenbrock)
    # The Rosenbrock function and its gradient are defined within JuMP's
    # @NLobjective macro, which handles automatic differentiation or
    # allows for user-defined derivatives. For complex non-linear functions,
    # it's often more straightforward to express them directly in the macro.

    # 2. Constraints
    # Ax = b  (e.g., sum of first 10 elements is 5, sum of last 15 is 10)
    A = zeros(2, n)
    A[1, 1:10] .= 1.0
    A[2, 11:n] .= 1.0
    b = [5.0, 10.0]

    # Bounds
    lower_bounds = fill(-2.0, n)
    upper_bounds = fill(2.0, n)

    # 3. Initial Guesses
    x0 = zeros(n)

    # Create a JuMP model
    model = Model(Ipopt.Optimizer)

    # Set Ipopt options (optional, but good for control)
    set_optimizer_attribute(model, "print_level", 3) # Suppress detailed IPOPT output
    set_optimizer_attribute(model, "tol", 1e-6) # Primal and dual tolerance
    # set_optimizer_attribute(model, "max_iter", 100000) # Max iterations, if needed

    # Define variables
    @variable(model, lower_bounds[i] <= x[i=1:n] <= upper_bounds[i], start = x0[i])

    # Define the nonlinear objective
    @NLobjective(model, Min,
        0.25 * (x[1] - 1)^2 +
        sum((x[i] - x[i-1]^2)^2 for i = 2:n) * 4
    )

    # Define linear equality constraints
    @constraint(model, con1, sum(A[1, j] * x[j] for j=1:n) == b[1])
    @constraint(model, con2, sum(A[2, j] * x[j] for j=1:n) == b[2])

    println("Solving constrained Rosenbrock problem with IPOPT...")
    start_time = time()
    optimize!(model)
    end_time = time()

    solve_time = end_time - start_time

    # Check termination status
    if termination_status(model) == MOI.LOCALLY_SOLVED
        println("IPOPT solved the problem successfully!")
        x_sol_ipopt = value.(x)
        obj_val_ipopt = objective_value(model)
        println("IPOPT Objective value: ", obj_val_ipopt)
        # Note: IPOPT provides duals for constraints, but not directly for bounds in the same way PDHG might.
        # For linear equality constraints, you can get the dual multipliers:
        # dual_con1 = dual(con1)
        # dual_con2 = dual(con2)
        # println("Dual for con1: ", dual_con1)
        # println("Dual for con2: ", dual_con2)

        return x_sol_ipopt, obj_val_ipopt, solve_time
    else
        println("IPOPT did not solve the problem successfully. Status: ", termination_status(model))
        return nothing, nothing, solve_time
    end
end

# --- PDHG with L-BFGS-B (from your example, adapted for direct comparison) ---
# NOTE: This is a placeholder for your actual pdhg_with_lbfgsb implementation.
# You need to provide the full definition of `pdhg_with_lbfgsb` and its dependencies
# (like the L-BFGS-B solver part, proximal operators, etc.) for this to run.

# For demonstration, let's assume you have a dummy pdhg_with_lbfgsb for now
# or you can paste your actual implementation here.
function pdhg_with_lbfgsb(
    f,                  # Objective function f(x)
    grad_f!,            # In-place gradient of f(x)
    A::AbstractMatrix,
    b::AbstractVector,
    bounds::AbstractMatrix, # Bounds for L-BFGS-B
    x0::AbstractVector,
    y0::AbstractVector;
    tau::Real=0.0,
    sigma::Real=0.0,
    max_iter::Int=100,
    tol::Float64=1e-6
)
    # --- Setup ---
    x = copy(x0)
    y = copy(y0)
    x_bar = copy(x0)
    x_old = similar(x)

    # --- Step size selection ---
    if tau == 0.0 || sigma == 0.0
        op_norm_A = opnorm(A)
        # A smaller tau can help stabilize the L-BFGS-B solve
        _tau = 0.5 / op_norm_A
        _sigma = 0.1 / op_norm_A
    else
        _tau = tau
        _sigma = sigma
    end
    println("Using tau = $(round(_tau, sigdigits=4)) and sigma = $(round(_sigma, sigdigits=4))")

    # --- L-BFGS-B Optimizer Instance ---
    # The dimension is length(x), max corrections can be small (e.g., 10-20)
    optimizer = L_BFGS_B(length(x), 17)

    # --- Main Loop ---
    for k in 1:max_iter
        copyto!(x_old, x)

        # 1. Dual update
        y .+= _sigma * (A * x_bar - b)

        # 2. Primal subproblem solve with L-BFGS-B
        # Define the objective and gradient for this iteration's subproblem
        function subproblem_f(x_sub)
            return f(x_sub) + y' * (A * x_sub) + (0.5 / _tau) * norm(x_sub - x)^2
        end

        function subproblem_g!(g, x_sub)
            grad_f!(g, x_sub) # g = ∇f(x_sub)
            g .+= A' * y      # g += Aᵀy
            g .+= (1.0 / _tau) * (x_sub - x) # g += (1/τ)(x_sub - x)
        end

        # Solve the primal subproblem: min_{l<=x<=u} subproblem_f(x)
        # We use the current x as the initial guess for the subproblem.
        _, x_new = optimizer(
            subproblem_f, subproblem_g!, copy(x), bounds,
            m=5, factr=1e10, pgtol=1e-8, iprint=-1
        )
        x = x_new

        # 3. Extrapolation
        x_bar .= 2 * x - x_old

        # --- Convergence Check ---
        primal_residual = norm(A * x - b)
        g = similar(x)
        subproblem_g!(g, x_bar)  # Compute the gradient at x_bar
        dual_residual = norm(g)
        println("Iter: $k, Primal Residual: $(round(primal_residual, sigdigits=4)), Dual Residual: $(round(dual_residual, sigdigits=4))")

        if k > 1 && primal_residual < tol && dual_residual < tol
            println("\nConverged at iteration $k")
            break
        end
        if k == max_iter
            println("\nReached max iterations.")
        end
    end

    return x, y
end



# --- Main Execution ---
function main_comparison()
    # Run IPOPT
    n = 15000
    x_sol_ipopt, obj_val_ipopt, solve_time_ipopt = rosenbrock_problem_ipopt(n)

    println("\n--- IPOPT Results ---")
    println("IPOPT Solve Time: $(solve_time_ipopt) seconds")
    if obj_val_ipopt !== nothing
        println("IPOPT Optimal Objective: $(obj_val_ipopt)")
        # println("IPOPT Optimal x (first 5 elements): ", x_sol_ipopt[1:min(5, length(x_sol_ipopt))])
    end

    # --- Setup for PDHG (using your original main function structure) ---
    # Problem Dimension
    

    # 1. Objective Function (Rosenbrock)
    function f(x)
        y = 0.25 * (x[1] - 1)^2
        for i = 2:length(x)
            y += (x[i] - x[i-1]^2)^2
        end
        return 4 * y
    end

    # Gradient function g! as provided
    function g!(z, x)
        m = length(x)
        t₁ = x[2] - x[1]^2
        z[1] = 2 * (x[1] - 1) - 1.6e1 * x[1] * t₁
        for i = 2:m-1
            t₂ = t₁
            t₁ = x[i+1] - x[i]^2
            z[i] = 8 * t₂ - 1.6e1 * x[i] * t₁
        end
        z[m] = 8 * t₁
    end

    # 2. Constraints
    A = zeros(2, n)
    A[1, 1:10] .= 1.0
    A[2, 11:n] .= 1.0
    b = [5.0, 10.0]

    # Bounds for L-BFGS-B solver (assuming these are passed to your PDHG)
    bounds = zeros(3, n)
    for i in 1:n
        bounds[1, i] = 2  # Both lower and upper bounds
        bounds[2, i] = -2.0 # Lower bound
        bounds[3, i] = 2.0  # Upper bound
    end

    # 3. Initial Guesses
    x0_pdhg = zeros(n)
    y0_pdhg = zeros(size(A, 1)) # Dual variable for linear constraints

    # # --- Run PDHG Solver ---
    # println("\nSolving constrained Rosenbrock problem with PDHG + L-BFGS-B...")
    start_time_pdhg = time()
    x_sol_pdhg, y = pdhg_with_lbfgsb(
        f, g!, A, b, bounds, x0_pdhg, y0_pdhg, max_iter=100000, tol=1e-6
    )
    end_time_pdhg = time()
    actual_solve_time_pdhg = end_time_pdhg - start_time_pdhg

    # println("\n--- IPOPT Results ---")
    println("IPOPT Solve Time: $(solve_time_ipopt) seconds")
    if obj_val_ipopt !== nothing
        println("IPOPT Optimal Objective: $(obj_val_ipopt)")
        println("IPOPT Optimal x (first 5 elements): ", x_sol_ipopt[1:min(5, length(x_sol_ipopt))])
    end

    println("\n--- PDHG + L-BFGS-B Results ---")
    println("PDHG Solve Time: $(actual_solve_time_pdhg) seconds")
    println("PDHG Objective value: $(f(x_sol_pdhg))")
    println("PDHG Optimal x (first 5 elements): ", x_sol_pdhg[1:min(5, length(x_sol_pdhg))])
end

# Execute the comparison
main_comparison()