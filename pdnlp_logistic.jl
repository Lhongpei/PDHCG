using JuMP
using Ipopt
using LinearAlgebra
using LBFGSB
using Random

# --- Problem Definition --- Logistic Regression with Linear Constraints
# --- Problem Definition --- Logistic Regression with Linear Constraints
function logistic_regression_problem_ipopt(n = 10, m = 100)
    # Generate synthetic data
    Random.seed!(42)
    A_data = randn(m, n)
    y = 2 * rand(m) .- 1  # Labels in {-1, 1}
    # Linear constraints
    C1 = ones(1, n)  # x1 + x2 + ... + xn = 0.5
    d1 = [0.5]
    C2 = reshape([(-1)^(i+1) for i in 1:n], 1, n)  # x1 - x2 + x3 - x4 + ... + (-1)^(n+1) xn = 0
    d2 = [0.0]
    # Linear constraints
    # Inequality constraints
    A1 = ones(1, n)  # x1 + x2 + ... + xn <= 1
    b1 = [1.0]
    A2 = reshape([(-1)^(i+1) for i in 1:n],1,n)  # x1 - x2 + x3 - x4 + ... + (-1)^(n+1) xn <= 0.5
    b2 = [0.5]
    A3 = reshape(collect(1:n) ,1,n) # x1 + 2x2 + 3x3 + ... + nxn <= 2
    b3 = [2.0]
    # println("Size of C1: ", size(C1))
    # println("Size of C2: ", size(C2))
    # println("Size of A1: ", size(A1))
    # println("Size of A2: ", size(A2))
    # println("Size of A3: ", size(A3))
    A = [C1; C2; -A1; -A2]
    b = [d1; d2; -b1; -b2]


    # Bounds
    lower_bounds = fill(-1.0, n)
    upper_bounds = fill(1.0, n)

    # Initial guesses
    x0 = zeros(n)

    # Create a JuMP model
    model = Model(Ipopt.Optimizer)

    # Register the dot function with JuMP
    register(model, :dot, 2, dot; autodiff = true)

    # Set Ipopt options (optional, but good for control)
    set_optimizer_attribute(model, "print_level", 0) # Suppress detailed IPOPT output
    set_optimizer_attribute(model, "tol", 1e-6) # Primal and dual tolerance

    # Define variables
    @variable(model, lower_bounds[i] <= x[i=1:n] <= upper_bounds[i], start = x0[i])

    # Define the nonlinear objective
    @NLobjective(model, Min,
        sum(log(1 + exp(-y[i] * sum(A_data[i,j] * x[j] for j in 1:n))) for i = 1:m)
    )


    # Define linear constraints
    @constraint(model, con1, sum(A[1, j] * x[j] for j=1:n) == b[1])
    @constraint(model, con2, sum(A[2, j] * x[j] for j=1:n) == b[2])
    @constraint(model, con3, sum(A[3, j] * x[j] for j=1:n) >= b[2])
    @constraint(model, con4, sum(A[4, j] * x[j] for j=1:n) <= b[3])

    println("Solving constrained Logistic Regression problem with IPOPT...")
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
        return x_sol_ipopt, obj_val_ipopt, solve_time
    else
        println("IPOPT did not solve the problem successfully. Status: ", termination_status(model))
        return nothing, nothing, solve_time
    end
end


# --- PDHG with L-BFGS-B for Logistic Regression ---
function pdhg_with_lbfgsb(
    f,                  # Objective function f(x)
    grad_f!,            # In-place gradient of f(x)
    A::AbstractMatrix,
    b::AbstractVector,
    bounds::AbstractMatrix, # Bounds for L-BFGS-B
    x0::AbstractVector,
    y0::AbstractVector,
    num_equalities::Int;
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
        _tau = 10 / op_norm_A
        _sigma = 0.9 / op_norm_A
    else
        _tau = tau
        _sigma = sigma
    end
    println("Using tau = $(round(_tau, sigdigits=4)) and sigma = $(round(_sigma, sigdigits=4))")

    # --- L-BFGS-B Optimizer Instance ---
    optimizer = L_BFGS_B(length(x), 17)

    # --- Main Loop ---
    for k in 1:max_iter
        copyto!(x_old, x)

        # 1. Dual updat
        y .+= _sigma * (b - A * x_bar)
        y[num_equalities+1:end] .= max.(y[num_equalities+1:end], 0)  # Ensure non-negativity for inequality constraints

        # 2. Primal subproblem solve with L-BFGS-B
        function subproblem_f(x_sub)
            return f(x_sub) + y' * (b - A * x_sub) + (0.5 / _tau) * norm(x_sub - x)^2
        end

        function subproblem_g!(g, x_sub)
            grad_f!(g, x_sub) # g = ∇f(x_sub)
            g .-= A' * y      # g += Aᵀy
            g .+= (1.0 / _tau) * (x_sub - x) # g += (1/τ)(x_sub - x)
        end

        _, x_new = optimizer(
            subproblem_f, subproblem_g!, copy(x), bounds,
            m=5, factr=1e10, pgtol=1e-8, iprint=-1
        )
        x = x_new

        # 3. Extrapolation
        x_bar .= 2 * x - x_old
        if mod(k, 100) == 0
            residual = b - A * x_bar
            primal_residual = norm(residual[1:num_equalities]) + norm(residual[num_equalities+1:end][residual[num_equalities+1:end] .> 0])
            g = similar(x)
            subproblem_g!(g, x_bar)  # Compute the gradient at x_bar
            dual_residual = norm(g)
            println("Iter: $k, Primal Residual: $(round(primal_residual, sigdigits=4)), Dual Residual: $(round(dual_residual, sigdigits=4))")

            if k > 1 && primal_residual < tol && dual_residual < tol
                println("\nConverged at iteration $k")
                break
            end
        
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
    n = 5000
    m = 10000
    x_sol_ipopt, obj_val_ipopt, solve_time_ipopt = logistic_regression_problem_ipopt(n, m)

    println("\n--- IPOPT Results ---")
    println("IPOPT Solve Time: $(solve_time_ipopt) seconds")
    if obj_val_ipopt !== nothing
        println("IPOPT Optimal Objective: $(obj_val_ipopt)")
        println("IPOPT Optimal x (first 5 elements): ", x_sol_ipopt[1:min(5, length(x_sol_ipopt))])
    end

    # --- Setup for PDHG ---
    # Generate synthetic data
    Random.seed!(42)
    A_data = randn(m, n)
    y = 2 * rand(m) .- 1  # Labels in {-1, 1}

    # Objective function and gradient for Logistic Regression
    function f(x)
        return sum(log.(1 .+ exp.(-y .* (A_data * x))))
    end

    function grad_f!(g, x)
        g .= A_data' * (-y ./ (1 .+ exp.(y .* (A_data * x))))
    end

    # Linear constraints
    C1 = ones(1, n)  # x1 + x2 + ... + xn = 0.5
    d1 = [0.5]
    C2 = reshape([(-1)^(i+1) for i in 1:n], 1, n)  # x1 - x2 + x3 - x4 + ... + (-1)^(n+1) xn = 0
    d2 = [0.0]

    # Inequality constraints
    A1 = ones(1, n)  # x1 + x2 + ... + xn <= 1
    b1 = [1.0]
    A2 = reshape([(-1)^(i+1) for i in 1:n],1,n)  # x1 - x2 + x3 - x4 + ... + (-1)^(n+1) xn <= 0.5
    b2 = [0.5]
    # println("Size of C1: ", size(C1))
    # println("Size of C2: ", size(C2))
    # println("Size of A1: ", size(A1))
    # println("Size of A2: ", size(A2))
    # println("Size of A3: ", size(A3))
    A = [C1; C2; -A1; -A2]
    b = [d1; d2; -b1; -b2]

        # Bounds for L-BFGS-B solver
    bounds = zeros(3, n)
    for i in 1:n
        bounds[1, i] = 2
        bounds[2, i] = -1.0  # Lower bound
        bounds[3, i] = 1.0   # Upper bound
    end

    # Initial guesses
    x0_pdhg = zeros(n)
    y0_pdhg = zeros(size(A, 1)) # Dual variable for linear constraints

    # --- Run PDHG Solver ---
    println("\nSolving constrained Logistic Regression problem with PDHG + L-BFGS-B...")
    start_time_pdhg = time()
    x_sol_pdhg, dual = pdhg_with_lbfgsb(
        f, grad_f!, A, b, bounds, x0_pdhg, y0_pdhg, 2, max_iter=10000000, tol=1e-6
    )
    end_time_pdhg = time()
    actual_solve_time_pdhg = end_time_pdhg - start_time_pdhg

    println("\n--- PDHG + L-BFGS-B Results ---")
    println("PDHG Solve Time: $(actual_solve_time_pdhg) seconds")
    println(size(x_sol_pdhg))
    println(size(A_data))
    println("PDHG Objective value: $(f(x_sol_pdhg))")
    println("PDHG Optimal x (first 5 elements): ", x_sol_pdhg[1:min(5, length(x_sol_pdhg))])
end

# Execute the comparison
main_comparison()