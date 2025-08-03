using LinearAlgebra
using LBFGSB  # Make sure you have this package installed: using Pkg; Pkg.add("L_BFGS_B")

"""
    pdhg_with_lbfgsb(
        f, grad_f!, A, b, bounds, x0, y0;
        tau=nothing, sigma=nothing, max_iter=100, tol=1e-6
    )

PDHG algorithm where the primal subproblem is solved using L-BFGS-B.
"""
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
        _sigma = 0.5 / op_norm_A
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


# --- Example Problem: Rosenbrock with Equality Constraints ---
function main()
    # Problem Dimension
    n = 15000

    # 1. Objective Function (Rosenbrock)
    function f(x)
        y = 0.25 * (x[1] - 1)^2
        for i = 2:length(x)
            y += (x[i] - x[i-1]^2)^2
        end
        return 4 * y
    end

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
    # Ax = b  (e.g., sum of first 10 elements is 5, sum of last 15 is 10)
    A = zeros(2, n)
    A[1, 1:10] .= 1.0
    A[2, 11:n] .= 1.0
    b = [5.0, 10.0]

    # Bounds for L-BFGS-B solver
    bounds = zeros(3, n)
    for i in 1:n
        bounds[1, i] = 2  # Both lower and upper bounds
        bounds[2, i] = -2.0 # Lower bound
        bounds[3, i] = 2.0  # Upper bound
    end

    # 3. Initial Guesses
    x0 = zeros(n)
    y0 = zeros(size(A, 1))

    # --- Run the Solver ---
    println("Solving constrained Rosenbrock problem with PDHG + L-BFGS-B...")
    x_sol, y_sol = pdhg_with_lbfgsb(
        f, g!, A, b, bounds, x0, y0, max_iter=100000, tol=1e-5
    )

    println("\n--- Solution ---")
    println("Optimal x (first 5 elements): ", round.(x_sol[1:10]; digits=4))
    println("Optimal y: ", round.(y_sol; digits=4))
    println("\nConstraint check (Ax - b):")
    println(round.(A * x_sol - b; digits=6))
    println("\nObjective Value f(x): ", f(x_sol))
end

main()