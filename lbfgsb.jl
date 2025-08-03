using LBFGSB
using LinearAlgebra
using Printf
using SparseArrays
using IterativeSolvers  # For condition number estimation
using Random
# 1. Define a more complex quadratic problem
n = 30000  # Dimension of the problem

function create_complex_quadratic(n; κ=1e6, density=0.001, seed=123)
    Random.seed!(seed)
    
    # 1. Create a sparse random matrix with controlled density
    A = sprand(n, n, density)
    A = (A + A') / 2  # Make symmetric
    
    # # 2. Ensure positive definiteness with controlled condition number
    # λ = eigvals(Matrix(A))  # Convert to dense for small matrices, use Arpack for large
    # λ_min, λ_max = extrema(λ)
    # current_κ = λ_max / λ_min
    
    # # Adjust eigenvalues to achieve desired condition number
    # if current_κ > κ
    #     A = A / current_κ * κ
    # end
    
    # Add diagonal dominance to ensure strong convexity
    # D = spdiagm(0 => abs.(vec(sum(A, dims=2))) .+ 1.0I)
    A = A + spdiagm(0 => ones(n) *  4.0)
    
    # 3. Create a realistic gradient vector
    b = randn(n)
    
    # 4. Optional: Add block diagonal structure for more realism
    block_size = min(100, div(n, 10))
    for i in 1:block_size:n
        j = min(i+block_size-1, n)
        block = sprand(j-i+1, j-i+1, 0.1)
        A[i:j, i:j] += (block + block') / 2
    end
    
    # Final check (for small matrices)
    if n <= 1000
        @assert minimum(eigvals(Matrix(A))) > 0 "Matrix not positive definite"
    end
    
    return A, b
end

# Example usage
n = 30000
A, b = create_complex_quadratic(n, κ=1e6, density=0.0001)
l = fill(5.0, n)  # Lower bounds
u = fill(10.0, n)  # Upper bounds
# Quadratic function and gradient
f(x) = 0.5 * dot(x, A * x) - dot(b, x)
function g!(storage, x)
    mul!(storage, A, x)
    storage .-= b
end

# 2. Optimization setup
x_initial = zeros(n)
bounds = zeros(3, n)
bounds[1, :] .= 2
bounds[2, :] .= l
bounds[3, :] .= u

# 3. L-BFGS-B optimization
println("\nRunning L-BFGS-B optimization...")
optimizer = L_BFGS_B(n, 17)

@time begin
    result = optimizer(f, g!, x_initial, bounds, 
                      m=17, factr=1e12, pgtol=1e-5, 
                      iprint=1, maxiter=10000)
    f_min_lbfgsb, x_min_lbfgsb = result
end

println("L-BFGS-B Minimum Value: ", f_min_lbfgsb)

# 4. Preconditioned Conjugate Gradient
function preconditioned_cg(A, b; x_init=zeros(n), max_iter=10000, tol=1e-6, l = nothing, u = nothing)
    if l !== nothing || u !== nothing
        require_project = true
        x_init = clamp.(x_init, l, u)  # 初始点投影到可行域
    end
    x = x_init
    r = b - A * x
    M = Diagonal(1 ./ diag(A))  # Jacobi preconditioner
    z = M * r
    p = z
    rs_old = dot(r, z)
    iterations = 0
    
    for i in 1:max_iter
        iterations += 1
        Ap = A * p
        alpha = rs_old / dot(p, Ap)
        x = x + alpha * p
        if require_project
            x = clamp.(x, l, u)  # 投影到约束域
        end
        r = r - alpha * Ap
        z = M * r
        rs_new = dot(r, z)
        
        if sqrt(rs_new) < tol
            break
        end
        
        p = z + (rs_new/rs_old) * p
        rs_old = rs_new
        
        if i % 10 == 0
            println("Iteration $i: Residual Norm = $(sqrt(rs_new)), Objective = $(f(x))")
        end
    end
    return x, iterations
end

function preconditioned_projected_BB(
    A, b, l, u;
    x_init=zeros(length(b)),
    max_iter=10000,
    tol=1e-6,
    M
)
    x = clamp.(x_init, l, u)  # 初始点投影到可行域
    r = b - A * x
    g = -r  # 负梯度方向
    pg = M * g  # 预条件梯度
    x_prev = similar(x)
    g_prev = similar(g)
    iterations = 0
    
    for i in 1:max_iter
        iterations += 1
        
        # 保存上一步的变量
        copyto!(x_prev, x)
        copyto!(g_prev, g)
        
        # BB步长计算
        if i > 1
            s = x - x_prev
            y = g - g_prev
            alpha = dot(s, M * s) / dot(s, y)  # 预条件BB步长
            alpha = clamp(alpha, 1e-8, 1e8)     # 数值稳定
        else
            alpha = 1.0 / norm(A, 2)  # 初始步长
        end
        
        # 更新解并投影
        delta_x = alpha .* pg
        x .= x .- delta_x
        x .= clamp.(x, l, u)  # 投影到约束域
        
        # 计算新梯度
        r .= b - A * x
        g .= -r
        pg .= M * g  # 预条件梯度
        
        # 收敛检查
        res_norm = norm(delta_x)
        if i % 10 == 0
            println("Iteration $i: Residual Norm = $res_norm, Objective = $(f(x))")
        end
        if res_norm < tol
            break
        end
    end
    
    return x, iterations
end

# println("\nRunning Preconditioned Conjugate Gradient...")
# @time begin
#     x_min_pcg, pcg_iters = preconditioned_projected_BB(
#         A, b, l, u;
#         x_init=x_initial,
#         max_iter=10000,
#         tol=1e-6,
#         M=Diagonal(ones(n))# Jacobi preconditioner
#     )
#     f_min_pcg = f(x_min_pcg)
# end

@time begin
    x_min_pcg, pcg_iters = preconditioned_cg(A, b; 
        x_init=x_initial, 
        max_iter=10000, 
        tol=1e-6,
        l = l,
        u = u)
    f_min_pcg = f(x_min_pcg)
end

println("PCG Minimum Value: ", f_min_pcg)
println("PCG Iterations: ", pcg_iters)

# 5. Solution comparison
println("\nSolution Difference (L-BFGS-B vs PCG): ", norm(x_min_lbfgsb - x_min_pcg))

# 6. Practical matrix analysis
println("\nMatrix Properties:")
println("Size: ", size(A))
println("Sparsity: ", nnz(A) / length(A))

# Estimate condition number using power iteration
function estimate_condition_number(A, n_iters=10)
    v = rand(size(A, 2))
    v /= norm(v)
    σ_max = σ_min = 0.0
    
    for i in 1:n_iters
        Av = A * v
        σ = norm(Av)
        v = Av / σ
        
        if i == 1
            σ_max = σ_min = σ
        else
            σ_max = max(σ_max, σ)
            σ_min = min(σ_min, σ)
        end
    end
    σ_max / σ_min
end

# println("\nEstimating condition number (this may take a while)...")
# @time cond_estimate = estimate_condition_number(A)
# println("Condition number estimate: ", cond_estimate)