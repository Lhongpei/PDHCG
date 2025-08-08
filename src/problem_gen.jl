using SparseArrays  
function generate_randomQP_problem(n::Int, seed::Int=1, sparsity::Float64=1e-8)
    Random.seed!(seed)
    m = Int(n)

    # Generate problem data
    P = sprandn(n, n, sparsity)
    rowval = collect(1:n)
    colptr = collect(1:n+1)
    nzval = ones(n)
    P = P * P' + 1e-02 * SparseMatrixCSC(n, n, colptr, rowval, nzval)
    q = randn(n)
    A = sprandn(m, n, sparsity)

    v = randn(n)   # Fictitious solution
    delta = rand(m)  # To get inequality
    ru = A * v + delta
    rl = -Inf * ones(m)
    lb = -Inf * ones(n)
    ub = Inf * ones(n)
     
    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        -A,
        -A',
        -ru,
        0,
    )
end

function generate_lasso_problem(n::Int, seed::Int=1, sparsity::Float64=1e-2)
    # Set random seed
    Random.seed!(seed)

    # Initialize parameters
    m = Int(n * 0.5)
    Ad = sprandn(m, n, sparsity)
    x_true = (rand(n) .> 0.5) .* randn(n) ./ sqrt(n)
    bd = Ad * x_true + randn(m)
    lambda_max = norm(Ad' * bd, Inf)
    lambda_param = (1/5) * lambda_max

    # Construct the QP problem
    rowval_m = collect(1:m)
    colptr_m = collect(1:m+1)
    nzval_m = ones(m)
    P = blockdiag(spzeros(n, n), SparseMatrixCSC(m, m, colptr_m, rowval_m, nzval_m .* 2), spzeros(n, n))
    q = vcat(zeros(m + n), lambda_param * ones(n))
    rowval_n = collect(1:n)
    colptr_n = collect(1:n+1)
    nzval_n = ones(n)
    In = SparseMatrixCSC(n, n, colptr_n, rowval_n, nzval_n)
    Onm = spzeros(n, m)
    A = vcat(hcat(Ad, -SparseMatrixCSC(m, m, colptr_m, rowval_m, nzval_m), spzeros(m, n)),
             hcat(In, Onm, -In),
             hcat(-In, Onm, -In))
    rl = vcat(bd, -Inf * ones(n), -Inf * ones(n))
    ru = vcat(bd, zeros(n), zeros(n))
    lb = -Inf * ones(2*n+m)
    ub = Inf * ones(2*n+m)

    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        -A,
        -A',
        -ru,
        m,
    )
end

function generate_svm_problem(n::Int, seed::Int=1, Sparsity::Float64=1e-2)
    # 设置随机种子
    Random.seed!(seed)

    # 初始化属性
    n_features = n               # 特征数量
    m_data = Int(n_features*0.5)    # 数据点数量
    N_half = Int(m_data * 0.5)
    gamma_val = 1.0
    b_svm_val = vcat(ones(N_half), -ones(N_half))

    # 生成数据
    A_upp = sprandn(N_half, n_features, Sparsity)
    A_low = sprandn(N_half, n_features, Sparsity)
    A_svm_val = vcat(A_upp / sqrt(n_features) .+ (A_upp .!= 0) / n_features,
                     A_low / sqrt(n_features) .- (A_low .!= 0) / n_features)

    # 生成 QP 问题
    P = spdiagm(0 => vcat(ones(n_features), zeros(m_data)))
    q = vcat(zeros(n_features), (gamma_val) * ones(m_data))

    rowval1 = collect(1:length(b_svm_val))
    colptr1 = collect(1:length(b_svm_val)+1)
    rowval2 = collect(1:m_data)
    colptr2 = collect(1:m_data+1)
    nzval2 = ones(m_data)

    A = hcat(-SparseMatrixCSC(colptr1, rowval1, b_svm_val) * A_svm_val, SparseMatrixCSC(colptr2, rowval2, nzval2))
    ru = ones(m_data)

    lb = vcat(-Inf * ones(n_features), zeros(m_data))
    ub = vcat(Inf * ones(n_features), Inf * ones(m_data))

    println("norm_A")
    println(norm(A))
    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        A,
        A',
        ru,
        0,
    )
end

function generate_portfolio_problem(n::Int, seed::Int=1, sparsity::Float64=0.15)
    Random.seed!(seed)
    
    n_assets = n 
    k = Int(n*1)
    F = sprandn(n_assets, k, sparsity)
    D = spdiagm(0 => rand(n_assets) .* sqrt(k))
    mu = randn(n_assets)
    gamma = 1.0

    # Generate QP problem
    rowval1 = collect(1:n_assets)
    colptr1 = collect(1:n_assets + 1)
    nzval1 = rand(n_assets) .* sqrt(k) .* 2

    rowval2 = collect(n_assets + 1:k + n_assets)
    colptr2 = collect(n_assets + 2:k + n_assets + 1)
    nzval2 = ones(k) .* 2

    rowval = vcat(rowval1, rowval2)
    colptr = vcat(colptr1, colptr2)
    nzval = vcat(nzval1, nzval2)

    rand(n_assets) .* sqrt(k)

    rowval_k = collect(1:k)
    colptr_k = collect(1:k + 1)
    nzval_k = ones(k)

    P = SparseMatrixCSC(n_assets + k, n_assets + k, colptr, rowval, nzval)
    q = vcat(-mu ./ gamma, zeros(k))
    A = vcat(
        hcat(sparse(ones(1, n_assets)), spzeros(1, k)),
        hcat(F', -SparseMatrixCSC(k, k, colptr_k, rowval_k, nzval_k)),
    )
    ru = vcat(1.0, zeros(k))

    lb = vcat(zeros(n_assets), -Inf * ones(k))
    ub = vcat(ones(n_assets), Inf * ones(k))

    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        -A,
        -A',
        -ru,
        k+1,
    )
end
function generate_mpc_problem(n::Int, seed:: Int = 1)
    """
    Generate a model predictive control (MPC) problem with a quadcopter model.
    Inputs:
    - n: Prediction horizon, here the num of columns of the problem is 16*n
    - seed: Random seed
    """
    
    speye(N) = spdiagm(ones(N))
    Random.seed!(seed)
    # Discrete time model of a quadcopter
    Ad = [1       0       0   0   0   0   0.1     0       0    0       0       0;
        0       1       0   0   0   0   0       0.1     0    0       0       0;
        0       0       1   0   0   0   0       0       0.1  0       0       0;
        0.0488  0       0   1   0   0   0.0016  0       0    0.0992  0       0;
        0      -0.0488  0   0   1   0   0      -0.0016  0    0       0.0992  0;
        0       0       0   0   0   1   0       0       0    0       0       0.0992;
        0       0       0   0   0   0   1       0       0    0       0       0;
        0       0       0   0   0   0   0       1       0    0       0       0;
        0       0       0   0   0   0   0       0       1    0       0       0;
        0.9734  0       0   0   0   0   0.0488  0       0    0.9846  0       0;
        0      -0.9734  0   0   0   0   0      -0.0488  0    0       0.9846  0;
        0       0       0   0   0   0   0       0       0    0       0       0.9846] |> sparse
    Bd = [0      -0.0726  0       0.0726;
        -0.0726  0       0.0726  0;
        -0.0152  0.0152 -0.0152  0.0152;
        0      -0.0006 -0.0000  0.0006;
        0.0006  0      -0.0006  0;
        0.0106  0.0106  0.0106  0.0106;
        0      -1.4512  0       1.4512;
        -1.4512  0       1.4512  0;
        -0.3049  0.3049 -0.3049  0.3049;
        0      -0.0236  0       0.0236;
        0.0236  0      -0.0236  0;
        0.2107  0.2107  0.2107  0.2107] |> sparse
    (nx, nu) = size(Bd)

    # Constraints
    u0 = 10.5916
    umin = [9.6, 9.6, 9.6, 9.6] .- u0
    umax = [13, 13, 13, 13] .- u0
    xmin = [[-pi/6, -pi/6, -Inf, -Inf, -Inf, -1]; -Inf .* ones(6)][:]
    xmax = [[pi/6,  pi/6,  Inf,  Inf,  Inf, Inf]; Inf .* ones(6)][:]

    # Objective function
    Q = spdiagm([0, 0, 10, 10, 10, 10, 0, 0, 0, 5, 5, 5])
    QN = Q
    R = 0.1 * speye(nu)

    # Initial and reference states
    x0 = zeros(12)
    xr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Prediction horizon
    N = n

    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # - quadratic objective
    P = blockdiag(kron(speye(N), Q), QN, kron(speye(N), R))
    # - linear objective
    q = [repeat(-Q * xr, N); -QN * xr; zeros(N*nu)]
    # - linear dynamics
    Ax = kron(speye(N + 1), -speye(nx)) + kron(spdiagm(-1 => ones(N)), Ad)
    Bu = kron([spzeros(1, N); speye(N)], Bd)
    Aeq = [Ax Bu]
    leq = [-x0; zeros(N * nx)]
    # - input and state constraints

    lineq = [repeat(xmin, N + 1); repeat(umin, N)]
    uineq = [repeat(xmax, N + 1); repeat(umax, N)]
    # - OSQP constraints
    


    return QuadraticProgrammingProblem(
        size(Aeq, 2),
        size(Aeq, 1),
        lineq,
        uineq,
        Vector{Bool}(isfinite.(lineq)),
        Vector{Bool}(isfinite.(uineq)),
        P,
        q,
        0.0,
        Aeq,
        Aeq',
        leq,
        size(Aeq, 1),
        )
    end