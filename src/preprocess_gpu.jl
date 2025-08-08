function ruiz_rescaling_gpu(
    problem::CuQuadraticProgrammingProblem,
    num_iterations::Int,
    p::Float64 = Inf,
)
    num_constraints, num_variables = size(problem.constraint_matrix)

    # 全部初始化到 GPU 上
    cum_constraint_rescaling = CUDA.ones(Float64, num_constraints)
    cum_variable_rescaling   = CUDA.ones(Float64, num_variables)
    cum_constant_rescaling   = 1.0  # 标量留在 CPU 没问题

    for _ in 1:num_iterations
        constraint_matrix = problem.constraint_matrix
        objective_matrix  = problem.objective_matrix
        objective_vector  = problem.objective_vector
        right_hand_side   = problem.right_hand_side

        # ===== 计算 variable_rescaling =====
        if p == Inf
            # 逐列最大值
            max_cmat_cols = vec(maximum(abs.(constraint_matrix), dims=1))
            max_omat_cols = vec(maximum(abs.(objective_matrix), dims=1))
            # 第三个分量是 0.0*abs.(objective_vector) 实际上就是 0
            variable_rescaling = sqrt.(max.(max_cmat_cols, max_omat_cols))
        else
            error("Only p=Inf is implemented for GPU version.")
        end
        variable_rescaling .= ifelse.(variable_rescaling .== 0.0, 1.0, variable_rescaling)

        # ===== 计算 constraint_rescaling =====
        if num_constraints == 0
            constraint_rescaling = CuArray{Float64}(undef, 0)
        else
            if p == Inf
                max_cmat_rows = vec(maximum(abs.(constraint_matrix), dims=2))
                constraint_rescaling = sqrt.(max_cmat_rows)
            else
                error("Only p=Inf is implemented for GPU version.")
            end
            constraint_rescaling .= ifelse.(constraint_rescaling .== 0.0, 1.0, constraint_rescaling)
        end

        # ===== 计算 constant_rescaling =====
        constant_rescaling = sqrt(max(
            maximum(abs.(objective_vector)),
            maximum(abs.(right_hand_side)),
        ))
        constant_rescaling = constant_rescaling == 0.0 ? 1.0 : constant_rescaling

        # ===== 缩放问题 =====
        scale_problem(problem, constraint_rescaling, variable_rescaling, constant_rescaling)

        # ===== 累乘更新 =====
        cum_constraint_rescaling .*= constraint_rescaling
        cum_variable_rescaling   .*= variable_rescaling
        cum_constant_rescaling   *= constant_rescaling
    end

    return cum_constraint_rescaling, cum_variable_rescaling, cum_constant_rescaling
end