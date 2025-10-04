struct ConstantStepsizeParams end

struct PdhcgParameters
    l_inf_ruiz_iterations::Int
    l2_norm_rescaling::Bool
    pock_chambolle_alpha::Union{Float64,Nothing}
    primal_importance::Float64
    scale_invariant_initial_primal_weight::Bool
    verbosity::Int64
    record_iteration_stats::Bool
    termination_evaluation_frequency::Int32
    termination_criteria::TerminationCriteria
    restart_params::RestartParameters
    step_size_policy_params::Union{
        ConstantStepsizeParams,
    }
    online_precondition_band_primal::Union{Nothing,Int64}
    online_precondition_band_dual::Union{Nothing,Int64}
end
mutable struct QP_constant_paramter
    Q_origin::SparseMatrixCSC{Float64,Int64}
    Q_scaled::SparseMatrixCSC{Float64,Int64}
end

mutable struct PdhcgSolverState
    current_primal_solution::Vector{Float64}
    current_dual_solution::Vector{Float64}
    delta_primal::Vector{Float64}
    delta_dual::Vector{Float64}
    current_primal_product::Vector{Float64}
    current_dual_product::Vector{Float64}
    current_primal_obj_product::Vector{Float64} 
    solution_weighted_avg::SolutionWeightedAverage 
    step_size::Float64
    primal_weight::Float64
    numerical_error::Bool
    cumulative_kkt_passes::Float64
    total_number_iterations::Int64
    CG_total_extra::Int64
    required_ratio::Union{Float64,Nothing}
    ratio_step_sizes::Union{Float64,Nothing}
    l2_norm_objective_matrix::Float64
    l2_norm_constraint_matrix::Float64
    current_gradient::Vector{Float64}
    current_direction::Vector{Float64}
    CG_switch::Bool
end

function define_norms(
    primal_size::Int64,
    dual_size::Int64,
    step_size::Float64,
    primal_weight::Float64,
)
    return 1 / step_size * primal_weight, 1 / step_size / primal_weight
end
  

function pdhcg_specific_log(
    iteration::Int64,
    current_primal_solution::Vector{Float64},
    current_dual_solution::Vector{Float64},
    step_size::Float64,
    required_ratio::Union{Float64,Nothing},
    primal_weight::Float64,
)
    Printf.@printf(
        "   %5d norms=(%9g, %9g) inv_step_size=%9g ",
        iteration,
        norm(current_primal_solution),
        norm(current_dual_solution),
        1 / step_size,
    )
    if !isnothing(required_ratio)
        Printf.@printf(
        "   primal_weight=%18g  inverse_ss=%18g\n",
        primal_weight,
        required_ratio
        )
    else
        Printf.@printf(
        "   primal_weight=%18g \n",
        primal_weight,
        )
    end
end

function pdhcg_final_log(
    problem::QuadraticProgrammingProblem,
    avg_primal_solution::Vector{Float64},
    avg_dual_solution::Vector{Float64},
    verbosity::Int64,
    iteration::Int64,
    CG_total_iteration::Int64,
    termination_reason::TerminationReason,
    last_iteration_stats::IterationStats,
)

    if verbosity >= 2
        
        println("Avg solution:")
        Printf.@printf(
            "  pr_infeas=%12g pr_obj=%15.10g dual_infeas=%12g dual_obj=%15.10g\n",
            last_iteration_stats.convergence_information[1].l_inf_primal_residual,
            last_iteration_stats.convergence_information[1].primal_objective,
            last_iteration_stats.convergence_information[1].l_inf_dual_residual,
            last_iteration_stats.convergence_information[1].dual_objective
        )
        Printf.@printf(
            "  primal norms: L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
            norm(avg_primal_solution, 1),
            norm(avg_primal_solution),
            norm(avg_primal_solution, Inf)
        )
        Printf.@printf(
            "  dual norms:   L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
            norm(avg_dual_solution, 1),
            norm(avg_dual_solution),
            norm(avg_dual_solution, Inf)
        )
    end

    generic_final_log(
        problem,
        avg_primal_solution,
        avg_dual_solution,
        last_iteration_stats,
        verbosity,
        iteration,
        CG_total_iteration,
        termination_reason,
    )
end

function power_method_failure_probability(
    dimension::Int64,
    epsilon::Float64,
    k::Int64,
)
    if k < 2 || epsilon <= 0.0
        return 1.0
    end
    return min(0.824, 0.354 / sqrt(epsilon * (k - 1))) * sqrt(dimension) * (1.0 - epsilon)^(k - 1 / 2) 
end

function estimate_maximum_singular_value(
    matrix::SparseMatrixCSC{Float64,Int64};
    probability_of_failure = 0.01::Float64,
    desired_relative_error = 0.1::Float64,
    seed::Int64 = 1,
)
    epsilon = 1.0 - (1.0 - desired_relative_error)^2
    x = randn(Random.MersenneTwister(seed), size(matrix, 2))
    x .= x / norm(x, 2)
    temp = Vector{Float64}(undef, size(matrix, 1))
    number_of_power_iterations = 0
    num_col = size(matrix, 2)
    while power_method_failure_probability(
        num_col,
        epsilon,
        number_of_power_iterations,
    ) > probability_of_failure
        mul!(temp, matrix, x)
        mul!(x, matrix', temp)
        x .= x / norm(x, 2)
        number_of_power_iterations += 1
    end

    mul!(temp, matrix, x)
    max_singular_value = sqrt(dot(x, matrix'*temp))
    return max_singular_value, number_of_power_iterations
end


"""
Compute primal solution in the next iteration
"""

function compute_next_primal_solution!(
    problem::QuadraticProgrammingProblem,
    current_primal_solution::Vector{Float64},
    current_dual_product::Vector{Float64},
    current_primal_obj_product::Vector{Float64},
    total_iteration::Int64,
    step_size::Float64,
    primal_weight::Float64,
    CG_bound::Float64,
    current_gradient::Vector{Float64},
    current_direction::Vector{Float64},
    next_primal::Vector{Float64},
    CG_product::Vector{Float64},
    next_primal_product::Vector{Float64},
    next_primal_obj_product::Vector{Float64},
    first_iter::Bool
)

max_CG_iter = 20
current_gradient .= current_primal_obj_product .+ problem.objective_vector .-current_dual_product
current_direction .= current_gradient

if first_iter
    next_primal .= next_primal .- (step_size/primal_weight).* current_direction
    CG_iter = 0
else
    gg = dot(current_gradient,current_gradient)
    CG_iter = 1
    next_primal .= current_primal_solution

    while CG_iter <= max_CG_iter
        gkgk = gg
        CG_product .= problem.objective_matrix * current_direction .+(primal_weight / step_size).*current_direction
        dHd = dot(current_direction, CG_product)
        alpha = gg / dHd
        next_primal .= next_primal .- alpha.* current_direction
        current_gradient .= current_gradient .- alpha.*CG_product
        gg = dot(current_gradient,current_gradient)

        if sqrt(gg) <= 0.05*CG_bound
            break
        end  

        current_direction .= (gg /gkgk) .* current_direction .+ current_gradient
        CG_iter += 1
    end
end
    next_primal_product .= problem.constraint_matrix * next_primal

     if mod(total_iteration,40)==1 || first_iter
         next_primal_obj_product .= problem.objective_matrix * next_primal
     else
        next_primal_obj_product .= current_gradient .- problem.objective_vector .+ current_dual_product .- (primal_weight / step_size).*(next_primal .- current_primal_solution)
     end

    CG_iter = min(CG_iter,max_CG_iter)

    return CG_iter
end

function projection!(
    primal::Vector{Float64},
    variable_lower_bound::Vector{Float64},
    variable_upper_bound::Vector{Float64},
  )
    for idx in 1:length(primal)
      primal[idx] = min(
        variable_upper_bound[idx],
        max(variable_lower_bound[idx], primal[idx]),
      )
    end
  end

function compute_next_primal_solution_gd_BB!(
    problem::QuadraticProgrammingProblem,
    current_primal_solution::Vector{Float64},
    current_dual_product::Vector{Float64},
    current_primal_obj_product::Vector{Float64},
    last_gradient::Vector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    CG_bound::Float64,
    current_gradient::Vector{Float64},
    inner_delta_primal::Vector{Float64},
    next_primal::Vector{Float64},
    last_primal::Vector{Float64},
    next_primal_product::Vector{Float64},
    next_primal_obj_product::Vector{Float64},
    norm_Q::Float64,
    first_iter::Bool,
)

    max_CG_iter = 100
    k = 1
    alpha = 1.0/(norm_Q+primal_weight / step_size)
    last_primal .= current_primal_solution
    current_gradient .= current_primal_obj_product.+ problem.objective_vector .-current_dual_product
    last_gradient .=current_gradient
    next_primal .= current_primal_solution .- alpha .*current_gradient
    projection!(next_primal,problem.variable_lower_bound,problem.variable_upper_bound)

    while k <= max_CG_iter
    inner_delta_primal .= next_primal.-last_primal
    gg = dot(inner_delta_primal,inner_delta_primal)

    if sqrt(gg)<=min(0.05*CG_bound, 1e-2)*alpha
        break
    end
    
    current_gradient .= problem.objective_matrix * next_primal .+(primal_weight / step_size).*(next_primal.-current_primal_solution).+ problem.objective_vector .-current_dual_product
    alpha = gg/dot(inner_delta_primal,current_gradient.-last_gradient)
    last_primal.=next_primal
    last_gradient.=current_gradient
    next_primal .= next_primal .- alpha.*current_gradient

    projection!(next_primal,problem.variable_lower_bound,problem.variable_upper_bound)
    k +=1
    end

    next_primal_obj_product .= problem.objective_matrix * next_primal
    next_primal_product .= problem.constraint_matrix * next_primal
    CG_iter = min(k,max_CG_iter)

    return CG_iter
end
"""
Kernel to compute dual solution in the next iteration
"""
function compute_next_dual_solution_kernel!(
    right_hand_side::Vector{Float64},
    current_dual_solution::Vector{Float64},
    current_primal_product::Vector{Float64},
    next_primal_product::Vector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    num_equalities::Int64,
    num_constraints::Int64,
    next_dual::Vector{Float64},
)
    for tx in 1:num_equalities
        next_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - next_primal_product[tx])
    end

    for tx in (num_equalities + 1):num_constraints
        next_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - next_primal_product[tx])
        next_dual[tx] = max(next_dual[tx], 0.0)
    end
    return 
end

"""
Compute dual solution in the next iteration
"""
function compute_next_dual_solution!(
    problem::QuadraticProgrammingProblem,
    current_dual_solution::Vector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    next_primal_product::Vector{Float64},
    current_primal_product::Vector{Float64},
    next_dual::Vector{Float64},
    next_dual_product::Vector{Float64},
)
    compute_next_dual_solution_kernel!(
        problem.right_hand_side,
        current_dual_solution,
        current_primal_product,
        next_primal_product,
        step_size,
        primal_weight,
        problem.num_equalities,
        problem.num_constraints,
        next_dual,
    )
    next_dual_product .= problem.constraint_matrix_t * next_dual
end

"""
Update primal and dual solutions
"""
function update_solution_in_solver_state!(
    solver_state::PdhcgSolverState,
    buffer_state::BufferState,
)
    solver_state.delta_primal .= buffer_state.next_primal .- solver_state.current_primal_solution
    solver_state.delta_dual .= buffer_state.next_dual .- solver_state.current_dual_solution


    solver_state.current_primal_solution .= copy(buffer_state.next_primal)
    solver_state.current_dual_solution .= copy(buffer_state.next_dual)
    solver_state.current_dual_product .= copy(buffer_state.next_dual_product)
    solver_state.current_primal_product .= copy(buffer_state.next_primal_product)
    solver_state.current_primal_obj_product .= copy(buffer_state.next_primal_obj_product)

    weight = 1 / (1.0 + solver_state.solution_weighted_avg.primal_solutions_count)
    add_to_solution_weighted_average!(
        solver_state.solution_weighted_avg,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        weight,
        solver_state.current_primal_product,
        solver_state.current_dual_product,
        solver_state.current_primal_obj_product,
    )
end

function compute_interaction_and_movement(
    solver_state::PdhcgSolverState,
    buffer_state::BufferState,
)
    buffer_state.delta_primal .= buffer_state.next_primal .- solver_state.current_primal_solution
    buffer_state.delta_dual .= buffer_state.next_dual .- solver_state.current_dual_solution
    buffer_state.delta_dual_product .= buffer_state.next_dual_product .- solver_state.current_dual_product
    buffer_state.delta_primal_product .= buffer_state.next_primal_product .- solver_state.current_primal_product
    buffer_state.delta_primal_obj_product .= buffer_state.next_primal_obj_product .- solver_state.current_primal_obj_product



    primal_dual_interaction = dot(buffer_state.delta_primal, buffer_state.delta_dual_product)
    interaction = abs(primal_dual_interaction)
    norm_delta_primal = dot(buffer_state.delta_primal,buffer_state.delta_primal)
    norm_delta_dual = dot(buffer_state.delta_dual,buffer_state.delta_dual)
        

        
    movement = 0.5 * solver_state.primal_weight * norm_delta_primal + (0.5 / solver_state.primal_weight) * norm_delta_dual
    return interaction, movement
end

function take_step!(
    step_params::ConstantStepsizeParams,
    problem::QuadraticProgrammingProblem,
    solver_state::PdhcgSolverState,
    buffer_state::BufferState,
)
    
    step_size = solver_state.step_size
    done = false
    k = 1
    while !done
        k += 1
        if k>=20
            break
        end
        solver_state.total_number_iterations += 1
        first_iter = false
        
        if solver_state.total_number_iterations <= 1
            first_iter = true
        end

    if solver_state.CG_switch
        
        CG_extra = compute_next_primal_solution!(
            problem,
            solver_state.current_primal_solution,
            solver_state.current_dual_product,
            solver_state.current_primal_obj_product,
            solver_state.total_number_iterations,
            step_size,
            solver_state.primal_weight,
            buffer_state.CG_bound,
            solver_state.current_gradient,
            solver_state.current_direction,
            buffer_state.next_primal,
            buffer_state.CG_product,
            buffer_state.next_primal_product,
            buffer_state.next_primal_obj_product,
            first_iter,
        )
    else

        CG_extra = compute_next_primal_solution_gd_BB!(
            problem,
            solver_state.current_primal_solution,
            solver_state.current_dual_product,
            solver_state.current_primal_obj_product,
            buffer_state.delta_primal_obj_product,
            step_size,
            solver_state.primal_weight,
            buffer_state.CG_bound,
            solver_state.current_gradient,
            solver_state.current_direction,
            buffer_state.next_primal,
            buffer_state.CG_product,
            buffer_state.next_primal_product,
            buffer_state.next_primal_obj_product,
            solver_state.l2_norm_objective_matrix,
            first_iter,
        )
    end
        solver_state.CG_total_extra += CG_extra

    compute_next_dual_solution!(
        problem,
        solver_state.current_dual_solution,
        step_size,
        solver_state.primal_weight,
        buffer_state.next_primal_product,
        solver_state.current_primal_product,
        buffer_state.next_dual,
        buffer_state.next_dual_product,
    )

    solver_state.cumulative_kkt_passes += 1
    interaction, movement = compute_interaction_and_movement(
        solver_state,
        buffer_state,
    )


    if interaction > 0
        step_size_limit = movement / interaction
        if movement == 0.0
            solver_state.numerical_error = true
            break
        end
    else
        step_size_limit = Inf
    end

    if step_size <= step_size_limit
        update_solution_in_solver_state!(
            solver_state, 
            buffer_state,
        )
        done = true
    end
    first_term = (1 - 1/(solver_state.total_number_iterations + 1)^(0.3)) * step_size_limit
    second_term = (1 + 1/(solver_state.total_number_iterations + 1)^(0.6)) * step_size
    step_size = min(first_term, second_term)
    
    
end

    solver_state.step_size = step_size


end

"""
Main algorithm
"""
function optimize(
    params::PdhcgParameters,
    original_problem::QuadraticProgrammingProblem;
    initial_primal::Union{Nothing, Vector{Float64}} = nothing,
    initial_dual::Union{Nothing, Vector{Float64}} = nothing,
    )
    validate(original_problem)
    qp_cache = cached_quadratic_program_info(original_problem)
    original_norm_Q = estimate_maximum_singular_value(original_problem.objective_matrix)
    flag_update_CG_bound = false
    stopkkt_Q = true

    empty_lb_inf = isempty(findall(original_problem.variable_lower_bound.>-Inf))
    empty_ub_inf = isempty(findall(original_problem.variable_upper_bound.<Inf))
    CG_switch = empty_lb_inf && empty_ub_inf

    if original_problem.num_equalities >= 1
        
        G =  original_problem.constraint_matrix[1:original_problem.num_equalities,:]
        G_square = G'*G
        q =  original_problem.right_hand_side[1:original_problem.num_equalities]
        norm_G = estimate_maximum_singular_value(G_square)
        rho = 0.1*original_norm_Q[1]/(norm_G[1])
        
        original_problem.objective_matrix = original_problem.objective_matrix .+ rho .* G_square
        original_problem.objective_vector = original_problem.objective_vector .- rho .* G'*q
        end

    scaled_problem = rescale_problem(
        params.l_inf_ruiz_iterations,
        params.l2_norm_rescaling,
        params.pock_chambolle_alpha,
        params.verbosity,
        original_problem,
    )
    if stopkkt_Q 

        scaled_Q =
        sparse(Diagonal(1 ./ scaled_problem.variable_rescaling)) *
        original_problem.objective_matrix *
        sparse(Diagonal(1 ./ scaled_problem.variable_rescaling))

        QP_constant = QP_constant_paramter(original_problem.objective_matrix,scaled_Q)
    end

    primal_size = length(scaled_problem.scaled_qp.variable_lower_bound)
    dual_size = length(scaled_problem.scaled_qp.right_hand_side)
    num_eq = scaled_problem.scaled_qp.num_equalities
    if params.primal_importance <= 0 || !isfinite(params.primal_importance)
        error("primal_importance must be positive and finite")
    end

    d_problem = scaled_problem.scaled_qp

    norm_Q, number_of_power_iterations_Q = estimate_maximum_singular_value(d_problem.objective_matrix)
    norm_A, number_of_power_iterations_A = estimate_maximum_singular_value(d_problem.constraint_matrix)
    
    if isnothing(initial_primal)
        initial_primal_point = zeros(Float64, primal_size)
        initial_primal_product = zeros(Float64, dual_size)
        initial_primal_obj_product = zeros(Float64, primal_size)
    else
        initial_primal_point = initial_primal
        initial_primal_product = scaled_problem.scaled_qp.constraint_matrix * initial_primal_point
        initial_primal_obj_product = scaled_problem.scaled_qp.objective_matrix * initial_primal_point
    end

    if isnothing(initial_dual)
        initial_dual_point = zeros(Float64, dual_size)
        initial_dual_product = zeros(Float64, primal_size)
    else
        initial_dual_point = initial_dual
        initial_dual_product = scaled_problem.scaled_qp.constraint_matrix' * initial_dual_point
    end
    solver_state = PdhcgSolverState(
        initial_primal_point,               # current_primal_solution
        initial_dual_point,                 # current_dual_solution
        zeros(Float64, primal_size),   # delta_primal
        zeros(Float64, dual_size),     # delta_dual
        initial_primal_product,             # current_primal_product
        initial_dual_product,               # current_dual_product
        initial_primal_obj_product,         # current_primal_obj_product
        initialize_solution_weighted_average(primal_size, dual_size),
        0.0,                 # step_size
        1.0,                 # primal_weight
        false,               # numerical_error
        0.0,                 # cumulative_kkt_passes
        0,                   # total_number_iterations
        0,
        nothing,
        nothing,
        norm_Q,
        norm_A,
        zeros(Float64, primal_size),    # current_gradient
        zeros(Float64, primal_size),    # current_direction
        CG_switch,
    )

    buffer_state = BufferState(
        zeros(Float64, primal_size),      # next_primal
        zeros(Float64, dual_size),        # next_dual
        zeros(Float64, primal_size),      # delta_primal
        zeros(Float64, dual_size),        # delta_dual
        zeros(Float64, dual_size),        # next_primal_product
        zeros(Float64, dual_size),        # delta_primal_product
        zeros(Float64, primal_size),      # next_dual_product
        zeros(Float64, primal_size),      # delta_dual_product
        zeros(Float64, primal_size),      # next_primal_obj_product
        zeros(Float64, primal_size),      # delta_next_primal_obj_product
        zeros(Float64, primal_size),      # CG_gradient,
        zeros(Float64, primal_size),      # CG_product,
        1e-5,                             # CG_bound
    )

    buffer_avg = BufferAvgState(
        zeros(Float64, primal_size),      # avg_primal_solution
        zeros(Float64, dual_size),        # avg_dual_solution
        zeros(Float64, dual_size),        # avg_primal_product
        zeros(Float64, primal_size),      # avg_dual_product
        zeros(Float64, primal_size),      # avg_primal_gradient
        zeros(Float64, primal_size),      # avg_primal_obj_product
    )

    buffer_original = BufferOriginalSol(
        zeros(Float64, primal_size),      # primal
        zeros(Float64, dual_size),        # dual
        zeros(Float64, dual_size),        # primal_product
        zeros(Float64, primal_size),      # dual_product
        zeros(Float64, primal_size),      # primal_gradient
        zeros(Float64, primal_size),      # primal_obj_product
    )

    buffer_kkt = BufferKKTState(
        zeros(Float64, primal_size),      # primal
        zeros(Float64, dual_size),        # dual
        zeros(Float64, dual_size),        # primal_product
        zeros(Float64, primal_size),      # primal_gradient
        zeros(Float64, primal_size),      # primal_obj_product
        zeros(Float64, primal_size),      # lower_variable_violation
        zeros(Float64, primal_size),      # upper_variable_violation
        zeros(Float64, dual_size),        # constraint_violation
        zeros(Float64, primal_size),      # dual_objective_contribution_array
        zeros(Float64, primal_size),      # reduced_costs_violations
        DualStats(
            0.0,
            zeros(Float64, dual_size - num_eq),
            zeros(Float64, primal_size),
        ),
        0.0,                              # dual_res_inf
    )

    buffer_primal_gradient = scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product
    buffer_primal_gradient .+= solver_state.current_primal_obj_product

    
    solver_state.cumulative_kkt_passes += number_of_power_iterations_Q + number_of_power_iterations_A

    if params.scale_invariant_initial_primal_weight
        solver_state.primal_weight = select_initial_primal_weight(
            scaled_problem.scaled_qp,
            1.0,
            1.0,
            params.primal_importance,
            params.verbosity,
        )
    else
        solver_state.primal_weight = params.primal_importance
    end
    solver_state.step_size = 1/norm_A

    KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

    primal_weight_update_smoothing = params.restart_params.primal_weight_update_smoothing 

    iteration_stats = IterationStats[]
    start_time = time()
    time_spent_doing_basic_algorithm = 0.0

    last_restart_info = create_last_restart_info(
        scaled_problem.scaled_qp,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        solver_state.current_primal_product,
        solver_state.current_dual_product,
        buffer_primal_gradient,
        solver_state.current_primal_obj_product,
    )

    # For termination criteria:
    termination_criteria = params.termination_criteria
    iteration_limit = termination_criteria.iteration_limit
    termination_evaluation_frequency = params.termination_evaluation_frequency

    # This flag represents whether a numerical error occurred during the algorithm
    # if it is set to true it will trigger the algorithm to terminate.
    solver_state.numerical_error = false
    if params.verbosity > 1
        display_iteration_stats_heading()
    end

    

    iteration = 0
    while true
        iteration += 1

        if mod(iteration - 1, termination_evaluation_frequency) == 0 ||
            iteration == iteration_limit + 1 ||
            iteration <= 10 ||
            solver_state.numerical_error
            
            solver_state.cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

            ### average ###
            if solver_state.numerical_error || solver_state.solution_weighted_avg.primal_solutions_count == 0 || solver_state.solution_weighted_avg.dual_solutions_count == 0
                buffer_avg.avg_primal_solution .= copy(solver_state.current_primal_solution)
                buffer_avg.avg_dual_solution .= copy(solver_state.current_dual_solution)
                buffer_avg.avg_primal_product .= copy(solver_state.current_primal_product)
                buffer_avg.avg_dual_product .= copy(solver_state.current_dual_product)
                buffer_avg.avg_primal_gradient .= copy(buffer_primal_gradient)
                buffer_avg.avg_primal_obj_product .= copy(solver_state.current_primal_obj_product) 
            else

                compute_average!(solver_state.solution_weighted_avg, buffer_avg, d_problem)
            end

            ### KKT ###            
            if stopkkt_Q
                current_primal_obj_product = QP_constant.Q_scaled * solver_state.current_primal_solution
                avg_primal_obj_product = QP_constant.Q_scaled * buffer_avg.avg_primal_solution
                
                buffer_primal_gradient .= scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product
                buffer_primal_gradient .+= current_primal_obj_product

                buffer_avg_primal_gradient = scaled_problem.scaled_qp.objective_vector .- buffer_avg.avg_dual_product .+ avg_primal_obj_product
                
                current_iteration_stats_avg = evaluate_unscaled_iteration_stats(
                    scaled_problem,
                    qp_cache,
                    params.termination_criteria,
                    params.record_iteration_stats,
                    buffer_avg.avg_primal_solution,
                    buffer_avg.avg_dual_solution,
                    iteration,
                    time() - start_time,
                    solver_state.cumulative_kkt_passes,
                    termination_criteria.eps_optimal_absolute,
                    termination_criteria.eps_optimal_relative,
                    solver_state.step_size,
                    solver_state.primal_weight,
                    POINT_TYPE_AVERAGE_ITERATE, 
                    buffer_avg.avg_primal_product,
                    buffer_avg.avg_dual_product,
                    buffer_avg_primal_gradient,
                    avg_primal_obj_product,
                    buffer_original,
                    buffer_kkt,
                )
                current_iteration_stats_current = evaluate_unscaled_iteration_stats(
                    scaled_problem,
                    qp_cache,
                    params.termination_criteria,
                    params.record_iteration_stats,
                    solver_state.current_primal_solution,
                    solver_state.current_dual_solution,
                    iteration,
                    time() - start_time,
                    solver_state.cumulative_kkt_passes,
                    termination_criteria.eps_optimal_absolute,
                    termination_criteria.eps_optimal_relative,
                    solver_state.step_size,
                    solver_state.primal_weight,
                    POINT_TYPE_AVERAGE_ITERATE, 
                    solver_state.current_primal_product,
                    solver_state.current_dual_product,
                    buffer_primal_gradient,
                    current_primal_obj_product,
                    buffer_original,
                    buffer_kkt,
                )

            else
            current_iteration_stats_avg = evaluate_unscaled_iteration_stats(
                scaled_problem,
                qp_cache,
                params.termination_criteria,
                params.record_iteration_stats,
                buffer_avg.avg_primal_solution,
                buffer_avg.avg_dual_solution,
                iteration,
                time() - start_time,
                solver_state.cumulative_kkt_passes,
                termination_criteria.eps_optimal_absolute,
                termination_criteria.eps_optimal_relative,
                solver_state.step_size,
                solver_state.primal_weight,
                POINT_TYPE_AVERAGE_ITERATE, 
                buffer_avg.avg_primal_product,
                buffer_avg.avg_dual_product,
                buffer_avg.avg_primal_gradient,
                buffer_avg.avg_primal_obj_product,
                buffer_original,
                buffer_kkt,
            )
            current_iteration_stats_current = evaluate_unscaled_iteration_stats(
                scaled_problem,
                qp_cache,
                params.termination_criteria,
                params.record_iteration_stats,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                iteration,
                time() - start_time,
                solver_state.cumulative_kkt_passes,
                termination_criteria.eps_optimal_absolute,
                termination_criteria.eps_optimal_relative,
                solver_state.step_size,
                solver_state.primal_weight,
                POINT_TYPE_AVERAGE_ITERATE, 
                solver_state.current_primal_product,
                solver_state.current_dual_product,
                buffer_primal_gradient,
                solver_state.current_primal_obj_product,
                buffer_original,
                buffer_kkt,
            )
            end
            c_i_current = current_iteration_stats_current.convergence_information[1]
            c_i_avg = current_iteration_stats_avg.convergence_information[1]

            current_kkt_err = max(c_i_current.relative_optimality_gap, c_i_current.relative_l_inf_primal_residual,c_i_current.relative_l_inf_dual_residual)
            avg_kkt_err = max(c_i_avg.relative_optimality_gap,c_i_avg.relative_l_inf_primal_residual, c_i_avg.relative_l_inf_dual_residual)

            if current_kkt_err >= avg_kkt_err
            current_iteration_stats = current_iteration_stats_avg
            kkt_err = avg_kkt_err
            else
            current_iteration_stats = current_iteration_stats_current
            kkt_err = current_kkt_err
            end

            method_specific_stats = current_iteration_stats.method_specific_stats
            method_specific_stats["time_spent_doing_basic_algorithm"] =
                time_spent_doing_basic_algorithm

            primal_norm_params, dual_norm_params = define_norms(
                primal_size,
                dual_size,
                solver_state.step_size,
                solver_state.primal_weight,
            )
            
            ### check termination criteria ###
            termination_reason = check_termination_criteria(
                termination_criteria,
                qp_cache,
                current_iteration_stats,
            )
            if solver_state.numerical_error && termination_reason == false
                termination_reason = TERMINATION_REASON_NUMERICAL_ERROR
            end

            # If we're terminating, record the iteration stats to provide final
            # solution stats.
            if params.record_iteration_stats || termination_reason != false
                push!(iteration_stats, current_iteration_stats)
            end

            # Print table.
            if print_to_screen_this_iteration(
                termination_reason,
                iteration,
                params.verbosity,
                termination_evaluation_frequency,
            )
                display_iteration_stats(current_iteration_stats)
            end

            if termination_reason != false
                # ** Terminate the algorithm **
                # This is the only place the algorithm can terminate. Please keep it this way.
                
                avg_primal_solution = buffer_avg.avg_primal_solution
                avg_dual_solution = buffer_avg.avg_dual_solution

                pdhcg_final_log(
                    scaled_problem.scaled_qp,
                    avg_primal_solution,
                    avg_dual_solution,
                    params.verbosity,
                    iteration,
                    solver_state.CG_total_extra,
                    termination_reason,
                    current_iteration_stats,
                )
                #println(solver_state.CG_total_extra)
                return unscaled_saddle_point_output(
                    scaled_problem,
                    solver_state.current_primal_solution,
                    solver_state.current_dual_solution,
                    termination_reason,
                    iteration - 1,
                    iteration_stats,
                    solver_state.CG_total_extra,
                )
            end

            buffer_primal_gradient .= scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product
            buffer_primal_gradient .+= solver_state.current_primal_obj_product

            current_iteration_stats.restart_used = run_restart_scheme(
                scaled_problem.scaled_qp,
                buffer_state,
                solver_state.solution_weighted_avg,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                last_restart_info,
                iteration - 1,
                primal_norm_params,
                dual_norm_params,
                solver_state.primal_weight,
                params.verbosity,
                params.restart_params,
                solver_state.current_primal_product,
                solver_state.current_dual_product,
                buffer_avg,
                buffer_kkt,
                buffer_primal_gradient,
                solver_state.current_primal_obj_product,
            )

            if current_iteration_stats.restart_used != RESTART_CHOICE_NO_RESTART
                solver_state.primal_weight = compute_new_primal_weight(
                    last_restart_info,
                    solver_state.primal_weight,
                    primal_weight_update_smoothing,
                    params.verbosity,
                )
            end

        end

        
      
        if params.verbosity >= 6 && print_to_screen_this_iteration(
            false, # termination_reason
            iteration,
            params.verbosity,
            termination_evaluation_frequency,
        )
            pdhcg_specific_log(
                iteration,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                solver_state.step_size,
                solver_state.required_ratio,
                solver_state.primal_weight,
            )
          end
        time_spent_doing_basic_algorithm_checkpoint = time()
        take_step!(params.step_size_policy_params, d_problem, solver_state, buffer_state)

        time_spent_doing_basic_algorithm += time() - time_spent_doing_basic_algorithm_checkpoint
    end
end