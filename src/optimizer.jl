function write_vector_to_file(filename, vector)
    open(filename, "w") do io
      for x in vector
        println(io, x)
      end
    end
end
function warm_up(qp::PDHCG.QuadraticProgrammingProblem, gpu_flag::Bool,)
    restart_params = PDHCG.construct_restart_parameters(
        PDHCG.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        PDHCG.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.2,                    # primal_weight_update_smoothing
    )

    termination_params_warmup = PDHCG.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = 1.0e-3,
        eps_optimal_relative = 1.0e-3,
        time_sec_limit = Inf,
        iteration_limit = 10,
        kkt_matrix_pass_limit = Inf,
    )

    params_warmup = PDHCG.PdhcgParameters(
        10,
        true,
        1.0,
        1.0,
        true,
        0,
        true,
        40,
        termination_params_warmup,
        restart_params,
        PDHCG.ConstantStepsizeParams(),
    )
    if gpu_flag
        PDHCG.optimize_gpu(params_warmup, qp);
    else
        PDHCG.optimize(params_warmup, qp);
    end
end
function _solve(
        qp::QuadraticProgrammingProblem,
        parameters::PdhcgParameters,
        gpu_flag::Bool = false,
        save_flag::Bool = false,
        saved_name::Union{String, Nothing} = nothing,
        output_dir::Union{String, Nothing} = nothing;
        initial_primal::Union{Vector{Float64}, Nothing} = nothing,
        initial_dual::Union{Vector{Float64}, Nothing} = nothing,
    )
    if save_flag
        if output_dir isnothing
            output_dir = "./"
        end
        if !isdir(output_dir)
            mkpath(output_dir)
        end
        instance_name = saved_name
    end
    if gpu_flag
        output = optimize_gpu(parameters, qp, initial_primal = initial_primal, initial_dual = initial_dual)
    else
        output = optimize(parameters, qp, initial_primal = initial_primal, initial_dual = initial_dual)
    end
    log = SolveLog()
    #log.instance_name = instance_name
    log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
    log.primal_solution = output.primal_solution
    log.dual_solution = output.dual_solution
    log.termination_reason = output.termination_reason
    log.termination_string = output.termination_string
    log.iteration_count = output.iteration_count
    log.CG_total_iteration = output.CG_total_iteration
    log.solve_time_sec = output.iteration_stats[end].cumulative_time_sec
    log.solution_stats = output.iteration_stats[end]
    log.objective_value = output.iteration_stats[end].convergence_information[1].primal_objective
    kkt_error =  Vector{Float64}()
    for i = 1:length(output.iteration_stats)
        c_i_current = output.iteration_stats[i].convergence_information[1]
        current_kkt_err = norm([c_i_current.relative_optimality_gap, c_i_current.relative_l2_primal_residual, c_i_current.relative_l2_dual_residual])
        
        push!(kkt_error,current_kkt_err)
    end
    log.kkt_error = kkt_error

    log.solution_type = POINT_TYPE_AVERAGE_ITERATE
    if save_flag
        summary_output_path = joinpath(output_dir, instance_name * "_summary.json")
        open(summary_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end

        log.iteration_stats = output.iteration_stats
        full_log_output_path =
            joinpath(output_dir, instance_name * "_full_log.json.gz")
        GZip.open(full_log_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end

        primal_output_path = joinpath(output_dir, instance_name * "_primal.txt")
        write_vector_to_file(primal_output_path, output.primal_solution)

        dual_output_path = joinpath(output_dir, instance_name * "_dual.txt")
        write_vector_to_file(dual_output_path, output.dual_solution)
    end
    return log
end 

function restartParams(
    artificial_restart_threshold::Float64 = 0.2,
    sufficient_reduction::Float64 = 0.2,
    necessary_reduction::Float64 = 0.8,
    primal_weight_update_smoothing::Float64 = 0.2,
)
    return construct_restart_parameters(
        ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        artificial_restart_threshold,
        sufficient_reduction,
        necessary_reduction,
        primal_weight_update_smoothing,
    )
end

function pdhcgSolve(
    qp::QuadraticProgrammingProblem;
    gpu_flag::Bool = false,
    warm_up_flag::Bool = false,
    verbose_level::Int64 = 2,
    time_limit::Float64 = 3600.0,
    relat_error_tolerance::Float64 = 1e-6,
    iteration_limit ::Int64 = Int64(typemax(Int32)),
    ruiz_rescaling_iters::Int64 = 10,
    l2_norm_rescaling_flag::Bool = false,
    pock_chambolle_alpha::Float64 = 1.0,
    artificial_restart_threshold::Float64 = 0.2,
    sufficient_reduction::Float64 = 0.2,
    necessary_reduction::Float64 = 0.8,
    primal_weight_update_smoothing::Float64 = 0.2,
    save_flag::Bool = false,
    saved_name::Union{String, Nothing} = nothing,
    output_dir::Union{String, Nothing} = nothing,
    warm_start_flag::Bool = false,
    initial_primal::Union{Vector{Float64}, Nothing} = nothing,
    initial_dual::Union{Vector{Float64}, Nothing} = nothing,
)
    if warm_up_flag
        qpw = copy(qp)
        oldstd = stdout
        redirect_stdout(devnull)
        warm_up(qpw, gpu_flag);
        redirect_stdout(oldstd)
    end
    restart_params = construct_restart_parameters(
        ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        artificial_restart_threshold,
        sufficient_reduction,
        necessary_reduction,
        primal_weight_update_smoothing,
    )

    termination_params = construct_termination_criteria(
        eps_optimal_absolute = relat_error_tolerance,
        eps_optimal_relative = relat_error_tolerance,
        time_sec_limit = time_limit,
        iteration_limit = iteration_limit,
        kkt_matrix_pass_limit = typemax(Int32),
    )

    params = PdhcgParameters(
        ruiz_rescaling_iters,
        l2_norm_rescaling_flag,
        pock_chambolle_alpha,
        1.0,
        false,
        verbose_level,
        true,
        40,
        termination_params,
        restart_params,
        PDHCG.ConstantStepsizeParams(), 
    )
    if !warm_start_flag
        return _solve(qp, params, gpu_flag, save_flag, saved_name, output_dir)
    end
    return _solve(qp, params, gpu_flag, save_flag, saved_name, output_dir, initial_primal = initial_primal, initial_dual = initial_dual)
end