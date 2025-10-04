mutable struct CuPdhcgSolverState
	current_primal_solution::CuVector{Float64}
	current_dual_solution::CuVector{Float64}
	delta_primal::CuVector{Float64}
	delta_dual::CuVector{Float64}
	current_primal_product::CuVector{Float64}
	current_dual_product::CuVector{Float64}
	current_primal_obj_product::CuVector{Float64}
	solution_weighted_avg::CuSolutionWeightedAverage
	step_size::Float64
	primal_weight::Float64
	numerical_error::Bool
	cumulative_kkt_passes::Float64
	total_number_iterations::Int64
	CG_total_extra::Int64
	required_ratio::Union{Float64, Nothing}
	ratio_step_sizes::Union{Float64, Nothing}
	l2_norm_objective_matrix::Float64
	l2_norm_constraint_matrix::Float64
	current_gradient::CuVector{Float64}
	current_direction::CuVector{Float64}
	CG_switch::Bool
	primal_precondition::Union{Nothing, CuArray{Float64, 1}}
	primal_adagrad_state::Union{Nothing, CuArray{Float64, 1}}
	dual_precondition::Union{Nothing, CuArray{Float64, 1}}
	dual_adagrad_state::Union{Nothing, CuArray{Float64, 1}}
end
mutable struct QP_constant_paramter_gpu
	Q_origin::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
	Q_scaled::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
end
mutable struct CuBufferState
	next_primal::CuVector{Float64}
	next_dual::CuVector{Float64}
	delta_primal::CuVector{Float64}
	delta_dual::CuVector{Float64}
	next_primal_product::CuVector{Float64}
	delta_primal_product::CuVector{Float64}
	next_dual_product::CuVector{Float64}
	delta_dual_product::CuVector{Float64}
	next_primal_obj_product::CuVector{Float64}
	delta_primal_obj_product::CuVector{Float64}
	CG_product::CuVector{Float64}
	CG_bound::Float64
end

function pdhcg_specific_log(
	iteration::Int64,
	current_primal_solution::CuVector{Float64},
	current_dual_solution::CuVector{Float64},
	step_size::Float64,
	required_ratio::Union{Float64, Nothing},
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

"""
Kernel to compute primal solution in the next iteration
"""

function compute_next_primal_solution_kernel_update!(
	next_primal::CuDeviceVector{Float64},
	current_gradient::CuDeviceVector{Float64},
	current_direction::CuDeviceVector{Float64},
	CG_product::CuDeviceVector{Float64},
	alpha::Float64,
)
	tx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
	if tx <= length(next_primal)
		@inbounds begin
			next_primal[tx] = next_primal[tx] - alpha * current_direction[tx]
			current_gradient[tx] = current_gradient[tx] - alpha * CG_product[tx]
		end
	end
	return
end

"""
Compute primal solution in the next iteration
"""
function compute_next_primal_solution!(
	problem::CuQuadraticProgrammingProblem,
	current_primal_solution::CuVector{Float64},
	current_dual_product::CuVector{Float64},
	current_primal_obj_product::CuVector{Float64},
	total_iteration::Int64,
	step_size::Float64,
	primal_weight::Float64,
	CG_bound::Float64,
	current_gradient::CuVector{Float64},
	current_direction::CuVector{Float64},
	next_primal::CuVector{Float64},
	CG_product::CuVector{Float64},
	next_primal_product::CuVector{Float64},
	next_primal_obj_product::CuVector{Float64},
	first_iter::Bool,
)
	max_CG_iter = 20

	current_gradient .= current_primal_obj_product .+ problem.objective_vector .- current_dual_product
	current_direction .= current_gradient

	if first_iter
		alpha = step_size / primal_weight
		next_primal .= next_primal .- (step_size / primal_weight) .* current_direction
		CG_iter = 0
	else
		gg = CUDA.dot(current_gradient, current_gradient)
		CG_iter = 1
		next_primal .= current_primal_solution

		while CG_iter <= max_CG_iter
			gkgk = gg
			CUDA.CUSPARSE.mv!('N', 1.0, problem.objective_matrix, current_direction, 0.0, CG_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
			CG_product .+= (primal_weight / step_size) .* current_direction
			dHd = CUDA.dot(current_direction, CG_product)
			alpha = gg / dHd

			next_primal .= next_primal .- alpha .* current_direction
			current_gradient .= current_gradient .- alpha .* CG_product
			gg = CUDA.dot(current_gradient, current_gradient)

			if sqrt(gg) <= min(0.05 * CG_bound, 1e-2)
				break
			end

			current_direction .= (gg / gkgk) .* current_direction .+ current_gradient
			CG_iter += 1
		end
	end

	CUDA.CUSPARSE.mv!('N', 1.0, problem.constraint_matrix, next_primal, 0.0, next_primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
	if total_iteration % 40 == 1 || first_iter
		CUDA.CUSPARSE.mv!('N', 1.0, problem.objective_matrix, next_primal, 0.0, next_primal_obj_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
	else
		next_primal_obj_product .= current_gradient .- problem.objective_vector .+ current_dual_product .- (primal_weight / step_size) .* (next_primal .- current_primal_solution)
	end

	CG_iter = min(CG_iter, max_CG_iter)
	return CG_iter
end
function projection!(
	primal::CuArray{Float64, 1},
	variable_lower_bound::CuArray{Float64, 1},
	variable_upper_bound::CuArray{Float64, 1},
)
	CUDA.@sync @. primal = clamp(primal, variable_lower_bound, variable_upper_bound)
end

function primal_kernel1!(
	current_gradient::CuDeviceVector{Float64},
	next_primal::CuDeviceVector{Float64},
	current_primal_solution::CuDeviceVector{Float64},
	objective_vector::CuDeviceVector{Float64},
	current_dual_product::CuDeviceVector{Float64},
	current_primal_obj_product::CuDeviceVector{Float64},
	last_gradient::CuDeviceVector{Float64},
	variable_lower_bound::CuDeviceVector{Float64},
	variable_upper_bound::CuDeviceVector{Float64},
	alpha::Float64,
	N::Int64,
)
	tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
	if tid <= N
		gradient_cache = current_primal_obj_product[tid] + objective_vector[tid] - current_dual_product[tid]
		next_primal_cache = current_primal_solution[tid] - alpha * gradient_cache
		current_gradient[tid] = gradient_cache
		last_gradient[tid] = gradient_cache
		next_primal[tid] = clamp(next_primal_cache, variable_lower_bound[tid], variable_upper_bound[tid])
	end
	return
end
#current_gradient .= current_gradient .+ (primal_weight / step_size) .* (next_primal .- current_primal_solution) .+ problem.objective_vector .- current_dual_product
function primal_kernel2!(
	current_gradient::CuDeviceVector{Float64},
	next_primal::CuDeviceVector{Float64},
	current_primal_solution::CuDeviceVector{Float64},
	objective_vector::CuDeviceVector{Float64},
	current_dual_product::CuDeviceVector{Float64},
	alpha::Float64,
	N::Int64,
)
	tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
	if tid <= N
		current_gradient[tid] = current_gradient[tid] + alpha * (next_primal[tid] - current_primal_solution[tid]) + objective_vector[tid] - current_dual_product[tid]
	end
	return
end
function primal_kernel4!(
	current_gradient::CuDeviceVector{Float64},
	next_primal::CuDeviceVector{Float64},
	current_primal_solution::CuDeviceVector{Float64},
	objective_vector::CuDeviceVector{Float64},
	current_dual_product::CuDeviceVector{Float64},
	primal_precondition::CuDeviceVector{Float64},
	alpha::Float64,
	N::Int64,
)
	tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
	if tid <= N
		current_gradient[tid] = current_gradient[tid] + alpha * (next_primal[tid] - current_primal_solution[tid]) / primal_precondition[tid] + objective_vector[tid] - current_dual_product[tid]
	end
	return
end
# CUDA.copyto!(last_primal, next_primal)
# CUDA.copyto!(last_gradient, current_gradient)
# next_primal .= next_primal .- alpha .* current_gradient
# projection!(next_primal, problem.variable_lower_bound, problem.variable_upper_bound)
function primal_kernel3!(
	current_gradient::CuDeviceVector{Float64},
	next_primal::CuDeviceVector{Float64},
	last_gradient::CuDeviceVector{Float64},
	last_primal::CuDeviceVector{Float64},
	variable_lower_bound::CuDeviceVector{Float64},
	variable_upper_bound::CuDeviceVector{Float64},
	alpha::Float64,
	N::Int64
)
	tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
	if tid <= N
		primal_cache = next_primal[tid]
		gradient_cache = current_gradient[tid] 
		next_primal_cache = primal_cache - alpha * gradient_cache
		last_primal[tid] = primal_cache
		last_gradient[tid] = gradient_cache
		next_primal[tid] = clamp(next_primal_cache, variable_lower_bound[tid], variable_upper_bound[tid])
	end
	return
end
function compute_next_primal_solution_gd_BB!(
	problem::CuQuadraticProgrammingProblem,
	current_primal_solution::CuArray{Float64, 1},
	current_dual_product::CuArray{Float64, 1},
	current_primal_obj_product::CuArray{Float64, 1},
	last_gradient::CuArray{Float64, 1},
	step_size::Float64,
	primal_weight::Float64,
	CG_bound::Float64,
	current_gradient::CuArray{Float64, 1},
	inner_delta_primal::CuArray{Float64, 1},
	next_primal::CuArray{Float64, 1},
	last_primal::CuArray{Float64, 1},
	next_primal_product::CuArray{Float64, 1},
	next_primal_obj_product::CuArray{Float64, 1},
	norm_Q::Float64,
	first_iter::Bool,
	primal_precondition::Union{Nothing, CuArray{Float64, 1}} = nothing,
)

	max_CG_iter = 100
	k = 1
	alpha = 1.0 / (norm_Q + primal_weight / step_size)

	CUDA.@sync begin
		CUDA.copyto!(last_primal, current_primal_solution)
		# current_gradient .= current_primal_obj_product .+ problem.objective_vector .- current_dual_product
		# next_primal .= current_primal_solution .- alpha .* current_gradient
		@cuda threads=ThreadPerBlock blocks=cld(length(current_primal_solution), ThreadPerBlock) primal_kernel1!(
			current_gradient,
			next_primal,
			current_primal_solution,
			problem.objective_vector,
			current_dual_product,
			current_primal_obj_product,
			last_gradient,
			problem.variable_lower_bound,
			problem.variable_upper_bound,
			alpha,
			length(current_primal_solution),
		)
	end
	# CUDA.copyto!(last_gradient, current_gradient)
	
	# projection!(next_primal, problem.variable_lower_bound, problem.variable_upper_bound)
	while k <= max_CG_iter
		inner_delta_primal .= next_primal .- last_primal

		CUDA.@sync begin
			gg = CUDA.dot(inner_delta_primal, inner_delta_primal)
			CUDA.CUSPARSE.mv!('N', 1.0, problem.objective_matrix, next_primal, 0.0, current_gradient, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		end
		if sqrt(gg) <= min(0.05 * CG_bound, 1e-2) * alpha
			break
		end
		inv_primal_step_size = primal_weight / step_size
		# current_gradient .= current_gradient .+ (primal_weight / step_size) .* (next_primal .- current_primal_solution) .+ problem.objective_vector .- current_dual_product
		if isnothing(primal_precondition)	
			CUDA.@sync begin
				@cuda threads=ThreadPerBlock blocks=cld(length(current_gradient), ThreadPerBlock) primal_kernel2!(
					current_gradient,
					next_primal,
					current_primal_solution,
					problem.objective_vector,
					current_dual_product,
					inv_primal_step_size,
					length(current_gradient),
				)
			end
		else
			CUDA.@sync begin
				@cuda threads=ThreadPerBlock blocks=cld(length(current_gradient), ThreadPerBlock) primal_kernel4!(
					current_gradient,
					next_primal,
					current_primal_solution,
					problem.objective_vector,
					current_dual_product,
					primal_precondition,
					inv_primal_step_size,
					length(current_gradient),
				)
			end
		end
		alpha = gg / CUDA.dot(inner_delta_primal, current_gradient .- last_gradient)

		# CUDA.copyto!(last_primal, next_primal)
		# CUDA.copyto!(last_gradient, current_gradient)
		# next_primal .= next_primal .- alpha .* current_gradient
		# projection!(next_primal, problem.variable_lower_bound, problem.variable_upper_bound)
		CUDA.@sync begin
			@cuda threads=ThreadPerBlock blocks=cld(length(next_primal), ThreadPerBlock) primal_kernel3!(
				current_gradient,
				next_primal,
				last_gradient,
				last_primal,
				problem.variable_lower_bound,
				problem.variable_upper_bound,
				alpha,
				length(next_primal),
			)
		end
		k += 1
	end

	CUDA.@sync begin
		CUDA.CUSPARSE.mv!('N', 1.0, problem.objective_matrix, next_primal, 0.0, next_primal_obj_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		CUDA.CUSPARSE.mv!('N', 1.0, problem.constraint_matrix, next_primal, 0.0, next_primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
	end
	CG_iter = min(k, max_CG_iter)
	return CG_iter
end

"""
Kernel to compute dual solution in the next iteration
"""
function compute_next_dual_solution_kernel!(
	right_hand_side::CuDeviceVector{Float64},
	current_dual_solution::CuDeviceVector{Float64},
	current_primal_product::CuDeviceVector{Float64},
	next_primal_product::CuDeviceVector{Float64},
	step_size::Float64,
	primal_weight::Float64,
	num_equalities::Int64,
	num_constraints::Int64,
	next_dual::CuDeviceVector{Float64},
)
	tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
	if tx <= num_equalities
		@inbounds begin
			next_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - 2 * next_primal_product[tx] + current_primal_product[tx])
		end
	elseif (num_equalities + 1) <= tx <= num_constraints
		@inbounds begin
			next_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - 2 * next_primal_product[tx] + current_primal_product[tx])
			next_dual[tx] = max(next_dual[tx], 0.0)
		end
	end
	return
end
function compute_next_dual_solution_kernel_with_diag_precond!(
	right_hand_side::CuDeviceVector{Float64},
	current_dual_solution::CuDeviceVector{Float64},
	current_primal_product::CuDeviceVector{Float64},
	next_primal_product::CuDeviceVector{Float64},
	step_size::Float64,
	primal_weight::Float64,
	num_equalities::Int64,
	num_constraints::Int64,
	next_dual::CuDeviceVector{Float64},
	diagonal_precondition::CuDeviceVector{Float64},
	norm_precondition::Float64 = 1.0,
)	
	tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
	if tx <= num_equalities
		# diagonal_precondition[tx] -= (primal_weight * step_size) * (right_hand_side[tx] - current_primal_product[tx]) * (right_hand_side[tx] - next_primal_product[tx]) / (norm_precondition ^ 2 + 1e-8) 
		next_dual[tx] = current_dual_solution[tx] + diagonal_precondition[tx] * (primal_weight * step_size) * (right_hand_side[tx] - 2 * next_primal_product[tx] + current_primal_product[tx]) 
		
		# diagonal_precondition[tx] = max(diagonal_precondition[tx], 0.0)
	elseif (num_equalities + 1) <= tx <= num_constraints
		# diagonal_precondition[tx] -= (primal_weight * step_size) * (right_hand_side[tx] - current_primal_product[tx]) * (next_dual[tx] > 0.0 ? next_dual[tx] : 0.0) * (right_hand_side[tx] - next_primal_product[tx]) / (norm_precondition ^ 2 + 1e-8)
		next_dual[tx] = current_dual_solution[tx] + diagonal_precondition[tx] * (primal_weight * step_size) * (right_hand_side[tx] - 2 * next_primal_product[tx] + current_primal_product[tx])
		
		# diagonal_precondition[tx] = max(diagonal_precondition[tx], 0.0)
		next_dual[tx] = max(next_dual[tx], 0.0)
	end
	return
end
"""
Compute dual solution in the next iteration
"""
function compute_next_dual_solution!(
	problem::CuQuadraticProgrammingProblem,
	current_dual_solution::CuVector{Float64},
	step_size::Float64,
	primal_weight::Float64,
	next_primal_product::CuVector{Float64},
	current_primal_product::CuVector{Float64},
	next_dual::CuVector{Float64},
	next_dual_product::CuVector{Float64},
	dual_precondition::Union{Nothing, CuVector{Float64}} = nothing,
)
	NumBlockDual = ceil(Int64, problem.num_constraints / ThreadPerBlock)
	if isnothing(dual_precondition)
		CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockDual compute_next_dual_solution_kernel!(
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
		CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t, next_dual, 0, next_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
	else
		norm_precondition = CUDA.norm(problem.right_hand_side - current_primal_product)
		# println("norm_precondition: ", norm_precondition)
		CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockDual compute_next_dual_solution_kernel_with_diag_precond!(
			problem.right_hand_side,
			current_dual_solution,
			current_primal_product,
			next_primal_product,
			step_size,
			primal_weight,
			problem.num_equalities,
			problem.num_constraints,
			next_dual,
			dual_precondition,
			norm_precondition,
		)
		CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t, next_dual, 0, next_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)

	end
end

"""
Update primal and dual solutions
"""
function update_solution_in_solver_state!(
	solver_state::CuPdhcgSolverState,
	buffer_state::CuBufferState,
)
	CUDA.@sync begin
		solver_state.delta_primal .= buffer_state.next_primal .- solver_state.current_primal_solution
		solver_state.delta_dual .= buffer_state.next_dual .- solver_state.current_dual_solution

		solver_state.current_primal_solution .= buffer_state.next_primal
		solver_state.current_dual_solution .= buffer_state.next_dual
		solver_state.current_dual_product .= buffer_state.next_dual_product
		solver_state.current_primal_product .= buffer_state.next_primal_product
		solver_state.current_primal_obj_product .= buffer_state.next_primal_obj_product

		weight = 1 / (1.0 + solver_state.solution_weighted_avg.primal_solutions_count)
	end

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

function equal_to_a_minus_b_kerel!(
	a::CuDeviceVector{Float64},
	b::CuDeviceVector{Float64},
	c::CuDeviceVector{Float64},
	N::Int64
)
	tx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
	if tx <= N
		c[tx] = a[tx] - b[tx] 
	end
	return
end
function equal_to_a_minus_b_kernel!(
        a::CuDeviceVector{Float64},
        b::CuDeviceVector{Float64},
        c::CuDeviceVector{Float64},
        N::Int64)
    tx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if tx <= N
        @inbounds c[tx] = a[tx] - b[tx]
    end
    return
end



function compute_interaction_and_movement(
	solver_state::CuPdhcgSolverState,
	buffer_state::CuBufferState,
)

	# CUDA.@sync begin
	# 	buffer_state.delta_primal .= buffer_state.next_primal .- solver_state.current_primal_solution
	# 	buffer_state.delta_dual .= buffer_state.next_dual .- solver_state.current_dual_solution
	# 	buffer_state.delta_dual_product .= buffer_state.next_dual_product .- solver_state.current_dual_product
	# 	buffer_state.delta_primal_product .= buffer_state.next_primal_product .- solver_state.current_primal_product
	# 	buffer_state.delta_primal_obj_product .= buffer_state.next_primal_obj_product .- solver_state.current_primal_obj_product
	# end
	function launch!(a, b, c)
        len = length(c)
        if len > 0
            blocks = cld(len, ThreadPerBlock)
            @cuda threads=ThreadPerBlock blocks=blocks equal_to_a_minus_b_kernel!(a, b, c, len)
        end
    end

    CUDA.@sync begin
        launch!(buffer_state.next_primal,
                solver_state.current_primal_solution,
                buffer_state.delta_primal)

        launch!(buffer_state.next_dual,
                solver_state.current_dual_solution,
                buffer_state.delta_dual)

        launch!(buffer_state.next_dual_product,
                solver_state.current_dual_product,
                buffer_state.delta_dual_product)

        launch!(buffer_state.next_primal_product,
                solver_state.current_primal_product,
                buffer_state.delta_primal_product)

        launch!(buffer_state.next_primal_obj_product,
                solver_state.current_primal_obj_product,
                buffer_state.delta_primal_obj_product)
    end
	
	CUDA.@sync begin
		primal_dual_interaction = CUDA.dot(buffer_state.delta_primal, buffer_state.delta_dual_product)
		norm_delta_primal = CUDA.dot(buffer_state.delta_primal, buffer_state.delta_primal)
	end
	CUDA.@sync begin
		interaction = abs.(primal_dual_interaction)
		norm_delta_dual = CUDA.dot(buffer_state.delta_dual, buffer_state.delta_dual)
	end
	movement = 0.5 * solver_state.primal_weight * norm_delta_primal + (0.5 / solver_state.primal_weight) * norm_delta_dual
	return interaction, movement
end
"""
Take PDHCG step with ConstantStepsize
"""
function take_step!(
	step_params::ConstantStepsizeParams,
	problem::CuQuadraticProgrammingProblem,
	solver_state::CuPdhcgSolverState,
	buffer_state::CuBufferState,
	time_dict ::Dict{String, Float64}
)
	step_size = solver_state.step_size
	done = false
	k = 1
	while !done
		k += 1
		if k >= 20
			break
		end
		solver_state.total_number_iterations += 1
		first_iter = false

		if solver_state.total_number_iterations <= 1
			first_iter = true
		end
		start_inner_algorithm_time = time()
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
		time_dict["Primal algorithm time"] += time() - start_inner_algorithm_time
		solver_state.CG_total_extra += CG_extra

		start_dual_solution_time = time()
		compute_next_dual_solution!(
			problem,
			solver_state.current_dual_solution,
			step_size,
			solver_state.primal_weight,
			buffer_state.next_primal_product,
			solver_state.current_primal_product,
			buffer_state.next_dual,
			buffer_state.next_dual_product,
			solver_state.dual_precondition,
		)
		time_dict["Dual algorithm time"] += time() - start_dual_solution_time
		
		start_cal_new_primal_weight_time = time()	
		solver_state.cumulative_kkt_passes += 1
		interaction, movement = compute_interaction_and_movement(
			solver_state,
			buffer_state,
		)


		if interaction > 0
			step_size_limit = movement / interaction
			if movement == 0.0
				# The algorithm will terminate at the beginning of the next iteration
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

		first_term = (1 - 1 / (solver_state.total_number_iterations + 1)^(0.3)) * step_size_limit
		second_term = (1 + 1 / (solver_state.total_number_iterations + 1)^(0.6)) * step_size
		step_size = min(first_term, second_term)
		time_dict["Stepsize&Weight update time"] += time() - start_cal_new_primal_weight_time

	end

	solver_state.step_size = step_size


end


"""
Main algorithm
# """
# 	#Detect if the input problem is a GPU problem
# 	if typeof(original_problem) == QuadraticProgrammingProblem
# 		println("The input problem is a CPU problem, we will convert it to a GPU problem")
# 		original_problem = qp_cpu_to_gpu(original_problem)
# 	end
# 	solving_time_details = Dict{String, Float64}()
# 	start_time_root = time()
# 	validate_gpu(original_problem)
# 	qp_cache = cached_quadratic_program_info(original_problem)
# 	original_norm_Q = cu_estimate_maximum_singular_value(original_problem.objective_matrix)
# 	stopkkt_Q = true  
# 	empty_lb_inf = all(original_problem.variable_lower_bound .> -Inf)
# 	empty_ub_inf = all(original_problem.variable_upper_bound .< Inf)
# 	CG_switch = empty_lb_inf && empty_ub_inf
# 	if original_problem.num_equalities >= 1

# 		G = original_problem.constraint_matrix[1:original_problem.num_equalities, :]
# 		G_square = G' * G
# 		q = original_problem.right_hand_side[1:original_problem.num_equalities]
# 		norm_G = cu_estimate_maximum_singular_value(G_square)

# 		rho = 0.1 * original_norm_Q[1] / (norm_G[1])
# 		original_problem.objective_matrix = original_problem.objective_matrix .+ rho .* G_square
# 		original_problem.objective_vector = original_problem.objective_vector .- rho .* G' * q
# 	end
function estimate_maximum_singular_value(
    gmat::CuSparseMatrixCSR{Float64};  # 输入已经在 GPU 上
    probability_of_failure::Float64 = 0.01,
    desired_relative_error::Float64 = 0.1,
    seed::Int64 = 1,
)

    epsilon = 1.0 - (1.0 - desired_relative_error)^2

    # 初始化随机向量到 GPU
    rng = MersenneTwister(seed)
    x = cu(randn(rng, size(gmat, 2)))
    x ./= norm(x, 2)

    temp = CuArray{Float64}(undef, size(gmat, 1))
    gmatT = transpose(gmat)  # GPU 上的转置

    number_of_power_iterations = 0
    num_col = size(gmat, 2)

    # 你的概率函数（可以和 CPU 版一样）
    power_method_failure_probability(num_col, epsilon, k) = exp(-k * epsilon / num_col)

    # 幂迭代
    while power_method_failure_probability(num_col, epsilon, number_of_power_iterations) > probability_of_failure
        mul!(temp, gmat, x)    # temp = A * x
        mul!(x, gmatT, temp)   # x = A' * temp
        x ./= norm(x, 2)       # 归一化
        number_of_power_iterations += 1
    end

    mul!(temp, gmat, x)
    max_singular_value = sqrt(dot(x, gmatT * temp))

    return max_singular_value, number_of_power_iterations
end
function optimize_gpu(
	params::PdhcgParameters,
	original_problem::QuadraticProgrammingProblem;
	initial_primal::Union{Nothing, Vector{Float64}} = nothing,
	initial_dual::Union{Nothing, Vector{Float64}} = nothing,
	initial_diagonal_precondition_primal::Union{Nothing, Vector{Float64}} = nothing,
	initial_diagonal_precondition_dual::Union{Nothing, Vector{Float64}} = nothing,
)
	solving_time_details = Dict{String, Float64}()
	start_time_root = time()
	validate(original_problem)
	qp_cache = cached_quadratic_program_info(original_problem)
	original_norm_Q = estimate_maximum_singular_value(original_problem.objective_matrix)
	stopkkt_Q = true  #true means stop kkt is calculate by Q
	empty_lb_inf = isempty(findall(original_problem.variable_lower_bound .> -Inf))
	empty_ub_inf = isempty(findall(original_problem.variable_upper_bound .< Inf))
	CG_switch = empty_lb_inf && empty_ub_inf
	if original_problem.num_equalities >= 1

		G = original_problem.constraint_matrix[1:original_problem.num_equalities, :]
		G_square = G' * G
		q = original_problem.right_hand_side[1:original_problem.num_equalities]
		norm_G = estimate_maximum_singular_value(G_square)

		rho = 0.1 * original_norm_Q[1] / (norm_G[1])
		original_problem.objective_matrix = original_problem.objective_matrix .+ rho .* G_square
		original_problem.objective_vector = original_problem.objective_vector .- rho .* G' * q
	end

	scaled_problem = rescale_problem(
		params.l_inf_ruiz_iterations,
		params.l2_norm_rescaling,
		params.pock_chambolle_alpha,
		params.verbosity,
		original_problem,
	)

	if stopkkt_Q

		dim = length(scaled_problem.variable_rescaling)
		rowval = collect(1:dim)
		colptr = collect(1:dim+1)
		nzval = 1 ./ scaled_problem.variable_rescaling
		diag_tmp = SparseMatrixCSC(dim, dim, colptr, rowval, nzval)
		scaled_Q = diag_tmp * original_problem.objective_matrix * diag_tmp

		d_objective_matrix = CUDA.CUSPARSE.CuSparseMatrixCSR(original_problem.objective_matrix)
		d_scaled_Q = CUDA.CUSPARSE.CuSparseMatrixCSR(scaled_Q)

		QP_constant = QP_constant_paramter_gpu(d_objective_matrix, d_scaled_Q)

	end

	primal_size = length(scaled_problem.scaled_qp.variable_lower_bound)
	dual_size = length(scaled_problem.scaled_qp.right_hand_side)
	num_eq = scaled_problem.scaled_qp.num_equalities
	if params.primal_importance <= 0 || !isfinite(params.primal_importance)
		error("primal_importance must be positive and finite")
	end
	rescaled_time = time() 
	solving_time_details["rescaling_time"] = rescaled_time - start_time_root
	print_timing_banner("Rescaling time: ", solving_time_details["rescaling_time"])

	d_scaled_problem = scaledqp_cpu_to_gpu(scaled_problem)
	cpu_to_gpu_time = time()
	solving_time_details["cpu_to_gpu_time"] = cpu_to_gpu_time - rescaled_time
	print_timing_banner("CPU to GPU time: ", solving_time_details["cpu_to_gpu_time"])

	d_problem = d_scaled_problem.scaled_qp

	norm_Q, number_of_power_iterations_Q = estimate_maximum_singular_value(scaled_problem.scaled_qp.objective_matrix)
	norm_A, number_of_power_iterations_A = estimate_maximum_singular_value(scaled_problem.scaled_qp.constraint_matrix)
	estimate_singular_value_time = time()
	solving_time_details["estimate_singular_value_time"] = estimate_singular_value_time - cpu_to_gpu_time
	print_timing_banner("Estimate singular value time: ", solving_time_details["estimate_singular_value_time"])

	if isnothing(initial_primal)
		initial_primal_point = CUDA.zeros(Float64, primal_size)
		initial_primal_product = CUDA.zeros(Float64, dual_size)
		initial_primal_obj_product = CUDA.zeros(Float64, primal_size)
	else
		initial_primal_point = CUDA.CuArray(initial_primal)
		initial_primal_product = CUDA.CuArray(scaled_problem.scaled_qp.constraint_matrix * initial_primal)
		initial_primal_obj_product = CUDA.CuArray(scaled_problem.scaled_qp.objective_matrix * initial_primal)
	end

	if isnothing(initial_dual)
		initial_dual_point = CUDA.zeros(Float64, dual_size)
		initial_dual_product = CUDA.zeros(Float64, primal_size)
	else
		initial_dual_point = CUDA.CuArray(initial_dual)
		initial_dual_product = CUDA.CuArray(scaled_problem.scaled_qp.constraint_matrix' * initial_dual)
	end

	if isnothing(params.online_precondition_band_primal)
		primal_precondition = nothing
		primal_adagrad_state = nothing
	else
		primal_precondition = CUDA.ones(Float64, primal_size * (2 * params.online_precondition_band + 1))
		primal_adagrad_state = CUDA.zeros(Float64, primal_size * (2 * params.online_precondition_band + 1))
	end
	if isnothing(params.online_precondition_band_dual)
		dual_precondition = nothing
		dual_adagrad_state = nothing
	else
		dual_precondition = CUDA.ones(Float64, dual_size * (2 * params.online_precondition_band_dual + 1))
		dual_adagrad_state = CUDA.zeros(Float64, dual_size * (2 * params.online_precondition_band_dual + 1))
	end

	# initialization
	solver_state = CuPdhcgSolverState(
		initial_primal_point,               # current_primal_solution
		initial_dual_point,                 # current_dual_solution
		CUDA.zeros(Float64, primal_size),   # delta_primal
		CUDA.zeros(Float64, dual_size),     # delta_dual
		initial_primal_product,             # current_primal_product
		initial_dual_product,               # current_dual_product
		initial_primal_obj_product,         # current_primal_obj_product
		cu_initialize_solution_weighted_average(primal_size, dual_size),
		0.0,                 # step_size
		1.0,                 # primal_weight
		false,               # numerical_error
		0.0,                 # cumulative_kkt_passes
		0,                   # total_number_iterations
		0,                   # CG_number_iterations
		nothing,
		nothing,
		norm_Q,
		norm_A,
		CUDA.zeros(Float64, primal_size),    # current_gradient
		CUDA.zeros(Float64, primal_size),    # current_direction
		CG_switch,
		primal_precondition,                  # primal_precondition
		primal_adagrad_state,                 # primal_adagrad_state
		dual_precondition,                    # dual_precondition
		dual_adagrad_state,                   # dual_adagrad_state
	)

	buffer_state = CuBufferState(
		CUDA.zeros(Float64, primal_size),      # next_primal
		CUDA.zeros(Float64, dual_size),        # next_dual
		CUDA.zeros(Float64, primal_size),      # delta_primal
		CUDA.zeros(Float64, dual_size),        # delta_dual
		CUDA.zeros(Float64, dual_size),        # next_primal_product
		CUDA.zeros(Float64, dual_size),        # delta_primal_product
		CUDA.zeros(Float64, primal_size),      # next_dual_product
		CUDA.zeros(Float64, primal_size),      # delta_dual_product
		CUDA.zeros(Float64, primal_size),      # next_primal_obj_product
		CUDA.zeros(Float64, primal_size),      # delta_next_primal_obj_product
		CUDA.zeros(Float64, primal_size),      # CG_product,
		1e-5,                                  # CG_bound
	)

	buffer_avg = CuBufferAvgState(
		CUDA.zeros(Float64, primal_size),      # avg_primal_solution
		CUDA.zeros(Float64, dual_size),        # avg_dual_solution
		CUDA.zeros(Float64, dual_size),        # avg_primal_product
		CUDA.zeros(Float64, primal_size),      # avg_dual_product
		CUDA.zeros(Float64, primal_size),      # avg_primal_gradient
		CUDA.zeros(Float64, primal_size),      # avg_primal_obj_product
	)

	buffer_original = CuBufferOriginalSol(
		CUDA.zeros(Float64, primal_size),      # primal
		CUDA.zeros(Float64, dual_size),        # dual
		CUDA.zeros(Float64, dual_size),        # primal_product
		CUDA.zeros(Float64, primal_size),      # dual_product
		CUDA.zeros(Float64, primal_size),      # primal_gradient
		CUDA.zeros(Float64, primal_size),      # primal_obj_product
	)

	buffer_kkt = CuBufferKKTState(
		CUDA.zeros(Float64, primal_size),      # primal
		CUDA.zeros(Float64, dual_size),        # dual
		CUDA.zeros(Float64, dual_size),        # primal_product
		CUDA.zeros(Float64, primal_size),      # primal_gradient
		CUDA.zeros(Float64, primal_size),      # primal_obj_product
		CUDA.zeros(Float64, primal_size),      # lower_variable_violation
		CUDA.zeros(Float64, primal_size),      # upper_variable_violation
		CUDA.zeros(Float64, dual_size),        # constraint_violation
		CUDA.zeros(Float64, primal_size),      # dual_objective_contribution_array
		CUDA.zeros(Float64, primal_size),      # reduced_costs_violations
		CuDualStats(
			0.0,
			CUDA.zeros(Float64, dual_size - num_eq),
			CUDA.zeros(Float64, primal_size),
		),
		0.0,                                   # dual_res_inf
	)

	buffer_primal_gradient = CUDA.zeros(Float64, primal_size)
	buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product
	buffer_primal_gradient .+= solver_state.current_primal_obj_product
	current_primal_obj_product_Q = CUDA.zeros(Float64, primal_size)
	avg_primal_obj_product_Q = CUDA.zeros(Float64, primal_size)
	avg_primal_gradient = CUDA.zeros(Float64, primal_size)

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

	solver_state.step_size = 1 / norm_A

	KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

	primal_weight_update_smoothing = params.restart_params.primal_weight_update_smoothing

	iteration_stats = IterationStats[]
	start_time = time()
	time_spent_doing_basic_algorithm = 0.0

	last_restart_info = create_last_restart_info(
		d_scaled_problem.scaled_qp,
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

	intialized_end_time = time()
	solving_time_details["initialization_time"] = intialized_end_time - estimate_singular_value_time
	print_timing_banner("Initialization time: ", solving_time_details["initialization_time"])
	solving_time_details["total_time_usage_initialization"] = intialized_end_time - start_time_root
	print_timing_banner("Total time usage during initialization: ", solving_time_details["total_time_usage_initialization"])
	iteration = 0
	solver_state.numerical_error = false
	if params.verbosity > 1
		display_iteration_stats_heading()
	end
	solving_time_details["Primal algorithm time"] = 0.0
	solving_time_details["KKT evaluation time"] = 0.0
	solving_time_details["Restart evaluation time"] = 0.0
	solving_time_details["Stepsize&Weight update time"] = 0.0
	solving_time_details["Dual algorithm time"] = 0.0

	while true
		iteration += 1

		if mod(iteration - 1, termination_evaluation_frequency) == 0 ||
		   iteration == iteration_limit + 1 ||
		   iteration <= 10 ||
		   solver_state.numerical_error

			solver_state.cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

			### average ###
			if solver_state.numerical_error || solver_state.solution_weighted_avg.primal_solutions_count == 0 || solver_state.solution_weighted_avg.dual_solutions_count == 0
				buffer_avg.avg_primal_solution .= solver_state.current_primal_solution
				buffer_avg.avg_dual_solution .= solver_state.current_dual_solution
				buffer_avg.avg_primal_product .= solver_state.current_primal_product
				buffer_avg.avg_dual_product .= solver_state.current_dual_product
				buffer_avg.avg_primal_gradient .= buffer_primal_gradient
				buffer_avg.avg_primal_obj_product .= solver_state.current_primal_obj_product
			else
				compute_average!(solver_state.solution_weighted_avg, buffer_avg, d_problem)
			end

			### KKT ###
			start_kkt_time = time()
			if stopkkt_Q

				CUDA.CUSPARSE.mv!('N', 1.0, QP_constant.Q_scaled, solver_state.current_primal_solution, 0.0, current_primal_obj_product_Q, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
				CUDA.CUSPARSE.mv!('N', 1.0, QP_constant.Q_scaled, buffer_avg.avg_primal_solution, 0.0, avg_primal_obj_product_Q, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)

				buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product
				buffer_primal_gradient .+= current_primal_obj_product_Q
				avg_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- buffer_avg.avg_dual_product .+ avg_primal_obj_product_Q

				current_iteration_stats_avg = evaluate_unscaled_iteration_stats(
					d_scaled_problem,
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
					avg_primal_gradient,
					avg_primal_obj_product_Q,
					buffer_original,
					buffer_kkt,
				)
				current_iteration_stats_current = evaluate_unscaled_iteration_stats(
					d_scaled_problem,
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
					current_primal_obj_product_Q,
					buffer_original,
					buffer_kkt,
				)
			else
				current_iteration_stats_avg = evaluate_unscaled_iteration_stats(
					d_scaled_problem,
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
					d_scaled_problem,
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

			current_kkt_err = max(c_i_current.relative_optimality_gap, c_i_current.relative_l_inf_primal_residual, c_i_current.relative_l_inf_dual_residual)
			avg_kkt_err = max(c_i_avg.relative_optimality_gap, c_i_avg.relative_l_inf_primal_residual, c_i_avg.relative_l_inf_dual_residual)
			
			if current_kkt_err >= avg_kkt_err
				current_iteration_stats = current_iteration_stats_avg
				kkt_err = avg_kkt_err
			else
				current_iteration_stats = current_iteration_stats_current
				kkt_err = current_kkt_err
			end

			solving_time_details["KKT evaluation time"] += time() - start_kkt_time

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
				
				main_algo_end_time = time()
				avg_primal_solution = zeros(primal_size)
				avg_dual_solution = zeros(dual_size)
				gpu_to_cpu!(
					buffer_avg.avg_primal_solution,
					buffer_avg.avg_dual_solution,
					avg_primal_solution,
					avg_dual_solution,
				)

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
				
				output = unscaled_saddle_point_output(
					scaled_problem,
					Vector(solver_state.current_primal_solution),
					Vector(solver_state.current_dual_solution),
					termination_reason,
					iteration - 1,
					iteration_stats,
					solver_state.CG_total_extra,
				)
				post_process_time = time()
				solving_time_details["main_algo_time"] = output.iteration_stats[end].cumulative_time_sec
				solving_time_details["post_process_time"] = post_process_time - main_algo_end_time
				print_timing_banner("PDHCG algorithm time: ", solving_time_details["main_algo_time"])
				solving_time_details["Basic algorithm time"] = time_spent_doing_basic_algorithm
				print_timing_banner("Basic algorithm time: ", time_spent_doing_basic_algorithm)
				print_timing_banner("KKT evaluation time: ", solving_time_details["KKT evaluation time"])
				print_timing_banner("Primal algorithm time: ", solving_time_details["Primal algorithm time"])
				print_timing_banner("Dual algorithm time: ", solving_time_details["Dual algorithm time"])
				print_timing_banner("Post processing time: ", solving_time_details["post_process_time"])
				print_timing_banner("Stepsize&Weight update time: ", solving_time_details["Stepsize&Weight update time"])
				return output
			end

			buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product
			buffer_primal_gradient .+= solver_state.current_primal_obj_product

			start_restart_time = time()
			current_iteration_stats.restart_used = run_restart_scheme(
				d_scaled_problem.scaled_qp,
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

				#solver_state.step_size = 0.99 * 2 / (norm_Q / solver_state.primal_weight + sqrt(4*norm_A^2 + norm_Q^2 / solver_state.primal_weight^2))
			end
			solving_time_details["Restart evaluation time"] += time() - start_restart_time

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
		take_step!(params.step_size_policy_params, d_problem, solver_state, buffer_state, solving_time_details)

		time_spent_doing_basic_algorithm += time() - time_spent_doing_basic_algorithm_checkpoint
	end

end
