function readFile(filename::String; fix_format::Bool = false)
	"""
	Reads an MPS or QPS file using the QPSReader package and transforms it into a
	`QuadraticProgrammingProblem` struct.
	"""
	return qps_reader_to_standard_form(filename, fixed_format = fix_format)
end

function generateProblem(problem_type::String; n = 100, density = 0.1, seed = 0)
	"""
	Generate a problem of the specified type.
	You can set the scale `n` and density `density` of the problem.
	Returns a `QuadraticProgrammingProblem`` struct.
	"""
	if problem_type == "randomqp"
		return generate_randomQP_problem(n, seed, density)
	elseif problem_type == "lasso"
		return generate_lasso_problem(n, seed, density)
	elseif problem_type == "svm"
		return generate_svm_problem(n, seed, density)
	elseif problem_type == "portfolio"
		return generate_portfolio_problem(n, seed, density)
	elseif problem_type == "mpc"
		return generate_mpc_problem(n, seed)
	else
		error("Unknown problem type: $problem_type")
	end

end

function constructProblem(
	objective_matrix:: SparseMatrixCSC{Float64,Int64},
	objective_vector::Vector{Float64},
	objective_constant::Float64,
	constraint_matrix::SparseMatrixCSC{Float64,Int64},
	constraint_lower_bound::Vector{Float64};
	variable_lower_bound::Union{Vector{Float64}, Nothing} = nothing,
	variable_upper_bound::Union{Vector{Float64}, Nothing} = nothing,
	num_equality_constraints::Int64 = 0
)
	"""
	Construct a problem from the given matrices and vectors.
	Returns a `QuadraticProgrammingProblem`` struct.
	"""
	if isnothing(variable_lower_bound)
		variable_lower_bound = -Inf * ones(size(objective_matrix, 2))
	end
	if isnothing(variable_upper_bound)
		variable_upper_bound = Inf * ones(size(objective_matrix, 2))
	end
	return QuadraticProgrammingProblem(
		size(objective_matrix, 2),
		size(constraint_matrix, 1),
		variable_lower_bound,
		variable_upper_bound,
		Vector{Bool}(isfinite.(variable_lower_bound)),
		Vector{Bool}(isfinite.(variable_upper_bound)),
		objective_matrix,
		objective_vector,
		objective_constant,
		constraint_matrix,
		constraint_matrix',
		constraint_lower_bound,
		num_equality_constraints,
	)
end
