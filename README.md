

# PDHCG QP Solver 
This Repo contains the codebase of the paper "A Restarted Primal-Dual Hybrid Conjugate Gradient Method for Large-Scale Quadratic Programming".
## Notice
This Repo will **NOT** be further maintained. A newer and stronger version of PDHCG, see [PDHCG-II](https://github.com/Lhongpei/PDHCG-II), has been released! 
## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2024.0983

https://doi.org/10.1287/ijoc.2024.0983.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{pdhcg,
  author =        {Y. Huang, W. Zhang, H. Li, H. Liu, D. Ge and Y. Ye},
  publisher =     {INFORMS Journal on Computing},
  title =         {A Restarted Primal-Dual Hybrid Conjugate Gradient Method for Large-Scale Quadratic Programming},
  year =          {2025},
  doi =           {10.1287/ijoc.2024.0983.cd},
  url =           {https://github.com/INFORMSJoC/2024.0983},
  note =          {Available for download at https://github.com/INFORMSJoC/2024.0983},
}  
```
# Usage
## 1. Problem Creation

You can define a `QuadraticProgrammingProblem` in one of three ways: reading from a file, generating a standard problem type, or constructing it directly from matrices and vectors.

### `readFile`
Reads a QP problem from a file in MPS or QPS format.

**Signature:**
```julia
function readFile(filename::String; fix_format::Bool = false)
```

**Description:**
This function utilizes the `QPSReader.jl` package to parse a file and transforms it into the `QuadraticProgrammingProblem` standard form required by the solver.

**Arguments:**
- `filename::String`: The path to the MPS or QPS file.
- `fix_format::Bool` (optional, default: `false`): A flag passed to the underlying reader to handle potential format inconsistencies.

**Returns:**
A `QuadraticProgrammingProblem` struct.

---

### `generateProblem`
Generates a QP problem of a specified standard type.

**Signature:**
```julia
function generateProblem(problem_type::String; n=100, density=0.1, seed=0)
```

**Description:**
A convenience function to create instances of common QP problems like LASSO, Support Vector Machines (SVM), etc. This is useful for testing and benchmarking.

**Arguments:**
- `problem_type::String`: The type of problem to generate. Supported values are: `"randomqp"`, `"lasso"`, `"svm"`, `"portfolio"`, `"mpc"`.
- `n::Int` (optional, default: `100`): The size or scale of the problem.
- `density::Float64` (optional, default: `0.1`): The density of the problem matrices (where applicable).
- `seed::Int` (optional, default: `0`): A random seed for reproducibility.

**Returns:**
A `QuadraticProgrammingProblem` struct.

---

### `constructProblem`
Constructs a QP problem directly from its constituent matrices and vectors.

**Signature:**
```julia
function constructProblem(
    objective_matrix, 
    objective_vector, 
    objective_constant, 
    constraint_matrix,
    constraint_lower_bound, 
    variable_lower_bound=nothing, 
    variable_upper_bound=nothing
)
```

**Description:**
This is the most direct way to define a custom QP problem. If variable bounds are not provided, they default to `-Inf` and `+Inf` respectively.

**Arguments:**
- `objective_matrix`: The quadratic term matrix (Q or H).
- `objective_vector`: The linear term vector (c or f).
- `objective_constant`: A scalar constant in the objective function.
- `constraint_matrix`: The matrix for linear constraints (A).
- `constraint_lower_bound`: The lower bound vector for the constraints (l).
- `variable_lower_bound` (optional): The lower bound vector for the primal variables (x).
- `variable_upper_bound` (optional): The upper bound vector for the primal variables (x).

**Returns:**
A `QuadraticProgrammingProblem` struct.

## 2. Solver Execution

The primary function for solving a problem is `pdhcgSolve`, which wraps a lower-level `_solve` function and provides extensive configuration options.

### `pdhcgSolve`
The main entry point for solving a `QuadraticProgrammingProblem`.

**Signature:**
```julia
function pdhcgSolve(
    qp::QuadraticProgrammingProblem;
    gpu_flag::Bool = false,
    warm_up_flag::Bool = false,
    online_precondition_band::Union{Int64, Nothing} = nothing,
    verbose_level::Int64 = 2,
    time_limit::Float64 = 3600.0,
    relat_error_tolerance::Float64 = 1e-6,
    iteration_limit::Int64 = Int64(typemax(Int32)),
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
    initial_diagonal_precondition::Union{Vector{Float64}, Nothing} = nothing,
)
```

**Description:**
This function configures and runs the PDHCG optimizer on a given `qp` problem. It allows for fine-grained control over termination criteria, restart strategies, rescaling, and hardware acceleration.

**Arguments:**
- `qp::QuadraticProgrammingProblem`: The problem to be solved.

**Execution Control:**
- `gpu_flag::Bool`: If true, the solver will run on the GPU. Default: `false`.
- `warm_up_flag::Bool`: If true, performs a short, preliminary run to mitigate compilation overhead. Default: `false`.
- `verbose_level::Int64`: Controls the verbosity of the solver's log output. `0` for silent, `1` for summary, `2` for detailed. Default: `2`.

**Termination Criteria:**
- `time_limit::Float64`: The maximum wall-clock time in seconds. Default: `3600.0`.
- `relat_error_tolerance::Float64`: The desired relative tolerance for termination. Default: `1e-6`.
- `iteration_limit::Int64`: The maximum number of iterations. Default: `typemax(Int32)`.

**Algorithm Parameters:**
- `ruiz_rescaling_iters::Int64`: Number of Ruiz rescaling iterations to perform. Default: `10`.
- `l2_norm_rescaling_flag::Bool`: Whether to perform L2 norm rescaling. Default: `false`.
- `pock_chambolle_alpha::Float64`: The step size parameter alpha for the Pock-Chambolle algorithm. Default: `1.0`.
- `online_precondition_band::Int64`: (GPU only) The band size for online preconditioning. `nothing` disables it. Default: `nothing`.
- `artificial_restart_threshold::Float64`: Threshold for triggering an artificial restart. Default: `0.2`.
- `sufficient_reduction::Float64`: Sufficient reduction factor for adaptive restarts. Default: `0.2`.
- `necessary_reduction::Float64`: Necessary reduction factor for adaptive restarts. Default: `0.8`.
- `primal_weight_update_smoothing::Float64`: Smoothing factor for primal weight updates during restarts. Default: `0.2`.

**Output and Logging:**
- `save_flag::Bool`: If true, saves the solution and logs to files. Default: `false`.
- `saved_name::String`: The base name for the output files (e.g., "my_problem"). Required if `save_flag` is true. Default: `nothing`.
- `output_dir::String`: The directory where results will be saved. Default: `nothing` (current directory).

**Warm-Starting:**
- `warm_start_flag::Bool`: If true, uses the provided initial solutions. Default: `false`.
- `initial_primal::Vector{Float64}`: An initial guess for the primal solution vector. Default: `nothing`.
- `initial_dual::Vector{Float64}`: An initial guess for the dual solution vector. Default: `nothing`.
- `initial_diagonal_precondition::Vector{Float64}`: An initial guess for the diagonal preconditioner. Default: `nothing`.

**Returns:**
A `SolveLog` struct containing the solution, statistics, and termination information.


# Solver Log Documentation

## Overview
The `SolveLog` struct provides comprehensive information about the solver's execution and results. It contains detailed statistics about the optimization process, termination reasons, and solution quality metrics.

## Main Components

### `SolveLog` Structure
```julia
mutable struct SolveLog
    instance_name::String
    command_line_invocation::String
    termination_reason::TerminationReason
    termination_string::String
    iteration_count::Int32
    CG_total_iteration::Int64
    solve_time_sec::Float64
    solution_stats::IterationStats
    solution_type::PointType
    iteration_stats::Vector{IterationStats}
    kkt_error::Vector{Float64}
    primal_solution::Union{Vector{Float64}, Nothing}
    dual_solution::Union{Vector{Float64}, Nothing}
    objective_value::Union{Float64, Nothing}
end
```

### Key Fields Explanation

#### 1. Basic Information
- `instance_name`: Name of the optimization problem
- `command_line_invocation`: Command used to invoke the solver (if applicable)

#### 2. Termination Information
- `termination_reason`: Why the solver stopped (see `TerminationReason` enum below)
- `termination_string`: Additional details about termination
- `iteration_count`: Total iterations performed
- `CG_total_iteration`: Total conjugate gradient iterations (if applicable)
- `solve_time_sec`: Total runtime in seconds

#### 3. Solution Quality
- `solution_stats`: Final iteration statistics (see `IterationStats` below)
- `solution_type`: Type of point that caused termination
- `kkt_error`: KKT error metrics
- `primal_solution`: Final primal solution vector
- `dual_solution`: Final dual solution vector
- `objective_value`: Final objective value

### `TerminationReason` Enum
```julia
@enum TerminationReason begin
    TERMINATION_REASON_UNSPECIFIED
    TERMINATION_REASON_OPTIMAL         # Found optimal solution
    TERMINATION_REASON_TIME_LIMIT      # Exceeded time limit
    TERMINATION_REASON_ITERATION_LIMIT # Exceeded iteration limit
    TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT
    TERMINATION_REASON_NUMERICAL_ERROR
    TERMINATION_REASON_INVALID_PROBLEM # Problem data issues
    TERMINATION_REASON_OTHER           # Other unspecified reason
end
```

### `IterationStats` Structure
Contains detailed statistics for each iteration:
```julia
mutable struct IterationStats
    iteration_number::Int32
    convergence_information::Vector{ConvergenceInformation}
    cumulative_kkt_matrix_passes::Float64
    cumulative_rejected_steps::Int32
    cumulative_time_sec::Float64
    restart_used::RestartChoice
    step_size::Float64
    primal_weight::Float64
    method_specific_stats::Dict{String,Float64}
end
```

### `ConvergenceInformation` Structure
Measures solution quality at different points:
```julia
mutable struct ConvergenceInformation
    candidate_type::PointType
    primal_objective::Float64
    dual_objective::Float64
    corrected_dual_objective::Float64
    l_inf_primal_residual::Float64    # ∞-norm of primal constraint violations
    l2_primal_residual::Float64       # 2-norm of primal constraint violations
    l_inf_dual_residual::Float64      # ∞-norm of dual constraint violations
    l2_dual_residual::Float64         # 2-norm of dual constraint violations
    relative_l_inf_primal_residual::Float64
    relative_l2_primal_residual::Float64
    relative_l_inf_dual_residual::Float64
    relative_l2_dual_residual::Float64
    relative_optimality_gap::Float64
    l_inf_primal_variable::Float64    # ∞-norm of primal variables
    l2_primal_variable::Float64       # 2-norm of primal variables
    l_inf_dual_variable::Float64      # ∞-norm of dual variables
    l2_dual_variable::Float64         # 2-norm of dual variables
end
```

### `PointType` Enum
```julia
@enum PointType begin
    POINT_TYPE_UNSPECIFIED
    POINT_TYPE_CURRENT_ITERATE     # Current (x_k, y_k)
    POINT_TYPE_ITERATE_DIFFERENCE  # (x_{k+1} - x_k, y_{k+1} - y_k)
    POINT_TYPE_AVERAGE_ITERATE     # Average since last restart
    POINT_TYPE_NONE
end
```

### `RestartChoice` Enum
```julia
@enum RestartChoice begin
    RESTART_CHOICE_UNSPECIFIED
    RESTART_CHOICE_NO_RESTART
    RESTART_CHOICE_WEIGHTED_AVERAGE_RESET
    RESTART_CHOICE_RESTART_TO_AVERAGE
end
```

## Interpretation Guide

1. **Checking Solution Status**:
   - Examine `termination_reason` first
   - `TERMINATION_REASON_OPTIMAL` indicates successful convergence
   - Other values may indicate limits or problems

2. **Solution Quality Metrics**:
   - Small `l_inf_primal_residual` indicates constraint satisfaction
   - Small `relative_optimality_gap` indicates near-optimality
   - Compare `primal_objective` and `dual_objective` for duality gap

3. **Performance Analysis**:
   - `solve_time_sec` for total runtime
   - `iteration_count` for iteration efficiency
   - `iteration_stats` for detailed convergence history

4. **Algorithm Behavior**:
   - `step_size` and `primal_weight` show adaptation
   - `restart_used` indicates restart events
   - `kkt_error` tracks optimality conditions

## Example Usage
```julia
# After solving...
log = pdhcgSolve(qp_problem)

if log.termination_reason == TERMINATION_REASON_OPTIMAL
    println("Optimal solution found!")
    println("Objective value: ", log.objective_value)
    println("Primal residual: ", log.solution_stats.convergence_information[1].l_inf_primal_residual)
else
    println("Solver stopped due to: ", log.termination_string)
end
```

## Notes
- All norms and residuals assume the primal is a minimization problem
- The dual vector includes only linear constraint multipliers (not reduced costs)
- Relative residuals are scaled by `(eps_optimal_absolute / eps_optimal_relative)`
