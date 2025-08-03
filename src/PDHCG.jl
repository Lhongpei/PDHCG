module PDHCG

import LinearAlgebra
import Logging
import Printf
import SparseArrays
import Random

import GZip
import QPSReader
import Statistics
import StatsBase
import StructTypes
using LinearAlgebra
using Random
using SparseArrays

const KERNEL_DIR = joinpath(@__DIR__, "kernel/compiled_kernel/")
const Diagonal = LinearAlgebra.Diagonal
const diag = LinearAlgebra.diag
const dot = LinearAlgebra.dot
const norm = LinearAlgebra.norm
const mean = Statistics.mean
const median = Statistics.median
const nzrange = SparseArrays.nzrange
const nnz = SparseArrays.nnz
const nonzeros = SparseArrays.nonzeros
const rowvals = SparseArrays.rowvals
const sparse = SparseArrays.sparse
const SparseMatrixCSC = SparseArrays.SparseMatrixCSC
const spdiagm = SparseArrays.spdiagm
const spzeros = SparseArrays.spzeros
const quantile = Statistics.quantile
const sample = StatsBase.sample
const mul! = LinearAlgebra.mul!
const ThreadPerBlock = 128

include("problem_gen.jl")
include("quadratic_programming.jl")
include("solve_log.jl")
include("quadratic_programming_io.jl")
include("preprocess.jl")
include("termination.jl")
include("iteration_stats_utils.jl")
include("saddle_point.jl")
include("solver_core.jl")
include("data.jl")
include("optimizer.jl")
using CUDA
include("cpu_to_gpu.jl")
include("iteration_stats_utils_gpu.jl")
include("saddle_point_gpu.jl")
include("solver_gpu_core.jl")

# # Export functions that can be used directly by other languages with Base.@ccallable
# # C-callable wrappers
# Base.@ccallable function c_readFile(filename::Cstring; fix_format::Cint = 0)::Ptr{Cvoid}
#     filename_str = unsafe_string(filename)
#     fix_format_bool = fix_format != 0
#     problem = readFile(filename_str; fix_format = fix_format_bool)
#     return pointer_from_objref(problem)
# end

# Base.@ccallable function c_generateProblem(problem_type::Cstring; n::Cint = 100, density::Cdouble = 0.1, seed::Cint = 0)::Ptr{Cvoid}
#     problem_type_str = unsafe_string(problem_type)
#     problem = generateProblem(problem_type_str; n = n, density = density, seed = seed)
#     return pointer_from_objref(problem)
# end

# Base.@ccallable function c_constructProblem(
#     objective_matrix_ptr::Ptr{Cvoid},
#     objective_vector_ptr::Ptr{Cdouble},
#     objective_constant::Cdouble,
#     constraint_matrix_ptr::Ptr{Cvoid},
#     constraint_lower_bound_ptr::Ptr{Cdouble};
#     variable_lower_bound_ptr::Ptr{Cdouble} = C_NULL,
#     variable_upper_bound_ptr::Ptr{Cdouble} = C_NULL,
#     num_equality_constraints::Cint = 0
# )::Ptr{Cvoid}
#     # Convert pointers to Julia objects
#     objective_matrix = unsafe_pointer_to_objref(objective_matrix_ptr)::SparseMatrixCSC{Float64, Int64}
#     objective_vector = unsafe_wrap(Array, objective_vector_ptr, size(objective_matrix, 2))
#     constraint_matrix = unsafe_pointer_to_objref(constraint_matrix_ptr)::SparseMatrixCSC{Float64, Int64}
#     constraint_lower_bound = unsafe_wrap(Array, constraint_lower_bound_ptr, size(constraint_matrix, 1))

#     variable_lower_bound = if variable_lower_bound_ptr != C_NULL
#         unsafe_wrap(Array, variable_lower_bound_ptr, size(objective_matrix, 2))
#     else
#         nothing
#     end

#     variable_upper_bound = if variable_upper_bound_ptr != C_NULL
#         unsafe_wrap(Array, variable_upper_bound_ptr, size(objective_matrix, 2))
#     else
#         nothing
#     end

#     problem = constructProblem(
#         objective_matrix,
#         objective_vector,
#         objective_constant,
#         constraint_matrix,
#         constraint_lower_bound;
#         variable_lower_bound = variable_lower_bound,
#         variable_upper_bound = variable_upper_bound,
#         num_equality_constraints = num_equality_constraints
#     )
#     return pointer_from_objref(problem)
# end


end # module PDHCG
