module PDHCG

import LinearAlgebra
import Logging
import Printf
import SparseArrays
import Random
using CUDA
import GZip
import QPSReader
import Statistics
import StatsBase
import StructTypes
using LinearAlgebra
using Random
using SparseArrays
using Dates: now
using Dates
using CUDA.CUSPARSE
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
include("cpu_to_gpu.jl")
include("iteration_stats_utils_gpu.jl")
include("saddle_point_gpu.jl")
include("solver_gpu_core.jl")
include("preprocess_gpu.jl")
end # module PDHCG
