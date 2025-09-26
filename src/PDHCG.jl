module PDHCG

import Logging
import Printf
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
using ArgParse
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

function julia_main()::Cint
    println("Welcome to PDHCG solver (GPU-based Convex QP Solver)!")
    s = ArgParseSettings(
        description = "PDHCG solver (GPU-ready)",
        commands_are_required = false,
        version = "1.0"
    )

    @add_arg_table! s begin
        "filename"
            help = "QPS / MPS problem file"
            required = false     
            default = ""
        "--gpu", "-g"
            help = "use GPU"
            arg_type = Bool
            default = true
        "--warm-up"
            help = "run warm-up"
            arg_type = Bool
            default = true
        "--verbose", "-v"
            help = "verbosity (0-3)"
            arg_type = Int
            default = 2
        "--time-limit"
            help = "time limit (s)"
            arg_type = Float64
            default = 3600.0
        "--rel-tol"
            help = "relative tolerance"
            arg_type = Float64
            default = 1e-6
        "--iter-limit"
            help = "max iterations"
            arg_type = Int
            default = typemax(Int32)
        "--ruiz-iters"
            help = "Ruiz rescaling iterations"
            arg_type = Int
            default = 10
        "--l2-rescale"
            help = "use L2 rescaling"
            action = :store_true
        "--pock-alpha"
            help = "Pock-Chambolle α"
            arg_type = Float64
            default = 1.0
        "--restart-thresh"
            help = "artificial restart threshold"
            arg_type = Float64
            default = 0.2
        "--suff-red"
            help = "sufficient reduction"
            arg_type = Float64
            default = 0.2
        "--nece-red"
            help = "necessary reduction"
            arg_type = Float64
            default = 0.8
        "--primal-smooth"
            help = "primal weight smoothing"
            arg_type = Float64
            default = 0.2
        "--save"
            help = "save result"
            action = :store_true
        "--saved-name"
            help = "output file name"
            arg_type = String
            default = ""
        "--output-dir"
            help = "output directory"
            arg_type = String
            default = ""
    end

    args = parse_args(s)

    isempty(args["filename"]) && return 0

    result = PDHCG.pdhcgSolveFile(
        args["filename"];
        gpu_flag                      = args["gpu"],
        warm_up_flag                  = args["warm-up"],
        verbose_level                 = args["verbose"],
        time_limit                    = args["time-limit"],
        relat_error_tolerance         = args["rel-tol"],
        iteration_limit               = args["iter-limit"],
        ruiz_rescaling_iters          = args["ruiz-iters"],
        l2_norm_rescaling_flag        = args["l2-rescale"],
        pock_chambolle_alpha          = args["pock-alpha"],
        artificial_restart_threshold  = args["restart-thresh"],
        sufficient_reduction          = args["suff-red"],
        necessary_reduction           = args["nece-red"],
        primal_weight_update_smoothing= args["primal-smooth"],
        save_flag                     = args["save"],
        saved_name                    = isempty(args["saved-name"]) ? nothing : args["saved-name"],
        output_dir                    = isempty(args["output-dir"]) ? nothing : args["output-dir"],
    )

    return 0
end



end # module PDHCG
