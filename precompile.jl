using DataFrames
using CSV
include("src/PDHCG.jl")
# ENV["JULIA_PROJECT"] = "pdhcg_env"
dataset_dir = expanduser("~/QP_datasets/MM_benchmark")
result_df = DataFrame(
    dataset = String[],
    time_cost_sec = Float64[],
    objective_value = Float64[],
    outer_iteration_count = Int[],
    inner_iteration_count = Int[]
)
for file in readdir(dataset_dir)
    if endswith(file, ".QPS")
        qp = PDHCG.readFile(joinpath(dataset_dir, "CONT-100.QPS"))
        log = PDHCG.pdhcgSolve(qp, gpu_flag=true, warm_up_flag=true, online_precondition_band_dual=nothing, verbose_level=2, time_limit = 600.)
        time_cost = log.solve_time_sec
        obj = log.objective_value
        outer_iter = log.iteration_count
        inner_iter = log.CG_total_iteration
        push!(result_df, (file, time_cost, obj, outer_iter, inner_iter))
        #save  data frame to csv
        CSV.write("results.csv", result_df, append=false)
        println("Processed file: $file, Time: $time_cost sec, Objective: $obj, Outer Iterations: $outer_iter, Inner Iterations: $inner_iter")
        break
    end
end

