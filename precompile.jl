ENV["CUDA_VISIBLE_DEVICES"] = "3"
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
warm_up_flag = true
for file in readdir(dataset_dir)  # Skip the first 15 files for warm-up
    global warm_up_flag
    if endswith(file, ".QPS")
        if file == "BOYD1.QPS"
            continue  # Skip this file as it is known to cause issues
        end
        # file = "DUALC1.QPS"  # For testing purposes, we can use a specific file
        qp = PDHCG.readFile(joinpath(dataset_dir, file))
        log = PDHCG.pdhcgSolve(qp, gpu_flag=true, warm_up_flag=warm_up_flag, online_precondition_band_dual=0, verbose_level=6, time_limit = 600.)
        time_cost = log.solve_time_sec
        obj = log.objective_value
        outer_iter = log.iteration_count
        inner_iter = log.CG_total_iteration
        push!(result_df, (file, time_cost, obj, outer_iter, inner_iter))
        #save  data frame to csv
        CSV.write("results_test.csv", result_df, append=false)
        println("Processed file: $file, Time: $time_cost sec, Objective: $obj, Outer Iterations: $outer_iter, Inner Iterations: $inner_iter")
        # warm_up_flag = false
        # break
    end

end

