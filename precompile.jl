# using DataFrames
# using CSV
using PDHCG
# ENV["JULIA_PROJECT"] = "pdhcg_env"
dataset_dir = expanduser("~/QP_datasets/MM_benchmark")
k = 1
files = readdir(dataset_dir)
for i in 1:k:1#length(files)
    file = files[i]
    if endswith(file, ".QPS")
        qp = PDHCG.readFile(joinpath(dataset_dir, file))
        log = PDHCG.pdhcgSolve(qp, gpu_flag=true, warm_up_flag=true, online_precondition_band_dual=nothing, verbose_level=2, time_limit = 600.)
        time_cost = log.solve_time_sec
        obj = log.objective_value
        outer_iter = log.iteration_count
        inner_iter = log.CG_total_iteration
    end
end


