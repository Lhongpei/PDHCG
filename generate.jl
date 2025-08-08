using DataFrames
using CSV
include("src/PDHCG.jl")
# ENV["JULIA_PROJECT"] = "pdhcg_env"

qp = PDHCG.generate_randomQP_problem(50000, 423, 1e-3)
log = PDHCG.pdhcgSolve(qp, gpu_flag=true, warm_up_flag=true, online_precondition_band_dual=nothing, verbose_level=2, time_limit = 20.)
time_cost = log.solve_time_sec
obj = log.objective_value
outer_iter = log.iteration_count
inner_iter = log.CG_total_iteration
