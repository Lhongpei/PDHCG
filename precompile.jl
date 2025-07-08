
include("src/PDHCG.jl")

qp = PDHCG.generateProblem("randomqp", n=100, density=0.001, seed=100)
PDHCG.pdhcgSolve(qp, gpu_flag=true, warm_up_flag=true)
