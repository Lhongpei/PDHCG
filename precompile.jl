
using PDHCG
qp = PDHCG.generateProblem("randomqp", n=10000, density=0.001, seed=100)
PDHCG.pdhcgSolve(qp, gpu_flag=true, warm_up_flag=true)
