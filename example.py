import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
from PDHCG import PDHCG
import pandas as pd

solver = PDHCG(name="Example QP Solver")

# Set parameters
solver.setParams(gpu_flag=True, verbose_level=3, ruiz_rescaling_iters=10, time_limit=600.0, primal_weight_update_smoothing=0.01)

# Define a custom problem
objective_matrix = np.eye(10)
objective_vector = np.ones(10)
constraint_matrix = np.random.randn(5, 10)
constraint_lower_bound = np.zeros(5)
dict_result = {}
dataset_root = "../QP_datasets/MM_benchmark/"
for file in os.listdir(dataset_root)[1:]:
    print(file)
    solver.read(os.path.join(dataset_root, file))
    solver.solve()
    dict_result['file'] = {
        'objective_value': solver.objective_value,
        'iteration': solver.iteration_count,
        'run_time': solver.solve_time_sec
    }
print("Start solving the problem")
# Solve the problem
df = pd.DataFrame(dict_result)
df.to_csv("result.csv", index=False)

# Output results
# print("Objective Value:", solver.objective_value)
# print("Primal Solution:", solver.primal_solution)
# print("Dual Solution:", solver.dual_solution)