import numpy as np
from PDHCG import PDHCG

def test_pdhcg_cpu():
    print("Running CPU test for PDHCG...")
    solver = PDHCG(name="Example QP Solver")
    solver.setParams(gpu_flag=False, verbose_level=0)
    solver.read("example/example1.QPS")
    solver.solve()
    
    print(f"Solver Status (CPU): {solver.status}")
    assert solver.status == "OPTIMAL"
    print("CPU test passed.")

def test_pdhcg_gpu():
    print("Running GPU test for PDHCG...")
    solver = PDHCG(name="Example QP Solver")
    solver.setParams(gpu_flag=True, verbose_level=0)
    solver.read("example/example1.QPS")
    solver.solve()
    
    print(f"Solver Status (GPU): {solver.status}")
    assert solver.status == "OPTIMAL"
    print("GPU test passed.")
