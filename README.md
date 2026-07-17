# PDHCG: A First-Order Solver for Quadratic Conic Programming with Multi-GPU Acceleration

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![PyPI version](https://badge.fury.io/py/pdhcg.svg)](https://pypi.org/project/pdhcg/) [![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://lhongpei.github.io/PDHCG-II) [![Publication](https://img.shields.io/badge/DOI-10.1287/ijoc.2024.0983-B31B1B.svg)](https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0983) [![arXiv](https://img.shields.io/badge/arXiv-2602.23967-b31b1b.svg)](https://arxiv.org/abs/2602.23967) [![qpsolvers](https://img.shields.io/badge/qpsolvers-supported-brightgreen.svg)](https://github.com/qpsolvers/qpsolvers) [![CVXPY](https://img.shields.io/badge/CVXPY-supported-brightgreen.svg)](https://www.cvxpy.org/)


**PDHCG** is a high-performance, GPU-accelerated implementation of the Primal-Dual Hybrid Gradient (PDHG) algorithm designed for solving large-scale Convex Quadratic Programming (QP) problems.

For a detailed explanation of the methodology, please refer to our papers: [A Restarted Primal-Dual Hybrid Conjugate Gradient Method for Large-Scale Quadratic Programming](https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0983) and [PDHCG-II: An Enhanced Version of PDHCG for Large-Scale Convex QP](https://arxiv.org/abs/2602.23967).


---

## Problem Formulation

PDHCG solves convex quadratic programs in the following form, which allows a flexible input of the quadratic objective matrix as a sparse component plus a structured low-rank component:

```math
\begin{aligned}
\min_{x} \quad & \frac{1}{2}x^\top (Q + R^\top D R) x + c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & \ell_v \le x_I \le u_v, \\
                  & x_J \in \mathcal{K} \quad \text{for cone blocks } J.
\end{aligned}
```

- $Q$ is the sparse symmetric quadratic component (optional).
- $R \in \mathbb{R}^{k\times n}$ is a tall low-rank factor (optional, $k$ = rank).
- $D \in \mathbb{R}^{k\times k}$ is an optional middle matrix that scales / weights / signs the low-rank term. When omitted it defaults to the identity, recovering the standard $Q + R^\top R$ formulation. $D$ may be **diagonal, sparse, dense, or indefinite** — the backend auto-detects the cheapest runtime representation.
- Convex conic constraints (Standard SOC, Rotated SOC, Exponential cone) are supported as additional restrictions on the variable set.


## Installation (C++ Executable)

To use the standalone C++ solver, you must compile the project using CMake.

### Requirements
* **GPU:** NVIDIA GPU with CUDA 12.4+.
* **Build Tools:** CMake (≥ 3.20), GCC, NVCC.
* **Distributed (Optional):** MPI (e.g., OpenMPI) and NCCL for multi-GPU support.

### Build from Source
Clone the repository and compile the project using CMake.
```bash
git clone https://github.com/Lhongpei/PDHCG-II.git
cd PDHCG-II
cmake -S . -B build
cmake --build build --clean-first
```
This will create the solver binary at `./build/bin/pdhcg`.

If your system has multiple CUDA versions or the default nvcc is outdated (e.g., in `/usr/bin/nvcc`), you should explicitly specify the path to your modern CUDA compiler using the CUDACXX environment variable.
```bash
git clone https://github.com/Lhongpei/PDHCG-II.git
cd PDHCG-II
# Replace '/your/path/to/nvcc' with the actual path, e.g., /usr/local/cuda-12.6/bin/nvcc
CUDACXX=/your/path/to/nvcc cmake -S . -B build
cmake --build build --clean-first
```

### Build with Multi-GPU Support

To enable distributed multi-GPU solving, turn on the `PDHCG_COMPILE_DISTRIBUTED` CMake option:

```bash
cmake -S . -B build -DPDHCG_COMPILE_DISTRIBUTED=ON
cmake --build build --clean-first
```

This requires MPI and NCCL to be installed on your system.

##  Usage (C++ Executable)

Run the solver from the command line:

```bash
./build/bin/pdhcg <FILE_NAME> <OUTPUT_DIR> [OPTIONS]
```

### Command Line Arguments

**Positional Arguments:**

1. `<FILE_NAME>`: Path to the input problem file (supports `.mps`, `.qps`, `.cbf`, and gzip-compressed variants).
2. `<OUTPUT_DIR>`: Directory where solution files will be saved.

Solver Parameters:
| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| -h, --help | flag | Display the help message. | N/A |
| -v, --verbose | int | Verbosity level: 0 (Silent), 1 (Summary), 2 (Detailed). | 1 |
| --time_limit | double | Time limit in seconds. | 3600.0 |
| --iter_limit | int | Iteration limit. | 2147483647 |
| --eps_opt | double | Relative optimality tolerance. | 1e-4 |
| --eps_feas | double | Relative feasibility tolerance. | 1e-4 |
| --eps_infeas_detect | double | Infeasibility detection tolerance. | 1e-12 |
| --l_inf_ruiz_iter | int | Iterations for L-inf Ruiz rescaling. | 10 |
| --pock_chambolle_alpha | double | Value for Pock-Chambolle step size parameter $\alpha$. | 1.0 |
| --no_pock_chambolle | flag | Disable Pock-Chambolle rescaling (enabled by default). | false |
| --no_bound_obj_rescaling | flag | Disable bound objective rescaling (enabled by default). | false |
| --heterogeneous_cone_scaling | flag | Keep per-element cone scaling instead of forcing one scaling value per cone block. | false |
| --sv_max_iter | int | Max iterations for singular value estimation (Power Method). | 5000 |
| --sv_tol | double | Tolerance for singular value estimation. | 1e-4 |
| --eval_freq | int | Frequency of termination criteria evaluation (in iterations). | 200 |
| --artificial_restart_threshold | double | Threshold for artificial restart. | 0.36 |
| --sufficient_reduction_for_restart | double | Sufficient reduction factor to justify a restart. | 0.2 |
| --necessary_reduction_for_restart | double | Necessary reduction factor required for a restart. | 0.8 |
| --opt_norm | string | Norm for optimality criteria (l2 or linf). | linf |
| --inner_iter_limit | int | Max iterations for the inner solver. | 1000 |
| --inner_init_tol | double | Initial tolerance for the inner solver. | 1e-3 |
| --inner_min_tol | double | Minimum tolerance for the inner solver. | 1e-9 |
| --no_diag_precond | flag | Disable the Jacobi diagonal preconditioner used in the inner subproblem (enabled by default). | false |
| --soc_form | string | Cone formulation for QCQP transformations: rotated or standard. | rotated |

**Distributed Options** (only available when built with `-DPDHCG_COMPILE_DISTRIBUTED=ON`):
| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| --grid_size | string | 2D grid dimensions, format `r,c` (e.g., `2,4`). Must satisfy `r*c = num_procs`. | auto |
| --partition_method | string | Matrix partition strategy: `uniform` or `nnz`. | nnz |
| --permute_method | string | Permutation strategy: `none`, `random`, or `block`. | block |
| --permute_block_size | int | Block size for block permutation. | 256 |

### Multi-GPU Usage

When built with distributed support, the same binary automatically detects whether it is launched with multiple MPI ranks and switches to the distributed solver:

```bash

# Multi-GPU on 4 GPUs
mpirun -n 4 ./build/bin/pdhcg problem.mps ./output

# Multi-GPU with a custom 2x2 process grid
mpirun -n 4 ./build/bin/pdhcg problem.mps ./output --grid_size 2,2
```

---

## Python Interface

> PDHCG is now officially supported as a backend in the popular [`qpsolvers`](https://github.com/qpsolvers/qpsolvers) ecosystem (v4.11.0+).

PDHCG provides a user-friendly Python interface that allows you to define, solve, and analyze QP problems using familiar libraries like NumPy and SciPy.

For detailed instructions on how to use the Python interface, including installation, modeling, and examples, please see the [Python Interface README](./python/README.md).

### Quick Example in Python

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import Model

# Example: minimize 0.5 * x'(Q + R^T D R)x + c'x
# subject to l <= A x <= u,  lb <= x <= ub
# (D defaults to identity, recovering the classic Q + R^T R form.)

# 1. Define Standard QP terms
Q = sp.csc_matrix([[1.0, -1.0], [-1.0, 2.0]])
c = np.array([-2.0, -6.0])

# 2. Define Low-Rank Matrix R
# Let's add a term 0.5 * ||Rx||^2 where R = [[1, 0]]
# This adds 0.5 * (x1)^2 to the objective
R = sp.csc_matrix([[1.0, 0.0]])

# 3. (Optional) Middle matrix D in 0.5 * x^T R^T D R x. Pass a 1-D array
# for a diagonal D, a 2-D array for dense D, or a scipy.sparse matrix.
# Omit entirely (or pass None) to use D = identity.
# D = np.array([2.5])  # e.g., weight the low-rank term by 2.5

# 4. Define Constraints
A = sp.csc_matrix([[1.0, 1.0], [-1.0, 2.0], [2.0, 1.0]])
l = np.array([-np.inf, -np.inf, -np.inf])
u = np.array([2.0, 2.0, 3.0])
lb = np.zeros(2)
ub = np.array([np.inf, np.inf])

# 5. Create QP model. Q, R and D are all optional.
m = Model(objective_matrix=Q,
          objective_matrix_low_rank=R,
          # objective_matrix_low_rank_middle=D,  # uncomment to use D
          objective_vector=c,
          constraint_matrix=A,
          constraint_lower_bound=l,
          constraint_upper_bound=u,
          variable_lower_bound=lb,
          variable_upper_bound=ub)

# 5. Set solver parameters (0=Silent, 1=Summary, 2=Detailed)
m.setParams(LogLevel=2)

# Solve
m.optimize()

# Print results
print(f"Status: {m.Status}")
print(f"Objective: {m.ObjVal:.4f}")
if m.X is not None:
    print(f"Primal Solution: {m.X}")
```

## Citation
If you use this software or method in your research, please cite our paper:
```
@misc{li2026pdhcgiienhancedversionpdhcg,
      title={PDHCG-II: An Enhanced Version of PDHCG for Large-Scale Convex QP},
      author={Hongpei Li and Yicheng Huang and Huikang Liu and Dongdong Ge and Yinyu Ye},
      year={2026},
      eprint={2602.23967},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2602.23967},
}
```

## Acknowledgments

This solver is built upon the infrastructure of [cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx) (originally developed by Haihao Lu). We gratefully acknowledge this project for providing the high-performance CUDA-C framework for Linear Programming (LP) that serves as the foundation for this QP solver.



---

## License

Copyright 2024-2026 Hongpei Li, Haihao Lu.

Licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
