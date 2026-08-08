# PDHCG: A First-Order Solver for Quadratic Conic Programming with Multi-GPU Acceleration

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![PyPI version](https://badge.fury.io/py/pdhcg.svg)](https://pypi.org/project/pdhcg/) [![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://lhongpei.github.io/PDHCG) [![Publication](https://img.shields.io/badge/DOI-10.1287/ijoc.2024.0983-B31B1B.svg)](https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0983) [![arXiv](https://img.shields.io/badge/arXiv-2602.23967-b31b1b.svg)](https://arxiv.org/abs/2602.23967) [![qpsolvers](https://img.shields.io/badge/qpsolvers-supported-brightgreen.svg)](https://github.com/qpsolvers/qpsolvers) [![CVXPY](https://img.shields.io/badge/CVXPY-supported-brightgreen.svg)](https://www.cvxpy.org/)


**PDHCG** is a high-performance, GPU-accelerated implementation of the Primal-Dual Hybrid Gradient (PDHG) algorithm for large-scale convex quadratic and quadratic conic programming.

For a detailed explanation of the methodology, please refer to our papers: [A Restarted Primal-Dual Hybrid Conjugate Gradient Method for Large-Scale Quadratic Programming](https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0983) and [PDHCG: An Enhanced First-Order Solver for Large-Scale Convex QP](https://arxiv.org/abs/2602.23967).


---

## Problem Formulation

PDHCG solves convex quadratic conic programs in the following form, with a sparse quadratic objective component and an optional structured low-rank component:

```math
\begin{aligned}
\min_{x} \quad & \frac{1}{2}x^\top (Q + R^\top D R) x + c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & Fx + g \in \mathcal{K}_a, \\
                  & \ell_v \le x \le u_v, \\
                  & x_J \in \mathcal{K}_v \quad \text{for variable-cone blocks } J.
\end{aligned}
```

- $Q$ is the sparse symmetric quadratic component (optional).
- $R \in \mathbb{R}^{k\times n}$ is a tall low-rank factor (optional, $k$ = rank).
- $D \in \mathbb{R}^{k\times k}$ is an optional middle matrix that scales / weights / signs the low-rank term. When omitted it defaults to the identity, recovering the standard $Q + R^\top R$ formulation. $D$ may be **diagonal, sparse, dense, or indefinite** — the backend auto-detects the cheapest runtime representation.
- Standard SOC, Rotated SOC, Exponential, and Power cones are supported both on variable blocks and through native affine constraints $Fx + g \in \mathcal{K}_a$.


## Installation (C++ Executable)

To use the standalone C++ solver, you must compile the project using CMake.

### Requirements
* **GPU:** NVIDIA GPU with CUDA 12.4+.
* **Build Tools:** CMake (≥ 3.20), GCC, NVCC.
* **Distributed (Optional):** MPI (e.g., OpenMPI) and NCCL for multi-GPU support.

### Build from Source
Clone the repository and compile the project using CMake.
```bash
git clone https://github.com/Lhongpei/PDHCG.git
cd PDHCG
cmake -S . -B build
cmake --build build --clean-first
```
This will create the solver binary at `./build/bin/pdhcg`.

If your system has multiple CUDA versions or the default nvcc is outdated (e.g., in `/usr/bin/nvcc`), you should explicitly specify the path to your modern CUDA compiler using the CUDACXX environment variable.
```bash
git clone https://github.com/Lhongpei/PDHCG.git
cd PDHCG
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

Distributed conic solves preserve every cone as an ordered permutation unit. Small cones, exponential/power cones,
cones with cone-preserving scaling disabled, unsupported fixed-slot patterns, and cones requiring a diagonal-Q metric
stay on one column GPU. Large uniformly scaled SOC/RSOC blocks can span column GPUs; projection and dual-residual
kernels reduce only per-cone statistics with NCCL and parallelize each local cone slice across multiple CUDA blocks.
Raw QCQPs are reformulated on rank 0 before this partitioning step.

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
| --curtis_reid_iter | int | Iterations for Curtis-Reid log-domain matrix scaling; 0 disables it. | 0 |
| --l_inf_ruiz_iter | int | Iterations for L-inf Ruiz rescaling. | 10 |
| --pock_chambolle_alpha | double | Value for Pock-Chambolle step size parameter $\alpha$. | 1.0 |
| --no_pock_chambolle | flag | Disable Pock-Chambolle rescaling (enabled by default). | false |
| --no_bound_obj_rescaling | flag | Disable bound objective rescaling (enabled by default). | false |
| --no_cone_preserving_scaling | flag | Keep coordinate-wise scaling within cone blocks. | false |
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

#### Cone scaling aggregation

With the default cone-preserving scaling,
PDHCG first computes a positive candidate scale `d_j` for every coordinate,
then broadcasts one scale over each cone block `B`. Define

\[
d_{\max}=\max_{j\in B}d_j,\qquad
d_{\mathrm{rms}}=\sqrt{\frac{1}{|B|}\sum_{j\in B}d_j^2}.
\]

The block scale is

| Scaling phase | Block size <= 8 | Block size > 8 |
| :--- | :--- | :--- |
| Ruiz | `d_max` | `d_rms` |
| Pock-Chambolle | `d_rms` | `sqrt(d_max * d_rms)` |

This preserves cone geometry while avoiding max-dominated scaling on large
blocks. The aggregation follows the
[`:phase_taper` strategy in HPR-SOCP](https://github.com/PolyU-IOR/HPR-SOCP/blob/0cccff309957e41225646a5e5d0bf811fe899daa/src/utils/scaling.jl#L462-L469),
with its GPU implementation in `src/kernels.jl` at the same commit. PDHCG
applies the rule to both variable and affine cone blocks. Setting
`--no_cone_preserving_scaling` bypasses block aggregation.

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

PDHCG provides a user-friendly Python interface for quadratic and quadratic conic problems using NumPy and SciPy.

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
      title={PDHCG: An Enhanced First-Order Solver for Large-Scale Convex QP},
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
