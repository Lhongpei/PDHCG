# **Python Interface for PDHCG**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/pdhcg.svg)](https://pypi.org/project/pdhcg/)
[![Publication](https://img.shields.io/badge/DOI-10.1287/ijoc.2024.0983-B31B1B.svg)](https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0983)
[![arXiv](https://img.shields.io/badge/arXiv-2608.09159-b31b1b.svg)](https://arxiv.org/abs/2608.09159)

This is the Python interface to **[`PDHCG`](../README.md)**, a GPU-accelerated first-order solver for large-scale quadratic and quadratic conic programming.
It provides a high-level, Pythonic API using NumPy and SciPy data structures.

## Installation

### Requirements
- Python ≥ 3.8
- NumPy ≥ 1.21
- SciPy ≥ 1.8
- An NVIDIA GPU with CUDA support (≥12.4 required)
- A C/C++ toolchain with GCC and NVCC

!!! note "CUDA Version and SpMVOp"
    PDHCG automatically detects your CUDA version at compile time:

    - **CUDA 13+**: Uses cuSPARSE **SpMVOp** for improved performance.
    - **CUDA 12.x**: Falls back to the standard **SpMV** API. No manual intervention is required.

!!! note "Multi-GPU support"
    The Python interface currently supports single-GPU solving only. For multi-GPU distributed solving, build the C++ executable with `-DPDHCG_COMPILE_DISTRIBUTED=ON` and launch it via `mpirun` (see the [main README](../README.md)).

### Install
Install from PyPI:

```bash
pip install pdhcg
```


Or build from source:

```bash
git clone https://github.com/Lhongpei/PDHCG.git
cd PDHCG
pip install .
```

If your system has multiple CUDA installations or the default nvcc (typically in `/usr/bin/nvcc`) is outdated, you must explicitly point to your modern CUDA compiler using environment variables. This ensures the build system bypasses the system default.

```Bash
# Replace '/your/path/to/nvcc' with your actual path
# Example: export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
export CUDACXX=/your/path/to/nvcc
export SKBUILD_CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/your/path/to/nvcc"

pip install pdhcg
```

## Quick Start

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import Model, PDHCG

# Example: minimize 0.5 * x'(Q + R^T D R)x + c'x
# subject to l <= A x <= u,  lb <= x <= ub
# (D defaults to identity, recovering 0.5 * x'Qx + 0.5 * ||Rx||^2.)

# 1. Define Standard QP terms
Q = sp.csc_matrix([[1.0, -1.0], [-1.0, 2.0]])
c = np.array([-2.0, -6.0])

# 2. Define Low-Rank Matrix R
# Let's add a term 0.5 * ||Rx||^2 where R = [[1, 0]]
# This effectively adds 0.5 * (x1)^2 to the objective
R = sp.csc_matrix([[1.0, 0.0]])

# 3. (Optional) Middle matrix D. Pass a 1-D numpy array for a diagonal D,
# a 2-D array for dense D, or a scipy.sparse matrix. Omit to use D = I.
# D = np.array([2.5])

# 4. Define Constraints
A = sp.csc_matrix([[1.0, 1.0], [-1.0, 2.0], [2.0, 1.0]])
l = np.array([-np.inf, -np.inf, -np.inf])
u = np.array([2.0, 2.0, 3.0])
lb = np.zeros(2)
ub = np.array([np.inf, np.inf])

# 5. Create QP model with Low-Rank term
m = Model(objective_matrix=Q,
          objective_matrix_low_rank=R,                # <--- Pass R here
          # objective_matrix_low_rank_middle=D,      # <--- and optionally D
          objective_vector=c,
          constraint_matrix=A,
          constraint_lower_bound=l,
          constraint_upper_bound=u,
          variable_lower_bound=lb,
          variable_upper_bound=ub)

# Set model sense
m.ModelSense = PDHCG.MINIMIZE

# Parameters can be set in multiple ways
m.Params.TimeLimit = 60        # attribute style
m.setParam("FeasibilityTol", 1e-6)
m.setParams(LogLevel=2, OptimalityTol=1e-8)

# Solve
m.optimize()

# Retrieve results
print("Status:", m.Status)
print("Objective:", m.ObjVal)
print("Primal solution:", m.X)
print("Dual solution:", m.Pi)
```

## Cone Constraints

Second-order, rotated second-order, exponential, and power cone constraints use
the columnar `ConeSpec` API:

```python
import numpy as np
from pdhcg import ConeSpec, ConeType, Model

cones = ConeSpec(ConeType.EXP, 3 * np.arange(num_cones, dtype=np.int32))
model = Model(objective_vector=c, constraint_matrix=A, variable_cones=cones)
```

The same object can be passed as `solve_once(..., cones=cones)`. See
[docs/python/quickstart.md](../docs/python/quickstart.md#quick-start-with-cone-constraints)
for a runnable example and [docs/python/model.md](../docs/python/model.md#cone-constraints)
for all fields and affine-cone input.

## CVXPY

Install PDHCG with the optional CVXPY dependency:

```bash
pip install "pdhcg[cvxpy]"
```

Import the backend once before solving:

```python
import cvxpy as cp
import pdhcg.cvxpy_backend  # Registers solver="PDHCG".

x = cp.Variable()
problem = cp.Problem(cp.Minimize(x), [x >= 1])
problem.solve(solver="PDHCG", eps=1e-6)
```

Quadratic objectives and Zero, NonNeg, SOC, ExpCone, and PowCone3D
constraints are supported. PSD and mixed-integer models are not supported.

## Modeling

The `Model` class represents a quadratic conic programming problem of the form:

$$
\begin{aligned}
\min_x \quad & \frac{1}{2} x^\top (Q + R^\top D R) x + c^\top x + c_0 \\
\text{s.t.} \quad & \ell \le A x \le u, \\
                    & Fx + g \in \mathcal{K}_a, \\
                    & x_J \in \mathcal{K}_v \quad \text{for variable-cone blocks } J, \\
                    & \text{lb} \le x \le \text{ub}.
\end{aligned}
$$

### Arguments

- **objective_matrix** (`Q`, optional): Quadratic part of the objective function. Both dense (`numpy.ndarray`) and sparse (`scipy.sparse.csr_matrix`) inputs are supported.
- **objective_matrix_low_rank** (`R`, optional): Low-rank factor matrix in the quadratic objective term. Both dense (`numpy.ndarray`) and sparse (`scipy.sparse.csr_matrix`) inputs are supported.
- **objective_matrix_low_rank_middle** (`D`, optional): Middle matrix in $R^\top D R$, $\text{rank}\times\text{rank}$. Accepts a 1-D numpy array (diagonal $D$), a 2-D numpy array (dense symmetric $D$), or a scipy sparse matrix. May be indefinite — useful for quasi-Newton updates, kernel-based formulations, and weighted least squares. Defaults to identity, recovering $R^\top R$.
- **objective_vector** (`c`): Linear part of the objective function.
- **constraint_matrix** (`A`): Coefficient matrix for the constraints. Both dense (`numpy.ndarray`) and sparse (`scipy.sparse.csr_matrix`) inputs are supported.
- **constraint_lower_bound** (`l`): Lower bounds for each constraint. Use `-np.inf` or `None` for no lower bound.
- **constraint_upper_bound** (`u`): Upper bounds for each constraint. Use `+np.inf` or `None` for no upper bound.
- **affine_cone_matrix** (`F`, optional): Matrix in the native affine-cone constraint $Fx + g \in \mathcal{K}_a$.
- **affine_cone_offset** (`g`, optional): Affine-cone offset. Defaults to zero.
- **affine_cones** (optional): `ConeSpec` covering every row of `F`.
- **variable_cones** (optional): `ConeSpec` describing cone blocks embedded in `x`.
- **variable_lower_bound** (`lb`, optional): Lower bounds for the decision variables. Defaults to `0` for all variables if not provided.
- **variable_upper_bound** (`ub`, optional): Upper bounds for the decision variables. Defaults to `+np.inf` for all variables if not provided.
- **objective_constant** (`c0`, optional): Constant offset in the objective function. Defaults to `0.0`.

To initialize a quadratic programming problem, you need to provide the objective matrix and vector, constraint matrix, and bounds on both constraints and variables.

```python
Q = sp.csc_matrix([[1.0, -1.0], [-1.0, 2.0]])
c = np.array([-2.0, -6.0])
A = sp.csc_matrix([[1.0, 1.0], [-1.0, 2.0], [2.0, 1.0]])
l = np.array([-np.inf, -np.inf, -np.inf])
u = np.array([2.0, 2.0, 3.0])
lb = np.zeros(2)
ub = np.array([np.inf, np.inf])


# Create QP model
m = Model(objective_matrix=Q,
          objective_vector=c,
          constraint_matrix=A,
          constraint_lower_bound=l,
          constraint_upper_bound=u,
          variable_lower_bound=lb,
          variable_upper_bound=ub)
```


## Model Sense

By default, `pdhcg` solves **minimization problems**.

To switch between minimization and maximization, set the attribute `ModelSense`:

```python
# Set model sense
m.ModelSense = PDHCG.MAXIMIZE
```

## Parameters

Solver parameters control termination criteria, logging, scaling, and restart behavior.

Below is a list of commonly used parameters, their internal keys, and descriptions.

| Alias | Internal Key | Type | Default | Description |
|---|---|---|---|---|
| `TimeLimit` | `time_sec_limit` | float | `3600.0` | Maximum wall-clock time in seconds. The solver terminates if the limit is reached. |
| `IterationLimit` | `iteration_limit` | int | `2147483647` | Maximum number of iterations. |
| `LogLevel`, `Verbosity` | `verbose` | int | `1` | Verbosity level: `0` (Silent), `1` (Summary), or `2` (Detailed iteration info). |
| `TermCheckFreq` | `termination_evaluation_frequency` | int | `200` | Frequency (in iterations) at which termination conditions are evaluated. |
| `OptimalityNorm` | `optimality_norm` | string | `"linf"` | Norm for optimality criteria. Use `"l2"` for L2 norm or `"linf"` for infinity norm. |
| `OptimalityTol` | `eps_optimal_relative` | float | `1e-4` | Relative tolerance for optimality gap. Solver stops if the relative primal-dual gap ≤ this value. |
| `FeasibilityTol` | `eps_feasible_relative` | float | `1e-4` | Relative feasibility tolerance for primal/dual residuals. |
| `CurtisReidIters` | `curtis_reid_iterations` | int | `0` | Number of Curtis-Reid log-domain scaling iterations. Set to `0` to disable. |
| `RuizIters` | `l_inf_ruiz_iterations` | int | `10` | Number of iterations for L∞ Ruiz scaling. Improves numerical conditioning. |
| `UsePCAlpha` | `has_pock_chambolle_alpha` | bool | `True` | Whether to use the Pock–Chambolle α step size adjustment. |
| `PCAlpha` | `pock_chambolle_alpha` | float | `1.0` | Value of the Pock–Chambolle α parameter. |
| `BoundObjRescaling` | `bound_objective_rescaling` | bool | `True` | Whether to rescale the objective vector during preprocessing. |
| `UseConePreservingScaling` | `use_cone_preserving_scaling` | bool | `True` | Whether to broadcast one scaling value over every cone block. |
| `RestartArtificialThresh` | `artificial_restart_threshold` | float | `0.36` | Threshold for artificial restart. |
| `RestartSufficientReduction` | `sufficient_reduction_for_restart` | float | `0.2` | Sufficient reduction factor to justify a restart. |
| `RestartNecessaryReduction` | `necessary_reduction_for_restart` | float | `0.8` | Necessary reduction factor required for a restart. |
| `RestartKp` | `k_p` | float | `0.99` | Proportional coefficient for PID-controlled primal weight updates. |
| `ReflectionCoeff` | `reflection_coefficient` | float | `1.0` | Reflection coefficient. |
| `SVMaxIter` | `sv_max_iter` | int | `5000` | Maximum number of iterations for the power method. |
| `SVTol`| `sv_tol` | float | `1e-4` | Termination tolerance for the power method. |
| `InnerIterLimit` | `inner_iter_limit` | int | `1000` | Maximum number of iterations for the inner solver. |
| `InnerInitTol` | `inner_init_tol` | float | `1e-3` | Initial tolerance for the inner solver. |
| `InnerMinTol` | `inner_min_tol` | float | `1e-9` | Minimum tolerance for the inner solver. |
| `DiagJacobiPrecond` | `diag_jacobi_precond` | bool | `True` | Whether to use the Jacobi diagonal preconditioner in the inner subproblem. Set to `False` to disable. |

They can be set in multiple ways:

```python
# Method 1: single parameter
m.setParam("TimeLimit", 300)
m.setParam("FeasibilityTol", 1e-6)

# Method 2: multiple parameters
m.setParams(TimeLimit=300, FeasibilityTol=1e-6)

# Method 3: attribute-style access
m.Params.TimeLimit = 300
m.Params.FeasibilityTol = 1e-6
```

## Solution Attributes

After calling `m.optimize()`, the solver stores results in a set of read-only attributes. These attributes provide access to primal/dual solutions, objective values, residuals, and runtime statistics.

### Attribute Reference

| Attribute | Type | Description |
|---|---|---|
| `Status` | str | Human-readable solver status (`"OPTIMAL"`, `"INFEASIBLE"`, `"UNBOUNDED"`, `"TIME_LIMIT"`, etc.). |
| `StatusCode` | int | Numeric status code (`OPTIMAL=1`, `INFEASIBLE=2`, `UNBOUNDED=3`, `ITERATION_LIMIT=4`, `TIME_LIMIT=5`, `UNSPECIFIED=-1`). |
| `ObjVal` | float | Primal objective value at termination (sign-adjusted according to `ModelSense`). |
| `DualObj` | float | Dual objective value at termination. |
| `Gap` | float | Absolute primal-dual gap. |
| `RelGap` | float | Relative primal-dual gap. |
| `X` | numpy.ndarray | Primal solution vector \(x\). May be `None` if no feasible solution was found. |
| `Pi` | numpy.ndarray | Dual solution vector (Lagrange multipliers). |
| `IterCount` | int | Number of iterations performed. |
| `Runtime` | float | Total wall-clock runtime in seconds. |
| `RescalingTime` | float | Time spent on preprocessing and rescaling (seconds). |
| `RelPrimalResidual` | float | Relative primal residual. |
| `RelDualResidual` | float | Relative dual residual. |
| `MaxPrimalRayInfeas` | float | Maximum primal ray infeasibility (indicator for infeasibility). |
| `MaxDualRayInfeas` | float | Maximum dual ray infeasibility. |
| `PrimalRayLinObj` | float | Linear objective value along a primal ray (used in infeasibility detection). |
| `DualRayObj` | float | Objective value along a dual ray (used in unboundedness detection). |

All solution-related information can then be queried directly from the `Model` object:

```python
m.optimize()

print("Status:", m.Status, "(code:", m.StatusCode, ")")
print("Primal objective:", m.ObjVal)
print("Dual objective:", m.DualObj)
print("Relative gap:", m.RelGap)
print("Iterations:", m.IterCount, " Runtime (s):", m.Runtime)

# Access solutions
print("Primal solution:", m.X)
print("Dual solution:", m.Pi)

# Check residuals
print("Primal residual:", m.RelPrimalResidual)
print("Dual residual:", m.RelDualResidual)
```

## Warm Start

`pdhcg` supports warm starting from user-provided primal and/or dual solutions.
This allows resuming from a previous iterate or reusing solutions from a related instance, often reducing iterations needed to reach optimality.

```python
# Warm starting solution
x_init = [1.0, 2.0]
pi_init = [1.0, -1.0, 0.0]

# Set warm start
m.setWarmStart(primal=x_init, dual=pi_init)

# Solve
m.optimize()
```

Both primal and dual arguments are optional. You may specify only one of them if desired:

```python
# Only provide primal start
m.setWarmStart(primal=x_init)

# Only provide dual start
m.setWarmStart(dual=pi_init)
```

If the warm-start vectors have incorrect dimensions, the solver automatically falls back to a cold start and issues a warning.

To clear existing warm-start values:

```python
m.clearWarmStart()
```

or

```python
m.setWarmStart(primal=None, dual=None)
```
