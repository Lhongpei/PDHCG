# Python Quick Start

PDHCG provides a user-friendly Python interface that allows you to define, solve, and analyze QP problems using familiar libraries like NumPy and SciPy.

## Basic Usage

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import Model

# Example: minimize 0.5 * x'(Q + R^T D R)x + c'x
# subject to l <= A x <= u,  lb <= x <= ub
# (D defaults to identity, i.e. 0.5 * x'(Q + R^T R)x + c'x.)

# 1. Define Standard QP terms
Q = sp.csc_matrix([[1.0, -1.0], [-1.0, 2.0]])
c = np.array([-2.0, -6.0])

# 2. Define Low-Rank Matrix R
# This adds 0.5 * ||Rx||^2 to the objective
R = sp.csc_matrix([[1.0, 0.0]])

# 3. (Optional) Middle matrix D for R^T D R; 1-D = diag, 2-D = dense, sparse OK.
# D = np.array([2.5])

# 4. Define Constraints
A = sp.csc_matrix([[1.0, 1.0], [-1.0, 2.0], [2.0, 1.0]])
l = np.array([-np.inf, -np.inf, -np.inf])
u = np.array([2.0, 2.0, 3.0])
lb = np.zeros(2)
ub = np.array([np.inf, np.inf])

# 5. Create QP model with Low-Rank term (R) and optional middle D
m = Model(objective_matrix=Q,
          objective_matrix_low_rank=R,
          # objective_matrix_low_rank_middle=D,
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

## Quick start with cone constraints

Conic constraints are passed to the lower-level [`solve_once`](../c/functions.md) entry point
via a `cones=` list of dicts. Each dict has `type` (`"soc"`, `"rsoc"`, or `"exp"`),
`start_idx`, and (for SOC/RSOC) `v_dim`. See [model.md](model.md#cone-constraints).

```python
import numpy as np
import scipy.sparse as sp
from pdhcg._core import solve_once

# min  z  s.t.  v = 3, w = 4, (v, w, z) in K_soc  =>  z = sqrt(v^2 + w^2) = 5
A = sp.csr_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
res = solve_once(
    None, None, A,
    np.array([0.0, 0.0, 1.0]),                # objective_vector
    0.0,                                      # objective_constant
    np.array([-np.inf, -np.inf, -np.inf]),    # variable_lower_bound
    np.array([np.inf, np.inf, np.inf]),       # variable_upper_bound
    np.array([3.0, 4.0]),                     # constraint_lower_bound
    np.array([3.0, 4.0]),                     # constraint_upper_bound
    cones=[{"type": "soc", "start_idx": 0, "v_dim": 1}],
)
print(res["Status"], res["X"])
```

## Model Creation

The `Model` class is the core interface for defining QP problems. The problem formulation is:

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2}x^\top (Q + R^\top D R) x + c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & \ell_v \le x \le u_v.
\end{aligned}
$$

### Required Parameters

- `objective_vector` ($c$): Linear coefficients of the objective function

### Optional Parameters

- `objective_matrix` ($Q$): Sparse quadratic coefficients
- `objective_matrix_low_rank` ($R$): Low-rank quadratic factor of shape $(k, n)$
- `objective_matrix_low_rank_middle` ($D$, $k\times k$): Middle matrix in $R^\top D R$. 1-D array → diagonal $D$; 2-D array → dense symmetric $D$; scipy sparse → sparse $D$. May be indefinite. Defaults to identity
- `constraint_matrix` ($A$): Linear constraint matrix
- `constraint_lower_bound` ($\ell_c$): Constraint lower bounds
- `constraint_upper_bound` ($u_c$): Constraint upper bounds
- `variable_lower_bound` ($\ell_v$): Variable lower bounds (default: $-\infty$)
- `variable_upper_bound` ($u_v$): Variable upper bounds (default: $+\infty$)
- `objective_constant`: Constant term in objective

## Setting Parameters

Solver parameters can be set individually or in batch:

```python
# Set individual parameter
m.setParam("TimeLimit", 3600)

# Set multiple parameters
m.setParams(
    TimeLimit=3600,
    IterationLimit=100000,
    LogLevel=1
)

# Or use the Params view
m.Params.TimeLimit = 3600
```

## Warm Starting

Provide initial solutions to speed up convergence:

```python
# Set warm start
m.setWarmStart(primal=x0, dual=y0)

# Clear warm start
m.clearWarmStart()
```

## Accessing Results

After calling `optimize()`, results are available through properties:

```python
m.optimize()

# Solution
print(m.X)          # Primal solution
print(m.Pi)         # Dual solution

# Objective
print(m.ObjVal)     # Primal objective value
print(m.DualObj)    # Dual objective value
print(m.Gap)        # Objective gap
print(m.RelGap)     # Relative gap

# Status
print(m.Status)         # Solution status string
print(m.StatusCode)     # Solution status code
print(m.IterCount)      # Number of iterations
print(m.Runtime)        # Runtime in seconds

# Residuals
print(m.RelPrimalResidual)  # Relative primal residual
print(m.RelDualResidual)    # Relative dual residual
```
