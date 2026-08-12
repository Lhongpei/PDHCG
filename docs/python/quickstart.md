# Python Quick Start

PDHCG provides a user-friendly Python interface for quadratic and quadratic conic problems using familiar NumPy and SciPy data structures.

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

Conic constraints use a columnar `ConeSpec`, whose arrays can describe many
blocks without allocating one Python object per cone. See
[model.md](model.md#cone-constraints).

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import ConeSpec, ConeType, Model

# min  z  s.t.  v = 3, w = 4, (v, w, z) in K_soc  =>  z = sqrt(v^2 + w^2) = 5
A = sp.csr_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
model = Model(
    objective_vector=np.array([0.0, 0.0, 1.0]),
    constraint_matrix=A,
    constraint_lower_bound=np.array([3.0, 4.0]),
    constraint_upper_bound=np.array([3.0, 4.0]),
    variable_cones=ConeSpec(
        ConeType.SOC,
        np.array([0], dtype=np.int32),
        v_dims=1,
    ),
)
model.optimize()
print(model.Status, model.X)
```

## CVXPY

Install the optional integration with `pip install "pdhcg[cvxpy]"`, then
import the backend once in each process:

```python
import cvxpy as cp
import pdhcg.cvxpy_backend  # Registers solver="PDHCG".

x = cp.Variable()
problem = cp.Problem(cp.Minimize(x), [x >= 1])
value = problem.solve(solver="PDHCG", eps=1e-6)

print(problem.status, value, x.value)
```

The backend preserves CVXPY's primal and dual conventions. It supports
quadratic objectives and Zero, NonNeg, SOC, PSD, ExpCone, and PowCone3D
constraints. Mixed-integer models are not supported.

## Model Creation

The `Model` class is the core interface for defining quadratic conic problems. The problem formulation is:

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2}x^\top (Q + R^\top D R) x + c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & Fx + g \in \mathcal{K}_a, \\
                  & x_J \in \mathcal{K}_v \quad \text{for variable-cone blocks } J, \\
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
- `affine_cone_matrix` ($F$): Matrix in the native affine-cone constraint $Fx + g \in \mathcal{K}_a$
- `affine_cone_offset` ($g$): Affine-cone offset; defaults to zero
- `affine_cones`: `ConeSpec` covering every row of $F$
- `variable_cones`: `ConeSpec` describing cone blocks embedded in $x$
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
