# Examples

## Python Examples

### Basic QP

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import Model

# Minimize 0.5 * x^T Q x + c^T x
# Subject to: A x <= b, x >= 0

Q = sp.csc_matrix([[2.0, 0.0], [0.0, 2.0]])
c = np.array([-2.0, -6.0])
A = sp.csc_matrix([[1.0, 1.0], [-1.0, 2.0], [2.0, 1.0]])
l = np.array([-np.inf, -np.inf, -np.inf])
u = np.array([2.0, 2.0, 3.0])
lb = np.zeros(2)

m = Model(
    objective_matrix=Q,
    objective_vector=c,
    constraint_matrix=A,
    constraint_lower_bound=l,
    constraint_upper_bound=u,
    variable_lower_bound=lb
)
m.optimize()

print(f"Solution: {m.X}")
print(f"Objective: {m.ObjVal}")
```

### Low-Rank Quadratic Term

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import Model

# Minimize 0.5 * x^T (Q + R^T R) x + c^T x
# R adds a low-rank component to the quadratic objective

n = 1000
r = 10  # rank of R

Q = sp.random(n, n, density=0.01, format='csc')
Q = Q + Q.T  # Make symmetric
R = np.random.randn(r, n)  # Low-rank matrix
c = np.random.randn(n)

m = Model(
    objective_matrix=Q,
    objective_matrix_low_rank=R,
    objective_vector=c
)
m.optimize()
```

### Low-Rank with a Middle Matrix D

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import Model

# Minimize 0.5 * x^T (Q + R^T D R) x + c^T x
# D may be diagonal (1-D), dense (2-D), or scipy.sparse;
# may also be indefinite. Defaults to identity if omitted.

n = 1000
r = 10  # rank
Q = None
R = np.random.randn(r, n)
c = np.random.randn(n)

# (a) Weighted least-squares-style: D = diag(w)
D_diag = np.array([0.5, 1.0, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
m = Model(
    objective_matrix=Q,
    objective_matrix_low_rank=R,
    objective_matrix_low_rank_middle=D_diag,
    objective_vector=c,
)
m.optimize()

# (b) Dense (possibly indefinite) D, e.g., from a quasi-Newton compact form
M = np.random.randn(r, r)
D_dense = 0.5 * (M + M.T)  # symmetric, no PSD requirement
m = Model(
    objective_matrix=Q,
    objective_matrix_low_rank=R,
    objective_matrix_low_rank_middle=D_dense,
    objective_vector=c,
)
m.optimize()
```

### Warm Starting

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import Model

# Solve once
Q = sp.csc_matrix([[2.0, 0.0], [0.0, 2.0]])
c = np.array([-2.0, -6.0])
m = Model(objective_matrix=Q, objective_vector=c)
m.optimize()

# Use solution as warm start for slightly modified problem
c_new = np.array([-2.5, -6.5])
m.setObjectiveVector(c_new)
m.setWarmStart(primal=m.X, dual=m.Pi)
m.optimize()
```

## C++ Examples

### Reading from MPS File

```bash
./build/pdhcg problem.mps ./output --time_limit 3600 --eps_opt 1e-6
```

### Command Line Options

```bash
# Silent mode, tight tolerance
./build/pdhcg problem.mps ./output -v 0 --eps_opt 1e-8 --eps_feas 1e-8

# With iteration limit
./build/pdhcg problem.mps ./output --iter_limit 100000

# Disable Pock-Chambolle rescaling
./build/pdhcg problem.mps ./output --no_pock_chambolle
```

## Multi-GPU Distributed Examples

These examples require the solver to be built with `-DPDHCG_COMPILE_DISTRIBUTED=ON`.

### Basic Multi-GPU Run

```bash
# Run on 4 GPUs
mpirun -n 4 ./build/pdhcg problem.mps ./output
```

### Custom Process Grid

By default, the solver attempts to infer a square-ish 2D process grid. You can explicitly set the grid dimensions:

```bash
# Use a 2x4 grid (8 GPUs total)
mpirun -n 8 ./build/pdhcg problem.mps ./output --grid_size 2,4
```

### Partition and Permutation Options

```bash
# Uniform row partitioning with block permutation
mpirun -n 4 ./build/pdhcg problem.mps ./output \
    --partition_method uniform \
    --permute_method block \
    --permute_block_size 512

# Nonzero-balanced partitioning with random permutation
mpirun -n 4 ./build/pdhcg problem.mps ./output \
    --partition_method nnz \
    --permute_method random
```

## Conic Examples

The conic interface accepts a columnar `ConeSpec` through `Model` or the
lower-level `solve_once` entry. See the [Python model API](python/model.md#cone-constraints)
for its fields and the [C API](c/functions.md) for cone layouts.

### Standard SOC

Minimise `z` over `(v, w, z) in K_soc` with `v = 3`, `w = 4`. Optimum recovers
`(3, 4, 5)` (the Euclidean norm `||(3,4)||_2`).

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import ConeSpec, ConeType
from pdhcg._core import solve_once

A = sp.csr_matrix([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])
c = np.array([0.0, 0.0, 1.0])
con_lb = np.array([3.0, 4.0]); con_ub = con_lb.copy()
INF = np.full(3, 1e30)

info = solve_once(
    None, None, A, c, 0.0,
    -INF, INF, con_lb, con_ub,
    cones=ConeSpec(ConeType.SOC, np.array([0], dtype=np.int32)),
)
print(info["X"])  # [3., 4., 5.]
```

Expected output:

```
Status: OPTIMAL
X: [3.0  4.0  5.0]
PrimalObj: 5.0
```

### SPARSE_Q coupled to SOC

Off-diagonal `Q` on the non-cone block `(a, b)` linearly coupled into a SOC
cone `(v, w, z)`. Variables are `(a, b, v, w, z)`; constraints pin `a = v`,
`b = w`; `Q = [[1, 0.5], [0.5, 1]]` on the `(a,b)` block. Optimum:
`a = 3, b = 4, v = 3, w = 4, z = 5`.

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import ConeSpec, ConeType
from pdhcg._core import solve_once

# (a, b, v, w, z)  with a - v = 0, b - w = 0
A = sp.csr_matrix([[1, 0, -1,  0, 0],
                   [0, 1,  0, -1, 0]], dtype=float)
Q = sp.csr_matrix([[1.0, 0.5, 0, 0, 0],
                   [0.5, 1.0, 0, 0, 0],
                   [0,   0,   0, 0, 0],
                   [0,   0,   0, 0, 0],
                   [0,   0,   0, 0, 0]])
c = np.array([-3.0, -4.0, 0.0, 0.0, 1.0])
con_lb = np.zeros(2); con_ub = np.zeros(2)
INF = np.full(5, 1e30)

info = solve_once(
    Q, None, A, c, 0.0,
    -INF, INF, con_lb, con_ub,
    cones=ConeSpec(ConeType.SOC, np.array([2], dtype=np.int32)),
)
print(info["X"])         # [3, 4, 3, 4, 5]
print(info["PrimalObj"]) # -1.5
```

Expected output:

```
Status: OPTIMAL
X: [3.0 4.0 3.0 4.0 5.0]
PrimalObj: -1.5
```

### Exponential cone with fixed `y` (Fisher-style)

Quasi-linear Fisher market, 2 buyers `×` 3 goods, hand-set utilities `u_ij`
and budgets `w_i`. Variables: `x_ij` (allocations), `v_i` (slack), and per
buyer an exp triple `(z_i, y_i, t_i)` with `y_i` pinned to 1. Cone
feasibility at the solution: `y_i * exp(z_i / y_i) <= t_i`.

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import ConeSpec, ConeType
from pdhcg._core import solve_once

u = np.array([[0.5, 1.0, 0.2], [0.3, 0.7, 1.0]])   # 2 buyers, 3 goods
w = np.array([1.0, 1.5])                            # budgets
b = np.array([1.0, 1.0, 1.0])                       # supplies
n, m = u.shape
nx, N = n*m, n*m + n + 3*n                          # x | v | (z,y,t)
v0, c0 = nx, nx + n

# Build A row by row: m supply rows, then n budget rows.
rows = sp.lil_matrix((m + n, N))
for j in range(m):
    rows[j, [i*m + j for i in range(n)]] = 1.0      # sum_i x_ij = b_j
for i in range(n):
    rows[m+i, i*m:(i+1)*m] = -u[i]                  # -u_i^T x_i - v_i + t_i = 0
    rows[m+i, v0+i] = -1.0
    rows[m+i, c0+3*i+2] = 1.0
A = rows.tocsr()

c = np.zeros(N)
for i in range(n):
    c[v0+i] = 1.0
    c[c0+3*i] = -w[i]                               # min sum v_i - w_i z_i
lb = np.full(N, -1e30); ub = np.full(N, 1e30)
lb[:nx] = 0.0; lb[v0:v0+n] = 0.0
con_b = np.concatenate([b, np.zeros(n)])

primal_start = np.zeros(N)
primal_start[c0+1::3] = 1.0                         # pin y_i = 1
fixed_mask = np.zeros(N, dtype=np.uint8)
fixed_mask[c0+1::3] = 1
cones = ConeSpec(
    ConeType.EXP,
    c0 + 3 * np.arange(n, dtype=np.int32),
    fixed_mask=fixed_mask,
)

info = solve_once(None, None, A, c, 0.0, lb, ub, con_b, con_b.copy(),
                  primal_start=primal_start, cones=cones)
x = info["X"]
for i in range(n):
    z, y, t = x[c0+3*i:c0+3*i+3]
    print(f"buyer {i}: y={y:.4f}  exp(z/y)={np.exp(z/y):.4f}  t={t:.4f}")
```

Expected output:

```
Status: OPTIMAL
buyer 0: y=1.0000  exp(z/y)=...  t=...   # y*exp(z/y) <= t
buyer 1: y=1.0000  exp(z/y)=...  t=...
```
