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
./build/bin/pdhcg problem.mps ./output --time_limit 3600 --eps_opt 1e-6
```

### Command Line Options

```bash
# Silent mode, tight tolerance
./build/bin/pdhcg problem.mps ./output -v 0 --eps_opt 1e-8 --eps_feas 1e-8

# With iteration limit
./build/bin/pdhcg problem.mps ./output --iter_limit 100000

# Disable Pock-Chambolle rescaling
./build/bin/pdhcg problem.mps ./output --no_pock_chambolle
```

## Multi-GPU Distributed Examples

These examples require the solver to be built with `-DPDHCG_COMPILE_DISTRIBUTED=ON`.

### Basic Multi-GPU Run

```bash
# Run on 4 GPUs
mpirun -n 4 ./build/bin/pdhcg problem.mps ./output
```

### Custom Process Grid

By default, the solver attempts to infer a square-ish 2D process grid. You can explicitly set the grid dimensions:

```bash
# Use a 2x4 grid (8 GPUs total)
mpirun -n 8 ./build/bin/pdhcg problem.mps ./output --grid_size 2,4
```

### Partition and Permutation Options

```bash
# Uniform row partitioning with block permutation
mpirun -n 4 ./build/bin/pdhcg problem.mps ./output \
    --partition_method uniform \
    --permute_method block \
    --permute_block_size 512

# Nonzero-balanced partitioning with random permutation
mpirun -n 4 ./build/bin/pdhcg problem.mps ./output \
    --partition_method nnz \
    --permute_method random
```
