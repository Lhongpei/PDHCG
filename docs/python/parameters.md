# Solver Parameters

PDHCG provides various parameters to control the solver behavior.

## Setting Parameters

Parameters can be set using `setParam()` or `setParams()`:

```python
m.setParam("TimeLimit", 3600)
m.setParams(TimeLimit=3600, LogLevel=1)
```

## Parameter Reference

### Termination Criteria

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `TimeLimit` | float | 3600.0 | Time limit in seconds |
| `IterationLimit` | int | 2147483647 | Maximum number of iterations |
| `OptTol` | float | 1e-4 | Relative optimality tolerance |
| `FeasTol` | float | 1e-4 | Relative feasibility tolerance |
| `InfeasTol` | float | 1e-10 | Infeasibility detection tolerance |

### Algorithm Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `RuizIterations` | int | 10 | Iterations for L-inf Ruiz rescaling |
| `PockChambolleAlpha` | float | 1.0 | Pock-Chambolle step size parameter |
| `UsePockChambolle` | bool | True | Enable Pock-Chambolle rescaling |
| `UseBoundObjectiveRescaling` | bool | True | Enable bound objective rescaling |
| `EvalFrequency` | int | 200 | Frequency of termination criteria evaluation |

### Inner Solver Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `InnerIterLimit` | int | 1000 | Max iterations for inner CG solver |
| `InnerInitTol` | float | 1e-3 | Initial tolerance for inner solver |
| `InnerMinTol` | float | 1e-9 | Minimum tolerance for inner solver |
| `DiagJacobiPrecond` | bool | True | Use the Jacobi diagonal preconditioner in the inner subproblem. Set to `False` to disable. |

### Singular Value Estimation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SVMaxIter` | int | 5000 | Max iterations for singular value estimation |
| `SVTol` | float | 1e-4 | Tolerance for singular value estimation |

### Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `LogLevel` | int | 1 | Verbosity: 0=Silent, 1=Summary, 2=Detailed |

### Norm Type

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `OptNorm` | str | "linf" | Norm for optimality: "l2" or "linf" |

!!! note "Cone constraints"
    Cone specs are not solver parameters; they are part of the problem and are
    passed via the `cones=` kwarg of `solve_once`. See [model.md](model.md#cone-constraints).
