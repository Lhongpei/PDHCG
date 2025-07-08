
# PDHCG-Python
The `PDHCG-Python` project provides a Python interface for PDHCG algorithm suggested in [Restarted Primal-Dual Hybrid Conjugate Gradient Method for Large-Scale Quadratic Programming](https://arxiv.org/abs/2405.16160).
y
Core functions is implemented by Julia, see [PDHCG.jl](https://github.com/Huangyc98/PDHCG.jl).

For datasets mentioned in the paper, please refer to [Datasets](https://github.com/Lhongpei/QP_datasets)
## Installation

**Prerequsite:** Install [Julia](https://julialang.org/downloads/).

Install through PyPI

```bash
pip install PDHCG
```
If you cannot install through PyPI, please clone this repo to use PDHCG.

## Usage

### 1. Import and Initialize

```python
from PDHCG import PDHCG  
solver = PDHCG(name="My QP Solver")
```

### 2. Setting Parameters

Set solver parameters either by individual `setParam` or batch `setParams` methods.

```python
solver.setParam("time_limit", 600)
solver.setParams(gpu_flag=True, verbose_level=2)
```


### 3. Defining the Problem

You can define a problem in three ways:

- **Read from File**:

  ```python
  solver.read("path/to/problem.qps", fixformat=False)
  ```

- **Generate Random Problem**:

  ```python
  solver.setGeneratedProblem(problem_type="QP", n=100, density=0.05, seed=42)
  ```

- **Construct from Scratch**:

  ```python
  from scipy.sparse import random

  # Define the problem components as numpy arrays or sparse matrices
  objective_matrix = random(100, 100, density=0.05).toarray()
  objective_vector = np.random.randn(100)
  constraint_matrix = random(50, 100, density=0.1).toarray()
  constraint_lower_bound = np.random.randn(50)
  
  solver.setConstructedProblem(
      objective_matrix=objective_matrix,
      objective_vector=objective_vector,
      objective_constant=0.0,
      constraint_matrix=constraint_matrix,
      constraint_lower_bound=constraint_lower_bound,
      num_equalities=10,
      variable_lower_bound=np.zeros(100),
      variable_upper_bound=np.ones(100) * 10,
      isfinite_variable_lower_bound=np.full(100, True),
      isfinite_variable_upper_bound=np.full(100, True)
  )
  ```

  Notice that all matrix will be strictly converted to sparse csc matrix.

### 4. Solving the Problem

```python
solver.solve()
```

### 5. Accessing Results

After solving, retrieve various results as properties:

- **`solver.primal_solution`**: Returns the primal solution after solving.
- **`solver.dual_solution`**: Returns the dual solution after solving.
- **`solver.objective_value`**: Returns the objective function's value.
- **`solver.iteration_count`**: Returns a dictionary with outer and inner iteration counts.
- **`solver.solve_time_sec`**: Solving time in seconds.
- **`solver.kkt_error`**: The KKT error for per iterations.
- **`solver.status`**: Termination status string.

## Interpreting the output

A table of iteration stats will be printed with the following headings.

### runtime

- `#iter`: the current iteration number.
- `#kkt`: the cumulative number of times the KKT matrix is multiplied.
- `seconds`: the cumulative solve time in seconds.

### residuals

- `pr norm`: the Euclidean norm of primal residuals (i.e., the constraint violation).
- `du norm`: the Euclidean norm of the dual residuals.
- `gap`: the gap between the primal and dual objective.

### solution information

- `pr obj`: the primal objective value.
- `pr norm`: the Euclidean norm of the primal variable vector.
- `du norm`: the Euclidean norm of the dual variable vector.

### relative residuals

- `rel pr`: the Euclidean norm of the primal residuals, relative to the right-hand side.
- `rel dul`: the Euclidean norm of the dual residuals, relative to the primal linear objective.
- `rel gap`: the relative optimality gap.
  
### At the end of the run, the following summary information will be printed

- Total Iterations: The total number of Primal-Dual iterations.

- CG  iteration: The total number of Conjugate Gradient iterations.

- Solving Status: Indicating if it found an optimal solution.

## Parameters

For more details of `Rescale` and `Restart` Parameters, please refer to [Restarted Primal-Dual Hybrid Conjugate Gradient Method for Large-Scale Quadratic Programming](https://arxiv.org/abs/2405.16160).

| Category   | Parameter                       | Default Value | Description                                                                                  |
|------------|---------------------------------|---------------|----------------------------------------------------------------------------------------------|
| **Basic**  | `gpu_flag`                      | `False`       | Whether to use the GPU for computations.                                                    |
|            | `warm_up_flag`                  | `False`       | If `True`, excludes compilation time from runtime measurements.                             |
|            | `verbose_level`                 | `2`           | Verbosity level (0-9). Higher values produce more detailed output.                          |
|            | `time_limit`                    | `3600.0`      | Maximum time limit for solving in seconds.                                                  |
|            | `relat_error_tolerance`         | `1e-6`        | Relative error tolerance for solution accuracy.                                             |
|            | `iteration_limit`               | `2**31 - 1`   | Maximum number of iterations allowed.                                                       |
| **Rescale**| `ruiz_rescaling_iters`          | `10`          | Number of iterations for Ruiz rescaling.                                                    |
|            | `l2_norm_rescaling_flag`        | `False`       | Enables L2 norm rescaling if `True`.                                                        |
|            | `pock_chambolle_alpha`          | `1.0`         | Alpha parameter for the Pock-Chambolle algorithm.                                           |
| **Restart**| `artificial_restart_threshold`  | `0.2`         | Threshold for artificial restart criteria.                                                  |
|            | `sufficient_reduction`          | `0.2`         | Minimum reduction required to consider a restart as sufficient.                             |
|            | `necessary_reduction`           | `0.8`         | Reduction required for restart necessity.                                                  |
|            | `primal_weight_update_smoothing`| `0.2`         | Smoothing parameter for primal-dual weight updates.                                         |
| **Log**    | `save_flag`                     | `False`       | If `True`, saves the solver's results to a file.                                           |
|            | `saved_name`                    | `None`        | Filename for saving the result (if `save_flag` is `True`).                                  |
|            | `output_dir`                    | `None`        | Directory path for saving results.                                                          |

## Example

Below is a complete example:

```python
import numpy as np
from PDHCG import PDHCG

solver = PDHCG(name="Example QP Solver")

# Set parameters
solver.setParams(gpu_flag=False, verbose_level=1)

# Define a custom problem
objective_matrix = np.eye(10)
objective_vector = np.ones(10)
constraint_matrix = np.random.randn(5, 10)
constraint_lower_bound = np.zeros(5)

solver.setConstructedProblem(
    objective_matrix=objective_matrix,
    objective_vector=objective_vector,
    objective_constant=0.0,
    constraint_matrix=constraint_matrix,
    constraint_lower_bound=constraint_lower_bound,
    num_equalities=2,
    variable_lower_bound=np.zeros(10),
    variable_upper_bound=np.ones(10) * 5,
    isfinite_variable_lower_bound=np.full(10, True),
    isfinite_variable_upper_bound=np.full(10, True)
)

# Solve the problem
solver.solve()

# Output results
print("Objective Value:", solver.objective_value)
print("Primal Solution:", solver.primal_solution)
print("Dual Solution:", solver.dual_solution)
```
