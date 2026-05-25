# C API Types

## Termination Reason

```c
typedef enum {
  TERMINATION_REASON_UNSPECIFIED,
  TERMINATION_REASON_OPTIMAL,
  TERMINATION_REASON_PRIMAL_INFEASIBLE,
  TERMINATION_REASON_DUAL_INFEASIBLE,
  TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED,
  TERMINATION_REASON_TIME_LIMIT,
  TERMINATION_REASON_ITERATION_LIMIT,
  TERMINATION_REASON_USER_INTERRUPT,
  TERMINATION_REASON_FEAS_POLISH_SUCCESS
} termination_reason_t;
```

| Value | Description |
|-------|-------------|
| `TERMINATION_REASON_UNSPECIFIED` | Unknown/unspecified status |
| `TERMINATION_REASON_OPTIMAL` | Optimal solution found |
| `TERMINATION_REASON_PRIMAL_INFEASIBLE` | Problem is primal infeasible |
| `TERMINATION_REASON_DUAL_INFEASIBLE` | Problem is dual infeasible |
| `TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED` | Problem is infeasible or unbounded |
| `TERMINATION_REASON_TIME_LIMIT` | Time limit reached |
| `TERMINATION_REASON_ITERATION_LIMIT` | Iteration limit reached |
| `TERMINATION_REASON_USER_INTERRUPT` | Solver interrupted by user signal |
| `TERMINATION_REASON_FEAS_POLISH_SUCCESS` | Feasibility polishing succeeded |

## Norm Type

```c
typedef enum {
  NORM_TYPE_L2 = 0,
  NORM_TYPE_L_INF = 1
} norm_type_t;
```

## Matrix Format

```c
typedef enum {
  matrix_dense = 0,
  matrix_csr = 1,
  matrix_csc = 2,
  matrix_coo = 3
} matrix_format_t;
```

## CSR Component

```c
typedef struct {
  int *row_ptr;
  int *col_ind;
  double *val;
} CsrComponent;
```

## Matrix Descriptor

```c
typedef struct {
  int m; // num_constraints
  int n; // num_variables
  matrix_format_t fmt;
  double zero_tolerance;  // treat abs(x) < zero_tolerance as zero

  union MatrixData {
    struct MatrixDense {    // Dense (row-major)
      const double *A;      // m*n
    } dense;

    struct MatrixCSR {      // CSR
      int nnz;
      const int *row_ptr;
      const int *col_ind;
      const double *vals;
    } csr;

    struct MatrixCSC {      // CSC
      int nnz;
      const int *col_ptr;
      const int *row_ind;
      const double *vals;
    } csc;

    struct MatrixCOO {      // COO
      int nnz;
      const int *row_ind;
      const int *col_ind;
      const double *vals;
    } coo;
  } data;
} matrix_desc_t;
```

## Quadratic Objective Type

```c
typedef enum {
  PDHCG_SPARSE_Q,
  PDHCG_DIAG_Q,
  PDHCG_LOW_RANK_PLUS_SPARSE_Q,
  PDHCG_LOW_RANK_Q,
  PDHCG_NON_Q
} quad_obj_type_t;
```

## QP Problem

```c
typedef struct {
  int num_variables;
  int num_constraints;
  int num_rank_lowrank_obj;
  double *variable_lower_bound;
  double *variable_upper_bound;
  double *objective_vector;
  double objective_constant;

  CsrComponent *constraint_matrix;
  int constraint_matrix_num_nonzeros;

  CsrComponent *objective_sparse_matrix;
  int objective_sparse_matrix_num_nonzeros;

  CsrComponent *objective_lowrank_matrix;
  int objective_lowrank_matrix_num_nonzeros;

  CsrComponent *objective_lowrank_middle_matrix;
  int objective_lowrank_middle_matrix_num_nonzeros;

  double *constraint_lower_bound;
  double *constraint_upper_bound;

  double *primal_start;
  double *dual_start;
} qp_problem_t;
```

## Restart Parameters

```c
typedef struct {
  double artificial_restart_threshold;
  double sufficient_reduction_for_restart;
  double necessary_reduction_for_restart;
  double k_p;
  double k_i;
  double k_d;
  double i_smooth;
} restart_parameters_t;
```

## Termination Criteria

```c
typedef struct {
  double eps_optimal_relative;
  double eps_feasible_relative;
  double eps_feas_polish_relative;
  double eps_infeasible;
  double time_sec_limit;
  int iteration_limit;
} termination_criteria_t;
```

## Inner Solver Parameters

```c
typedef struct {
  int iteration_limit;
  double initial_tolerance;
  double min_tolerance;
} inner_solver_parameters_t;
```

## Partition Method

```c
typedef enum {
  UNIFORM_PARTITION,
  NNZ_BALANCE_PARTITION,
} partition_method_t;
```

| Value | Description |
|-------|-------------|
| `UNIFORM_PARTITION` | Uniform row partitioning across the process grid |
| `NNZ_BALANCE_PARTITION` | Nonzero-balanced partitioning (default) |

## Permute Method

```c
typedef enum {
  NO_PERMUTATION,
  FULL_RANDOM_PERMUTATION,
  BLOCK_RANDOM_PERMUTATION,
} permute_method_t;
```

| Value | Description |
|-------|-------------|
| `NO_PERMUTATION` | No permutation applied |
| `FULL_RANDOM_PERMUTATION` | Full random permutation |
| `BLOCK_RANDOM_PERMUTATION` | Block-wise random permutation (default) |

## Grid Size

```c
typedef struct {
  int row_dims;
  int col_dims;
  bool decided;
} grid_size_t;
```

Describes the 2D process grid for distributed solving. If `decided` is `false`, the solver attempts to infer a suitable grid automatically.

## PDHG Parameters

```c
typedef struct {
  int l_inf_ruiz_iterations;
  bool has_pock_chambolle_alpha;
  double pock_chambolle_alpha;
  bool bound_objective_rescaling;
  int verbose;
  int termination_evaluation_frequency;
  int sv_max_iter;
  double sv_tol;
  termination_criteria_t termination_criteria;
  restart_parameters_t restart_params;
  double reflection_coefficient;
  bool feasibility_polishing;
  norm_type_t optimality_norm;
  inner_solver_parameters_t inner_solver_parameters;
  bool presolve;
  bool diag_jacobi_precond;
  partition_method_t partition_method;
  permute_method_t permute_method;
  grid_size_t grid_size;
  int permute_block_size;
} pdhg_parameters_t;
```

## PDHCG Result

```c
typedef struct {
  int num_variables;
  int num_constraints;
  int num_nonzeros;

  int num_reduced_variables;
  int num_reduced_constraints;
  int num_reduced_nonzeros;

  double *primal_solution;
  double *dual_solution;
  double *reduced_cost;

  int total_count;
  int total_inner_count;
  double rescaling_time_sec;
  double cumulative_time_sec;

  double absolute_primal_residual;
  double relative_primal_residual;
  double absolute_dual_residual;
  double relative_dual_residual;
  double primal_objective_value;
  double dual_objective_value;
  double objective_gap;
  double relative_objective_gap;
  double max_primal_ray_infeasibility;
  double max_dual_ray_infeasibility;
  double primal_ray_linear_objective;
  double dual_ray_objective;
  termination_reason_t termination_reason;
  double feasibility_polishing_time;
  int feasibility_iteration;

  // Presolve information
  double presolve_time;
  int presolve_status;
} pdhcg_result_t;
```
