/*
Copyright 2025 Haihao Lu
Copyright 2026 Hongpei Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "cbf_parser.h"
#include "mps_parser.h"
#include "pdhcg.h"
#include "presolve_wrapper.h"
#include "solver.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <getopt.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PDHCG_COMPILE_DISTRIBUTED
#include "distributed_solver.h"
#include <mpi.h>
#endif

char *get_output_path(const char *output_dir, const char *instance_name, const char *suffix)
{
    size_t path_len = strlen(output_dir) + strlen(instance_name) + strlen(suffix) + 2;
    char *full_path = safe_malloc(path_len * sizeof(char));
    snprintf(full_path, path_len, "%s/%s%s", output_dir, instance_name, suffix);
    return full_path;
}

/* Dispatch on file extension: .cbf(.gz) -> CBF parser, else MPS parser. */
static qp_problem_t *read_problem_file(const char *filename)
{
    size_t n = strlen(filename);
    const char *tail = filename;
    if (n > 3 && strcmp(filename + n - 3, ".gz") == 0)
        n -= 3;
    if (n > 4 && strncmp(tail + n - 4, ".cbf", 4) == 0)
        return read_cbf_file(filename);
    return read_mps_file(filename);
}

char *extract_instance_name(const char *filename)
{
    char *filename_copy = strdup(filename);
    if (filename_copy == NULL)
    {
        perror("Memory allocation failed");
        return NULL;
    }

    char *base = basename(filename_copy);
    char *dot = strchr(base, '.');
    if (dot)
    {
        *dot = '\0';
    }

    char *instance_name = strdup(base);
    free(filename_copy);
    return instance_name;
}

void save_solution(const double *data, int size, const char *output_dir, const char *instance_name, const char *suffix)
{
    char *file_path = get_output_path(output_dir, instance_name, suffix);
    if (file_path == NULL || data == NULL)
    {
        return;
    }

    FILE *outfile = fopen(file_path, "w");
    if (outfile == NULL)
    {
        perror("Error opening solution file");
        free(file_path);
        return;
    }

    for (int i = 0; i < size; ++i)
    {
        fprintf(outfile, "%.10g\n", data[i]);
    }

    fclose(outfile);
    free(file_path);
}

void save_solver_summary(const pdhcg_result_t *result, const char *output_dir, const char *instance_name)
{
    char *file_path = get_output_path(output_dir, instance_name, "_summary.txt");
    if (file_path == NULL)
    {
        return;
    }

    FILE *outfile = fopen(file_path, "w");
    if (outfile == NULL)
    {
        perror("Error opening summary file");
        free(file_path);
        return;
    }
    fprintf(outfile, "Termination Reason: %s\n", termination_reason_to_string(result->termination_reason));
    fprintf(outfile, "Runtime (sec): %e\n", result->cumulative_time_sec);
    fprintf(outfile, "Iterations Count: %d\n", result->total_count);
    if (result->total_inner_count > 0)
    {
        fprintf(outfile, "Inner Iterations Count: %d\n", result->total_inner_count);
    }
    fprintf(outfile, "Primal Objective Value: %e\n", result->primal_objective_value);
    fprintf(outfile, "Dual Objective Value: %e\n", result->dual_objective_value);
    fprintf(outfile, "Absolute Primal Residual: %e\n", result->absolute_primal_residual);
    fprintf(outfile, "Absolute Dual Residual: %e\n", result->absolute_dual_residual);
    fprintf(outfile, "Relative Primal Residual: %e\n", result->relative_primal_residual);
    fprintf(outfile, "Relative Dual Residual: %e\n", result->relative_dual_residual);
    fprintf(outfile, "Absolute Objective Gap: %e\n", result->objective_gap);
    fprintf(outfile, "Relative Objective Gap: %e\n", result->relative_objective_gap);
    fprintf(outfile, "Rows: %d\n", result->num_constraints);
    fprintf(outfile, "Columns: %d\n", result->num_variables);
    fprintf(outfile, "Nonzeros: %d\n", result->num_nonzeros);

    if (result->presolve_time > 0.0)
    {
        fprintf(outfile, "Presolve Status: %s\n", pdhcg_get_presolve_status_str(result->presolve_status));
        fprintf(outfile, "Presolve Time (sec): %e\n", result->presolve_time);
        fprintf(outfile, "Reduced Rows: %d\n", result->num_reduced_constraints);
        fprintf(outfile, "Reduced Columns: %d\n", result->num_reduced_variables);
        fprintf(outfile, "Reduced Nonzeros: %d\n", result->num_reduced_nonzeros);
    }

    if (result->feasibility_polishing_time > 0.0)
    {
        fprintf(outfile, "Feasibility Polishing Time (sec): %e\n", result->feasibility_polishing_time);
        fprintf(outfile, "Feasibility Polishing Iteration Count: %d\n", result->feasibility_iteration);
    }
    fclose(outfile);
    free(file_path);
}

void print_usage(const char *prog_name)
{
    fprintf(stderr, "Usage: %s [OPTIONS] <problem_file> <output_dir>\n\n", prog_name);

    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "  <problem_file>           Input problem: .mps, .qps, .cbf, and gzip-compressed variants.\n");
    fprintf(stderr, "  <output_dir>             Directory where output files will be saved. It will contain:\n");
    fprintf(stderr, "                             - <basename>_summary.txt\n");
    fprintf(stderr, "                             - <basename>_primal_solution.txt\n");
    fprintf(stderr, "                             - <basename>_dual_solution.txt\n\n");

    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help               Display this help message.\n");
    fprintf(stderr, "  -v, --verbose <int>      Verbosity level: 0=silent, 1=summary, 2=detailed (default: 1).\n");
    fprintf(stderr, "      --time_limit <sec>   Time limit in seconds (default: 3600.0).\n");
    fprintf(stderr, "      --iter_limit <int>   Iteration limit (default: %d).\n", INT32_MAX);
    fprintf(stderr, "      --eps_opt <float>    Relative optimality tolerance (default: 1e-4).\n");
    fprintf(stderr, "      --eps_feas <float>   Relative feasibility tolerance (default: 1e-4).\n");
    fprintf(stderr, "      --eps_infeas_detect  Infeasibility detection tolerance (default: 1e-12).\n");
    fprintf(stderr, "      --curtis_reid_iter   Iterations for Curtis-Reid scaling (default: 0, disabled).\n");
    fprintf(stderr, "      --l_inf_ruiz_iter    Iterations for L-inf Ruiz rescaling (default: 10).\n");
    fprintf(stderr, "      --no_pock_chambolle  Disable Pock-Chambolle rescaling (default: enabled).\n");
    fprintf(stderr, "      --pock_chambolle_alpha Value for Pock-Chambolle alpha (default: 1.0).\n");
    fprintf(stderr, "      --no_bound_obj_rescaling Disable bound objective rescaling.\n");
    fprintf(stderr, "      --no_cone_preserving_scaling Disable cone-preserving scaling.\n");
    fprintf(stderr, "      --eval_freq <int>    Termination evaluation frequency (default: 200).\n");
    fprintf(stderr, "      --artificial_restart_threshold Artificial restart threshold (default: 0.36).\n");
    fprintf(stderr,
            "      --sufficient_reduction_for_restart Sufficient reduction factor for restart (default: 0.2).\n");
    fprintf(stderr, "      --necessary_reduction_for_restart Necessary reduction factor for restart (default: 0.8).\n");
    fprintf(stderr, "      --sv_max_iter <int>  Max iterations for singular value estimation (default: 5000).\n");
    fprintf(stderr, "      --sv_tol <float>     Tolerance for singular value estimation (default: 1e-4).\n");
    fprintf(stderr, "      --opt_norm <type>    Norm for optimality criteria: l2 or linf (default: linf).\n");
    fprintf(stderr, "      --inner_iter_limit   Max iterations for the inner solver (default: 1000).\n");
    fprintf(stderr, "      --inner_init_tol     Initial tolerance for the inner solver (default: 1e-3).\n");
    fprintf(stderr, "      --inner_min_tol      Minimum tolerance for the inner solver (default: 1e-9).\n");
    fprintf(
        stderr,
        "      --no_diag_precond    Disable Jacobi diagonal preconditioner for inner subproblem (default: enabled).\n");
    fprintf(stderr, "      --soc_form <form>    QCQP cone formulation: 'rotated' or 'standard' (default: rotated).\n");

#ifdef PDHCG_COMPILE_DISTRIBUTED
    fprintf(stderr, "\nDistributed Options (MPI & NCCL):\n");
    fprintf(stderr, "      --grid_size r,c      Explicitly set the 2D grid dimensions (e.g., 2,4).\n");
    fprintf(stderr, "      --partition_method   'uniform' or 'nnz' (default: nnz).\n");
    fprintf(stderr, "      --permute_method     'none', 'random', or 'block' (default: block).\n");
    fprintf(stderr, "      --permute_block_size Block size for block permutation (default: 256).\n");
#endif
}

int run_pdhcg(int argc, char *argv[])
{
    cudaFree(0);
    pdhg_parameters_t params;
    set_default_parameters(&params);

    static struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                           {"verbose", required_argument, 0, 'v'},
                                           {"time_limit", required_argument, 0, 1001},
                                           {"iter_limit", required_argument, 0, 1002},
                                           {"eps_opt", required_argument, 0, 1003},
                                           {"eps_feas", required_argument, 0, 1004},
                                           {"eps_infeas_detect", required_argument, 0, 1005},
                                           {"eps_feas_polish", required_argument, 0, 1006},
                                           {"feasibility_polishing", no_argument, 0, 'f'},
                                           {"l_inf_ruiz_iter", required_argument, 0, 1007},
                                           {"pock_chambolle_alpha", required_argument, 0, 1008},
                                           {"no_pock_chambolle", no_argument, 0, 1009},
                                           {"no_bound_obj_rescaling", no_argument, 0, 1010},
                                           {"sv_max_iter", required_argument, 0, 1011},
                                           {"sv_tol", required_argument, 0, 1012},
                                           {"eval_freq", required_argument, 0, 1013},
                                           {"opt_norm", required_argument, 0, 1014},
                                           {"inner_iter_limit", required_argument, 0, 1015},
                                           {"inner_init_tol", required_argument, 0, 1016},
                                           {"inner_min_tol", required_argument, 0, 1017},
                                           {"presolve", required_argument, 0, 1018},
                                           {"no_diag_precond", no_argument, 0, 1019},
                                           {"soc_form", required_argument, 0, 1020},
                                           {"no_cone_preserving_scaling", no_argument, 0, 1021},
                                           {"artificial_restart_threshold", required_argument, 0, 1022},
                                           {"sufficient_reduction_for_restart", required_argument, 0, 1023},
                                           {"necessary_reduction_for_restart", required_argument, 0, 1024},
                                           {"curtis_reid_iter", required_argument, 0, 1025},
                                           {0, 0, 0, 0}};

    int opt;
    optind = 1;
    while ((opt = getopt_long(argc, argv, "hv:fp", long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'v':
                params.verbose = atoi(optarg);
                break;
            case 1001:
                params.termination_criteria.time_sec_limit = atof(optarg);
                break;
            case 1002:
                params.termination_criteria.iteration_limit = atoi(optarg);
                break;
            case 1003:
                params.termination_criteria.eps_optimal_relative = atof(optarg);
                break;
            case 1004:
                params.termination_criteria.eps_feasible_relative = atof(optarg);
                break;
            case 1005:
                params.termination_criteria.eps_infeasible = atof(optarg);
                break;
            case 1006:
                params.termination_criteria.eps_feas_polish_relative = atof(optarg);
                break;
            case 'f':
                params.feasibility_polishing = true;
                break;
            case 1007:
                params.l_inf_ruiz_iterations = atoi(optarg);
                break;
            case 1008:
                params.pock_chambolle_alpha = atof(optarg);
                break;
            case 1009:
                params.has_pock_chambolle_alpha = false;
                break;
            case 1010:
                params.bound_objective_rescaling = false;
                break;
            case 1011:
                params.sv_max_iter = atoi(optarg);
                break;
            case 1012:
                params.sv_tol = atof(optarg);
                break;
            case 1013:
                params.termination_evaluation_frequency = atoi(optarg);
                break;
            case 1014:
                if (strcmp(optarg, "l2") == 0)
                    params.optimality_norm = NORM_TYPE_L2;
                else if (strcmp(optarg, "linf") == 0)
                    params.optimality_norm = NORM_TYPE_L_INF;
                else
                {
                    fprintf(stderr, "Error: opt_norm must be 'l2' or 'linf'\n");
                    return 1;
                }
                break;
            case 1015:
                params.inner_solver_parameters.iteration_limit = atoi(optarg);
                break;
            case 1016:
                params.inner_solver_parameters.initial_tolerance = atof(optarg);
                break;
            case 1017:
                params.inner_solver_parameters.min_tolerance = atof(optarg);
                break;
            case 1018:
                params.presolve = (atoi(optarg) != 0);
                break;
            case 1019:
                params.diag_jacobi_precond = false;
                break;
            case 1020:
                if (strcmp(optarg, "rotated") == 0)
                    params.default_cone_type = CONE_ROTATED_SOC;
                else if (strcmp(optarg, "standard") == 0)
                    params.default_cone_type = CONE_STANDARD_SOC;
                else
                {
                    fprintf(stderr, "Error: soc_form must be 'rotated' or 'standard'\n");
                    return 1;
                }
                break;
            case 1021:
                params.use_cone_preserving_scaling = false;
                break;
            case 1022:
                params.restart_params.artificial_restart_threshold = atof(optarg);
                break;
            case 1023:
                params.restart_params.sufficient_reduction_for_restart = atof(optarg);
                break;
            case 1024:
                params.restart_params.necessary_reduction_for_restart = atof(optarg);
                break;
            case 1025:
                params.curtis_reid_iterations = atoi(optarg);
                break;
            case '?':
                return 1;
        }
    }

    if (argc - optind != 2)
    {
        fprintf(stderr, "Error: You must specify an input file and an output directory.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    const char *filename = argv[optind];
    const char *output_dir = argv[optind + 1];

    char *instance_name = extract_instance_name(filename);
    if (instance_name == NULL)
        return 1;

    qp_problem_t *problem = read_problem_file(filename);
    if (problem == NULL)
    {
        fprintf(stderr, "Failed to read or parse the file.\n");
        free(instance_name);
        return 1;
    }

    pdhcg_result_t *result = optimize(&params, problem);

    if (result == NULL)
    {
        fprintf(stderr, "Solver failed.\n");
    }
    else
    {
        save_solver_summary(result, output_dir, instance_name);
        save_solution(
            result->primal_solution, problem->num_variables, output_dir, instance_name, "_primal_solution.txt");
        save_solution(result->dual_solution, problem->num_constraints, output_dir, instance_name, "_dual_solution.txt");
        pdhcg_result_free(result);
    }

    qp_problem_free(problem);
    free(instance_name);

    return 0;
}

#ifdef PDHCG_COMPILE_DISTRIBUTED
int run_d_pdhcg(int argc, char *argv[])
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized)
    {
        MPI_Init(&argc, &argv);
    }

    int rank_global, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_global);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    pdhg_parameters_t params;
    set_default_parameters(&params);

    params.grid_size.row_dims = 0;
    params.grid_size.col_dims = 0;
    params.grid_size.decided = false;
    params.permute_block_size = 256;

    static struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                           {"verbose", required_argument, 0, 'v'},
                                           {"time_limit", required_argument, 0, 1001},
                                           {"iter_limit", required_argument, 0, 1002},
                                           {"eps_opt", required_argument, 0, 1003},
                                           {"eps_feas", required_argument, 0, 1004},
                                           {"eps_infeas_detect", required_argument, 0, 1005},
                                           {"eps_feas_polish", required_argument, 0, 1006},
                                           {"feasibility_polishing", no_argument, 0, 'f'},
                                           {"l_inf_ruiz_iter", required_argument, 0, 1007},
                                           {"pock_chambolle_alpha", required_argument, 0, 1008},
                                           {"no_pock_chambolle", no_argument, 0, 1009},
                                           {"no_bound_obj_rescaling", no_argument, 0, 1010},
                                           {"sv_max_iter", required_argument, 0, 1011},
                                           {"sv_tol", required_argument, 0, 1012},
                                           {"eval_freq", required_argument, 0, 1013},
                                           {"opt_norm", required_argument, 0, 1014},
                                           {"inner_iter_limit", required_argument, 0, 1015},
                                           {"inner_init_tol", required_argument, 0, 1016},
                                           {"inner_min_tol", required_argument, 0, 1017},
                                           {"presolve", required_argument, 0, 1018},
                                           {"no_diag_precond", no_argument, 0, 1019},
                                           {"no_cone_preserving_scaling", no_argument, 0, 1021},
                                           {"artificial_restart_threshold", required_argument, 0, 1022},
                                           {"sufficient_reduction_for_restart", required_argument, 0, 1023},
                                           {"necessary_reduction_for_restart", required_argument, 0, 1024},
                                           {"curtis_reid_iter", required_argument, 0, 1025},
                                           {"grid_size", required_argument, 0, 2001},
                                           {"partition_method", required_argument, 0, 2002},
                                           {"permute_method", required_argument, 0, 2003},
                                           {"permute_block_size", required_argument, 0, 2004},
                                           {0, 0, 0, 0}};

    int opt;
    optind = 1;
    while ((opt = getopt_long(argc, argv, "hv:fp", long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case 'h':
                if (rank_global == 0)
                    print_usage(argv[0]);
                MPI_Finalize();
                return 0;
            case 'v':
                params.verbose = atoi(optarg);
                break;
            case 1001:
                params.termination_criteria.time_sec_limit = atof(optarg);
                break;
            case 1002:
                params.termination_criteria.iteration_limit = atoi(optarg);
                break;
            case 1003:
                params.termination_criteria.eps_optimal_relative = atof(optarg);
                break;
            case 1004:
                params.termination_criteria.eps_feasible_relative = atof(optarg);
                break;
            case 1005:
                params.termination_criteria.eps_infeasible = atof(optarg);
                break;
            case 1006:
                params.termination_criteria.eps_feas_polish_relative = atof(optarg);
                break;
            case 'f':
                params.feasibility_polishing = true;
                break;
            case 1007:
                params.l_inf_ruiz_iterations = atoi(optarg);
                break;
            case 1008:
                params.pock_chambolle_alpha = atof(optarg);
                break;
            case 1009:
                params.has_pock_chambolle_alpha = false;
                break;
            case 1010:
                params.bound_objective_rescaling = false;
                break;
            case 1011:
                params.sv_max_iter = atoi(optarg);
                break;
            case 1012:
                params.sv_tol = atof(optarg);
                break;
            case 1013:
                params.termination_evaluation_frequency = atoi(optarg);
                break;
            case 1014:
                if (strcmp(optarg, "l2") == 0)
                    params.optimality_norm = NORM_TYPE_L2;
                else if (strcmp(optarg, "linf") == 0)
                    params.optimality_norm = NORM_TYPE_L_INF;
                else
                {
                    if (rank_global == 0)
                        fprintf(stderr, "Error: opt_norm must be 'l2' or 'linf'\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                break;
            case 1015:
                params.inner_solver_parameters.iteration_limit = atoi(optarg);
                break;
            case 1016:
                params.inner_solver_parameters.initial_tolerance = atof(optarg);
                break;
            case 1017:
                params.inner_solver_parameters.min_tolerance = atof(optarg);
                break;
            case 1018:
                params.presolve = (atoi(optarg) != 0);
                break;
            case 1019:
                params.diag_jacobi_precond = false;
                break;
            case 1021:
                params.use_cone_preserving_scaling = false;
                break;
            case 1022:
                params.restart_params.artificial_restart_threshold = atof(optarg);
                break;
            case 1023:
                params.restart_params.sufficient_reduction_for_restart = atof(optarg);
                break;
            case 1024:
                params.restart_params.necessary_reduction_for_restart = atof(optarg);
                break;
            case 1025:
                params.curtis_reid_iterations = atoi(optarg);
                break;
            case 2001: // --grid_size r,c
            {
                int r, c;
                if (sscanf(optarg, "%d,%d", &r, &c) == 2 && r > 0 && c > 0)
                {
                    if (r * c != world_size)
                    {
                        if (rank_global == 0)
                        {
                            fprintf(stderr, "\n[FATAL ERROR] MPI Grid Configuration Mismatch\n");
                            fprintf(stderr, "Command line input : --grid_size %s (Total: %d)\n", optarg, r * c);
                            fprintf(stderr, "MPI Runtime size   : -n %d\n", world_size);
                        }
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
                    params.grid_size.row_dims = r;
                    params.grid_size.col_dims = c;
                    params.grid_size.decided = true;
                }
                else
                {
                    if (rank_global == 0)
                        fprintf(stderr, "Error: Invalid grid_size format. Use r,c\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                break;
            }
            case 2002: // --partition_method
            {
                if (strcmp(optarg, "uniform") == 0)
                    params.partition_method = UNIFORM_PARTITION;
                else if (strcmp(optarg, "nnz") == 0 || strcmp(optarg, "nnz_balance") == 0)
                    params.partition_method = NNZ_BALANCE_PARTITION;
                else
                {
                    if (rank_global == 0)
                        fprintf(stderr, "Error: partition_method must be 'uniform' or 'nnz'\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                break;
            }
            case 2003: // --permute_method
            {
                if (strcmp(optarg, "none") == 0 || strcmp(optarg, "no") == 0)
                    params.permute_method = NO_PERMUTATION;
                else if (strcmp(optarg, "random") == 0 || strcmp(optarg, "full") == 0)
                    params.permute_method = FULL_RANDOM_PERMUTATION;
                else if (strcmp(optarg, "block") == 0)
                    params.permute_method = BLOCK_RANDOM_PERMUTATION;
                else
                {
                    if (rank_global == 0)
                        fprintf(stderr, "Error: permute_method must be 'none', 'random', or 'block'\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                break;
            }
            case 2004: // --permute_block_size
            {
                params.permute_block_size = atoi(optarg);
                if (params.permute_block_size <= 0)
                {
                    if (rank_global == 0)
                        fprintf(stderr, "Error: permute_block_size must be positive.\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                break;
            }
            case '?':
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                return 1;
        }
    }

    if (argc - optind != 2)
    {
        if (rank_global == 0)
        {
            fprintf(stderr, "Error: You must specify an input file and an output directory.\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *filename = argv[optind];
    const char *output_dir = argv[optind + 1];

    char *instance_name = extract_instance_name(filename);
    if (instance_name == NULL)
    {
        if (rank_global == 0)
            fprintf(stderr, "Error: Could not extract instance name from filename.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    qp_problem_t *problem = NULL;

    if (rank_global == 0)
    {
        if (params.verbose)
            printf("Rank 0: Loading problem file '%s'...\n", filename);
        problem = read_problem_file(filename);

        if (problem == NULL)
        {
            fprintf(stderr, "Rank 0: Failed to read or parse the file.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }
    }

    pdhcg_result_t *result = distributed_optimize(&params, problem);

    if (rank_global == 0)
    {
        if (result == NULL)
        {
            fprintf(stderr, "Solver failed (returned NULL result).\n");
        }
        else
        {
            save_solver_summary(result, output_dir, instance_name);
            save_solution(
                result->primal_solution, result->num_variables, output_dir, instance_name, "_primal_solution.txt");
            save_solution(
                result->dual_solution, result->num_constraints, output_dir, instance_name, "_dual_solution.txt");
            pdhcg_result_free(result);
        }
        if (problem)
            qp_problem_free(problem);
    }
    else
    {
        if (result)
            pdhcg_result_free(result);
    }

    free(instance_name);
    MPI_Finalize();
    return 0;
}
#endif // PDHCG_COMPILE_DISTRIBUTED

int main(int argc, char *argv[])
{
#ifdef PDHCG_COMPILE_DISTRIBUTED
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized)
    {
        MPI_Init(&argc, &argv);
    }

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size > 1)
    {
        return run_d_pdhcg(argc, argv);
    }
    else
    {
        int ret = run_pdhcg(argc, argv);
        MPI_Finalize();
        return ret;
    }
#else
    return run_pdhcg(argc, argv);
#endif
}
