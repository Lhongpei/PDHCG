# Copyright 2025 Haihao Lu
# Copyright 2026 Hongpei Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Objective
MINIMIZE = 1
MAXIMIZE = -1

# Status codes
OPTIMAL = 0
PRIMAL_INFEASIBLE = 1
DUAL_INFEASIBLE = 2
TIME_LIMIT = 3
ITERATION_LIMIT = 4
UNSPECIFIED = -1


# parameter name alias
_PARAM_ALIAS = {
    # limits / logging
    "TimeLimit": "time_sec_limit",
    "IterationLimit": "iteration_limit",
    "LogLevel": "verbose",
    "Verbosity": "verbose",
    "OutputFlag": "verbose",
    "LogToConsole": "verbose",
    # termination evaluation cadence
    "TermCheckFreq": "termination_evaluation_frequency",
    # tolerances
    "OptimalityTol": "eps_optimal_relative",
    "FeasibilityTol": "eps_feasible_relative",
    # scaling / step size
    "CurtisReidIters": "curtis_reid_iterations",
    "RuizIters": "l_inf_ruiz_iterations",
    "UsePCAlpha": "has_pock_chambolle_alpha",
    "PCAlpha": "pock_chambolle_alpha",
    "BoundObjRescaling": "bound_objective_rescaling",
    "UseConePreservingScaling": "use_cone_preserving_scaling",
    # restarts
    "RestartArtificialThresh": "artificial_restart_threshold",
    "RestartSufficientReduction": "sufficient_reduction_for_restart",
    "RestartNecessaryReduction": "necessary_reduction_for_restart",
    "RestartKp": "k_p",
    # reflection
    "ReflectionCoeff": "reflection_coefficient",
    # termination criteria
    "OptimalityNorm": "optimality_norm",
    # singular value estimation (power method)
    "SVMaxIter": "sv_max_iter",
    "SVTol": "sv_tol",
    # inner solver
    "InnerIterLimit": "inner_iter_limit",
    "InnerInitTol": "inner_init_tol",
    "InnerMinTol": "inner_min_tol",
    "DiagJacobiPrecond": "diag_jacobi_precond",
    # presolve
    "Presolve": "presolve",
}
