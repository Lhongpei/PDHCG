/*
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

#pragma once

#include "internal_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void project_cone_runtime(pdhg_solver_state_t *state, cone_runtime_t *runtime, double *vector, double *warm_start);

    void project_cone_runtime_diag_q(pdhg_solver_state_t *state, cone_runtime_t *runtime, double primal_step_size);

    void compute_cone_dual_residual(pdhg_solver_state_t *state, const double *effective_objective);

    void recompute_cone_reflection(pdhg_solver_state_t *state);

    void set_cone_dual_slack(pdhg_solver_state_t *state, const double *effective_objective);

#ifdef __cplusplus
}
#endif
