#pragma once

#include "internal_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void initialize_distributed_cones(pdhg_solver_state_t *state,
                                      const qp_problem_t *working_problem,
                                      const rescale_info_t *rescale_info);
    void free_distributed_cones(pdhg_solver_state_t *state);
    void project_distributed_cones(pdhg_solver_state_t *state, double *primal_solution);
    void recompute_distributed_cone_reflected(pdhg_solver_state_t *state,
                                              double *reflected_primal,
                                              const double *pdhg_primal,
                                              const double *current_primal);
    void compute_distributed_cone_dual_residual(pdhg_solver_state_t *state, const double *effective_objective);
    void set_distributed_cone_dual_slack(pdhg_solver_state_t *state,
                                         double *dual_slack,
                                         const double *effective_objective,
                                         const double *dual_product);

#ifdef __cplusplus
}
#endif
