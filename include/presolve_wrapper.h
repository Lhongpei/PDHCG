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

/* PDHCG-II optional PreFOS presolve adapter. */

#ifndef PDHCG_PRESOLVE_WRAPPER_H
#define PDHCG_PRESOLVE_WRAPPER_H

#include "pdhcg_types.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum
    {
        PDHCG_PRESOLVE_STATUS_UNCHANGED = 0,
        PDHCG_PRESOLVE_STATUS_REDUCED,
        PDHCG_PRESOLVE_STATUS_PRIMAL_INFEASIBLE,
        PDHCG_PRESOLVE_STATUS_ERROR,
        PDHCG_PRESOLVE_STATUS_NOT_AVAILABLE
    } pdhcg_presolve_status_t;

    /* PreFOS types stay private to the adapter implementation. */
    typedef struct
    {
        void *presolver;
        qp_problem_t *reduced_problem;
        bool problem_solved_during_presolve;
        double presolve_time;
        pdhcg_presolve_status_t presolve_status;
        int prefos_original_rows;
        double postsolve_tolerance;
    } pdhcg_presolve_info_t;

    /* Get the configured presolver version string. */
    const char *pdhcg_presolve_version(void);

    /* Check whether PDHCG was compiled with PreFOS. */
    int pdhcg_presolve_available(void);

    /* Get presolve status string */
    const char *pdhcg_get_presolve_status_str(int status);

    /* Presolve the unified LP/QP/conic model. */
    pdhcg_presolve_info_t *pdhcg_presolve(const qp_problem_t *original_prob, const pdhg_parameters_t *params);

    /* Create result from presolve (when problem is solved during presolve) */
    pdhcg_result_t *pdhcg_create_result_from_presolve(const pdhcg_presolve_info_t *info,
                                                      const qp_problem_t *original_prob);

    /* Postsolve to recover original solution */
    int pdhcg_postsolve(const pdhcg_presolve_info_t *info, pdhcg_result_t *result, const qp_problem_t *original_prob);

    /* Free presolve info */
    void pdhcg_presolve_info_free(pdhcg_presolve_info_t *info);

#ifdef __cplusplus
}
#endif

#endif /* PDHCG_PRESOLVE_WRAPPER_H */
