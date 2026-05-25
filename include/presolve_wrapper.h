/*
 * PDHCG-II PSQP Presolve Wrapper Header
 *
 * This header provides the interface for using PSQP presolver within PDHCG-II.
 */

#ifndef PDHCG_PRESOLVE_WRAPPER_H
#define PDHCG_PRESOLVE_WRAPPER_H

#include "pdhcg_types.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* Presolve information structure (similar to cuPDLPx)
 * Internal PSQP types are hidden using void* to avoid exposing PSQP headers */
    typedef struct
    {
        void *presolver; /* Actually Presolver* */
        void *settings;  /* Actually Settings* */
        qp_problem_t *reduced_problem;
        bool problem_solved_during_presolve;
        double presolve_time;
        int presolve_status;
    } pdhcg_presolve_info_t;

    /* Data structure for presolved problem */
    typedef struct
    {
        int success;
        int infeasible;
        int unbounded;

        size_t m;
        size_t n;
        size_t nnz;

        double *Ax;
        int *Ai;
        int *Ap;
        double *lhs;
        double *rhs;
        double *c;
        double *lbs;
        double *ubs;
        double obj_offset;

        int has_quad_qr;
        double *Qx;
        int *Qi;
        int *Qp;
        size_t Qnnz;
        double *Rx;
        int *Ri;
        int *Rp;
        size_t Rnnz;
        size_t k;

        void *presolver_handle;
    } PDHCG_PresolvedData;

    /* Presolve standard QP with P matrix */
    PDHCG_PresolvedData *pdhcg_presolve_qp(const double *Ax,
                                           const int *Ai,
                                           const int *Ap,
                                           size_t m,
                                           size_t n,
                                           size_t nnz,
                                           const double *lhs,
                                           const double *rhs,
                                           const double *lbs,
                                           const double *ubs,
                                           const double *c,
                                           const double *Px,
                                           const int *Pi,
                                           const int *Pp,
                                           size_t Pnnz);

    /* Presolve QP in QR format: P = Q + R^T R.
     * Note: PSQP does not support the optional middle matrix D from
     * Q + R^T D R; the solver auto-disables presolve when D != I. */
    PDHCG_PresolvedData *pdhcg_presolve_qr(const double *Ax,
                                           const int *Ai,
                                           const int *Ap,
                                           size_t m,
                                           size_t n,
                                           size_t nnz,
                                           const double *lhs,
                                           const double *rhs,
                                           const double *lbs,
                                           const double *ubs,
                                           const double *c,
                                           const double *Qx,
                                           const int *Qi,
                                           const int *Qp,
                                           size_t Qnnz,
                                           const double *Rx,
                                           const int *Ri,
                                           const int *Rp,
                                           size_t Rnnz,
                                           size_t k);

    /* Cleanup presolved data */
    void pdhcg_presolve_cleanup(PDHCG_PresolvedData *data);

    /* Get PSQP version string */
    const char *pdhcg_presolve_version(void);

    /* Check if PSQP is available */
    int pdhcg_presolve_available(void);

    /* Get presolve status string */
    const char *pdhcg_get_presolve_status_str(int status);

    /* Main presolve function (similar to cuPDLPx's pslp_presolve) */
    pdhcg_presolve_info_t *pdhcg_presolve(const qp_problem_t *original_prob, const pdhg_parameters_t *params);

    /* Create result from presolve (when problem is solved during presolve) */
    pdhcg_result_t *pdhcg_create_result_from_presolve(const pdhcg_presolve_info_t *info,
                                                      const qp_problem_t *original_prob);

    /* Postsolve to recover original solution */
    void pdhcg_postsolve(const pdhcg_presolve_info_t *info, pdhcg_result_t *result, const qp_problem_t *original_prob);

    /* Free presolve info */
    void pdhcg_presolve_info_free(pdhcg_presolve_info_t *info);

#ifdef __cplusplus
}
#endif

#endif /* PDHCG_PRESOLVE_WRAPPER_H */
