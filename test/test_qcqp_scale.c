/*
 * QCQP: min -sum(x_i)  s.t.  x_i^2 <= 1 for i=1..N
 *
 * Two lifts:
 *  A) aux: 3N cone + N orig = 4N vars. Cone (v_i, s_i, t_i), v_i = sqrt(2) x_i.
 *  B) no_aux: (x_i, s_i, t_i) triples reordered. 3N vars total.
 *
 * Both should give x_i = 1, obj = -N.
 * Compare wall time & iter count to measure gain of no-aux.
 */

#include "pdhcg.h"
#include "pdhcg_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double elapsed_sec(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
}

/* A: standard aux lift.
   vars = [x_1..x_N | v_1..v_N | s_1..s_N | t_1..t_N]   -- 4N vars
   rows: for each i: sqrt2 x_i - v_i = 0; s_i = 1; t_i = 1                       */
static qp_problem_t *build_aux(int N)
{
    const double SQRT2 = 1.4142135623730951;
    int nvar = 4 * N;
    int nrow = 3 * N;
    int nnz = 5 * N; /* per QC: [sqrt2, -1, 1, 1] = 4? Let me recount */
    /* rows: (i): sqrt2 x_i - v_i = 0 [2 nnz]; (N+i): s_i = 1 [1 nnz]; (2N+i): t_i = 1 [1 nnz] */
    /* total nnz per QC = 4, plus rearrangement */
    nnz = 4 * N;

    int *arow = (int *)calloc(nrow + 1, sizeof(int));
    int *acol = (int *)malloc(nnz * sizeof(int));
    double *aval = (double *)malloc(nnz * sizeof(double));
    int p = 0;
    for (int i = 0; i < N; ++i)
    {
        arow[i + 1] = arow[i] + 2;
        acol[p] = i;
        aval[p] = SQRT2;
        p++;
        acol[p] = N + i;
        aval[p] = -1.0;
        p++;
    }
    for (int i = 0; i < N; ++i)
    {
        arow[N + i + 1] = arow[N + i] + 1;
        acol[p] = 2 * N + i;
        aval[p] = 1.0;
        p++;
    }
    for (int i = 0; i < N; ++i)
    {
        arow[2 * N + i + 1] = arow[2 * N + i] + 1;
        acol[p] = 3 * N + i;
        aval[p] = 1.0;
        p++;
    }
    matrix_desc_t A = {0};
    A.m = nrow;
    A.n = nvar;
    A.fmt = matrix_csr;
    A.data.csr.nnz = nnz;
    A.data.csr.row_ptr = arow;
    A.data.csr.col_ind = acol;
    A.data.csr.vals = aval;

    double *c = (double *)calloc(nvar, sizeof(double));
    for (int i = 0; i < N; ++i)
        c[i] = -1.0;

    double *vlb = (double *)malloc(nvar * sizeof(double));
    double *vub = (double *)malloc(nvar * sizeof(double));
    for (int i = 0; i < nvar; ++i)
    {
        vlb[i] = -1e30;
        vub[i] = 1e30;
    }

    double *clb = (double *)malloc(nrow * sizeof(double));
    double *cub = (double *)malloc(nrow * sizeof(double));
    for (int i = 0; i < N; ++i)
    {
        clb[i] = 0.0;
        cub[i] = 0.0;
    } /* sqrt2 x - v = 0 */
    for (int i = 0; i < N; ++i)
    {
        clb[N + i] = 1.0;
        cub[N + i] = 1.0;
    } /* s = 1 */
    for (int i = 0; i < N; ++i)
    {
        clb[2 * N + i] = 1.0;
        cub[2 * N + i] = 1.0;
    } /* t = 1 */

    cone_spec_t *cones = (cone_spec_t *)calloc(N, sizeof(cone_spec_t));
    for (int i = 0; i < N; ++i)
    {
        cones[i].type = CONE_ROTATED_SOC;
        cones[i].start_idx = N + i * 1; /* v_i */
        /* WAIT: v_1..v_N are contiguous but s_i,t_i are not adjacent to v_i.
           This layout won't work with contiguous cone kernel. Need interleave. */
    }
    free(cones);

    /* Correct layout: interleave. vars = [x_1..x_N | (v_1,s_1,t_1), (v_2,s_2,t_2), ...] */
    nvar = 4 * N;
    /* Reindex: x_i at position i (i<N); (v_i, s_i, t_i) at positions N + 3i, N+3i+1, N+3i+2 */
    p = 0;
    /* Rebuild rows and cols with new positions */
    for (int i = 0; i < nnz; ++i)
        aval[i] = 0.0;
    for (int i = 0; i <= nrow; ++i)
        arow[i] = 0;
    for (int i = 0; i < N; ++i)
    {
        arow[i + 1] = arow[i] + 2;
        acol[p] = i;
        aval[p] = SQRT2;
        p++; /* sqrt2 x_i */
        acol[p] = N + 3 * i;
        aval[p] = -1.0;
        p++; /* -v_i */
    }
    for (int i = 0; i < N; ++i)
    {
        arow[N + i + 1] = arow[N + i] + 1;
        acol[p] = N + 3 * i + 1;
        aval[p] = 1.0;
        p++; /* s_i = 1 */
    }
    for (int i = 0; i < N; ++i)
    {
        arow[2 * N + i + 1] = arow[2 * N + i] + 1;
        acol[p] = N + 3 * i + 2;
        aval[p] = 1.0;
        p++; /* t_i = 1 */
    }
    A.data.csr.row_ptr = arow;
    A.data.csr.col_ind = acol;
    A.data.csr.vals = aval;

    cones = (cone_spec_t *)calloc(N, sizeof(cone_spec_t));
    for (int i = 0; i < N; ++i)
    {
        cones[i].type = CONE_ROTATED_SOC;
        cones[i].start_idx = N + 3 * i; /* (v_i, s_i, t_i) contiguous */
        cones[i].v_dim = 1;
        cones[i].is_fixed = NULL;
    }
    qp_problem_t *prob =
        create_qp_problem(c, NULL, NULL, NULL, &A, clb, cub, vlb, vub, NULL, N, cones, NULL, NULL, 0, NULL);
    free(cones);
    free(clb);
    free(cub);
    free(vlb);
    free(vub);
    free(c);
    free(arow);
    free(acol);
    free(aval);
    return prob;
}

/* B: no-aux, reordered.
   vars = [(x_1,s_1,t_1), (x_2,s_2,t_2), ...]  -- 3N vars
   rows: for each i: s_i = 1; t_i = 1        (no v-binding row!)                */
static qp_problem_t *build_no_aux(int N)
{
    int nvar = 3 * N;
    int nrow = 2 * N;
    int nnz = 2 * N;

    int *arow = (int *)calloc(nrow + 1, sizeof(int));
    int *acol = (int *)malloc(nnz * sizeof(int));
    double *aval = (double *)malloc(nnz * sizeof(double));
    int p = 0;
    for (int i = 0; i < N; ++i)
    {
        arow[i + 1] = arow[i] + 1;
        acol[p] = 3 * i + 1;
        aval[p] = 1.0;
        p++; /* s_i = 1 */
    }
    for (int i = 0; i < N; ++i)
    {
        arow[N + i + 1] = arow[N + i] + 1;
        acol[p] = 3 * i + 2;
        aval[p] = 1.0;
        p++; /* t_i = 1 */
    }
    matrix_desc_t A = {0};
    A.m = nrow;
    A.n = nvar;
    A.fmt = matrix_csr;
    A.data.csr.nnz = nnz;
    A.data.csr.row_ptr = arow;
    A.data.csr.col_ind = acol;
    A.data.csr.vals = aval;

    double *c = (double *)calloc(nvar, sizeof(double));
    for (int i = 0; i < N; ++i)
        c[3 * i] = -1.0; /* obj: -sum x_i */

    double *vlb = (double *)malloc(nvar * sizeof(double));
    double *vub = (double *)malloc(nvar * sizeof(double));
    for (int i = 0; i < nvar; ++i)
    {
        vlb[i] = -1e30;
        vub[i] = 1e30;
    }

    double *clb = (double *)malloc(nrow * sizeof(double));
    double *cub = (double *)malloc(nrow * sizeof(double));
    for (int i = 0; i < N; ++i)
    {
        clb[i] = 1.0;
        cub[i] = 1.0;
    } /* s_i = 1 */
    for (int i = 0; i < N; ++i)
    {
        clb[N + i] = 0.5;
        cub[N + i] = 0.5;
    } /* t_i = 1/2 -> x^2 <= 1 */

    cone_spec_t *cones = (cone_spec_t *)calloc(N, sizeof(cone_spec_t));
    for (int i = 0; i < N; ++i)
    {
        cones[i].type = CONE_ROTATED_SOC;
        cones[i].start_idx = 3 * i; /* (x_i, s_i, t_i) contiguous */
        cones[i].v_dim = 1;
        cones[i].is_fixed = NULL;
    }
    qp_problem_t *prob =
        create_qp_problem(c, NULL, NULL, NULL, &A, clb, cub, vlb, vub, NULL, N, cones, NULL, NULL, 0, NULL);
    free(cones);
    free(clb);
    free(cub);
    free(vlb);
    free(vub);
    free(c);
    free(arow);
    free(acol);
    free(aval);
    return prob;
}

static void run(const char *name, qp_problem_t *prob, int N, double eps)
{
    if (!prob)
    {
        fprintf(stderr, "[%s] create failed\n", name);
        return;
    }
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 0;
    params.termination_criteria.eps_optimal_relative = eps;
    params.termination_criteria.eps_feasible_relative = eps;
    params.termination_criteria.iteration_limit = 5000000;
    params.termination_criteria.time_sec_limit = 60.0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    pdhcg_result_t *res = solve_qp_problem(prob, &params);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double wall = elapsed_sec(t0, t1);
    if (res)
    {
        printf("%-10s N=%5d eps=%.0e  status=%d iter=%7d wall=%6.2fs obj=%.4f  primal=%.2e dual=%.2e\n",
               name,
               N,
               eps,
               (int)res->termination_reason,
               res->total_count,
               wall,
               res->primal_objective_value,
               res->relative_primal_residual,
               res->relative_dual_residual);
        pdhcg_result_free(res);
    }
    qp_problem_free(prob);
}

int main(void)
{
    int Ns[] = {100, 1000, 10000, 50000};
    double eps = 1e-6;
    for (int i = 0; i < (int)(sizeof(Ns) / sizeof(Ns[0])); ++i)
    {
        int N = Ns[i];
        run("aux", build_aux(N), N, eps);
        run("no_aux", build_no_aux(N), N, eps);
        printf("\n");
    }
    return 0;
}
