/*
 * End-to-end test: Fisher quasi-linear market solved via exponential-cone
 * formulation through the PDHCG conic API. Builds a random sparse buyer/
 * good utility instance, solves the min form, and verifies cone feasibility
 * (y*exp(z/y) <= t) plus the y_i = 1 constraint at the returned solution.
 */

#include "pdhcg.h"
#include "pdhcg_types.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef PDHCG_COMPILE_DISTRIBUTED
#include <mpi.h>
#endif

static int rand_int(int lo, int hi)
{
    return lo + rand() % (hi - lo + 1);
}
static double rand_unit(void)
{
    return (double)rand() / RAND_MAX;
}

typedef struct
{
    int n;
    int m;
    int nnz;
    int *row_ptr;
    int *col_ind;
    double *val;
} sparse_u_t;

static int cache_path(char *buf, size_t bufsz, int n, int m, double density, unsigned seed)
{
    const char *root = getenv("FISHER_CACHE_DIR");
    char default_root[512];
    if (!root || !root[0])
    {
        const char *tmpdir = getenv("TMPDIR");
        if (!tmpdir || !tmpdir[0])
            tmpdir = "/tmp";
        int written = snprintf(default_root, sizeof(default_root), "%s/pdhcg-fisher", tmpdir);
        if (written < 0 || (size_t)written >= sizeof(default_root))
            return 0;
        root = default_root;
    }
    if (mkdir(root, 0700) != 0 && errno != EEXIST)
        return 0;
    int written = snprintf(buf, bufsz, "%s/fisher_n%d_m%d_d%g_s%u.bin", root, n, m, density, seed);
    return written >= 0 && (size_t)written < bufsz;
}

#define CHECK_READ(buf, sz, n, f)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        size_t _got = fread((buf), (sz), (n), (f));                                                                    \
        if (_got != (size_t)(n))                                                                                       \
        {                                                                                                              \
            fprintf(stderr, "[cache] short read at %s:%d (got %zu/%zu)\n", __FILE__, __LINE__, _got, (size_t)(n));     \
            fclose(f);                                                                                                 \
            return 0;                                                                                                  \
        }                                                                                                              \
    } while (0)

static int try_load_cache(int n, int m, double density, unsigned seed, sparse_u_t *u_out, double **w_out)
{
    char path[512];
    if (!cache_path(path, sizeof(path), n, m, density, seed))
        return 0;
    FILE *f = fopen(path, "rb");
    if (!f)
        return 0;
    int hn, hm, hnnz;
    double hd;
    unsigned hs;
    CHECK_READ(&hn, sizeof(int), 1, f);
    if (hn != n)
    {
        fclose(f);
        return 0;
    }
    CHECK_READ(&hm, sizeof(int), 1, f);
    if (hm != m)
    {
        fclose(f);
        return 0;
    }
    CHECK_READ(&hd, sizeof(double), 1, f);
    if (hd != density)
    {
        fclose(f);
        return 0;
    }
    CHECK_READ(&hs, sizeof(unsigned), 1, f);
    if (hs != seed)
    {
        fclose(f);
        return 0;
    }
    CHECK_READ(&hnnz, sizeof(int), 1, f);
    int *row_ptr = (int *)malloc((n + 1) * sizeof(int));
    int *col_ind = (int *)malloc((size_t)hnnz * sizeof(int));
    double *val = (double *)malloc((size_t)hnnz * sizeof(double));
    double *w = (double *)malloc((size_t)n * sizeof(double));
    if (!row_ptr || !col_ind || !val || !w)
    {
        fprintf(stderr, "[cache] malloc failed\n");
        fclose(f);
        return 0;
    }
    CHECK_READ(row_ptr, sizeof(int), n + 1, f);
    CHECK_READ(col_ind, sizeof(int), hnnz, f);
    CHECK_READ(val, sizeof(double), hnnz, f);
    CHECK_READ(w, sizeof(double), n, f);
    fclose(f);
    u_out->n = n;
    u_out->m = m;
    u_out->nnz = hnnz;
    u_out->row_ptr = row_ptr;
    u_out->col_ind = col_ind;
    u_out->val = val;
    *w_out = w;
    fprintf(stderr, "[cache] loaded %s (nnz=%d)\n", path, hnnz);
    return 1;
}

#define CHECK_WRITE(buf, sz, n, f, path)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        size_t _got = fwrite((buf), (sz), (n), (f));                                                                   \
        if (_got != (size_t)(n))                                                                                       \
        {                                                                                                              \
            fprintf(stderr,                                                                                            \
                    "[cache] short write at %s:%d (got %zu/%zu) -- removing %s\n",                                     \
                    __FILE__,                                                                                          \
                    __LINE__,                                                                                          \
                    _got,                                                                                              \
                    (size_t)(n),                                                                                       \
                    (path));                                                                                           \
            fclose(f);                                                                                                 \
            remove(path);                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

static void save_cache(int n, int m, double density, unsigned seed, const sparse_u_t *u, const double *w)
{
    char path[512];
    if (!cache_path(path, sizeof(path), n, m, density, seed))
        return;
    FILE *f = fopen(path, "wb");
    if (!f)
    {
        fprintf(stderr, "[cache] could not open %s for writing\n", path);
        return;
    }
    CHECK_WRITE(&u->n, sizeof(int), 1, f, path);
    CHECK_WRITE(&u->m, sizeof(int), 1, f, path);
    CHECK_WRITE(&density, sizeof(double), 1, f, path);
    CHECK_WRITE(&seed, sizeof(unsigned), 1, f, path);
    CHECK_WRITE(&u->nnz, sizeof(int), 1, f, path);
    CHECK_WRITE(u->row_ptr, sizeof(int), n + 1, f, path);
    CHECK_WRITE(u->col_ind, sizeof(int), u->nnz, f, path);
    CHECK_WRITE(u->val, sizeof(double), u->nnz, f, path);
    CHECK_WRITE(w, sizeof(double), n, f, path);
    if (fclose(f) != 0)
    {
        fprintf(stderr, "[cache] fclose failed -- removing %s\n", path);
        remove(path);
        return;
    }
    fprintf(stderr, "[cache] wrote %s (nnz=%d)\n", path, u->nnz);
}

static sparse_u_t generate_u(int n, int m, double density, unsigned seed)
{
    srand(seed);
    sparse_u_t u;
    u.n = n;
    u.m = m;
    int max_nnz = (int)((double)n * m * density * 1.5 + n);
    int *row_ptr = (int *)malloc((n + 1) * sizeof(int));
    int *col_ind = (int *)malloc(max_nnz * sizeof(int));
    double *val = (double *)malloc(max_nnz * sizeof(double));

    int cnt = 0;
    row_ptr[0] = 0;
    int *picked = (int *)calloc(m, sizeof(int));
    for (int i = 0; i < n; ++i)
    {
        memset(picked, 0, m * sizeof(int));
        int row_nnz = 0;
        for (int j = 0; j < m; ++j)
        {
            if (rand_unit() < density)
            {
                if (!picked[j])
                {
                    col_ind[cnt] = j;
                    val[cnt] = rand_unit() + 0.1;
                    cnt++;
                    picked[j] = 1;
                    row_nnz++;
                }
            }
        }
        if (row_nnz == 0)
        {
            int j = rand_int(0, m - 1);
            col_ind[cnt] = j;
            val[cnt] = rand_unit() + 0.1;
            cnt++;
        }
        row_ptr[i + 1] = cnt;
    }
    free(picked);

    /* Every good must have at least one buyer for the market to be feasible. */
    int *good_seen = (int *)calloc(m, sizeof(int));
    for (int k = 0; k < cnt; ++k)
        good_seen[col_ind[k]] = 1;
    int extra_alloc = max_nnz - cnt;
    for (int j = 0; j < m && extra_alloc > 0; ++j)
    {
        if (!good_seen[j])
        {
            fprintf(stderr, "[generator] good %d has no buyer; raise density.\n", j);
            free(row_ptr);
            free(col_ind);
            free(val);
            free(good_seen);
            u.n = 0;
            return u;
        }
    }
    free(good_seen);

    u.nnz = cnt;
    u.row_ptr = row_ptr;
    u.col_ind = col_ind;
    u.val = val;
    return u;
}

static void free_u(sparse_u_t *u)
{
    free(u->row_ptr);
    free(u->col_ind);
    free(u->val);
}

int main(int argc, char **argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 50;
    int m = (argc > 2) ? atoi(argv[2]) : 20;
    double density = (argc > 3) ? atof(argv[3]) : 0.2;
    unsigned seed = (argc > 4) ? (unsigned)atoi(argv[4]) : 1u;
    double eps = (argc > 5) ? atof(argv[5]) : 1e-6;
    double time_limit = (argc > 6) ? atof(argv[6]) : 300.0;

    int distributed = 0;
    int rank = 0;
    int world_size = 1;
#ifdef PDHCG_COMPILE_DISTRIBUTED
    int initialized_mpi = 0;
    const char *distributed_env = getenv("PDHCG_FISHER_DISTRIBUTED");
    distributed = distributed_env && atoi(distributed_env) != 0;
    if (distributed)
    {
        MPI_Initialized(&initialized_mpi);
        if (!initialized_mpi)
            MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }
#endif

    pdhg_parameters_t params;
    set_default_parameters(&params);
    {
        const char *vs = getenv("PDHG_VERBOSE");
        params.verbose = vs ? atoi(vs) : 1;
    }
    params.termination_criteria.eps_optimal_relative = eps;
    params.termination_criteria.eps_feasible_relative = eps;
    params.termination_criteria.time_sec_limit = time_limit;
    params.termination_criteria.iteration_limit = 200000;
    params.feasibility_polishing = false;
#ifdef PDHCG_COMPILE_DISTRIBUTED
    if (distributed)
    {
        params.grid_size.decided = true;
        params.grid_size.row_dims = 1;
        params.grid_size.col_dims = world_size;
        params.partition_method = NNZ_BALANCE_PARTITION;
        params.permute_method = BLOCK_RANDOM_PERMUTATION;
        params.permute_block_size = 256;
    }

    if (distributed && rank != 0)
    {
        pdhcg_result_t *worker_result = solve_qp_problem_distributed(&params, NULL);
        int failed = worker_result != NULL;
        if (worker_result)
            pdhcg_result_free(worker_result);
        MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!initialized_mpi)
            MPI_Finalize();
        return failed;
    }
#endif

    if (rank == 0)
    {
        printf("Fisher quasi-linear: n=%d buyers, m=%d goods, density=%.6g, eps=%.1e\n", n, m, density, eps);
        if (distributed)
            printf("Distributed grid: 1 x %d GPUs\n", world_size);
    }
    sparse_u_t u;
    double *w = NULL;
    if (!try_load_cache(n, m, density, seed, &u, &w))
    {
        u = generate_u(n, m, density, seed);
        if (u.n == 0)
            return 1;
        w = (double *)malloc(n * sizeof(double));
        for (int i = 0; i < n; ++i)
            w[i] = rand_unit() + 0.1;
        save_cache(n, m, density, seed, &u, w);
    }
    if (rank == 0)
        printf("nnz(u)=%d\n", u.nnz);

    double *b = (double *)malloc(m * sizeof(double));
    double supply_each = 0.20 * (double)n;
    for (int j = 0; j < m; ++j)
        b[j] = supply_each;

    int nx = u.nnz;
    int nv = n;
    int nzyt = 3 * n;
    int nvar = nx + nv + nzyt;
    int x_off = 0;
    int v_off = nx;
    int cone_off = nx + nv;
    if (rank == 0)
        printf("variables: %d  (x:%d, v:%d, zyt:%d)\n", nvar, nx, nv, nzyt);

    int ncon = m + n;

    int A_nnz = u.nnz + 2 * n + u.nnz;
    int *A_row_ptr = (int *)malloc((ncon + 1) * sizeof(int));
    int *A_col_ind = (int *)malloc(A_nnz * sizeof(int));
    double *A_val = (double *)malloc(A_nnz * sizeof(double));

    int *good_cnt = (int *)calloc(m, sizeof(int));
    for (int i = 0; i < n; ++i)
        for (int k = u.row_ptr[i]; k < u.row_ptr[i + 1]; ++k)
            good_cnt[u.col_ind[k]]++;

    A_row_ptr[0] = 0;
    for (int j = 0; j < m; ++j)
        A_row_ptr[j + 1] = A_row_ptr[j] + good_cnt[j];
    for (int i = 0; i < n; ++i)
    {
        int row_nnz = 2 + (u.row_ptr[i + 1] - u.row_ptr[i]);
        A_row_ptr[m + i + 1] = A_row_ptr[m + i] + row_nnz;
    }
    if (A_row_ptr[ncon] != A_nnz)
    {
        fprintf(stderr, "row_ptr mismatch: got %d expected %d\n", A_row_ptr[ncon], A_nnz);
        return 1;
    }

    int *x_good = (int *)malloc(u.nnz * sizeof(int));
    int xk = 0;
    for (int i = 0; i < n; ++i)
        for (int k = u.row_ptr[i]; k < u.row_ptr[i + 1]; ++k)
            x_good[xk++] = u.col_ind[k];

    int *good_cursor = (int *)calloc(m, sizeof(int));
    for (int xk2 = 0; xk2 < u.nnz; ++xk2)
    {
        int j = x_good[xk2];
        int pos = A_row_ptr[j] + good_cursor[j]++;
        A_col_ind[pos] = x_off + xk2;
        A_val[pos] = 1.0;
    }
    free(good_cursor);
    free(good_cnt);

    int xk_running = 0;
    for (int i = 0; i < n; ++i)
    {
        int pos = A_row_ptr[m + i];
        for (int k = u.row_ptr[i]; k < u.row_ptr[i + 1]; ++k, ++xk_running)
        {
            A_col_ind[pos] = x_off + xk_running;
            A_val[pos] = -u.val[k];
            pos++;
        }
        A_col_ind[pos] = v_off + i;
        A_val[pos] = -1.0;
        pos++;
        A_col_ind[pos] = cone_off + 3 * i + 2;
        A_val[pos] = 1.0;
    }
    free(x_good);

    double *c = (double *)calloc(nvar, sizeof(double));
    for (int i = 0; i < n; ++i)
    {
        c[v_off + i] = 1.0;
        c[cone_off + 3 * i + 0] = -w[i];
    }

    double *var_lb = (double *)malloc(nvar * sizeof(double));
    double *var_ub = (double *)malloc(nvar * sizeof(double));
    for (int k = 0; k < nvar; ++k)
    {
        var_lb[k] = -1e30;
        var_ub[k] = 1e30;
    }
    for (int k = 0; k < nx; ++k)
        var_lb[x_off + k] = 0.0;
    for (int i = 0; i < n; ++i)
        var_lb[v_off + i] = 0.0;

    double *con_lb = (double *)malloc(ncon * sizeof(double));
    double *con_ub = (double *)malloc(ncon * sizeof(double));
    for (int j = 0; j < m; ++j)
    {
        con_lb[j] = b[j];
        con_ub[j] = b[j];
    }
    for (int i = 0; i < n; ++i)
    {
        con_lb[m + i] = 0.0;
        con_ub[m + i] = 0.0;
    }

    matrix_desc_t A;
    memset(&A, 0, sizeof(A));
    A.m = ncon;
    A.n = nvar;
    A.fmt = matrix_csr;
    A.data.csr.nnz = A_nnz;
    A.data.csr.row_ptr = A_row_ptr;
    A.data.csr.col_ind = A_col_ind;
    A.data.csr.vals = A_val;

    cone_spec_t *cones = (cone_spec_t *)malloc(n * sizeof(cone_spec_t));
    for (int i = 0; i < n; ++i)
    {
        cones[i].type = CONE_EXPONENTIAL;
        cones[i].start_idx = cone_off + 3 * i;
        cones[i].v_dim = 1;
        cones[i].is_fixed = NULL;
    }

    qp_problem_t *prob =
        create_qp_problem(c, NULL, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, n, cones, 0, NULL, NULL);
    if (!prob)
    {
        fprintf(stderr, "create_qp_problem failed\n");
        return 1;
    }

    for (int i = 0; i < n; ++i)
    {
        if (set_cone_fixed(prob, i, 1, 1.0) != 0)
            return 1;
    }

    pdhcg_result_t *res = NULL;
#ifdef PDHCG_COMPILE_DISTRIBUTED
    if (distributed)
    {
        res = solve_qp_problem_distributed(&params, prob);
    }
    else
#endif
    {
        res = solve_qp_problem(prob, &params);
    }

    int failed = 0;
    double max_cone_violation = 0.0;
    double max_y_dev = 0.0;
    if (rank == 0 && !res)
    {
        fprintf(stderr, "solve failed\n");
        failed = 1;
    }
    else if (rank == 0)
    {
        printf("\nSolver-reported time: %.6fs  iter=%d  status=%d (1=OPTIMAL)\n",
               res->cumulative_time_sec,
               res->total_count,
               (int)res->termination_reason);
        printf("Primal obj (min form): %.8f -> max form: %.8f\n",
               res->primal_objective_value,
               -res->primal_objective_value);

        /*
         * Audit the returned point in the original, unscaled model.  In
         * particular, y=1 makes each exponential cone an epigraph
         * exp(z) <= t.  Its KKT condition is
         *
         *   r_z + r_t exp(z) = 0, r_z <= 0, r_t >= 0,
         *
         * where r = c - A^T lambda.  This condition is independent of the
         * solver's internal residual implementation.
         */
        double *reduced_gradient = (double *)malloc((size_t)nvar * sizeof(double));
        double max_linear_residual = 0.0;
        double max_good_residual = 0.0;
        double max_utility_residual = 0.0;
        double max_rowwise_relative_residual = 0.0;
        double max_box_kkt = 0.0;
        double max_exp_kkt = 0.0;
        double max_exp_complementarity = 0.0;
        double independent_dual_objective = 0.0;
        int independent_dual_finite = reduced_gradient != NULL;
        if (!reduced_gradient)
        {
            fprintf(stderr, "KKT audit allocation failed\n");
            failed = 1;
        }
        else
        {
            memcpy(reduced_gradient, c, (size_t)nvar * sizeof(double));
            for (int row = 0; row < ncon; ++row)
            {
                double activity = 0.0;
                double lambda = res->dual_solution[row];
                for (int p = A_row_ptr[row]; p < A_row_ptr[row + 1]; ++p)
                {
                    int col = A_col_ind[p];
                    activity += A_val[p] * res->primal_solution[col];
                    reduced_gradient[col] -= A_val[p] * lambda;
                }
                double row_residual = fabs(activity - con_lb[row]);
                if (row_residual > max_linear_residual)
                    max_linear_residual = row_residual;
                if (row < m)
                {
                    if (row_residual > max_good_residual)
                        max_good_residual = row_residual;
                }
                else if (row_residual > max_utility_residual)
                {
                    max_utility_residual = row_residual;
                }
                double rowwise_relative = row_residual / (1.0 + fabs(con_lb[row]));
                if (rowwise_relative > max_rowwise_relative_residual)
                    max_rowwise_relative_residual = rowwise_relative;
                independent_dual_objective += lambda * con_lb[row];
            }

            for (int col = 0; col < cone_off; ++col)
            {
                double x = res->primal_solution[col];
                double r = reduced_gradient[col];
                double projected = fmax(var_lb[col], fmin(x - r, var_ub[col]));
                double violation = fabs(x - projected);
                if (violation > max_box_kkt)
                    max_box_kkt = violation;

                if (r < 0.0)
                    independent_dual_finite = 0;
            }

            for (int i = 0; i < n; ++i)
            {
                int idx = cone_off + 3 * i;
                double z = res->primal_solution[idx + 0];
                double y = res->primal_solution[idx + 1];
                double t = res->primal_solution[idx + 2];
                double rz = reduced_gradient[idx + 0];
                double ry = reduced_gradient[idx + 1];
                double rt = reduced_gradient[idx + 2];
                double ez = exp(z);
                double stationarity = fabs(rz + rt * ez);
                stationarity = fmax(stationarity, fmax(rz, 0.0));
                stationarity = fmax(stationarity, fmax(-rt, 0.0));
                if (stationarity > max_exp_kkt)
                    max_exp_kkt = stationarity;
                double complementarity = fabs(rt * (t - ez));
                if (complementarity > max_exp_complementarity)
                    max_exp_complementarity = complementarity;

                independent_dual_objective += ry * y;
                if (rt > 0.0 && rz < 0.0)
                {
                    independent_dual_objective += rz * log(-rz / rt) - rz;
                }
                else if (rt > 0.0 && rz == 0.0)
                {
                    /* The infimum is zero and is approached as z -> -inf. */
                }
                else if (!(rt == 0.0 && rz == 0.0))
                {
                    independent_dual_finite = 0;
                }
            }

            double independent_gap = INFINITY;
            if (independent_dual_finite)
            {
                independent_gap = fabs(res->primal_objective_value - independent_dual_objective) /
                    (1.0 + fabs(res->primal_objective_value) + fabs(independent_dual_objective));
            }
            printf("Internal KKT: rel_primal=%.9g rel_dual=%.9g rel_gap=%.9g dual_obj=%.17g\n",
                   res->relative_primal_residual,
                   res->relative_dual_residual,
                   res->relative_objective_gap,
                   res->dual_objective_value);
            printf("Independent KKT: linear_inf=%.9g good_inf=%.9g utility_inf=%.9g "
                   "rowwise_rel_inf=%.9g box_inf=%.9g exp_stationarity_inf=%.9g "
                   "exp_complementarity_inf=%.9g rel_gap=%.9g dual_obj=%.17g finite=%d\n",
                   max_linear_residual,
                   max_good_residual,
                   max_utility_residual,
                   max_rowwise_relative_residual,
                   max_box_kkt,
                   max_exp_kkt,
                   max_exp_complementarity,
                   independent_gap,
                   independent_dual_objective,
                   independent_dual_finite);
            free(reduced_gradient);
        }

        for (int i = 0; i < n; ++i)
        {
            double z = res->primal_solution[cone_off + 3 * i + 0];
            double y = res->primal_solution[cone_off + 3 * i + 1];
            double t = res->primal_solution[cone_off + 3 * i + 2];
            double yd = fabs(y - 1.0);
            if (yd > max_y_dev)
                max_y_dev = yd;
            double lhs = (y > 0.0) ? y * exp(z / y) : ((z <= 0.0) ? 0.0 : INFINITY);
            double viol = lhs - t;
            if (viol > max_cone_violation)
                max_cone_violation = viol;
        }
        printf("Exp cone max violation (y*exp(z/y) - t): %.3e\n", max_cone_violation);
        printf("Max |y - 1|: %.3e\n", max_y_dev);
        printf("FISHER_RESULT,n=%d,m=%d,density=%.8g,eps=%.8g,gpus=%d,time=%.9g,iter=%d,status=%d,"
               "objective=%.17g,cone_violation=%.9g,y_deviation=%.9g\n",
               n,
               m,
               density,
               eps,
               distributed ? world_size : 1,
               res->cumulative_time_sec,
               res->total_count,
               (int)res->termination_reason,
               res->primal_objective_value,
               max_cone_violation,
               max_y_dev);

        if (distributed &&
            (res->termination_reason != TERMINATION_REASON_OPTIMAL || max_cone_violation > 5e-6 || max_y_dev > 5e-6))
            failed = 1;
    }

    if (res)
        pdhcg_result_free(res);
    qp_problem_free(prob);
    free(cones);
    free(c);
    free(var_lb);
    free(var_ub);
    free(con_lb);
    free(con_ub);
    free(A_row_ptr);
    free(A_col_ind);
    free(A_val);
    free_u(&u);
    free(w);
    free(b);

#ifdef PDHCG_COMPILE_DISTRIBUTED
    if (distributed)
    {
        MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!initialized_mpi)
            MPI_Finalize();
    }
#endif
    return failed;
}
