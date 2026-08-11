#include "mps_parser.h"
#include "pdhcg.h"
#include "pdhcg_types.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p)
    {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    return p;
}

static void *xmalloc(size_t size)
{
    void *p = malloc(size ? size : 1);
    if (!p)
    {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    return p;
}

static int write_all(FILE *f, const void *ptr, size_t size, size_t count)
{
    if (count == 0)
        return 0;
    return fwrite(ptr, size, count, f) == count ? 0 : -1;
}

static int write_csr(FILE *f, const CsrComponent *csr, int rows, int nnz)
{
    if (write_all(f, csr && csr->row_ptr ? csr->row_ptr : NULL, sizeof(int32_t), (size_t)rows + 1) != 0)
        return -1;
    if (write_all(f, csr && csr->col_ind ? csr->col_ind : NULL, sizeof(int32_t), (size_t)nnz) != 0)
        return -1;
    if (write_all(f, csr && csr->val ? csr->val : NULL, sizeof(double), (size_t)nnz) != 0)
        return -1;
    return 0;
}

static int32_t one_if_nonnull(const void *p)
{
    return p ? 1 : 0;
}

static CsrComponent *copy_csr_with_rows(const CsrComponent *src, int old_rows, int new_rows, int nnz, double scale)
{
    CsrComponent *dst = (CsrComponent *)xcalloc(1, sizeof(CsrComponent));
    dst->row_ptr = (int *)xcalloc((size_t)new_rows + 1, sizeof(int));
    if (src && src->row_ptr)
    {
        memcpy(dst->row_ptr, src->row_ptr, ((size_t)old_rows + 1) * sizeof(int));
        int last = src->row_ptr[old_rows];
        for (int r = old_rows + 1; r <= new_rows; ++r)
            dst->row_ptr[r] = last;
    }
    if (nnz > 0)
    {
        dst->col_ind = (int *)xmalloc((size_t)nnz * sizeof(int));
        dst->val = (double *)xmalloc((size_t)nnz * sizeof(double));
        memcpy(dst->col_ind, src->col_ind, (size_t)nnz * sizeof(int));
        for (int i = 0; i < nnz; ++i)
            dst->val[i] = scale * src->val[i];
    }
    return dst;
}

static qp_problem_t *epigraph_objective_q_to_qc(const qp_problem_t *orig)
{
    if (!orig)
        return NULL;
    if (orig->objective_sparse_matrix_num_nonzeros <= 0)
        return NULL;
    if (orig->num_rank_lowrank_obj > 0 || orig->objective_lowrank_matrix_num_nonzeros > 0 ||
        orig->objective_lowrank_middle_matrix_num_nonzeros > 0)
    {
        fprintf(stderr, "objective epigraph for low-rank Q is not implemented\n");
        return NULL;
    }

    const int n_old = orig->num_variables;
    const int m_old = orig->num_constraints;
    const int n_new = n_old + 1;
    const int m_new = m_old + 1;
    const int eta_col = n_old;

    int lin_nnz = 0;
    for (int j = 0; j < n_old; ++j)
    {
        if (orig->objective_vector[j] != 0.0)
            lin_nnz++;
    }

    qp_problem_t *out = (qp_problem_t *)xcalloc(1, sizeof(qp_problem_t));
    out->num_variables = n_new;
    out->num_constraints = m_new;
    out->num_rank_lowrank_obj = 0;
    out->objective_sparse_matrix_num_nonzeros = 0;
    out->objective_lowrank_matrix_num_nonzeros = 0;
    out->objective_lowrank_middle_matrix_num_nonzeros = 0;
    out->objective_constant = 0.0;
    out->num_original_variables = orig->num_original_variables > 0 ? orig->num_original_variables : n_old;

    out->objective_vector = (double *)xcalloc((size_t)n_new, sizeof(double));
    out->objective_vector[eta_col] = 1.0;
    out->variable_lower_bound = (double *)xmalloc((size_t)n_new * sizeof(double));
    out->variable_upper_bound = (double *)xmalloc((size_t)n_new * sizeof(double));
    memcpy(out->variable_lower_bound, orig->variable_lower_bound, (size_t)n_old * sizeof(double));
    memcpy(out->variable_upper_bound, orig->variable_upper_bound, (size_t)n_old * sizeof(double));
    out->variable_lower_bound[eta_col] = -INFINITY;
    out->variable_upper_bound[eta_col] = INFINITY;

    out->constraint_lower_bound = (double *)xmalloc((size_t)m_new * sizeof(double));
    out->constraint_upper_bound = (double *)xmalloc((size_t)m_new * sizeof(double));
    memcpy(out->constraint_lower_bound, orig->constraint_lower_bound, (size_t)m_old * sizeof(double));
    memcpy(out->constraint_upper_bound, orig->constraint_upper_bound, (size_t)m_old * sizeof(double));
    out->constraint_lower_bound[m_old] = -INFINITY;
    out->constraint_upper_bound[m_old] = -orig->objective_constant;

    int a_nnz_old = orig->constraint_matrix_num_nonzeros;
    out->constraint_matrix_num_nonzeros = a_nnz_old + lin_nnz + 1;
    out->constraint_matrix = (CsrComponent *)xcalloc(1, sizeof(CsrComponent));
    out->constraint_matrix->row_ptr = (int *)xcalloc((size_t)m_new + 1, sizeof(int));
    memcpy(out->constraint_matrix->row_ptr, orig->constraint_matrix->row_ptr, ((size_t)m_old + 1) * sizeof(int));
    out->constraint_matrix->row_ptr[m_new] = out->constraint_matrix_num_nonzeros;
    out->constraint_matrix->col_ind = (int *)xmalloc((size_t)out->constraint_matrix_num_nonzeros * sizeof(int));
    out->constraint_matrix->val = (double *)xmalloc((size_t)out->constraint_matrix_num_nonzeros * sizeof(double));
    if (a_nnz_old > 0)
    {
        memcpy(out->constraint_matrix->col_ind, orig->constraint_matrix->col_ind, (size_t)a_nnz_old * sizeof(int));
        memcpy(out->constraint_matrix->val, orig->constraint_matrix->val, (size_t)a_nnz_old * sizeof(double));
    }
    int dst = a_nnz_old;
    for (int j = 0; j < n_old; ++j)
    {
        double cj = orig->objective_vector[j];
        if (cj == 0.0)
            continue;
        out->constraint_matrix->col_ind[dst] = j;
        out->constraint_matrix->val[dst] = cj;
        dst++;
    }
    out->constraint_matrix->col_ind[dst] = eta_col;
    out->constraint_matrix->val[dst] = -1.0;

    out->objective_sparse_matrix = (CsrComponent *)xcalloc(1, sizeof(CsrComponent));
    out->objective_sparse_matrix->row_ptr = (int *)xcalloc((size_t)n_new + 1, sizeof(int));
    out->objective_lowrank_matrix = (CsrComponent *)xcalloc(1, sizeof(CsrComponent));
    out->objective_lowrank_matrix->row_ptr = (int *)xcalloc(1, sizeof(int));
    out->objective_lowrank_middle_matrix = NULL;

    const int k_old = orig->num_quadratic_constraints;
    const int k_new = k_old + 1;
    out->num_quadratic_constraints = k_new;
    out->quadratic_constraint_row_indices = (int *)xcalloc((size_t)k_new, sizeof(int));
    out->quadratic_constraint_matrices = (CsrComponent **)xcalloc((size_t)k_new, sizeof(CsrComponent *));
    out->quadratic_constraint_matrix_num_nonzeros = (int *)xcalloc((size_t)k_new, sizeof(int));
    for (int k = 0; k < k_old; ++k)
    {
        out->quadratic_constraint_row_indices[k] = orig->quadratic_constraint_row_indices[k];
        out->quadratic_constraint_matrix_num_nonzeros[k] = orig->quadratic_constraint_matrix_num_nonzeros[k];
        out->quadratic_constraint_matrices[k] = copy_csr_with_rows(orig->quadratic_constraint_matrices[k],
                                                                   n_old,
                                                                   n_new,
                                                                   orig->quadratic_constraint_matrix_num_nonzeros[k],
                                                                   1.0);
    }
    out->quadratic_constraint_row_indices[k_old] = m_old;
    out->quadratic_constraint_matrix_num_nonzeros[k_old] = orig->objective_sparse_matrix_num_nonzeros;
    out->quadratic_constraint_matrices[k_old] = copy_csr_with_rows(
        orig->objective_sparse_matrix, n_old, n_new, orig->objective_sparse_matrix_num_nonzeros, 0.5);

    out->cones.num_cones = 0;
    out->cones.start_idx = NULL;
    out->cones.v_dim = NULL;
    out->cones.type = NULL;
    out->cones.is_fixed = NULL;
    out->primal_start = NULL;
    out->dual_start = NULL;
    return out;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s INPUT.mps[.gz] OUTPUT.bin [rotated|standard] [epigraph]\n", argv[0]);
        return 2;
    }
    const char *input = argv[1];
    const char *output = argv[2];
    cone_type_t form = CONE_ROTATED_SOC;
    int epigraph_objective = 0;
    for (int i = 3; i < argc; ++i)
    {
        if (strcmp(argv[i], "standard") == 0)
            form = CONE_STANDARD_SOC;
        else if (strcmp(argv[i], "rotated") == 0)
            form = CONE_ROTATED_SOC;
        else if (strcmp(argv[i], "epigraph") == 0 || strcmp(argv[i], "--epigraph-objective") == 0 ||
                 strcmp(argv[i], "epigraph-objective") == 0)
            epigraph_objective = 1;
        else
        {
            fprintf(stderr, "unknown option '%s'\n", argv[i]);
            return 2;
        }
    }

    qp_problem_t *orig = read_mps_file(input);
    if (!orig)
    {
        fprintf(stderr, "read_mps_file failed: %s\n", input);
        return 1;
    }
    qp_problem_t *base = orig;
    qp_problem_t *epig = NULL;
    int epigraph_done = 0;
    if (epigraph_objective && orig->objective_sparse_matrix_num_nonzeros > 0)
    {
        epig = epigraph_objective_q_to_qc(orig);
        if (!epig)
        {
            qp_problem_free(orig);
            fprintf(stderr, "objective epigraph failed: %s\n", input);
            return 1;
        }
        base = epig;
        epigraph_done = 1;
    }

    qp_problem_t *prob = base;
    int transformed = 0;
    if (base->num_quadratic_constraints > 0)
    {
        prob = qcqp_to_socp_qp(base, form);
        if (!prob)
        {
            if (epig)
                qp_problem_free(epig);
            qp_problem_free(orig);
            fprintf(stderr, "qcqp_to_socp_qp failed: %s\n", input);
            return 1;
        }
        transformed = 1;
    }

    FILE *f = fopen(output, "wb");
    if (!f)
    {
        perror(output);
        if (transformed)
            qp_problem_free(prob);
        if (epig)
            qp_problem_free(epig);
        qp_problem_free(orig);
        return 1;
    }

    const char magic[8] = {'P', 'D', 'H', 'Q', 'C', 'Q', '1', '\0'};
    int32_t header[18];
    memset(header, 0, sizeof(header));
    header[0] = 1;
    header[1] = prob->num_variables;
    header[2] = prob->num_constraints;
    header[3] = prob->constraint_matrix_num_nonzeros;
    header[4] = prob->objective_sparse_matrix_num_nonzeros;
    header[5] = prob->cones.num_cones;
    header[6] = prob->num_original_variables;
    header[7] = transformed;
    header[8] = orig->num_variables;
    header[9] = orig->num_constraints;
    header[10] = orig->constraint_matrix_num_nonzeros;
    header[11] = orig->objective_sparse_matrix_num_nonzeros;
    header[12] = orig->num_quadratic_constraints;
    header[13] = one_if_nonnull(prob->cones.is_fixed);
    header[14] = one_if_nonnull(prob->primal_start);
    header[15] = (int32_t)form;
    header[16] = prob->num_rank_lowrank_obj;
    header[17] = prob->objective_lowrank_matrix_num_nonzeros + prob->objective_lowrank_middle_matrix_num_nonzeros;

    int rc = 0;
    rc |= write_all(f, magic, sizeof(char), sizeof(magic));
    rc |= write_all(f, header, sizeof(int32_t), 18);
    rc |= write_all(f, &prob->objective_constant, sizeof(double), 1);
    rc |= write_all(f, prob->objective_vector, sizeof(double), (size_t)prob->num_variables);
    rc |= write_all(f, prob->variable_lower_bound, sizeof(double), (size_t)prob->num_variables);
    rc |= write_all(f, prob->variable_upper_bound, sizeof(double), (size_t)prob->num_variables);
    rc |= write_all(f, prob->constraint_lower_bound, sizeof(double), (size_t)prob->num_constraints);
    rc |= write_all(f, prob->constraint_upper_bound, sizeof(double), (size_t)prob->num_constraints);
    rc |= write_csr(f, prob->constraint_matrix, prob->num_constraints, prob->constraint_matrix_num_nonzeros);
    rc |= write_csr(f, prob->objective_sparse_matrix, prob->num_variables, prob->objective_sparse_matrix_num_nonzeros);
    rc |= write_all(f, prob->cones.start_idx, sizeof(int32_t), (size_t)prob->cones.num_cones);
    rc |= write_all(f, prob->cones.v_dim, sizeof(int32_t), (size_t)prob->cones.num_cones);
    rc |= write_all(f, prob->cones.type, sizeof(int32_t), (size_t)prob->cones.num_cones);
    if (prob->cones.is_fixed)
        rc |= write_all(f, prob->cones.is_fixed, sizeof(char), (size_t)prob->num_variables);
    if (prob->primal_start)
        rc |= write_all(f, prob->primal_start, sizeof(double), (size_t)prob->num_variables);

    if (fclose(f) != 0)
        rc = -1;

    fprintf(stderr,
            "{\"input\":\"%s\",\"output\":\"%s\",\"n\":%d,\"m\":%d,\"A_nnz\":%d,"
            "\"Q_nnz\":%d,\"cones\":%d,\"orig_n\":%d,\"orig_m\":%d,"
            "\"orig_qc\":%d,\"orig_Q_nnz\":%d,\"epigraph_objective\":%d,\"lowrank_nnz\":%d}\n",
            input,
            output,
            prob->num_variables,
            prob->num_constraints,
            prob->constraint_matrix_num_nonzeros,
            prob->objective_sparse_matrix_num_nonzeros,
            prob->cones.num_cones,
            orig->num_variables,
            orig->num_constraints,
            orig->num_quadratic_constraints,
            orig->objective_sparse_matrix_num_nonzeros,
            epigraph_done,
            header[17]);

    if (transformed)
        qp_problem_free(prob);
    if (epig)
        qp_problem_free(epig);
    qp_problem_free(orig);
    return rc == 0 ? 0 : 1;
}
