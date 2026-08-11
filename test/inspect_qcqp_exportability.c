#include "mps_parser.h"
#include "pdhcg.h"
#include "pdhcg_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int extract_diag_signed(const CsrComponent *Q, int n, int nnz_max, int *sign_out)
{
    int count = 0;
    int sign = 0;
    if (!Q || !Q->row_ptr)
        return -1;
    for (int row = 0; row < n; ++row)
    {
        int start = Q->row_ptr[row];
        int end = Q->row_ptr[row + 1];
        for (int k = start; k < end; ++k)
        {
            int col = Q->col_ind[k];
            double val = Q->val[k];
            if (col != row)
                return -1;
            if (val == 0.0)
                continue;
            int s = (val > 0.0) ? +1 : -1;
            if (sign == 0)
                sign = s;
            else if (sign != s)
                return -1;
            if (count >= nnz_max)
                return -1;
            count++;
        }
    }
    *sign_out = sign;
    return count;
}

static long count_nonzero_obj(const qp_problem_t *p)
{
    long nnz = 0;
    for (int j = 0; j < p->num_variables; ++j)
        if (p->objective_vector[j] != 0.0)
            nnz++;
    return nnz;
}

static int inspect_qc(const qp_problem_t *p,
                      int idx,
                      int base_n,
                      int base_m,
                      int base_a_nnz,
                      long *total_v,
                      long *num_pin,
                      int *first_bad,
                      const char **bad_reason)
{
    const int orig_k = p->num_quadratic_constraints;
    const int has_obj_q = p->objective_sparse_matrix_num_nonzeros > 0;
    const int is_obj_q = (has_obj_q && idx == orig_k);
    CsrComponent *Q = is_obj_q ? p->objective_sparse_matrix : p->quadratic_constraint_matrices[idx];
    int q_nnz = is_obj_q ? p->objective_sparse_matrix_num_nonzeros : p->quadratic_constraint_matrix_num_nonzeros[idx];
    int sign = 0;
    int diag_nnz = extract_diag_signed(Q, p->num_variables, q_nnz, &sign);
    if (diag_nnz < 0)
    {
        *first_bad = idx;
        *bad_reason = "non_diagonal_or_mixed_sign";
        return -1;
    }

    int row = is_obj_q ? p->num_constraints : p->quadratic_constraint_row_indices[idx];
    double lhs = is_obj_q ? -INFINITY : p->constraint_lower_bound[row];
    double rhs = is_obj_q ? -p->objective_constant : p->constraint_upper_bound[row];
    if (sign >= 0)
    {
        if (isfinite(lhs) || !isfinite(rhs))
        {
            *first_bad = idx;
            *bad_reason = "psd_requires_le_constraint";
            return -1;
        }
    }
    else
    {
        if (!isfinite(lhs) || isfinite(rhs))
        {
            *first_bad = idx;
            *bad_reason = "nsd_requires_ge_constraint";
            return -1;
        }
    }

    int row_nnz = 0;
    if (is_obj_q)
        row_nnz = (int)count_nonzero_obj(p) + 1;
    else
        row_nnz = p->constraint_matrix->row_ptr[row + 1] - p->constraint_matrix->row_ptr[row];
    if (row_nnz == 0)
        (*num_pin)++;
    (void)base_n;
    (void)base_m;
    (void)base_a_nnz;
    *total_v += diag_nnz;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s INPUT.mps[.gz]\n", argv[0]);
        return 2;
    }
    qp_problem_t *p = read_mps_file(argv[1]);
    if (!p)
    {
        fprintf(stderr, "read_mps_file failed: %s\n", argv[1]);
        return 1;
    }

    int has_obj_q = p->objective_sparse_matrix_num_nonzeros > 0;
    int base_n = p->num_variables + (has_obj_q ? 1 : 0);
    int base_m = p->num_constraints + (has_obj_q ? 1 : 0);
    long obj_lin_nnz = has_obj_q ? count_nonzero_obj(p) : 0;
    long base_a_nnz = p->constraint_matrix_num_nonzeros + (has_obj_q ? obj_lin_nnz + 1 : 0);
    int K = p->num_quadratic_constraints + (has_obj_q ? 1 : 0);

    long total_v = 0;
    long num_pin = 0;
    int first_bad = -1;
    const char *bad_reason = "";
    for (int i = 0; i < K; ++i)
    {
        if (inspect_qc(p, i, base_n, base_m, (int)base_a_nnz, &total_v, &num_pin, &first_bad, &bad_reason) != 0)
            break;
    }

    long n_ext = (long)base_n + total_v + 2L * K;
    long m_ext = (long)base_m + total_v + K;
    long a_ext = base_a_nnz + 2L * (K - num_pin) + 2L * total_v;
    long bytes = 8 + 72 + 8 + 8 * n_ext * 3 + 8 * m_ext * 2 + 4 * (m_ext + 1) + 12 * a_ext + 4 * (n_ext + 1) + 12 * 0 +
        12 * K + n_ext + 8 * n_ext;

    printf("{\"input\":\"%s\",\"n\":%d,\"m\":%d,\"A_nnz\":%d,\"obj_Q_nnz\":%d,"
           "\"qc\":%d,\"base_n\":%d,\"base_m\":%d,\"base_A_nnz\":%ld,"
           "\"supported\":%s,\"first_bad_qc\":%d,\"bad_reason\":\"%s\","
           "\"total_v\":%ld,\"num_pin\":%ld,\"n_ext_est\":%ld,\"m_ext_est\":%ld,"
           "\"A_nnz_ext_est\":%ld,\"bin_bytes_est\":%ld}\n",
           argv[1],
           p->num_variables,
           p->num_constraints,
           p->constraint_matrix_num_nonzeros,
           p->objective_sparse_matrix_num_nonzeros,
           K,
           base_n,
           base_m,
           base_a_nnz,
           first_bad < 0 ? "true" : "false",
           first_bad,
           bad_reason,
           total_v,
           num_pin,
           n_ext,
           m_ext,
           a_ext,
           bytes);

    qp_problem_free(p);
    return first_bad < 0 ? 0 : 3;
}
