#include "mps_parser.h"
#include "pdhcg.h"
#include "pdhcg_types.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int diag_signed_count(const CsrComponent *Q, int n, int *sign_out)
{
    int count = 0;
    int sign = 0;
    if (!Q || !Q->row_ptr)
        return -1;
    for (int row = 0; row < n; ++row)
    {
        for (int k = Q->row_ptr[row]; k < Q->row_ptr[row + 1]; ++k)
        {
            int col = Q->col_ind[k];
            double val = Q->val[k];
            if (col != row)
                return -1;
            if (val == 0.0)
                continue;
            int s = (val > 0.0) ? 1 : -1;
            if (sign == 0)
                sign = s;
            else if (sign != s)
                return -1;
            count++;
        }
    }
    *sign_out = sign;
    return count;
}

static int row_nnz(const CsrComponent *A, int row)
{
    if (!A || !A->row_ptr)
        return 0;
    return A->row_ptr[row + 1] - A->row_ptr[row];
}

static int objective_linear_nnz(const qp_problem_t *p)
{
    int nnz = 0;
    for (int j = 0; j < p->num_variables; ++j)
        if (p->objective_vector && p->objective_vector[j] != 0.0)
            nnz++;
    return nnz;
}

static void inspect_qc(const qp_problem_t *p,
                       const CsrComponent *Q,
                       int q_nnz,
                       int row,
                       int is_epigraph_obj,
                       long long *total_v,
                       int *num_pin,
                       int *unsupported_diag,
                       int *unsupported_bound)
{
    (void)q_nnz;
    int sign = 0;
    int k = diag_signed_count(Q, p->num_variables + (is_epigraph_obj ? 1 : 0), &sign);
    if (k < 0)
    {
        (*unsupported_diag)++;
        return;
    }

    double lhs = is_epigraph_obj ? -INFINITY : p->constraint_lower_bound[row];
    double rhs = is_epigraph_obj ? -p->objective_constant : p->constraint_upper_bound[row];
    if (sign >= 0)
    {
        if (isfinite(lhs) || !isfinite(rhs))
            (*unsupported_bound)++;
    }
    else
    {
        if (!isfinite(lhs) || isfinite(rhs))
            (*unsupported_bound)++;
    }

    *total_v += k;
    if (!is_epigraph_obj && row_nnz(p->constraint_matrix, row) == 0)
        (*num_pin)++;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: %s INPUT.mps[.gz] [epigraph]\n", argv[0]);
        return 2;
    }
    int epigraph = argc >= 3;
    qp_problem_t *p = read_mps_file(argv[1]);
    if (!p)
    {
        fprintf(stderr, "read_mps_file failed\n");
        return 1;
    }

    long long total_v = 0;
    int num_pin = 0;
    int unsupported_diag = 0;
    int unsupported_bound = 0;
    for (int i = 0; i < p->num_quadratic_constraints; ++i)
    {
        inspect_qc(p,
                   p->quadratic_constraint_matrices[i],
                   p->quadratic_constraint_matrix_num_nonzeros[i],
                   p->quadratic_constraint_row_indices[i],
                   0,
                   &total_v,
                   &num_pin,
                   &unsupported_diag,
                   &unsupported_bound);
    }

    int add_obj_q = epigraph && p->objective_sparse_matrix_num_nonzeros > 0;
    if (add_obj_q)
    {
        int sign = 0;
        int k = diag_signed_count(p->objective_sparse_matrix, p->num_variables, &sign);
        if (k < 0)
            unsupported_diag++;
        else
        {
            if (sign < 0)
                unsupported_bound++;
            total_v += k;
        }
    }

    long long n_base = (long long)p->num_variables + (add_obj_q ? 1 : 0);
    long long m_base = (long long)p->num_constraints + (add_obj_q ? 1 : 0);
    long long a_base = p->constraint_matrix_num_nonzeros;
    if (add_obj_q)
        a_base += objective_linear_nnz(p) + 1;
    int K = p->num_quadratic_constraints + add_obj_q;
    long long n_ext = n_base + total_v + 2LL * K;
    long long m_ext = m_base + total_v + K;
    long long nnz_ext = a_base + 2LL * total_v + 2LL * (K - num_pin);

    printf("{\"input\":\"%s\",\"n\":%d,\"m\":%d,\"A_nnz\":%d,"
           "\"obj_Q_nnz\":%d,\"lowrank_nnz\":%d,\"qc\":%d,"
           "\"epigraph_obj\":%d,\"K_eff\":%d,\"total_v\":%lld,"
           "\"num_pin\":%d,\"unsupported_diag\":%d,\"unsupported_bound\":%d,"
           "\"n_ext_est\":%lld,\"m_ext_est\":%lld,\"A_nnz_ext_est\":%lld}\n",
           argv[1],
           p->num_variables,
           p->num_constraints,
           p->constraint_matrix_num_nonzeros,
           p->objective_sparse_matrix_num_nonzeros,
           p->objective_lowrank_matrix_num_nonzeros + p->objective_lowrank_middle_matrix_num_nonzeros,
           p->num_quadratic_constraints,
           add_obj_q,
           K,
           total_v,
           num_pin,
           unsupported_diag,
           unsupported_bound,
           n_ext,
           m_ext,
           nnz_ext);

    qp_problem_free(p);
    return 0;
}
