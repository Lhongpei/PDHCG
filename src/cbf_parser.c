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

/* MOSEK Conic Benchmark Format (CBF) parser.
   Supports variable-side cones L=, L+, L-, F, Q, QR, EXP;
   constraint-side cones L=, L+, L-, Q, QR, EXP.
   Nonlinear constraint-side cones are converted to auxiliary cone variables
   y in K plus linear equalities A x - y = -b.
   Rejects: PSDVAR, PSDCON, HCOORD, DCOORD, FCOORD, OBJFCOORD, INT,
   POW, POW*, EXP*, CHANGE blocks. */

#include "cbf_parser.h"
#include "utils.h"
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define LINE_BUF_SIZE 8192

typedef enum
{
    CBF_CONE_FREE = 0, /* F  : free domain */
    CBF_CONE_ZERO,     /* L= : fixed at zero */
    CBF_CONE_LPOS,     /* L+ : x >= 0 */
    CBF_CONE_LNEG,     /* L- : x <= 0 */
    CBF_CONE_SOC,      /* Q  : standard SOC, x[0] >= ||x[1:]|| */
    CBF_CONE_RSOC,     /* QR : rotated SOC, 2*x[0]*x[1] >= ||x[2:]||^2 */
    CBF_CONE_EXP,      /* EXP: x[0] >= x[1] * exp(x[2]/x[1]), x[1] > 0 */
} cbf_cone_t;

typedef struct
{
    cbf_cone_t type;
    int dim;
    int start; /* start index in original CBF ordering */
} cbf_block_t;

typedef struct
{
    bool is_gz;
    gzFile gz;
    FILE *fp;
    char *buf;
    int line_no;
} cbf_reader_t;

static cbf_reader_t *cbf_open(const char *filename)
{
    cbf_reader_t *r = (cbf_reader_t *)safe_calloc(1, sizeof(cbf_reader_t));
    r->buf = (char *)safe_malloc(LINE_BUF_SIZE);
    size_t n = strlen(filename);
    if (n > 3 && strcmp(filename + n - 3, ".gz") == 0)
    {
        r->is_gz = true;
        r->gz = gzopen(filename, "rb");
        if (!r->gz)
        {
            free(r->buf);
            free(r);
            return NULL;
        }
    }
    else
    {
        r->is_gz = false;
        r->fp = fopen(filename, "r");
        if (!r->fp)
        {
            free(r->buf);
            free(r);
            return NULL;
        }
    }
    return r;
}

static void cbf_close(cbf_reader_t *r)
{
    if (!r)
        return;
    if (r->is_gz && r->gz)
        gzclose(r->gz);
    if (!r->is_gz && r->fp)
        fclose(r->fp);
    free(r->buf);
    free(r);
}

/* Reads one raw line into r->buf. Returns NULL on EOF. */
static char *cbf_getline_raw(cbf_reader_t *r)
{
    char *s = r->is_gz ? gzgets(r->gz, r->buf, LINE_BUF_SIZE) : fgets(r->buf, LINE_BUF_SIZE, r->fp);
    if (!s)
        return NULL;
    r->line_no++;
    return s;
}

/* Reads the next non-blank, non-comment line. Strips trailing whitespace.
   Returns NULL on EOF. */
static char *cbf_next_line(cbf_reader_t *r)
{
    while (1)
    {
        char *s = cbf_getline_raw(r);
        if (!s)
            return NULL;
        char *p = s;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        /* Strip trailing whitespace / newline */
        size_t len = strlen(s);
        while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r' || s[len - 1] == ' ' || s[len - 1] == '\t'))
            s[--len] = '\0';
        return s;
    }
}

static bool cbf_parse_cone_name(const char *tok, cbf_cone_t *out)
{
    if (strcmp(tok, "F") == 0)
    {
        *out = CBF_CONE_FREE;
        return true;
    }
    if (strcmp(tok, "L=") == 0)
    {
        *out = CBF_CONE_ZERO;
        return true;
    }
    if (strcmp(tok, "L+") == 0)
    {
        *out = CBF_CONE_LPOS;
        return true;
    }
    if (strcmp(tok, "L-") == 0)
    {
        *out = CBF_CONE_LNEG;
        return true;
    }
    if (strcmp(tok, "Q") == 0)
    {
        *out = CBF_CONE_SOC;
        return true;
    }
    if (strcmp(tok, "QR") == 0)
    {
        *out = CBF_CONE_RSOC;
        return true;
    }
    if (strcmp(tok, "EXP") == 0)
    {
        *out = CBF_CONE_EXP;
        return true;
    }
    return false;
}

static bool cbf_is_lp_cone(cbf_cone_t ct)
{
    return ct == CBF_CONE_FREE || ct == CBF_CONE_ZERO || ct == CBF_CONE_LPOS || ct == CBF_CONE_LNEG;
}

static bool cbf_is_nonlinear_cone(cbf_cone_t ct)
{
    return ct == CBF_CONE_SOC || ct == CBF_CONE_RSOC || ct == CBF_CONE_EXP;
}

static int cbf_internal_cone_slots(cbf_cone_t ct, int dim)
{
    if (ct == CBF_CONE_SOC)
        return dim;
    if (ct == CBF_CONE_RSOC)
        return dim;
    if (ct == CBF_CONE_EXP)
        return 3;
    return dim;
}

static int cbf_internal_cone_v_dim(cbf_cone_t ct, int dim)
{
    if (ct == CBF_CONE_SOC)
        return dim - 2;
    if (ct == CBF_CONE_RSOC)
        return dim - 2;
    return 1; /* EXP */
}

static cone_type_t cbf_internal_cone_type(cbf_cone_t ct)
{
    if (ct == CBF_CONE_SOC)
        return CONE_STANDARD_SOC;
    if (ct == CBF_CONE_RSOC)
        return CONE_ROTATED_SOC;
    return CONE_EXPONENTIAL;
}

static int cbf_map_cone_component(cbf_cone_t ct, int dim, int base, int local_idx)
{
    if (ct == CBF_CONE_SOC)
    {
        /* CBF Q: (t, v...). Internal SOC: [v..., w, z]. */
        if (local_idx == 0)
            return base + dim - 1;
        if (local_idx == dim - 1)
            return base + dim - 2;
        return base + local_idx - 1;
    }
    if (ct == CBF_CONE_RSOC)
    {
        /* CBF QR: (s, t, v...). Internal RSOC: [v..., s, t]. */
        int vd = dim - 2;
        if (local_idx == 0)
            return base + vd;
        if (local_idx == 1)
            return base + vd + 1;
        return base + local_idx - 2;
    }
    /* CBF EXP: (x0, x1, x2), internal: (r1=x2, r2=x1, r3=x0). */
    return base + (2 - local_idx);
}

/* Read n blocks of "CONE_NAME dim" pairs. Rejects unsupported cones. */
static cbf_block_t *cbf_read_blocks(cbf_reader_t *r, int nblk, int *total_out, const char *ctx)
{
    cbf_block_t *blk = (cbf_block_t *)safe_malloc(nblk * sizeof(cbf_block_t));
    int total = 0;
    for (int i = 0; i < nblk; ++i)
    {
        char *ln = cbf_next_line(r);
        if (!ln)
        {
            fprintf(stderr, "[cbf] %s: unexpected EOF in cone list at block %d\n", ctx, i);
            free(blk);
            return NULL;
        }
        char cone_name[16];
        int dim = 0;
        if (sscanf(ln, "%15s %d", cone_name, &dim) != 2 || dim <= 0)
        {
            fprintf(stderr, "[cbf] %s: bad cone line '%s'\n", ctx, ln);
            free(blk);
            return NULL;
        }
        cbf_cone_t ct;
        if (!cbf_parse_cone_name(cone_name, &ct))
        {
            fprintf(stderr, "[cbf] %s: unsupported cone '%s' (need L=/L+/L-/F/Q/QR/EXP)\n", ctx, cone_name);
            free(blk);
            return NULL;
        }
        if (ct == CBF_CONE_SOC && dim < 2)
        {
            fprintf(stderr, "[cbf] %s: Q dim must be >= 2, got %d\n", ctx, dim);
            free(blk);
            return NULL;
        }
        if (ct == CBF_CONE_RSOC && dim < 3)
        {
            fprintf(stderr, "[cbf] %s: QR dim must be >= 3, got %d\n", ctx, dim);
            free(blk);
            return NULL;
        }
        if (ct == CBF_CONE_EXP && dim != 3)
        {
            fprintf(stderr, "[cbf] %s: EXP dim must be 3, got %d\n", ctx, dim);
            free(blk);
            return NULL;
        }
        blk[i].type = ct;
        blk[i].dim = dim;
        blk[i].start = total;
        total += dim;
    }
    *total_out = total;
    return blk;
}

/* Skip an unsupported block of `nnz` coord lines. */
static bool cbf_skip_lines(cbf_reader_t *r, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (!cbf_next_line(r))
        {
            fprintf(stderr, "[cbf] EOF while skipping block (line %d of %d)\n", i, n);
            return false;
        }
    }
    return true;
}

typedef struct
{
    /* Header */
    int ver;
    int objsense_neg; /* 1 if MAX (negate objective) */

    /* Variable cones (CBF ordering) */
    cbf_block_t *var_blocks;
    int num_var_blocks;
    int num_vars;

    /* Constraint cones (CBF ordering) */
    cbf_block_t *con_blocks;
    int num_con_blocks;
    int num_cons;

    /* Objective */
    double obj_constant;
    double *obj_c; /* [num_vars] linear coefficients (CBF ordering) */

    /* Constraint matrix in COO (row/col/val) — CBF ordering */
    int nnz_A;
    int cap_A;
    int *A_row;
    int *A_col;
    double *A_val;

    /* b vector for CBF Ax + b ∈ K_con (CBF ordering) */
    double *b;

    /* OBJQCOORD symmetric fill; objective is 0.5 * x^T Q x. CBF variable ordering. */
    int nnz_Q;
    int cap_Q;
    int *Q_row;
    int *Q_col;
    double *Q_val;
} cbf_state_t;

static void cbf_state_free(cbf_state_t *s)
{
    free(s->var_blocks);
    free(s->con_blocks);
    free(s->obj_c);
    free(s->A_row);
    free(s->A_col);
    free(s->A_val);
    free(s->b);
    free(s->Q_row);
    free(s->Q_col);
    free(s->Q_val);
}

static bool cbf_read_ver(cbf_reader_t *r, cbf_state_t *s)
{
    char *ln = cbf_next_line(r);
    if (!ln)
    {
        fprintf(stderr, "[cbf] EOF after VER header\n");
        return false;
    }
    if (sscanf(ln, "%d", &s->ver) != 1)
    {
        fprintf(stderr, "[cbf] bad VER line '%s'\n", ln);
        return false;
    }
    return true;
}

static bool cbf_read_objsense(cbf_reader_t *r, cbf_state_t *s)
{
    char *ln = cbf_next_line(r);
    if (!ln)
    {
        fprintf(stderr, "[cbf] EOF after OBJSENSE header\n");
        return false;
    }
    char sense[16];
    if (sscanf(ln, "%15s", sense) != 1)
    {
        fprintf(stderr, "[cbf] bad OBJSENSE '%s'\n", ln);
        return false;
    }
    if (strcmp(sense, "MIN") == 0)
        s->objsense_neg = 0;
    else if (strcmp(sense, "MAX") == 0)
        s->objsense_neg = 1;
    else
    {
        fprintf(stderr, "[cbf] OBJSENSE must be MIN or MAX, got '%s'\n", sense);
        return false;
    }
    return true;
}

static bool cbf_read_var(cbf_reader_t *r, cbf_state_t *s)
{
    if (s->var_blocks)
    {
        fprintf(stderr, "[cbf] duplicate VAR block\n");
        return false;
    }
    char *ln = cbf_next_line(r);
    if (!ln)
        return false;
    int n = 0, k = 0;
    if (sscanf(ln, "%d %d", &n, &k) != 2 || n < 0 || k < 0)
    {
        fprintf(stderr, "[cbf] bad VAR header '%s'\n", ln);
        return false;
    }
    int total = 0;
    s->var_blocks = cbf_read_blocks(r, k, &total, "VAR");
    if (!s->var_blocks)
        return false;
    if (total != n)
    {
        fprintf(stderr, "[cbf] VAR cone sum %d != n=%d\n", total, n);
        return false;
    }
    s->num_var_blocks = k;
    s->num_vars = n;
    s->obj_c = (double *)safe_calloc(n, sizeof(double));
    return true;
}

static bool cbf_read_con(cbf_reader_t *r, cbf_state_t *s)
{
    if (s->con_blocks)
    {
        fprintf(stderr, "[cbf] duplicate CON block\n");
        return false;
    }
    char *ln = cbf_next_line(r);
    if (!ln)
        return false;
    int m = 0, k = 0;
    if (sscanf(ln, "%d %d", &m, &k) != 2 || m < 0 || k < 0)
    {
        fprintf(stderr, "[cbf] bad CON header '%s'\n", ln);
        return false;
    }
    int total = 0;
    s->con_blocks = cbf_read_blocks(r, k, &total, "CON");
    if (!s->con_blocks)
        return false;
    if (total != m)
    {
        fprintf(stderr, "[cbf] CON cone sum %d != m=%d\n", total, m);
        return false;
    }
    s->num_con_blocks = k;
    s->num_cons = m;
    s->b = (double *)safe_calloc(m, sizeof(double));
    return true;
}

static bool cbf_read_objacoord(cbf_reader_t *r, cbf_state_t *s)
{
    char *ln = cbf_next_line(r);
    if (!ln)
        return false;
    int nnz = 0;
    if (sscanf(ln, "%d", &nnz) != 1 || nnz < 0)
    {
        fprintf(stderr, "[cbf] bad OBJACOORD header '%s'\n", ln);
        return false;
    }
    if (!s->obj_c)
    {
        fprintf(stderr, "[cbf] OBJACOORD before VAR\n");
        return false;
    }
    for (int i = 0; i < nnz; ++i)
    {
        ln = cbf_next_line(r);
        if (!ln)
        {
            fprintf(stderr, "[cbf] OBJACOORD truncated at %d/%d\n", i, nnz);
            return false;
        }
        int col;
        double val;
        if (sscanf(ln, "%d %lf", &col, &val) != 2)
        {
            fprintf(stderr, "[cbf] bad OBJACOORD entry '%s'\n", ln);
            return false;
        }
        if (col < 0 || col >= s->num_vars)
        {
            fprintf(stderr, "[cbf] OBJACOORD col %d out of range [0,%d)\n", col, s->num_vars);
            return false;
        }
        s->obj_c[col] += val;
    }
    return true;
}

static bool cbf_read_objbcoord(cbf_reader_t *r, cbf_state_t *s)
{
    char *ln = cbf_next_line(r);
    if (!ln)
        return false;
    double val;
    if (sscanf(ln, "%lf", &val) != 1)
    {
        fprintf(stderr, "[cbf] bad OBJBCOORD '%s'\n", ln);
        return false;
    }
    s->obj_constant += val;
    return true;
}

static void cbf_reserve_Q(cbf_state_t *s, int need)
{
    if (s->cap_Q >= need)
        return;
    int new_cap = (s->cap_Q > 0) ? s->cap_Q : 16;
    while (new_cap < need)
        new_cap *= 2;
    s->Q_row = (int *)safe_realloc(s->Q_row, (size_t)new_cap * sizeof(int));
    s->Q_col = (int *)safe_realloc(s->Q_col, (size_t)new_cap * sizeof(int));
    s->Q_val = (double *)safe_realloc(s->Q_val, (size_t)new_cap * sizeof(double));
    s->cap_Q = new_cap;
}

/* OBJQCOORD: nnz lines of "i j val". Off-diagonal entries are symmetric-filled
   into (i,j) and (j,i). Objective is 0.5 * x^T Q x + c^T x + f. */
static bool cbf_read_objqcoord(cbf_reader_t *r, cbf_state_t *s)
{
    char *ln = cbf_next_line(r);
    if (!ln)
        return false;
    int nnz = 0;
    if (sscanf(ln, "%d", &nnz) != 1 || nnz < 0)
    {
        fprintf(stderr, "[cbf] bad OBJQCOORD header '%s'\n", ln);
        return false;
    }
    if (!s->var_blocks)
    {
        fprintf(stderr, "[cbf] OBJQCOORD before VAR\n");
        return false;
    }
    /* Worst case: all off-diagonal, so up to 2*nnz internal entries. */
    cbf_reserve_Q(s, s->nnz_Q + 2 * nnz);
    for (int k = 0; k < nnz; ++k)
    {
        ln = cbf_next_line(r);
        if (!ln)
        {
            fprintf(stderr, "[cbf] OBJQCOORD truncated at %d/%d\n", k, nnz);
            return false;
        }
        int i, j;
        double val;
        if (sscanf(ln, "%d %d %lf", &i, &j, &val) != 3)
        {
            fprintf(stderr, "[cbf] bad OBJQCOORD entry '%s'\n", ln);
            return false;
        }
        if (i < 0 || i >= s->num_vars || j < 0 || j >= s->num_vars)
        {
            fprintf(stderr, "[cbf] OBJQCOORD entry (%d,%d) out of range\n", i, j);
            return false;
        }
        if (i == j)
        {
            s->Q_row[s->nnz_Q] = i;
            s->Q_col[s->nnz_Q] = j;
            s->Q_val[s->nnz_Q] = val;
            s->nnz_Q++;
        }
        else
        {
            s->Q_row[s->nnz_Q] = i;
            s->Q_col[s->nnz_Q] = j;
            s->Q_val[s->nnz_Q] = val;
            s->nnz_Q++;
            s->Q_row[s->nnz_Q] = j;
            s->Q_col[s->nnz_Q] = i;
            s->Q_val[s->nnz_Q] = val;
            s->nnz_Q++;
        }
    }
    return true;
}

static void cbf_reserve_A(cbf_state_t *s, int need)
{
    if (s->cap_A >= need)
        return;
    int new_cap = (s->cap_A > 0) ? s->cap_A : 16;
    while (new_cap < need)
        new_cap *= 2;
    s->A_row = (int *)safe_realloc(s->A_row, (size_t)new_cap * sizeof(int));
    s->A_col = (int *)safe_realloc(s->A_col, (size_t)new_cap * sizeof(int));
    s->A_val = (double *)safe_realloc(s->A_val, (size_t)new_cap * sizeof(double));
    s->cap_A = new_cap;
}

static bool cbf_read_acoord(cbf_reader_t *r, cbf_state_t *s)
{
    char *ln = cbf_next_line(r);
    if (!ln)
        return false;
    int nnz = 0;
    if (sscanf(ln, "%d", &nnz) != 1 || nnz < 0)
    {
        fprintf(stderr, "[cbf] bad ACOORD header '%s'\n", ln);
        return false;
    }
    if (!s->con_blocks || !s->var_blocks)
    {
        fprintf(stderr, "[cbf] ACOORD before VAR/CON\n");
        return false;
    }
    cbf_reserve_A(s, s->nnz_A + nnz);
    for (int i = 0; i < nnz; ++i)
    {
        ln = cbf_next_line(r);
        if (!ln)
        {
            fprintf(stderr, "[cbf] ACOORD truncated at %d/%d\n", i, nnz);
            return false;
        }
        int row, col;
        double val;
        if (sscanf(ln, "%d %d %lf", &row, &col, &val) != 3)
        {
            fprintf(stderr, "[cbf] bad ACOORD entry '%s'\n", ln);
            return false;
        }
        if (row < 0 || row >= s->num_cons || col < 0 || col >= s->num_vars)
        {
            fprintf(stderr, "[cbf] ACOORD entry (%d,%d) out of range\n", row, col);
            return false;
        }
        s->A_row[s->nnz_A] = row;
        s->A_col[s->nnz_A] = col;
        s->A_val[s->nnz_A] = val;
        s->nnz_A++;
    }
    return true;
}

static bool cbf_read_bcoord(cbf_reader_t *r, cbf_state_t *s)
{
    char *ln = cbf_next_line(r);
    if (!ln)
        return false;
    int nnz = 0;
    if (sscanf(ln, "%d", &nnz) != 1 || nnz < 0)
    {
        fprintf(stderr, "[cbf] bad BCOORD header '%s'\n", ln);
        return false;
    }
    if (!s->b)
    {
        fprintf(stderr, "[cbf] BCOORD before CON\n");
        return false;
    }
    for (int i = 0; i < nnz; ++i)
    {
        ln = cbf_next_line(r);
        if (!ln)
        {
            fprintf(stderr, "[cbf] BCOORD truncated at %d/%d\n", i, nnz);
            return false;
        }
        int row;
        double val;
        if (sscanf(ln, "%d %lf", &row, &val) != 2)
        {
            fprintf(stderr, "[cbf] bad BCOORD entry '%s'\n", ln);
            return false;
        }
        if (row < 0 || row >= s->num_cons)
        {
            fprintf(stderr, "[cbf] BCOORD row %d out of range\n", row);
            return false;
        }
        s->b[row] += val;
    }
    return true;
}

static bool cbf_read_int_block(cbf_reader_t *r)
{
    char *ln = cbf_next_line(r);
    if (!ln)
        return false;
    int n;
    if (sscanf(ln, "%d", &n) != 1)
        return false;
    fprintf(stderr, "[cbf] INT block found (n=%d): integer variables not supported\n", n);
    return false;
}

/* Consume the whole CHANGE block (v3+): CHANGE header + arbitrary follow-up subblocks
   until EOF. We treat any post-CHANGE data as ignored — base problem only. */
static void cbf_consume_change(cbf_reader_t *r)
{
    while (cbf_next_line(r))
    {
        /* discard */
    }
}

/* Compute qp_problem column layout:
     [ LP vars | VAR cone-block vars | auxiliary CON cone-block vars ]
   Return arrays:
     lp_offset[b] = column start of block b if LP, else -1
     cone_offset[b] = column start of VAR block b if cone, else -1
     con_cone_offset[b] = column start of CON block b if nonlinear cone, else -1
     total_vars = final variable count
     cbf_to_qp[i] = mapping from CBF variable index i to internal qp column index
     con_to_qp[r] = mapping from nonlinear CBF constraint component r to its aux y column */
static void cbf_build_layout(const cbf_state_t *s,
                             int **lp_off_out,
                             int **cone_off_out,
                             int **con_cone_off_out,
                             int *total_vars_out,
                             int **cbf_to_qp_out,
                             int **con_to_qp_out)
{
    int nb = s->num_var_blocks;
    int *lp_off = (int *)safe_malloc(nb * sizeof(int));
    int *cone_off = (int *)safe_malloc(nb * sizeof(int));
    for (int i = 0; i < nb; ++i)
    {
        lp_off[i] = -1;
        cone_off[i] = -1;
    }
    int col = 0;
    /* LP-side first, preserving CBF order for F, L=, L+, L- blocks. */
    for (int i = 0; i < nb; ++i)
    {
        cbf_cone_t ct = s->var_blocks[i].type;
        if (cbf_is_lp_cone(ct))
        {
            lp_off[i] = col;
            col += s->var_blocks[i].dim;
        }
    }
    /* Then cone-slot blocks. Each takes its CBF cone dimension. */
    for (int i = 0; i < nb; ++i)
    {
        cbf_cone_t ct = s->var_blocks[i].type;
        if (!cbf_is_nonlinear_cone(ct))
            continue;
        cone_off[i] = col;
        col += cbf_internal_cone_slots(ct, s->var_blocks[i].dim);
    }
    int *cbf_to_qp = (int *)safe_malloc(s->num_vars * sizeof(int));
    for (int i = 0; i < nb; ++i)
    {
        cbf_cone_t ct = s->var_blocks[i].type;
        int start = s->var_blocks[i].start;
        int dim = s->var_blocks[i].dim;
        if (cbf_is_lp_cone(ct))
        {
            int base = lp_off[i];
            for (int j = 0; j < dim; ++j)
                cbf_to_qp[start + j] = base + j;
        }
        else
        {
            int base = cone_off[i];
            for (int j = 0; j < dim; ++j)
                cbf_to_qp[start + j] = cbf_map_cone_component(ct, dim, base, j);
        }
    }

    int ncb = s->num_con_blocks;
    int *con_cone_off = (int *)safe_malloc((size_t)(ncb > 0 ? ncb : 1) * sizeof(int));
    for (int i = 0; i < ncb; ++i)
        con_cone_off[i] = -1;
    for (int i = 0; i < ncb; ++i)
    {
        cbf_cone_t ct = s->con_blocks[i].type;
        if (!cbf_is_nonlinear_cone(ct))
            continue;
        con_cone_off[i] = col;
        col += cbf_internal_cone_slots(ct, s->con_blocks[i].dim);
    }

    int *con_to_qp = (int *)safe_malloc((size_t)(s->num_cons > 0 ? s->num_cons : 1) * sizeof(int));
    for (int r = 0; r < s->num_cons; ++r)
        con_to_qp[r] = -1;
    for (int i = 0; i < ncb; ++i)
    {
        cbf_cone_t ct = s->con_blocks[i].type;
        if (!cbf_is_nonlinear_cone(ct))
            continue;
        int start = s->con_blocks[i].start;
        int dim = s->con_blocks[i].dim;
        int base = con_cone_off[i];
        for (int j = 0; j < dim; ++j)
            con_to_qp[start + j] = cbf_map_cone_component(ct, dim, base, j);
    }

    *lp_off_out = lp_off;
    *cone_off_out = cone_off;
    *con_cone_off_out = con_cone_off;
    *total_vars_out = col;
    *cbf_to_qp_out = cbf_to_qp;
    *con_to_qp_out = con_to_qp;
}

/* Sort (row, col) coordinates and coalesce duplicates into a CSR matrix.
   Returns malloc'd CsrComponent. */
static CsrComponent *
cbf_coo_to_csr(int m, int n, int nnz, const int *rows, const int *cols, const double *vals, int *out_nnz)
{
    (void)n;
    /* Simple bucket sort by row. */
    int *row_count = (int *)safe_calloc(m + 1, sizeof(int));
    for (int i = 0; i < nnz; ++i)
        row_count[rows[i] + 1]++;
    for (int i = 0; i < m; ++i)
        row_count[i + 1] += row_count[i];

    int *row_ptr = (int *)safe_malloc((m + 1) * sizeof(int));
    memcpy(row_ptr, row_count, (m + 1) * sizeof(int));

    int alloc_nnz = nnz > 0 ? nnz : 1;
    int *col_ind = (int *)safe_malloc((size_t)alloc_nnz * sizeof(int));
    double *val = (double *)safe_malloc((size_t)alloc_nnz * sizeof(double));
    int *cursor = row_count;
    for (int i = 0; i < nnz; ++i)
    {
        int r = rows[i];
        int pos = cursor[r]++;
        col_ind[pos] = cols[i];
        val[pos] = vals[i];
    }
    free(row_count);

    /* Sort each row's entries by col and coalesce duplicates. */
    int write = 0;
    for (int r = 0; r < m; ++r)
    {
        int s = row_ptr[r];
        int e = row_ptr[r + 1];
        /* Insertion sort — CBF rows are typically short. */
        for (int i = s + 1; i < e; ++i)
        {
            int c = col_ind[i];
            double v = val[i];
            int j = i - 1;
            while (j >= s && col_ind[j] > c)
            {
                col_ind[j + 1] = col_ind[j];
                val[j + 1] = val[j];
                j--;
            }
            col_ind[j + 1] = c;
            val[j + 1] = v;
        }
        int new_s = write;
        int i = s;
        while (i < e)
        {
            int c = col_ind[i];
            double acc = val[i];
            int j = i + 1;
            while (j < e && col_ind[j] == c)
            {
                acc += val[j];
                j++;
            }
            if (acc != 0.0)
            {
                col_ind[write] = c;
                val[write] = acc;
                write++;
            }
            i = j;
        }
        row_ptr[r] = new_s;
    }
    row_ptr[m] = write;
    *out_nnz = write;

    CsrComponent *csr = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
    csr->row_ptr = row_ptr;
    csr->col_ind = col_ind;
    csr->val = val;
    return csr;
}

static qp_problem_t *cbf_finalize(cbf_state_t *s)
{
    int *lp_off, *cone_off, *con_cone_off, *cbf_to_qp, *con_to_qp;
    int total_vars;
    cbf_build_layout(s, &lp_off, &cone_off, &con_cone_off, &total_vars, &cbf_to_qp, &con_to_qp);

    qp_problem_t *out = (qp_problem_t *)safe_calloc(1, sizeof(qp_problem_t));
    out->num_variables = total_vars;
    out->num_constraints = s->num_cons;
    out->num_original_variables = total_vars;
    out->objective_constant = s->obj_constant;

    out->objective_vector = (double *)safe_calloc(total_vars, sizeof(double));
    out->variable_lower_bound = (double *)safe_malloc(total_vars * sizeof(double));
    out->variable_upper_bound = (double *)safe_malloc(total_vars * sizeof(double));
    for (int i = 0; i < total_vars; ++i)
    {
        out->variable_lower_bound[i] = -INFINITY;
        out->variable_upper_bound[i] = INFINITY;
    }
    double c_sign = s->objsense_neg ? -1.0 : 1.0;
    for (int i = 0; i < s->num_vars; ++i)
        out->objective_vector[cbf_to_qp[i]] = c_sign * s->obj_c[i];

    /* Variable bounds from LP-cone membership. */
    for (int b = 0; b < s->num_var_blocks; ++b)
    {
        cbf_cone_t ct = s->var_blocks[b].type;
        int base = lp_off[b];
        if (base < 0)
            continue;
        int dim = s->var_blocks[b].dim;
        if (ct == CBF_CONE_LPOS)
        {
            for (int j = 0; j < dim; ++j)
                out->variable_lower_bound[base + j] = 0.0;
        }
        else if (ct == CBF_CONE_ZERO)
        {
            for (int j = 0; j < dim; ++j)
            {
                out->variable_lower_bound[base + j] = 0.0;
                out->variable_upper_bound[base + j] = 0.0;
            }
        }
        else if (ct == CBF_CONE_LNEG)
        {
            for (int j = 0; j < dim; ++j)
                out->variable_upper_bound[base + j] = 0.0;
        }
        /* CBF_CONE_FREE: leave -inf/inf. */
    }

    /* Cone-block descriptors. */
    int num_cones = 0;
    for (int b = 0; b < s->num_var_blocks; ++b)
    {
        cbf_cone_t ct = s->var_blocks[b].type;
        if (cbf_is_nonlinear_cone(ct))
            num_cones++;
    }
    for (int b = 0; b < s->num_con_blocks; ++b)
    {
        cbf_cone_t ct = s->con_blocks[b].type;
        if (cbf_is_nonlinear_cone(ct))
            num_cones++;
    }
    out->cones.num_cones = num_cones;
    if (num_cones > 0)
    {
        out->cones.start_idx = (int *)safe_malloc(num_cones * sizeof(int));
        out->cones.v_dim = (int *)safe_malloc(num_cones * sizeof(int));
        out->cones.type = (cone_type_t *)safe_malloc(num_cones * sizeof(cone_type_t));
        out->cones.power_alpha = NULL;
        int k = 0;
        for (int b = 0; b < s->num_var_blocks; ++b)
        {
            cbf_cone_t ct = s->var_blocks[b].type;
            if (!cbf_is_nonlinear_cone(ct))
                continue;
            out->cones.start_idx[k] = cone_off[b];
            out->cones.v_dim[k] = cbf_internal_cone_v_dim(ct, s->var_blocks[b].dim);
            out->cones.type[k] = cbf_internal_cone_type(ct);
            k++;
        }
        for (int b = 0; b < s->num_con_blocks; ++b)
        {
            cbf_cone_t ct = s->con_blocks[b].type;
            if (!cbf_is_nonlinear_cone(ct))
                continue;
            out->cones.start_idx[k] = con_cone_off[b];
            out->cones.v_dim[k] = cbf_internal_cone_v_dim(ct, s->con_blocks[b].dim);
            out->cones.type[k] = cbf_internal_cone_type(ct);
            k++;
        }
    }

    /* Build A (num_cons x total_vars) from COO in internal column indexing.
       CBF states: A x + b ∈ K_con, i.e., (Ax + b) participates in the cone.
       For LP CON cones:
         F   ->  no row restriction
         L=  ->  A x = -b   (l = u = -b)
         L+  ->  A x >= -b  (l = -b, u = +inf)
         L-  ->  A x <= -b  (l = -inf, u = -b) */
    int con_aux_nnz = 0;
    for (int b = 0; b < s->num_con_blocks; ++b)
    {
        if (cbf_is_nonlinear_cone(s->con_blocks[b].type))
            con_aux_nnz += s->con_blocks[b].dim;
    }
    int work_nnz = s->nnz_A + con_aux_nnz;
    int *rows = (int *)safe_malloc((size_t)(work_nnz > 0 ? work_nnz : 1) * sizeof(int));
    int *cols = (int *)safe_malloc((size_t)(work_nnz > 0 ? work_nnz : 1) * sizeof(int));
    double *vals = (double *)safe_malloc((size_t)(work_nnz > 0 ? work_nnz : 1) * sizeof(double));
    for (int i = 0; i < s->nnz_A; ++i)
    {
        rows[i] = s->A_row[i];
        cols[i] = cbf_to_qp[s->A_col[i]];
        vals[i] = s->A_val[i];
    }
    int write = s->nnz_A;
    for (int b = 0; b < s->num_con_blocks; ++b)
    {
        cbf_cone_t ct = s->con_blocks[b].type;
        if (!cbf_is_nonlinear_cone(ct))
            continue;
        int start = s->con_blocks[b].start;
        int dim = s->con_blocks[b].dim;
        for (int j = 0; j < dim; ++j)
        {
            rows[write] = start + j;
            cols[write] = con_to_qp[start + j];
            vals[write] = -1.0;
            write++;
        }
    }
    int final_nnz = 0;
    CsrComponent *A = cbf_coo_to_csr(s->num_cons, total_vars, work_nnz, rows, cols, vals, &final_nnz);
    free(rows);
    free(cols);
    free(vals);
    out->constraint_matrix = A;
    out->constraint_matrix_num_nonzeros = final_nnz;

    out->constraint_lower_bound = (double *)safe_malloc((size_t)(s->num_cons > 0 ? s->num_cons : 1) * sizeof(double));
    out->constraint_upper_bound = (double *)safe_malloc((size_t)(s->num_cons > 0 ? s->num_cons : 1) * sizeof(double));
    for (int b = 0; b < s->num_con_blocks; ++b)
    {
        cbf_cone_t ct = s->con_blocks[b].type;
        int start = s->con_blocks[b].start;
        int dim = s->con_blocks[b].dim;
        for (int j = 0; j < dim; ++j)
        {
            int r = start + j;
            double neg_b = -s->b[r];
            if (ct == CBF_CONE_FREE)
            {
                out->constraint_lower_bound[r] = -INFINITY;
                out->constraint_upper_bound[r] = INFINITY;
            }
            else if (ct == CBF_CONE_ZERO || cbf_is_nonlinear_cone(ct))
            {
                out->constraint_lower_bound[r] = neg_b;
                out->constraint_upper_bound[r] = neg_b;
            }
            else if (ct == CBF_CONE_LPOS)
            {
                out->constraint_lower_bound[r] = neg_b;
                out->constraint_upper_bound[r] = INFINITY;
            }
            else
            { /* CBF_CONE_LNEG */
                out->constraint_lower_bound[r] = -INFINITY;
                out->constraint_upper_bound[r] = neg_b;
            }
        }
    }

    out->num_quadratic_constraints = 0;
    out->quadratic_constraint_row_indices = NULL;
    out->quadratic_constraint_matrices = NULL;
    out->quadratic_constraint_matrix_num_nonzeros = NULL;

    /* MAX objsense negates Q too since max f = min -f. */
    if (s->nnz_Q > 0)
    {
        int *qrows = (int *)safe_malloc((size_t)s->nnz_Q * sizeof(int));
        int *qcols = (int *)safe_malloc((size_t)s->nnz_Q * sizeof(int));
        double *qvals = (double *)safe_malloc((size_t)s->nnz_Q * sizeof(double));
        for (int i = 0; i < s->nnz_Q; ++i)
        {
            qrows[i] = cbf_to_qp[s->Q_row[i]];
            qcols[i] = cbf_to_qp[s->Q_col[i]];
            qvals[i] = c_sign * s->Q_val[i];
        }
        int q_final_nnz = 0;
        CsrComponent *Q = cbf_coo_to_csr(total_vars, total_vars, s->nnz_Q, qrows, qcols, qvals, &q_final_nnz);
        free(qrows);
        free(qcols);
        free(qvals);
        out->objective_sparse_matrix = Q;
        out->objective_sparse_matrix_num_nonzeros = q_final_nnz;
    }

    free(lp_off);
    free(cone_off);
    free(con_cone_off);
    free(cbf_to_qp);
    free(con_to_qp);
    return out;
}

qp_problem_t *read_cbf_file(const char *filename)
{
    cbf_reader_t *r = cbf_open(filename);
    if (!r)
    {
        fprintf(stderr, "[cbf] cannot open '%s'\n", filename);
        return NULL;
    }

    cbf_state_t s = {0};
    s.ver = -1;

    bool ok = true;
    while (ok)
    {
        char *ln = cbf_next_line(r);
        if (!ln)
            break;
        char kw[32];
        if (sscanf(ln, "%31s", kw) != 1)
        {
            fprintf(stderr, "[cbf] bad keyword at line %d\n", r->line_no);
            ok = false;
            break;
        }
        if (strcmp(kw, "VER") == 0)
        {
            ok = cbf_read_ver(r, &s);
        }
        else if (strcmp(kw, "OBJSENSE") == 0)
        {
            ok = cbf_read_objsense(r, &s);
        }
        else if (strcmp(kw, "VAR") == 0)
        {
            ok = cbf_read_var(r, &s);
        }
        else if (strcmp(kw, "CON") == 0)
        {
            ok = cbf_read_con(r, &s);
        }
        else if (strcmp(kw, "OBJACOORD") == 0)
        {
            ok = cbf_read_objacoord(r, &s);
        }
        else if (strcmp(kw, "OBJBCOORD") == 0)
        {
            ok = cbf_read_objbcoord(r, &s);
        }
        else if (strcmp(kw, "OBJQCOORD") == 0)
        {
            ok = cbf_read_objqcoord(r, &s);
        }
        else if (strcmp(kw, "ACOORD") == 0)
        {
            ok = cbf_read_acoord(r, &s);
        }
        else if (strcmp(kw, "BCOORD") == 0)
        {
            ok = cbf_read_bcoord(r, &s);
        }
        else if (strcmp(kw, "INT") == 0)
        {
            ok = cbf_read_int_block(r);
        }
        else if (strcmp(kw, "CHANGE") == 0)
        {
            /* v3+ modification section: base problem is complete; ignore rest. */
            cbf_consume_change(r);
            break;
        }
        else if (strcmp(kw, "PSDVAR") == 0 || strcmp(kw, "PSDCON") == 0 || strcmp(kw, "OBJFCOORD") == 0 ||
                 strcmp(kw, "FCOORD") == 0 || strcmp(kw, "HCOORD") == 0 || strcmp(kw, "DCOORD") == 0)
        {
            fprintf(stderr, "[cbf] block '%s' not supported (PSD/free-matrix variables)\n", kw);
            ok = false;
        }
        else
        {
            fprintf(stderr, "[cbf] unknown keyword '%s' at line %d\n", kw, r->line_no);
            ok = false;
        }
    }

    cbf_close(r);

    if (!ok)
    {
        cbf_state_free(&s);
        return NULL;
    }
    if (s.num_var_blocks == 0 || !s.var_blocks)
    {
        fprintf(stderr, "[cbf] missing VAR block\n");
        cbf_state_free(&s);
        return NULL;
    }
    if (!s.con_blocks)
    {
        /* No constraints — synthesize empty CON. */
        s.num_con_blocks = 0;
        s.num_cons = 0;
        s.b = (double *)safe_calloc(1, sizeof(double)); /* placeholder */
    }

    qp_problem_t *out = cbf_finalize(&s);
    cbf_state_free(&s);
    return out;
}
