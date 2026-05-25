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

#ifndef SPMV_BACKEND_H
#define SPMV_BACKEND_H

#include <cusparse.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        cusparseSpMatDescr_t mat;
        cusparseDnVecDescr_t vec_x;
        cusparseDnVecDescr_t vec_y;
        void *buffer;
        void *descr;
        void *plan;
        int num_rows;
        int num_nonzeros;
    } pdhcg_spmv_ctx_t;

    bool pdhcg_use_spmvop_by_default(void);

    pdhcg_spmv_ctx_t *pdhcg_spmv_ctx_create(cusparseHandle_t sparse_handle,
                                            int num_rows,
                                            int num_cols,
                                            int num_nonzeros,
                                            int *row_ptr,
                                            int *col_ind,
                                            double *val,
                                            cusparseDnVecDescr_t vec_x,
                                            cusparseDnVecDescr_t vec_y);

    void pdhcg_spmv_ctx_destroy(pdhcg_spmv_ctx_t *ctx);

    void pdhcg_spmv_execute(cusparseHandle_t sparse_handle,
                            pdhcg_spmv_ctx_t *ctx,
                            const double *alpha,
                            const double *beta,
                            const double *x,
                            double *y);

#ifdef __cplusplus
}
#endif

#endif // PDHCG_MV_BACKEND_H
