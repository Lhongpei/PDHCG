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
#include "cusparse_compat.h"
#include "spmv_backend.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdlib.h>

// #define FUSE_ELEMENT_OP 0
// #ifdef PDHCG_USE_SPMVOP
// #ifndef PDHCG_DISTRIBUTED
// #define FUSE_ELEMENT_OP 1
// #endif
// #endif

extern "C" bool pdhcg_use_spmvop_by_default(void)
{
#if PDHCG_USE_SPMVOP
    return true;
#else
    return false;
#endif
}

static void pdhcg_spmv_buffer_size(cusparseHandle_t sparse_handle,
                                   cusparseSpMatDescr_t mat,
                                   cusparseDnVecDescr_t vec_x,
                                   cusparseDnVecDescr_t vec_y,
                                   size_t *buffer_size)
{
#if PDHCG_USE_SPMVOP
    CUSPARSE_CHECK(cusparseSpMVOp_bufferSize(
        sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, mat, vec_x, vec_y, vec_y, CUDA_R_64F, buffer_size));
#else
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(sparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           mat,
                                           vec_x,
                                           &HOST_ZERO,
                                           vec_y,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           buffer_size));
#endif
}

static void pdhcg_spmv_prepare(cusparseHandle_t sparse_handle,
                               cusparseSpMatDescr_t mat,
                               cusparseDnVecDescr_t vec_x,
                               cusparseDnVecDescr_t vec_y,
                               void *buffer,
                               void **descr,
                               void **plan)
{
#if PDHCG_USE_SPMVOP
    cusparseSpMVOpDescr_t local_descr = NULL;
    cusparseSpMVOpPlan_t local_plan = NULL;
    CUSPARSE_CHECK(cusparseSpMVOp_createDescr(
        sparse_handle, &local_descr, CUSPARSE_OPERATION_NON_TRANSPOSE, mat, vec_x, vec_y, vec_y, CUDA_R_64F, buffer));
    CUSPARSE_CHECK(cusparseSpMVOp_createPlan(sparse_handle, local_descr, &local_plan, NULL, 0));
    *descr = (void *)local_descr;
    *plan = (void *)local_plan;
#else
    (void)descr;
    (void)plan;
    CUSPARSE_CHECK(cusparseSpMV_preprocess(sparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           mat,
                                           vec_x,
                                           &HOST_ZERO,
                                           vec_y,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           buffer));
#endif
}

extern "C" pdhcg_spmv_ctx_t *pdhcg_spmv_ctx_create(cusparseHandle_t sparse_handle,
                                                   int num_rows,
                                                   int num_cols,
                                                   int num_nonzeros,
                                                   int *row_ptr,
                                                   int *col_ind,
                                                   double *val,
                                                   cusparseDnVecDescr_t vec_x,
                                                   cusparseDnVecDescr_t vec_y)
{
    pdhcg_spmv_ctx_t *ctx = (pdhcg_spmv_ctx_t *)safe_calloc(1, sizeof(pdhcg_spmv_ctx_t));
    ctx->vec_x = vec_x;
    ctx->vec_y = vec_y;
    ctx->num_rows = num_rows;
    ctx->num_nonzeros = num_nonzeros;
    size_t buffer_size = 0;

    if (num_nonzeros == 0)
    {
        return ctx;
    }

    CUSPARSE_CHECK(cusparseCreateCsr(&ctx->mat,
                                     num_rows,
                                     num_cols,
                                     num_nonzeros,
                                     row_ptr,
                                     col_ind,
                                     val,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));

    pdhcg_spmv_buffer_size(sparse_handle, ctx->mat, ctx->vec_x, ctx->vec_y, &buffer_size);
    if (buffer_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&ctx->buffer, buffer_size));
    }

    pdhcg_spmv_prepare(sparse_handle, ctx->mat, ctx->vec_x, ctx->vec_y, ctx->buffer, &ctx->descr, &ctx->plan);

    return ctx;
}

extern "C" void pdhcg_spmv_ctx_destroy(pdhcg_spmv_ctx_t *ctx)
{
    if (ctx == NULL)
        return;

    if (ctx->num_nonzeros == 0)
    {
        free(ctx);
        return;
    }

#if PDHCG_USE_SPMVOP
    if (ctx->descr)
        CUSPARSE_CHECK(cusparseSpMVOp_destroyDescr((cusparseSpMVOpDescr_t)ctx->descr));
    if (ctx->plan)
        CUSPARSE_CHECK(cusparseSpMVOp_destroyPlan((cusparseSpMVOpPlan_t)ctx->plan));
#endif

    if (ctx->buffer)
        CUDA_CHECK(cudaFree(ctx->buffer));
    if (ctx->mat)
        CUSPARSE_CHECK(cusparseDestroySpMat(ctx->mat));

    free(ctx);
}

extern "C" void pdhcg_spmv_execute(cusparseHandle_t sparse_handle,
                                   pdhcg_spmv_ctx_t *ctx,
                                   const double *alpha,
                                   const double *beta,
                                   const double *x,
                                   double *y)
{
    (void)alpha;
    (void)x;
    if (ctx->num_nonzeros == 0)
    {
        double beta_host = (beta != NULL) ? *beta : 0.0;
        if (beta_host == 0.0)
        {
            CUDA_CHECK(cudaMemset(y, 0, (size_t)ctx->num_rows * sizeof(double)));
        }
        else if (beta_host != 1.0)
        {
            CUDA_CHECK(cudaMemset(y, 0, (size_t)ctx->num_rows * sizeof(double)));
        }
        return;
    }

    CUSPARSE_CHECK(cusparseDnVecSetValues(ctx->vec_x, (void *)x));
    CUSPARSE_CHECK(cusparseDnVecSetValues(ctx->vec_y, y));

#if PDHCG_USE_SPMVOP
    CUSPARSE_CHECK(cusparseSpMVOp(
        sparse_handle, (cusparseSpMVOpPlan_t)ctx->plan, alpha, beta, ctx->vec_x, ctx->vec_y, ctx->vec_y));
#else
    CUSPARSE_CHECK(cusparseSpMV(sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                alpha,
                                ctx->mat,
                                ctx->vec_x,
                                beta,
                                ctx->vec_y,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                ctx->buffer));
#endif
}
