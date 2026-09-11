#pragma once

#include <cusparse.h>

// cusparseSpMVOp_bufferSize was introduced in cuSPARSE 12.7.3.
// Older CUDA/cuSPARSE versions should use the standard cusparseSpMV path.
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12703
#define PDHCG_USE_SPMVOP 1
#else
#define PDHCG_USE_SPMVOP 0
#endif

// CUDA Toolkit 13.3 / cuSPARSE 12.8.1 added the alg parameter to
// cusparseSpMVOp_bufferSize and cusparseSpMVOp_createDescr. CUDA 13.2 /
// cuSPARSE 12.7.10 still uses the older signature without alg.
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12801
#define PDHCG_CUSPARSE_SPMVOP_HAS_ALG_PARAM 1
#else
#define PDHCG_CUSPARSE_SPMVOP_HAS_ALG_PARAM 0
#endif

#if !PDHCG_USE_SPMVOP
// The SpMVOp types were added to cusparse.h before the functions
#if !defined(CUSPARSE_VERSION) || CUSPARSE_VERSION < 12700
typedef void *cusparseSpMVOpDescr_t;
typedef void *cusparseSpMVOpPlan_t;
#endif
#endif
