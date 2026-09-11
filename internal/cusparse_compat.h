#pragma once

#include <cusparse.h>

// CUSPARSE_VERSION encoding:
//     major * 1000 + minor * 100 + patch
//
// Examples:
//     CUDA Toolkit 13.2 ships cuSPARSE 12.7.10:
//         CUSPARSE_VERSION = 12 * 1000 + 7 * 100 + 10 = 12710
//     CUDA Toolkit 13.3 ships cuSPARSE 12.8.2:
//         CUSPARSE_VERSION = 12 * 1000 + 8 * 100 + 2 = 12802

// cusparseSpMVOp_bufferSize was introduced in cuSPARSE 12.7.3
// (CUDA 13.1 Update 1). Older CUDA/cuSPARSE versions should use the
// standard cusparseSpMV path.
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12703
#define PDHCG_USE_SPMVOP 1
#else
#define PDHCG_USE_SPMVOP 0
#endif

// CUDA Toolkit 13.3 / cuSPARSE 12.8.2 added the alg parameter to
// cusparseSpMVOp_bufferSize and cusparseSpMVOp_createDescr. CUDA 13.2 /
// cuSPARSE 12.7.10 still uses the older signature without alg.
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12802
#define PDHCG_CUSPARSE_SPMVOP_HAS_ALG_PARAM 1
#else
#define PDHCG_CUSPARSE_SPMVOP_HAS_ALG_PARAM 0
#endif

#if !PDHCG_USE_SPMVOP
// The SpMVOp types were added to cusparse.h before the functions
// (e.g. CUDA 13.1 base has the types but not the functions).
// Only provide fallback typedefs for cuSPARSE versions that lack them entirely.
#if !defined(CUSPARSE_VERSION) || CUSPARSE_VERSION < 12700
typedef void *cusparseSpMVOpDescr_t;
typedef void *cusparseSpMVOpPlan_t;
#endif
#endif
