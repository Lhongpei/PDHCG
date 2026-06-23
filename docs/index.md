# PDHCG-II

PDHCG-II is a high-performance, GPU-accelerated implementation of the Primal-Dual Hybrid Gradient (PDHG) algorithm designed for solving large-scale Convex Quadratic Programming (QP) problems.

## Problem Formulation

PDHCG solves quadratic programs in the following form:

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2}x^\top (Q + R^\top D R) x + c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & \ell_v \le x \le u_v, \\
                  & x_J \in \mathcal{K} \quad \text{for cone blocks } J.
\end{aligned}
$$

Where:

- $Q$ is a sparse symmetric matrix (optional)
- $R \in \mathbb{R}^{k\times n}$ is a low-rank factor of rank $k$ (optional)
- $D \in \mathbb{R}^{k\times k}$ is an optional middle matrix; defaults to the identity, recovering the standard $Q + R^\top R$ form. May be diagonal, sparse, dense, or indefinite — the backend auto-detects the cheapest representation
- $A$ is the constraint matrix
- $c$ is the linear objective vector
- $\ell_c, u_c$ are constraint bounds
- $\ell_v, u_v$ are variable bounds
- $\mathcal{K}$ is a convex cone (Standard SOC, Rotated SOC, or Exponential) applied to a contiguous block of variables

## Key Features

- **GPU Acceleration**: Fully leverages NVIDIA CUDA for extreme-scale QP problems
- **Flexible Problem Structure**: Supports sparse, low-rank, and middle-weighted low-rank ($R^\top D R$) quadratic terms — alone or combined
- **Conic constraints**: native SOC / Rotated SOC / Exponential cone projection, fully GPU-accelerated; works with LP, diagonal Q, and sparse Q objectives
- **High Performance**: Competitive with commercial solvers on large-scale problems
- **SpMVOp Auto-Detection**: Automatically uses cuSPARSE SpMVOp on CUDA 13+ while falling back to standard SpMV on CUDA 12.x
- **Multi-GPU Distributed Solving**: Supports parallel solving across multiple GPUs via MPI and NCCL (optional, enabled at compile time)

## Quick Links

- [Installation Guide](installation.md)
- [Python API Reference](python/quickstart.md)
- [C API Reference](c/overview.md)
- [Examples](examples.md)
- [Conic constraints usage](examples.md#conic-examples)

## Citation

If you use this software in your research, please cite:

```bibtex
@misc{li2026pdhcgiienhancedversionpdhcg,
      title={PDHCG-II: An Enhanced Version of PDHCG for Large-Scale Convex QP},
      author={Hongpei Li and Yicheng Huang and Huikang Liu and Dongdong Ge and Yinyu Ye},
      year={2026},
      eprint={2602.23967},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2602.23967},
}
```

## License

Copyright 2024-2026 Hongpei Li, Haihao Lu.

Licensed under the Apache License, Version 2.0.
