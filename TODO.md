# Technical Debt

## Distributed Model Transport

Status: deferred

`distribute_data_bcast_then_partition` currently broadcasts the complete unscaled
`working_problem`, then broadcasts `rescale_info`, whose payload contains another
complete copy in `scaled_problem`. Non-root ranks can therefore hold both global
models, the serialization buffer, and their local partitions at the same time.
This increases startup traffic and peak host memory, and can cause avoidable OOMs
on large instances.

Follow-up direction:

- Broadcast only the scaled model plus the small amount of original-model metadata
  required for reporting, warm starts, cone metadata, and original norms.
- Recover original quantities from scaling factors where that is exact.
- Measure peak memory and distribution time before and after the change.

## Processed Problem Representation

Status: deferred

`processed_qp_problem_t` mirrors most fields of `qp_problem_t` as borrowed pointers.
The repeated field lists in preprocessing, distributed partitioning, and cleanup
make ownership harder to audit and create maintenance work whenever `qp_problem_t`
changes.

Follow-up direction:

- Replace it with a small objective-specific derived representation containing only
  quadratic type, diagonal data, and low-rank middle data.
- Pass the owning `qp_problem_t` separately wherever the original arrays are needed.
- Make owned and borrowed fields explicit in the type and its destructor.

## Cone Projection Hot Paths

Status: deferred

The common free SOC/RSOC path has thread, warp, and multi-block grid kernels, and
split cones use multi-GPU reductions. Three less common paths still have limited
parallelism:

- A large local SOC/RSOC with fixed endpoints falls back to one warp because the
  grid projection and residual kernels do not implement fixed ball sections.
- Diagonal-Q cone proximal steps currently have only one-thread-per-cone kernels.
- Complementarity for a large local affine cone is reduced by one CUDA block.

Follow-up direction:

- Add grid reductions for fixed SOC/RSOC sections and their stationarity and
  complementarity residuals.
- Add warp/grid weighted proximal kernels for diagonal-Q SOC/RSOC blocks.
- Reuse the large-cone reduction machinery for local affine complementarity.
- Move the host launch adapters and dispatch tables from `pdhg_core_op.cu` into a
  dedicated cone-dispatch module once these kernel families are complete.
