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
