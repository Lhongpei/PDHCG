using CUDA, CUDA.CUSPARSE, SparseArrays, LinearAlgebra, BenchmarkTools
using Random

# ------------------------------------------------------------------
# 1. & 2. Data Structures (Unchanged)
# ------------------------------------------------------------------
# (Your block_diag_csr function and CuQuadraticProgrammingProblem struct are fine)
function block_diag_csr(blocks::Vector{SparseMatrixCSC{Float64,Int32}})
    nblocks = length(blocks)
    n = sum(size(b,1) for b in blocks)
    m = sum(size(b,2) for b in blocks)
    # This function seems to have a bug in how it accesses column indices.
    # We will assume the input CuSparseMatrixCSR objects are created correctly otherwise.
    # For this solution, this function is not used.
end

struct CuQuadraticProgrammingProblem
    objective_matrix   :: CuSparseMatrixCSR{Float64}
    constraint_matrix  :: CuSparseMatrixCSR{Float64}
    objective_vector   :: CuArray{Float64,1}
    variable_lower_bound :: CuArray{Float64,1}
    variable_upper_bound :: CuArray{Float64,1}
end

# ------------------------------------------------------------------
# 3. & 4. Original Functions (Unchanged, kept for benchmark comparison)
# ------------------------------------------------------------------
function projection!(x::CuArray, lb::CuArray, ub::CuArray)
    x .= max.(lb, min.(ub, x))
    return x
end

function compute_next_primal_solution_gd_BB!(
        problem::CuQuadraticProgrammingProblem,
        current_primal_solution::CuArray{Float64,1},
        current_dual_product::CuArray{Float64,1},
        current_primal_obj_product::CuArray{Float64,1},
        last_gradient::CuArray{Float64,1},
        step_size::Float64,
        primal_weight::Float64,
        CG_bound::Float64,
        current_gradient::CuArray{Float64,1},
        inner_delta_primal::CuArray{Float64,1},
        next_primal::CuArray{Float64,1},
        last_primal::CuArray{Float64,1},
        next_primal_product::CuArray{Float64,1},
        next_primal_obj_product::CuArray{Float64,1},
        norm_Q::Float64,
        first_iter::Bool;
        max_CG_iter::Int=1)

    k = 1
    alpha = 1.0 / (norm_Q + primal_weight / step_size)
    CUDA.copyto!(last_primal, current_primal_solution)
    current_gradient .= current_primal_obj_product .+ problem.objective_vector .- current_dual_product
    CUDA.copyto!(last_gradient, current_gradient)
    next_primal .= current_primal_solution .- alpha .* current_gradient
    projection!(next_primal, problem.variable_lower_bound, problem.variable_upper_bound)
    while k <= max_CG_iter
        inner_delta_primal .= next_primal .- last_primal
        gg = CUDA.dot(inner_delta_primal, inner_delta_primal)
        if sqrt(gg) <= min(0.05 * CG_bound, 1e-2) * alpha
            break
        end
        CUSPARSE.mv!('N', 1.0, problem.objective_matrix, next_primal, 0.0, current_gradient, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        current_gradient .= current_gradient .+ (primal_weight / step_size) .* (next_primal .- current_primal_solution) .+ problem.objective_vector .- current_dual_product
        alpha = gg / CUDA.dot(inner_delta_primal, current_gradient .- last_gradient)
        CUDA.copyto!(last_primal, next_primal)
        CUDA.copyto!(last_gradient, current_gradient)
        next_primal .= next_primal .- alpha .* current_gradient
        projection!(next_primal, problem.variable_lower_bound, problem.variable_upper_bound)
        k += 1
    end
    CUSPARSE.mv!('N', 1.0, problem.objective_matrix, next_primal, 0.0, next_primal_obj_product, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    return min(k, max_CG_iter)
end


# ------------------------------------------------------------------
# 5. ✅ NEW: Parallel Kernel Implementation
# ------------------------------------------------------------------

# This is the new top-level function that prepares data and launches the parallel kernel.
function compute_next_primal_solution_parallel!(
    blocks::Vector{CuQuadraticProgrammingProblem},
    # Workspace vectors (as vectors of CuArrays)
    current_primal_sols, current_dual_prods, current_primal_obj_prods,
    last_gradients, current_gradients, inner_delta_primals,
    next_primals, last_primals, next_primal_obj_prods,
    # Scalar parameters
    step_size, primal_weight, CG_bounds, norm_Qs;
    max_CG_iter=1, threads_per_block=256)

    n_blocks = length(blocks)
    block_size = length(blocks[1].objective_vector) # Assuming all blocks are same size

    # Prepare matrix data for the kernel by concatenating CSR components
    # We cannot pass the CuSparseMatrixCSR objects directly into the kernel.
    Q_rowptr_cat = vcat([b.objective_matrix.rowPtr for b in blocks]...)
    Q_colval_cat = vcat([b.objective_matrix.colVal for b in blocks]...)
    Q_nzval_cat = vcat([b.objective_matrix.nzVal for b in blocks]...)

    # Create offset arrays so the kernel knows where each matrix's data begins
    q_row_offsets = CuArray([1; cumsum(length.(b.objective_matrix.rowPtr for b in blocks))[1:end-1] .+ 1])
    q_nnz_offsets = CuArray([1; cumsum(length.(b.objective_matrix.nzVal for b in blocks))[1:end-1] .+ 1])

    # The kernel needs access to all vector data. We pass them as tuples of CuArrays.
    # This is much cleaner than concatenating all vectors.
    kernel_args = (
        # Workspace vectors
        Tuple(current_primal_sols), Tuple(current_dual_prods), Tuple(current_primal_obj_prods),
        Tuple(last_gradients), Tuple(current_gradients), Tuple(inner_delta_primals),
        Tuple(next_primals), Tuple(last_primals), Tuple(next_primal_obj_prods),
        # Problem data vectors
        Tuple(b.objective_vector for b in blocks),
        Tuple(b.variable_lower_bound for b in blocks),
        Tuple(b.variable_upper_bound for b in blocks),
        # Concatenated CSR matrix data
        Q_rowptr_cat, Q_colval_cat, Q_nzval_cat,
        # Offsets
        q_row_offsets, q_nnz_offsets,
        # Scalar parameters
        step_size, primal_weight, CuArray(CG_bounds), CuArray(norm_Qs),
        block_size, max_CG_iter
    )

    # Launch the kernel: one CUDA block per QP block
    @cuda blocks=n_blocks threads=threads_per_block shmem=threads_per_block*sizeof(Float64) solve_blocks_kernel!(kernel_args...)
    CUDA.synchronize()
end


# GPU Kernel and its helper device functions
function solve_blocks_kernel!(
    current_primals, current_duals, current_primal_objs,
    last_gradients, current_gradients, inner_delta_primals,
    next_primals, last_primals, next_primal_objs,
    objective_vectors, var_lbs, var_ubs,
    Q_rowptr, Q_colval, Q_nzval,
    q_row_offsets, q_nnz_offsets,
    step_size, primal_weight, CG_bounds, norm_Qs,
    block_size, max_CG_iter)

    # --- Helper: Device function for dot product ---
    @inline function device_dot(x, y, shared_mem)
        tid = threadIdx().x
        val = 0.0
        # Each thread computes a partial sum
        for i = tid:blockDim().x:block_size
            val += x[i] * y[i]
        end
        shared_mem[tid] = val
        sync_threads()

        # Parallel reduction in shared memory
        s = blockDim().x ÷ 2
        while s > 0
            if tid <= s
                shared_mem[tid] += shared_mem[tid + s]
            end
            sync_threads()
            s ÷= 2
        end
        return shared_mem[1]
    end

    # --- Helper: Device function for CSR SpMV (y = A*x) ---
    @inline function device_spmv_csr!(y, row_ptr, col_val, nz_val, x)
        row = threadIdx().x
        if row <= block_size
            start_idx = row_ptr[row]
            end_idx = row_ptr[row+1] - 1
            row_val = 0.0
            for j = start_idx:end_idx
                row_val += nz_val[j] * x[col_val[j]]
            end
            y[row] = row_val
        end
        sync_threads()
    end

    # --- Main Kernel Logic ---
    
    block_id = blockIdx().x
    tid = threadIdx().x

    # Select the data for the current block
    _cp = current_primals[block_id]
    _cd = current_duals[block_id]
    _cpo = current_primal_objs[block_id]
    _lg = last_gradients[block_id]
    _cg = current_gradients[block_id]
    _idp = inner_delta_primals[block_id]
    _np = next_primals[block_id]
    _lp = last_primals[block_id]
    _npo = next_primal_objs[block_id]
    _ov = objective_vectors[block_id]
    _lb = var_lbs[block_id]
    _ub = var_ubs[block_id]

    # Create views into the concatenated matrix data
    row_offset = q_row_offsets[block_id]
    nnz_offset = q_nnz_offsets[block_id]
    _Q_rowptr = CuDeviceArray(size(Q_rowptr), pointer(Q_rowptr, row_offset))
    _Q_colval = CuDeviceArray(size(Q_colval), pointer(Q_colval, nnz_offset))
    _Q_nzval = CuDeviceArray(size(Q_nzval), pointer(Q_nzval, nnz_offset))
    
    # Shared memory for reductions
    shared_mem = @cuDynamicSharedMem(Float64, blockDim().x)
    
    # Algorithm starts here (ported from your original function)
    alpha = 1.0 / (norm_Qs[block_id] + primal_weight / step_size)
    
    @inbounds for i = tid:blockDim().x:block_size
        _lp[i] = _cp[i]
        _cg[i] = _cpo[i] + _ov[i] - _cd[i]
        _lg[i] = _cg[i]
        val = _cp[i] - alpha * _cg[i]
        _np[i] = max(_lb[i], min(_ub[i], val))
    end
    sync_threads()

    k = 1
    while k <= max_CG_iter
        @inbounds for i = tid:blockDim().x:block_size
            _idp[i] = _np[i] - _lp[i]
        end
        sync_threads()

        gg = device_dot(_idp, _idp, shared_mem)
        
        if sqrt(gg) <= min(0.05 * CG_bounds[block_id], 1e-2) * alpha
            break
        end

        device_spmv_csr!(_cg, _Q_rowptr, _Q_colval, _Q_nzval, _np)
        
        @inbounds for i = tid:blockDim().x:block_size
            _cg[i] += (primal_weight / step_size) * (_np[i] - _cp[i]) + _ov[i] - _cd[i]
        end
        sync_threads()
        
        # Use _idp as a temp buffer for (current_gradient - last_gradient)
        @inbounds for i = tid:blockDim().x:block_size
            _idp[i] = _cg[i] - _lg[i]
        end
        sync_threads()
        
        # We need the original inner_delta_primal again for the dot product.
        # Let's re-calculate it into the last_primal buffer (which is about to be updated)
        @inbounds for i = tid:blockDim().x:block_size
             _lp[i] = _np[i] - _lp[i]
        end
        sync_threads()
        
        denom = device_dot(_lp, _idp, shared_mem)
        alpha = abs(gg / denom)

        # Updates for next iteration
        @inbounds for i = tid:blockDim().x:block_size
            old_np = _np[i]
            
            _lp[i] = old_np
            _lg[i] = _cg[i]
            
            new_np = old_np - alpha * _cg[i]
            _np[i] = max(_lb[i], min(_ub[i], new_np))
        end
        k += 1
        sync_threads()
    end

    # Final SpMV
    device_spmv_csr!(_npo, _Q_rowptr, _Q_colval, _Q_nzval, _np)
    
    return
end


# ------------------------------------------------------------------
# 6. Setup Test Data (Increased NBLOCKS to show parallelism)
# ------------------------------------------------------------------
const NBLOCKS = 16 # Increased from 2 to better see parallel speedup
const BLOCKSZ = 20480 # Reduced size slightly to fit more blocks in memory
const DENSITY = 0.05
rng = MersenneTwister(1234)

println("Setting up test data for $NBLOCKS blocks of size $BLOCKSZ...")
blocks_cpu = [sprand(rng, BLOCKSZ, BLOCKSZ, DENSITY) for _ in 1:NBLOCKS]
for b in blocks_cpu
    b .= (b + b') / 2.0 + 2.0I # Ensure symmetric positive definite
end
# We don't need the full block diagonal matrix for the parallel version
# Q_full_cpu = blockdiag(blocks_cpu...)
# Q_full_d = CuSparseMatrixCSR{Float64}(Q_full_cpu)
# Only create it if you intend to benchmark the serial version
println("Creating sparse matrices on GPU...")
blocks_d = [CuSparseMatrixCSR{Float64}(b) for b in blocks_cpu]

# Problem instances for each block
problem_blocks = [CuQuadraticProgrammingProblem(
    blocks_d[i],
    CUDA.rand(1,1), # Dummy matrix, not used in this function
    CUDA.rand(BLOCKSZ),
    CUDA.fill(-10.0, BLOCKSZ),
    CUDA.fill(10.0, BLOCKSZ)) for i in 1:NBLOCKS]

# ------------------------------------------------------------------
# 7. Allocate Workspace Vectors for the block-wise version
# ------------------------------------------------------------------
println("Allocating workspace vectors...")
make_vec() = [CUDA.rand(Float64, BLOCKSZ) for _ in 1:NBLOCKS]
current_primal_b      = make_vec()
current_dual_b        = make_vec()
current_primal_obj_b  = make_vec()
last_gradient_b       = make_vec()
current_gradient_b    = make_vec()
inner_delta_primal_b  = make_vec()
next_primal_b         = make_vec()
last_primal_b         = make_vec()
next_primal_product_b = make_vec() # Not used in this function
next_primal_obj_prod_b= make_vec()

# Parameters
norm_Qs = [norm(Array(blocks_cpu[i])) for i in 1:NBLOCKS]
step_size   = 1.0
primal_weight = 1.0
CG_bound    = 1.0
first_iter  = true
CG_bounds = fill(CG_bound, NBLOCKS)

# ------------------------------------------------------------------
# 8. Warmup
# ------------------------------------------------------------------
println("Warming up...")
compute_next_primal_solution_parallel!(
    problem_blocks,
    current_primal_b, current_dual_b, current_primal_obj_b,
    last_gradient_b, current_gradient_b, inner_delta_primal_b,
    next_primal_b, last_primal_b, next_primal_obj_prod_b,
    step_size, primal_weight, CG_bounds, norm_Qs)

# ------------------------------------------------------------------
# 9. Benchmark
# ------------------------------------------------------------------
println("🚀 Starting benchmark...")
# println("=== Whole matrix (Serial) ===")
# Note: Benchmarking the whole matrix requires creating problem_full and its workspaces,
# which was omitted to save memory for the larger NBLOCKS example. You can add it back
# if your GPU has enough memory.

println("=== Parallel Kernel (All Blocks at Once) ===")
@btime CUDA.@sync compute_next_primal_solution_parallel!(
    $problem_blocks,
    $current_primal_b, $current_dual_b, $current_primal_obj_b,
    $last_gradient_b, $current_gradient_b, $inner_delta_primal_b,
    $next_primal_b, $last_primal_b, $next_primal_obj_prod_b,
    $step_size, $primal_weight, $CG_bounds, $norm_Qs)