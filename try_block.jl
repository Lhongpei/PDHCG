using CUDA, CUDA.CUSPARSE, SparseArrays, LinearAlgebra, BenchmarkTools
using Random 
# ------------------------------------------------------------------
# 1. 构造一个分块对角的 CuSparseMatrixCSR
# ------------------------------------------------------------------
function block_diag_csr(blocks::Vector{SparseMatrixCSC{Float64,Int32}})
    nblocks = length(blocks)
    n = sum(size(b,1) for b in blocks)
    m = sum(size(b,2) for b in blocks)
    I = Int32[]
    J = Int32[]
    V = Float64[]
    row_off = col_off = 0
    for b in blocks
        r, c = size(b)
        append!(I, row_off .+ rowvals(b))
        append!(J, col_off .+ b.colptr[1:end-1][rowvals(b)])
        append!(V, nonzeros(b))
        row_off += r
        col_off += c
    end
    sparse(I, J, V, n, m, fmt=:csr)
end

# ------------------------------------------------------------------
# 2. 问题结构体（整体矩阵版）
# ------------------------------------------------------------------
struct CuQuadraticProgrammingProblem
    objective_matrix   :: CuSparseMatrixCSR{Float64}
    constraint_matrix  :: CuSparseMatrixCSR{Float64}
    objective_vector   :: CuArray{Float64,1}
    variable_lower_bound :: CuArray{Float64,1}
    variable_upper_bound :: CuArray{Float64,1}
end

# ------------------------------------------------------------------
# 3. 投影算子
# ------------------------------------------------------------------
function projection!(x::CuArray, lb::CuArray, ub::CuArray)
    x .= max.(lb, min.(ub, x))
    return x
end

# ------------------------------------------------------------------
# 4. 整体矩阵版本的 compute_next_primal_solution_gd_BB!
#    （与你给出的一致，仅加上类型约束以便复用）
# ------------------------------------------------------------------
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
    # CUSPARSE.mv!('N', 1.0, problem.constraint_matrix, next_primal, 0.0, next_primal_product, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    return min(k, max_CG_iter)
end

# ------------------------------------------------------------------
# 5. 分块并行版本：16 个 Stream，每块独立跑同一逻辑
# ------------------------------------------------------------------
function compute_next_primal_solution_gd_BB_blocks!(
        blocks::Vector{CuQuadraticProgrammingProblem},
        streams::Vector{CuStream},
        # 以下均为长度为 16 的 Vector{CuArray}
        current_primal_solutions, current_dual_products, current_primal_obj_products,
        last_gradients, current_gradients, inner_delta_primals,
        next_primals, last_primals, next_primal_products, next_primal_obj_products,
        step_size, primal_weight, CG_bounds, norm_Qs, first_iter)

    for (i, s) in enumerate(streams)
        
        CUDA.stream!(s)
        compute_next_primal_solution_gd_BB!(
            blocks[i],
            current_primal_solutions[i],
            current_dual_products[i],
            current_primal_obj_products[i],
            last_gradients[i],
            step_size,
            primal_weight,
            CG_bounds[i],
            current_gradients[i],
            inner_delta_primals[i],
            next_primals[i],
            last_primals[i],
            next_primal_products[i],
            next_primal_obj_products[i],
            norm_Qs[i],
            first_iter)
        
    end
    # 等待所有 Stream 完成
    CUDA.synchronize()
end

# ------------------------------------------------------------------
# 6. 构造测试数据
# ------------------------------------------------------------------
const NBLOCKS = 2
const BLOCKSZ = 204800
const DENSITY = 0.05
rng = MersenneTwister(1234)

blocks_cpu = [sprand(rng, BLOCKSZ, BLOCKSZ, DENSITY) for _ in 1:NBLOCKS]
for b in blocks_cpu               # 保证对称正定
    b .= (b + b') / 2.0 + 2.0I
end
Q_full_cpu = blockdiag(blocks_cpu...)  # 整体矩阵
Q_full_d = CuSparseMatrixCSR{Float64}(Q_full_cpu)

# 每块独立的问题
blocks_d = [CuSparseMatrixCSR{Float64}(b) for b in blocks_cpu]

# 构造问题实例
problem_full = CuQuadraticProgrammingProblem(
    Q_full_d,
    CuSparseMatrixCSR{Float64}(sprand(2*BLOCKSZ*NBLOCKS, BLOCKSZ*NBLOCKS, 0.01)),
    CUDA.rand(BLOCKSZ*NBLOCKS),
    CUDA.fill(-10.0, BLOCKSZ*NBLOCKS),
    CUDA.fill( 10.0, BLOCKSZ*NBLOCKS))

problem_blocks = [CuQuadraticProgrammingProblem(
    blocks_d[i],
    CuSparseMatrixCSR{Float64}(sprand(2*BLOCKSZ, BLOCKSZ, 0.01)),
    CUDA.rand(BLOCKSZ),
    CUDA.fill(-10.0, BLOCKSZ),
    CUDA.fill( 10.0, BLOCKSZ)) for i in 1:NBLOCKS]

# ------------------------------------------------------------------
# 7. 分配工作向量
# ------------------------------------------------------------------
n = BLOCKSZ * NBLOCKS
current_primal_full      = CUDA.rand(Float64, n)
println("type(current_primal_full)", typeof(current_primal_full))
current_dual_product_full= CUDA.rand(Float64, n)
current_primal_obj_full  = CUDA.rand(Float64, n)
last_gradient_full       = CUDA.rand(Float64, n)
current_gradient_full    = CUDA.rand(Float64, n)
inner_delta_primal_full  = CUDA.rand(Float64, n)
next_primal_full         = CUDA.rand(Float64, n)
last_primal_full         = CUDA.rand(Float64, n)
next_primal_product_full = CUDA.rand(Float64, n)
next_primal_obj_prod_full= CUDA.rand(Float64, n)

# 分块版本：16 个 CuArray
make_vec() = [CUDA.rand(Float64, BLOCKSZ) for _ in 1:NBLOCKS]
current_primal_b      = make_vec()
current_dual_b        = make_vec()
current_primal_obj_b  = make_vec()
last_gradient_b       = make_vec()
current_gradient_b    = make_vec()
inner_delta_primal_b  = make_vec()
next_primal_b         = make_vec()
last_primal_b         = make_vec()
next_primal_product_b = make_vec()
next_primal_obj_prod_b= make_vec()

streams = [CuStream() for _ in 1:NBLOCKS]

# 计算范数以确定 α
norm_Q_full = maximum([norm(Array(blocks_cpu[i])) for i in 1:NBLOCKS])
norm_Qs = [norm(Array(blocks_cpu[i])) for i in 1:NBLOCKS]

step_size   = 1.0
primal_weight = 1.0
CG_bound    = 1.0
first_iter  = true

# ------------------------------------------------------------------
# 8. 热身
# ------------------------------------------------------------------
compute_next_primal_solution_gd_BB!(
    problem_full, current_primal_full, current_dual_product_full,
    current_primal_obj_full, last_gradient_full, step_size, primal_weight,
    CG_bound, current_gradient_full, inner_delta_primal_full,
    next_primal_full, last_primal_full, next_primal_product_full,
    next_primal_obj_prod_full, norm_Q_full, first_iter)

compute_next_primal_solution_gd_BB_blocks!(
    problem_blocks, streams,
    current_primal_b, current_dual_b, current_primal_obj_b,
    last_gradient_b, current_gradient_b, inner_delta_primal_b,
    next_primal_b, last_primal_b, next_primal_product_b,
    next_primal_obj_prod_b,
    step_size, primal_weight, fill(CG_bound, NBLOCKS), norm_Qs, first_iter)

# ------------------------------------------------------------------
# 9. Benchmark
# ------------------------------------------------------------------
println("=== Whole matrix ===")
@btime compute_next_primal_solution_gd_BB!(
    $problem_full, $current_primal_full, $current_dual_product_full,
    $current_primal_obj_full, $last_gradient_full, $step_size, $primal_weight,
    $CG_bound, $current_gradient_full, $inner_delta_primal_full,
    $next_primal_full, $last_primal_full, $next_primal_product_full,
    $next_primal_obj_prod_full, $norm_Q_full, $first_iter)

println("=== 16 streams, block diagonal ===")
@btime compute_next_primal_solution_gd_BB_blocks!(
    $problem_blocks, $streams,
    $current_primal_b, $current_dual_b, $current_primal_obj_b,
    $last_gradient_b, $current_gradient_b, $inner_delta_primal_b,
    $next_primal_b, $last_primal_b, $next_primal_product_b,
    $next_primal_obj_prod_b,
    $step_size, $primal_weight, $(fill(CG_bound, NBLOCKS)), $norm_Qs, $first_iter)