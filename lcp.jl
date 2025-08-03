using LinearAlgebra, SparseArrays, JuMP # Required packages

# ========== Adjustable Parameters ==========
T = 5         # Number of stages
n = 100        # Dimension of decision variables per stage
Ξ = 10 # number of scenarios per stage
# ==========================================


N = n * (Ξ^T - 1) ÷ (Ξ - 1) # Total number of decision variables across all stages

function rand_diag_dom(n_dim::Int)
    A = randn(n_dim, n_dim) # Generate a random n_dim x n_dim matrix
    A = A + A' + 2n_dim * I # Make it symmetric and diagonally dominant to ensure SPD
    return Symmetric(A)     # Return as a Symmetric matrix for efficiency and correctness
end

I_rows, J_cols, V_vals = Int[], Int[], Float64[] # For building the sparse A_tilde matrix
b_vec_for_constraints = Float64[]               # Stores the -b_t(ξ) values

# Global row index for building the sparse matrix, starts from 1
row_idx_global = 1

for t in 1:T
    global V_vals, I_rows, J_cols, row_idx_global
    tot_ξ = Ξ^(t - 1) # Total number of scenarios up to stage t
    ξ_kind = randn(Ξ) # Random values for scenarios, can be adjusted as needed
    ξ_value = zeros(Float64, tot_ξ) # Placeholder for scenario values
    for ξ in 1:tot_ξ
        ξ_value[ξ] = ξ_kind[mod(ξ - 1, Ξ) + 1] # Assign the scenario value
    end
    accumulate_tot_ξ = (Ξ^(t - 1) - 1) ÷ (Ξ - 1) 
    accumulate_tot_dim = n * accumulate_tot_ξ # Total dimension for the current stage
    accumulate_tot_dim_next = n * (Ξ^t - 1) ÷ (Ξ - 1) # Total dimension for the next stage
    tot_dim = n * tot_ξ # Total dimension for the current stage
    Mt_block = rand_diag_dom(n)
    if t < T
        Mt_plus_1_block = rand_diag_dom(n) * 0.3 # Coupling matrix for the next stage
    end 
    random_noise_matrix_t = rand_diag_dom(n) # Random noise matrix for the current stage
    random_noise_matrix_t_next = rand_diag_dom(n)  # Random noise matrix for the next stage
    for ξ in 1:tot_ξ # ξ represents the current scenario's position in
        # Add rank-1 noise according to the scenario ξ
        random_noise_matrix = random_noise_matrix_t * ξ_value[ξ] # Random noise scaled by ξ_value
        for i in 1:n # Row index within the current block
            for j in 1:n # Column index within the current block
                # Global row index for the sparse matrix
                push!(I_rows, row_idx_global + i - 1)
                push!(J_cols, accumulate_tot_dim + j + (ξ - 1) * n)
                V_vals = push!(V_vals, Mt_block[i,j] + random_noise_matrix[i,j]) 
            end
        end
        if t < T
            random_noise_matrix_next = random_noise_matrix_t_next * ξ_value[ξ] # Random noise scaled by ξ_value
                for i in 1:n # Row index within the current block
                    for j in 1:n # Column index within the next block
                        push!(I_rows, row_idx_global + i - 1)
                        push!(J_cols, accumulate_tot_dim_next + j + (ξ - 1) * n)
                        V_vals = push!(V_vals, Mt_plus_1_block[i,j] + random_noise_matrix_next[i,j])
                    end
                end
        end
        row_idx_global += n # Update the global row index for the next block
    end
    b_t_val = randn(n * tot_ξ) # Random vector for the right-hand side of constraints
    append!(b_vec_for_constraints, -b_t_val) # Append the negative of b
end

A_tilde = sparse(I_rows, J_cols, V_vals, row_idx_global - 1, N)
Q_qp_objective = (A_tilde + A_tilde')
p_qp_objective = -b_vec_for_constraints
size_Q = size(Q_qp_objective)
nnz_Q = nnz(Q_qp_objective)
# --- 3. Output Dimension Information ---
println("Generated QP instance:")
println("  stages T    = ", T)
println("  dim   n     = ", n)
println("  scen  Ξ     = ", Ξ)
println("  vars  N     = ", N)
println("  constr      = ", size(A_tilde)) # Number of rows in A_tilde (total constraints)
println("  Q shape     = ", size_Q) # Shape of the quadratic objective matrix
println("  Q nnz       = ", nnz_Q) # Number of non-zeros in Q
println("  Sparsity    = ", nnz_Q / (size_Q[1] * size_Q[2])) # Sparsity of Q
# println("  dense Q whole matrix, = ", Matrix(Q_qp_objective)) # Display the dense version of Q for small matrices
# println(" type of Q = ", typeof(Q_qp_objective)) # Type of the quadratic objective matrix
# println("  type of A = ", typeof(A_tilde)) # Type of the constraint matrix
using PDHCG
constructed_problem = PDHCG.constructProblem(
    Q_qp_objective, p_qp_objective, 0.0, A_tilde, -b_vec_for_constraints,
)
println("The problem has been constructed successfully.")
log = PDHCG.pdhcgSolve(
    constructed_problem,
    gpu_flag = true,
    verbose_level = 6,)

time_cost = log.solve_time_sec
obj = log.objective_value
outer_iter = log.iteration_count
inner_iter = log.CG_total_iteration

println("PDHCG Solve Time: $time_cost seconds")
println("PDHCG Optimal Objective: $obj")
println("PDHCG Outer Iterations: $outer_iter")
println("PDHCG Inner Iterations: $inner_iter")

