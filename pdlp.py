import numpy as np

class PDLPSolver:
    """
    A PDLP-like solver for Linear Programs using the Primal-Dual Hybrid Gradient (PDHG) method.

    The problem is assumed to be in the form:
    minimize c^T x
    subject to A x = b
               x >= 0

    Key features include:
    - Core PDHG updates for primal (x) and dual (y) variables.
    - Diagonal preconditioning of the constraint matrix A, objective c, and RHS b.
    - Adaptive step size adjustment based on primal and dual residual norms.
    - Gap-based restart strategy to improve convergence.
    - Comprehensive convergence metrics including relative primal/dual residuals and gap.
    """

    def __init__(self, A, b, c,
                 max_iter=10000,
                 tol_primal_res=1e-6,
                 tol_dual_res=1e-6,
                 tol_gap=1e-6,
                 initial_sigma=1.0,
                 initial_tau=1.0,
                 theta=1.0, # Extrapolation parameter for PDHG (typically 1.0 for PDLP)
                 precondition=True,
                 adaptive_step_size=True,
                 restart_strategy="gap_based", # Options: "gap_based" or None
                 restart_cadence=100, # How often to check for restart
                 restart_threshold=0.99 # Restart if current gap is > this_threshold * best_gap
                ):
        """
        Initializes the PDLP solver.

        Args:
            A (np.ndarray): Constraint matrix (m x n).
            b (np.ndarray): Right-hand side vector (m,).
            c (np.ndarray): Objective function coefficients (n,).
            max_iter (int): Maximum number of iterations.
            tol_primal_res (float): Tolerance for relative primal residual.
            tol_dual_res (float): Tolerance for relative dual residual.
            tol_gap (float): Tolerance for relative primal-dual gap.
            initial_sigma (float): Initial primal step size.
            initial_tau (float): Initial dual step size.
            theta (float): Extrapolation parameter for the dual variable update.
                           A value of 1.0 corresponds to standard PDHG extrapolation.
            precondition (bool): If True, applies diagonal preconditioning to A, b, c.
            adaptive_step_size (bool): If True, dynamically adjusts sigma and tau.
            restart_strategy (str or None): Strategy for restarts. "gap_based" restarts
                                            if the primal-dual gap significantly worsens.
            restart_cadence (int): The number of iterations between restart checks.
            restart_threshold (float): Factor by which the current gap can exceed the best
                                       gap before a restart is triggered (e.g., 0.99 means
                                       if current_gap > 0.99 * best_gap, consider restarting).
        """
        self.A_orig = A.copy() # Store original A
        self.b_orig = b.copy() # Store original b
        self.c_orig = c.copy() # Store original c

        self.A = A
        self.b = b
        self.c = c

        self.m, self.n = A.shape

        self.max_iter = max_iter
        self.tol_primal_res = tol_primal_res
        self.tol_dual_res = tol_dual_res
        self.tol_gap = tol_gap

        self._initial_sigma = initial_sigma # Store for restarts
        self._initial_tau = initial_tau     # Store for restarts
        self.sigma = initial_sigma
        self.tau = initial_tau
        self.theta = theta

        self.precondition = precondition
        self.adaptive_step_size = adaptive_step_size
        self.restart_strategy = restart_strategy
        self.restart_cadence = restart_cadence
        self.restart_threshold = restart_threshold

        self.D_R = None # Row scaling matrix (for preconditioning)
        self.D_C = None # Column scaling matrix (for preconditioning)

    def _apply_preconditioning(self):
        """
        Applies diagonal preconditioning to A, b, c.
        This aims to improve the conditioning of the problem, leading to faster convergence.
        It scales rows and columns of A to have unit L2 norm.
        """
        print("Applying diagonal preconditioning...")
        # Calculate row norms of A
        row_norms = np.linalg.norm(self.A, axis=1)
        # Avoid division by zero for rows with all zeros
        row_norms[row_norms == 0] = 1.0
        self.D_R = np.diag(1.0 / row_norms)

        # Calculate column norms of A
        col_norms = np.linalg.norm(self.A, axis=0)
        # Avoid division by zero for columns with all zeros
        col_norms[col_norms == 0] = 1.0
        self.D_C = np.diag(1.0 / col_norms)

        # Apply scaling to A, b, c
        self.A = self.D_R @ self.A @ self.D_C
        self.b = self.D_R @ self.b
        self.c = self.D_C @ self.c

        # Adjust initial step sizes based on the scaled matrix norm.
        # A common heuristic is sigma * tau * ||A_scaled||_F^2 approx 1.
        # This helps in choosing initial steps that are appropriate for the scaled problem.
        norm_A_scaled = np.linalg.norm(self.A, ord='fro')
        if norm_A_scaled > 0:
            self.sigma = 1.0 / norm_A_scaled
            self.tau = 1.0 / norm_A_scaled
        print(f"Preconditioning applied. Initial sigma: {self.sigma:.2e}, initial tau: {self.tau:.2e}")

    def _calculate_metrics(self, x, y, use_original_problem=False):
        """
        Calculates primal residual, dual residual, and primal-dual gap.
        Can calculate using either the scaled problem (default) or the original problem.

        Args:
            x (np.ndarray): Current primal variable.
            y (np.ndarray): Current dual variable.
            use_original_problem (bool): If True, uses the original A_orig, b_orig, c_orig
                                         for metric calculation. Useful for final results.

        Returns:
            dict: A dictionary containing various convergence metrics.
        """
        if use_original_problem:
            A_curr, b_curr, c_curr = self.A_orig, self.b_orig, self.c_orig
            # If preconditioning was applied, transform x and y back to original scale
            if self.precondition:
                x_orig_scale = self.D_C @ x
                y_orig_scale = self.D_R @ y
            else:
                x_orig_scale = x
                y_orig_scale = y
        else:
            A_curr, b_curr, c_curr = self.A, self.b, self.c
            x_orig_scale = x # Already in scaled domain
            y_orig_scale = y # Already in scaled domain

        # Primal residual for equality constraint (Ax = b)
        primal_res_vec_eq = A_curr @ x_orig_scale - b_curr
        primal_res_norm = np.linalg.norm(primal_res_vec_eq)

        # Dual residual for inequality constraint (A^T y <= c)
        # This measures the violation of A^T y <= c
        dual_res_vec_ineq = A_curr.T @ y_orig_scale - c_curr
        dual_res_norm = np.linalg.norm(np.maximum(0, dual_res_vec_ineq))

        # Primal-Dual Gap
        primal_obj = c_curr.T @ x_orig_scale
        dual_obj = b_curr.T @ y_orig_scale
        primal_dual_gap = primal_obj - dual_obj

        # Relative residuals and gap for termination, using robust denominators
        # These denominators help in handling cases where norms might be small.
        norm_x = np.linalg.norm(x_orig_scale)
        norm_y = np.linalg.norm(y_orig_scale)
        norm_b = np.linalg.norm(b_curr)
        norm_c = np.linalg.norm(c_curr)
        norm_A_x = np.linalg.norm(A_curr @ x_orig_scale)
        norm_A_T_y = np.linalg.norm(A_curr.T @ y_orig_scale)

        denominator_primal = 1 + norm_b + norm_A_x
        denominator_dual = 1 + norm_c + norm_A_T_y
        denominator_gap = 1 + abs(primal_obj) + abs(dual_obj)

        rel_primal_res = primal_res_norm / denominator_primal
        rel_dual_res = dual_res_norm / denominator_dual
        rel_gap = abs(primal_dual_gap) / denominator_gap

        return {
            "primal_res_norm": primal_res_norm,
            "dual_res_norm": dual_res_norm,
            "primal_dual_gap": primal_dual_gap,
            "primal_obj": primal_obj,
            "dual_obj": dual_obj,
            "rel_primal_res": rel_primal_res,
            "rel_dual_res": rel_dual_res,
            "rel_gap": rel_gap
        }

    def _check_termination(self, metrics):
        """Checks if the termination criteria are met based on relative residuals and gap."""
        return (metrics["rel_primal_res"] < self.tol_primal_res and
                metrics["rel_dual_res"] < self.tol_dual_res and
                metrics["rel_gap"] < self.tol_gap)

    def solve(self):
        """
        Solves the Linear Program using the PDLP algorithm.

        Returns:
            tuple: (x_final, y_final, history, final_metrics)
                   x_final (np.ndarray): Optimal primal solution.
                   y_final (np.ndarray): Optimal dual solution.
                   history (list): List of dictionaries, each containing metrics for an iteration.
                   final_metrics (dict): Metrics at the termination of the solver.
        """
        # Apply preconditioning if enabled
        if self.precondition:
            self._apply_preconditioning()

        # Initialize primal and dual variables to zeros
        x = np.zeros(self.n)
        y = np.zeros(self.m)
        # y_prev is used for extrapolation (y_bar = y + theta * (y - y_prev))
        y_prev = np.zeros(self.m)

        # Variables to store the best solution found so far (for restarts)
        best_x = x.copy()
        best_y = y.copy()
        min_gap = np.inf # Stores the minimum primal-dual gap encountered

        # History to record metrics at each iteration
        history = []

        print("Starting PDLP iterations...")
        for k in range(self.max_iter):
            # 1. Extrapolated dual variable (y_bar)
            # This extrapolation helps in accelerating convergence.
            y_bar = y + self.theta * (y - y_prev)

            # 2. Primal update step
            # x^{k+1} = proj_{x>=0}(x^k - sigma_k * (c - A^T * y_bar^k))
            # The term (c - A^T * y_bar) can be seen as a subgradient of the primal objective.
            # np.maximum(0, ...) performs the projection onto the non-negative orthant.
            x_new_unprojected = x - self.sigma * (self.c - self.A.T @ y_bar)
            x_new = np.maximum(0, x_new_unprojected)

            # 3. Dual update step
            # y^{k+1} = y^k + tau_k * (b - A * x^{k+1})
            # The term (b - A * x_new) is the primal residual, driving dual feasibility.
            y_new = y + self.tau * (self.b - self.A @ x_new)

            # Update variables for the next iteration
            y_prev = y.copy() # Store current y to be y_prev in the next iteration
            x = x_new
            y = y_new

            # Calculate and store current iteration metrics
            metrics = self._calculate_metrics(x, y)
            history.append(metrics)

            # Print progress periodically
            if k % 100 == 0 or k == self.max_iter - 1:
                print(f"Iter {k}: RelPrimalRes={metrics['rel_primal_res']:.2e}, "
                      f"RelDualRes={metrics['rel_dual_res']:.2e}, "
                      f"RelGap={metrics['rel_gap']:.2e}")

            # Check for convergence
            if self._check_termination(metrics):
                print(f"Converged at iteration {k}.")
                break

            # Adaptive Step Size Adjustment
            # This heuristic aims to balance the progress made in primal and dual feasibility.
            # If one residual is much larger, the step sizes are adjusted to prioritize
            # reducing that residual, while maintaining a balance.
            if self.adaptive_step_size and k > 0:
                # Add a small epsilon to avoid division by zero
                ratio = metrics['primal_res_norm'] / (metrics['dual_res_norm'] + 1e-10)

                # Adjust step sizes: if primal residual is dominant, increase sigma (primal step),
                # decrease tau (dual step) and vice-versa.
                if ratio > 5: # Primal residual is much larger
                    self.sigma *= 1.1
                    self.tau /= 1.1
                elif ratio < 1/5: # Dual residual is much larger
                    self.sigma /= 1.1
                    self.tau *= 1.1
                # Else, step sizes remain unchanged for this iteration.

            # Restart Strategy (Gap-based)
            # If enabled, periodically checks if the primal-dual gap has significantly
            # worsened compared to the best gap found so far. If so, it resets the
            # iterates to the 'best_x' and 'best_y' found, and resets step sizes.
            if self.restart_strategy == "gap_based" and k > 0 and (k + 1) % self.restart_cadence == 0:
                current_gap = metrics['primal_dual_gap']
                # Update the best gap and corresponding solution
                if current_gap < min_gap:
                    min_gap = current_gap
                    best_x = x.copy()
                    best_y = y.copy()
                else:
                    # If current gap is significantly worse than the best, consider restarting
                    if min_gap != np.inf and current_gap > self.restart_threshold * min_gap:
                        print(f"Restarting at iteration {k}. Gap worsened. Resetting to best solution.")
                        x = best_x.copy()
                        y = best_y.copy()
                        y_prev = y.copy() # Reset y_prev for correct extrapolation after restart

                        # Reset step sizes to initial values (potentially re-calculated if preconditioned)
                        if self.precondition:
                            norm_A_scaled = np.linalg.norm(self.A, ord='fro')
                            if norm_A_scaled > 0:
                                self.sigma = 1.0 / norm_A_scaled
                                self.tau = 1.0 / norm_A_scaled
                        else:
                            self.sigma = self._initial_sigma
                            self.tau = self._initial_tau
                        min_gap = np.inf # Reset min_gap after restart to find new best

        # If preconditioning was applied, transform the final solution back to the original scale
        if self.precondition:
            x_final = self.D_C @ x
            y_final = self.D_R @ y
        else:
            x_final = x
            y_final = y

        # Calculate final metrics using the original problem data
        final_metrics = self._calculate_metrics(x_final, y_final, use_original_problem=True)

        print("\nOptimization Finished.")
        print(f"Final Primal Objective: {final_metrics['primal_obj']:.6f}")
        print(f"Final Dual Objective: {final_metrics['dual_obj']:.6f}")
        print(f"Final Primal-Dual Gap: {final_metrics['primal_dual_gap']:.4e}")
        print(f"Final Relative Primal Residual: {final_metrics['rel_primal_res']:.2e}")
        print(f"Final Relative Dual Residual: {final_metrics['rel_dual_res']:.2e}")
        print(f"Final Relative Gap: {final_metrics['rel_gap']:.2e}")

        return x_final, y_final, history, final_metrics

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Solving Example LP Problem ---")
    # Example LP Problem:
    # minimize x1 + x2
    # subject to:
    # 2x1 + x2 = 4
    # x1 + 2x2 = 5
    # x1, x2 >= 0

    # Optimal solution: x1 = 1, x2 = 2. Objective = 3.

    A_example = np.array([[2., 1.],
                          [1., 2.]])
    b_example = np.array([4., 5.])
    c_example = np.array([1., 1.])

    # Initialize and solve the LP using the PDLPSolver
    solver = PDLPSolver(A_example, b_example, c_example,
                        max_iter=5000, # Increased iterations for better convergence
                        tol_primal_res=1e-7,
                        tol_dual_res=1e-7,
                        tol_gap=1e-7,
                        precondition=True,
                        adaptive_step_size=True,
                        restart_strategy="gap_based",
                        restart_cadence=200,
                        restart_threshold=0.99)

    x_optimal, y_optimal, history, final_metrics = solver.solve()

    print("\n--- Results ---")
    print(f"Optimal Primal Solution (x): {x_optimal}")
    print(f"Optimal Dual Solution (y): {y_optimal}")
    print(f"Optimal Objective Value: {final_metrics['primal_obj']:.6f}")

    # Another example: LP with inequality constraints (converted to equality with slack)
    print("\n--- Solving LP with Inequality Constraints ---")
    # Minimize -x1 - 2x2
    # Subject to:
    # x1 + x2 <= 10  (becomes x1 + x2 + s1 = 10, s1 >= 0)
    # x1      <= 5   (becomes x1 + s2 = 5, s2 >= 0)
    # x2      <= 6   (becomes x2 + s3 = 6, s3 >= 0)
    # x1, x2 >= 0

    # Convert to standard form (Ax = b, x >= 0)
    # Original variables: x1, x2
    # Slack variables: s1, s2, s3
    # New variables: x_tilde = [x1, x2, s1, s2, s3]

    A_ineq_orig = np.array([[1., 1.],
                            [1., 0.],
                            [0., 1.]])
    b_ineq_orig = np.array([10., 5., 6.])
    c_ineq_orig = np.array([-1., -2.])

    # Construct A_tilde, b_tilde, c_tilde
    num_orig_vars = c_ineq_orig.shape[0]
    num_ineq_constraints = A_ineq_orig.shape[0]

    A_tilde = np.hstack((A_ineq_orig, np.eye(num_ineq_constraints)))
    b_tilde = b_ineq_orig
    c_tilde = np.hstack((c_ineq_orig, np.zeros(num_ineq_constraints)))

    solver_ineq = PDLPSolver(A_tilde, b_tilde, c_tilde,
                             max_iter=10000,
                             tol_primal_res=1e-7,
                             tol_dual_res=1e-7,
                             tol_gap=1e-7,
                             precondition=True,
                             adaptive_step_size=True,
                             restart_strategy="gap_based",
                             restart_cadence=200,
                             restart_threshold=0.99)

    x_tilde_optimal, y_ineq_optimal, history_ineq, final_metrics_ineq = solver_ineq.solve()

    print("\n--- Results for Inequality LP ---")
    print(f"Optimal Primal Solution (x_tilde): {x_tilde_optimal}")
    print(f"  (x1, x2) = ({x_tilde_optimal[0]:.4f}, {x_tilde_optimal[1]:.4f})")
    print(f"  (s1, s2, s3) = ({x_tilde_optimal[2]:.4f}, {x_tilde_optimal[3]:.4f}, {x_tilde_optimal[4]:.4f})")
    print(f"Optimal Dual Solution (y): {y_ineq_optimal}")
    print(f"Optimal Objective Value: {final_metrics_ineq['primal_obj']:.6f}")
    # Expected optimal for this problem: x1=4, x2=6. Objective = -16.
    # s1 = 10 - (4+6) = 0
    # s2 = 5 - 4 = 1
    # s3 = 6 - 6 = 0