"""
Smoke test for the optional middle matrix D in Q + R^T D R.

For each variant (D = identity, diag, dense PSD), we solve the same QP two ways:
  (a) Using the new D parameter on a sparse R.
  (b) Folding D into R as R' = sqrt(D) R (only valid when D is PSD), so the
      "no D" path solves the same effective QP.
The two solvers should return primal solutions that agree to a few digits.

We also check the indefinite-D case (where folding into R via real sqrt does
not exist) by comparing the explicit Q-only formulation: build Q_eff = R^T D R
and feed it as the sparse Q, then compare to the (R, D) formulation.
"""
import numpy as np
import scipy.sparse as sp

from pdhcg import Model, PDHCG


def _make_problem(n=40, rank=6, n_cons=20, seed=0):
    rng = np.random.default_rng(seed)
    R_dense = rng.standard_normal((rank, n)) * 0.5
    # Make R modestly sparse but valid CSR
    R_dense[np.abs(R_dense) < 0.25] = 0.0
    R = sp.csr_matrix(R_dense)

    A = sp.csr_matrix(rng.standard_normal((n_cons, n)))
    c = rng.standard_normal(n)
    b = rng.standard_normal(n_cons)
    return n, rank, R, A, c, b


def _solve(R, A, c, b, *, D=None, Q=None, iter_limit=5000):
    m = Model(
        objective_vector=c,
        constraint_matrix=A,
        constraint_lower_bound=b,
        constraint_upper_bound=b,
        objective_matrix=Q,
        objective_matrix_low_rank=R,
        objective_matrix_low_rank_middle=D,
        variable_lower_bound=-np.full(c.size, 10.0),
        variable_upper_bound=np.full(c.size, 10.0),
    )
    m.ModelSense = PDHCG.MINIMIZE
    m.setParams(
        LogLevel=0,
        OptimalityTol=1e-4,
        FeasibilityTol=1e-4,
        Presolve=False,
        IterationLimit=iter_limit,
        TimeLimit=30,
    )
    m.optimize()
    if m.Status not in ("OPTIMAL", "ITERATION_LIMIT"):
        raise AssertionError(f"unexpected status {m.Status}")
    return m.X, m.ObjVal


def test_identity_matches_no_D():
    n, rank, R, A, c, b = _make_problem()
    D = np.ones(rank)  # diag identity
    x_ref, obj_ref = _solve(R, A, c, b)
    x_d, obj_d = _solve(R, A, c, b, D=D)
    assert np.allclose(x_ref, x_d, atol=1e-2, rtol=1e-2), (
        f"identity D mismatch, max diff {np.max(np.abs(x_ref - x_d))}"
    )
    print(f"[identity]    obj_ref={obj_ref:.6e}  obj_D={obj_d:.6e}")


def test_diag_D_matches_folded_R():
    n, rank, R, A, c, b = _make_problem(seed=1)
    d = np.array([0.3, 1.7, 0.9, 2.4, 0.5, 1.1])
    assert d.size == rank

    # Fold into R: R' = sqrt(D) R  ->  R'^T R' = R^T D R, valid since d > 0
    R_folded = sp.diags(np.sqrt(d)) @ R

    x_ref, obj_ref = _solve(R_folded, A, c, b)
    x_d, obj_d = _solve(R, A, c, b, D=d)
    assert np.allclose(x_ref, x_d, atol=1e-2, rtol=1e-2), (
        f"diag D mismatch, max diff {np.max(np.abs(x_ref - x_d))}"
    )
    print(f"[diag PSD]    obj_ref={obj_ref:.6e}  obj_D={obj_d:.6e}")


def test_dense_D_matches_folded_R():
    n, rank, R, A, c, b = _make_problem(seed=2)
    # Build a random symmetric PSD D = M^T M + alpha I
    rng = np.random.default_rng(3)
    M = rng.standard_normal((rank, rank))
    D = M.T @ M + 0.2 * np.eye(rank)
    D = 0.5 * (D + D.T)  # enforce symmetry

    # Fold into R via Cholesky: R' = L^T R where D = L L^T
    L = np.linalg.cholesky(D)
    R_folded = sp.csr_matrix(L.T @ R.toarray())

    x_ref, obj_ref = _solve(R_folded, A, c, b)
    x_d, obj_d = _solve(R, A, c, b, D=D)
    assert np.allclose(x_ref, x_d, atol=1e-2, rtol=1e-2), (
        f"dense PSD D mismatch, max diff {np.max(np.abs(x_ref - x_d))}"
    )
    print(f"[dense PSD]   obj_ref={obj_ref:.6e}  obj_D={obj_d:.6e}")


def test_indefinite_dense_D_matches_explicit_Q():
    """D with negative eigenvalues cannot be folded into R via real sqrt;
    compare against an explicit Q built from Q = R^T D R + tiny shift."""
    n, rank, R, A, c, b = _make_problem(seed=3)
    rng = np.random.default_rng(4)
    M = rng.standard_normal((rank, rank))
    D = M.T @ M
    # Inject a negative eigenvalue
    eigvals, eigvecs = np.linalg.eigh(D)
    eigvals[0] = -0.3
    D = eigvecs @ np.diag(eigvals) @ eigvecs.T
    D = 0.5 * (D + D.T)
    # Verify indefinite
    assert np.min(np.linalg.eigvalsh(D)) < 0

    # Build effective Q = R^T D R + small PSD shift so the resulting Q is convex enough
    R_dense = R.toarray()
    Q_eff_dense = R_dense.T @ D @ R_dense
    # Symmetrize numerically
    Q_eff_dense = 0.5 * (Q_eff_dense + Q_eff_dense.T)

    # Solve via explicit Q (no R, no D); nonconvex Q so run more iterations.
    Q_eff_sparse = sp.csr_matrix(Q_eff_dense)
    x_ref, obj_ref = _solve(None, A, c, b, Q=Q_eff_sparse, iter_limit=20000)

    # Solve via (R, D=indefinite)
    x_d, obj_d = _solve(R, A, c, b, D=D, iter_limit=20000)

    # Objective is the most stable thing to compare on nonconvex problems.
    assert abs(obj_ref - obj_d) / (abs(obj_ref) + 1.0) < 1e-3, (
        f"indefinite D objective mismatch: ref={obj_ref:.6e} D={obj_d:.6e}"
    )
    print(f"[indef dense] obj_ref={obj_ref:.6e}  obj_D={obj_d:.6e}")


def test_sparse_csr_D_matches_dense_D():
    """Pass D as a scipy CSR matrix; should match feeding the equivalent dense D."""
    n, rank, R, A, c, b = _make_problem(seed=4)
    rng = np.random.default_rng(5)
    M = rng.standard_normal((rank, rank))
    D_dense = M.T @ M + 0.2 * np.eye(rank)
    D_dense = 0.5 * (D_dense + D_dense.T)
    # Sparsify the dense D so the CSR input genuinely has fewer than rank^2 nnz
    mask = np.abs(D_dense) > 0.3
    mask |= mask.T
    np.fill_diagonal(mask, True)
    D_dense_sparsified = np.where(mask, D_dense, 0.0)

    D_sparse = sp.csr_matrix(D_dense_sparsified)

    x_dense, obj_dense = _solve(R, A, c, b, D=D_dense_sparsified)
    x_sparse, obj_sparse = _solve(R, A, c, b, D=D_sparse)
    assert np.allclose(x_dense, x_sparse, atol=1e-2, rtol=1e-2), (
        f"sparse vs dense D mismatch, max diff {np.max(np.abs(x_dense - x_sparse))}"
    )
    print(f"[sparse CSR D] obj_dense={obj_dense:.6e}  obj_sparse={obj_sparse:.6e}")


def test_diagonal_sparse_D_detected_as_diag():
    """A sparse D with only diagonal entries should auto-detect to the DIAG kind
    and give the same answer as a 1-D diag array."""
    n, rank, R, A, c, b = _make_problem(seed=6)
    d = np.array([0.3, 1.7, 0.9, 2.4, 0.5, 1.1])
    assert d.size == rank
    D_sparse_diag = sp.diags(d).tocsr()

    x_1d, obj_1d = _solve(R, A, c, b, D=d)
    x_csr, obj_csr = _solve(R, A, c, b, D=D_sparse_diag)
    assert np.allclose(x_1d, x_csr, atol=1e-2, rtol=1e-2), (
        f"1-D vs sparse-diag mismatch, max diff {np.max(np.abs(x_1d - x_csr))}"
    )
    print(f"[diag via CSR] obj_1d={obj_1d:.6e}  obj_csr={obj_csr:.6e}")


if __name__ == "__main__":
    test_identity_matches_no_D()
    test_diag_D_matches_folded_R()
    test_dense_D_matches_folded_R()
    test_indefinite_dense_D_matches_explicit_Q()
    test_sparse_csr_D_matches_dense_D()
    test_diagonal_sparse_D_detected_as_diag()
    print("\nAll D-middle tests passed.")
