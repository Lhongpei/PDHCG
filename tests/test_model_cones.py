import numpy as np
import pytest
import scipy.sparse as sp
from pdhcg._core import solve_once

from pdhcg import ConeSpec, ConeType, Model


def _quiet_model(model: Model) -> Model:
    model.setParams(
        TimeLimit=30.0,
        FeasibilityTol=1e-6,
        OptimalityTol=1e-6,
        LogLevel=0,
    )
    return model


@pytest.mark.gpu
def test_model_variable_soc_columnar_input() -> None:
    model = _quiet_model(
        Model(
            objective_vector=np.array([0.0, 0.0, 1.0]),
            constraint_matrix=sp.csr_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            constraint_lower_bound=np.array([3.0, 4.0]),
            constraint_upper_bound=np.array([3.0, 4.0]),
            variable_cones=ConeSpec(
                ConeType.SOC,
                np.array([0], dtype=np.int32),
                v_dims=1,
            ),
        )
    )

    model.optimize()

    assert model.Status == "OPTIMAL"
    np.testing.assert_allclose(model.X, [3.0, 4.0, 5.0], atol=2e-4)


def test_model_rejects_legacy_cone_dicts() -> None:
    with pytest.raises(TypeError, match="ConeSpec"):
        Model(
            objective_vector=np.zeros(3),
            variable_cones=[{"type": "soc", "start_idx": 0, "v_dim": 1}],
        )


@pytest.mark.gpu
def test_low_level_rejects_legacy_cone_dicts() -> None:
    with pytest.raises(ValueError, match="ConeSpec"):
        solve_once(
            None,
            None,
            sp.csr_matrix((0, 3)),
            np.zeros(3),
            cones=[{"type": "soc", "start_idx": 0, "v_dim": 1}],
        )


@pytest.mark.gpu
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "csr"])
def test_model_native_affine_soc_columnar_input(sparse: bool) -> None:
    affine_matrix = np.array([[0.0], [0.0], [1.0]])
    if sparse:
        affine_matrix = sp.csr_matrix(affine_matrix)
    model = _quiet_model(
        Model(
            objective_vector=np.array([1.0]),
            affine_cone_matrix=affine_matrix,
            affine_cone_offset=np.array([3.0, 4.0, 0.0]),
            affine_cones=ConeSpec(
                ConeType.SOC,
                np.array([0], dtype=np.int32),
                v_dims=1,
            ),
        )
    )

    model.optimize()

    assert model.Status == "OPTIMAL"
    np.testing.assert_allclose(model.X, [5.0], atol=2e-4)


@pytest.mark.gpu
def test_model_variable_psd_svec_input() -> None:
    sqrt_two = np.sqrt(2.0)
    model = _quiet_model(
        Model(
            objective_vector=np.array([0.0, 0.0, 1.0]),
            constraint_matrix=sp.csr_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            constraint_lower_bound=np.array([1.0, 2.0 * sqrt_two]),
            constraint_upper_bound=np.array([1.0, 2.0 * sqrt_two]),
            variable_cones=ConeSpec(ConeType.PSD, np.array([0], dtype=np.int32), v_dims=2),
        )
    )

    model.optimize()

    assert model.Status == "OPTIMAL"
    np.testing.assert_allclose(model.X, [1.0, 2.0 * sqrt_two, 4.0], atol=8e-4)


@pytest.mark.gpu
def test_model_native_affine_psd_svec_input() -> None:
    sqrt_two = np.sqrt(2.0)
    model = _quiet_model(
        Model(
            objective_vector=np.array([1.0]),
            affine_cone_matrix=sp.csr_matrix([[1.0], [0.0], [0.0]]),
            affine_cone_offset=np.array([0.0, sqrt_two, 1.0]),
            affine_cones=ConeSpec(ConeType.PSD, np.array([0], dtype=np.int32), v_dims=2),
        )
    )

    model.optimize()

    assert model.Status == "OPTIMAL"
    np.testing.assert_allclose(model.X, [1.0], atol=8e-4)


@pytest.mark.gpu
def test_cvxpy_constant_exp_rows_use_equalities_instead_of_fixed_slots() -> None:
    cp = pytest.importorskip("cvxpy")
    import pdhcg.cvxpy_backend  # noqa: F401

    z = cp.Variable()
    problem = cp.Problem(cp.Minimize(z), [cp.ExpCone(0.0, 1.0, z)])

    value = problem.solve(solver="PDHCG", eps=1e-6, verbose=False)

    assert problem.status == cp.OPTIMAL
    assert value == pytest.approx(1.0, abs=5e-4)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("constraint_kind", "expected_dual"),
    [("nonnegative", 1.0), ("equality", -1.0)],
)
def test_cvxpy_linear_dual_signs(constraint_kind: str, expected_dual: float) -> None:
    cp = pytest.importorskip("cvxpy")
    import pdhcg.cvxpy_backend  # noqa: F401

    x = cp.Variable()
    constraint = x >= 1.0 if constraint_kind == "nonnegative" else x == 1.0
    problem = cp.Problem(cp.Minimize(x), [constraint])

    problem.solve(solver="PDHCG", eps=1e-7, verbose=False)

    assert problem.status == cp.OPTIMAL
    assert x.value == pytest.approx(1.0, abs=5e-5)
    assert constraint.dual_value == pytest.approx(expected_dual, abs=5e-5)


@pytest.mark.gpu
def test_cvxpy_soc_dual_sign_and_order() -> None:
    cp = pytest.importorskip("cvxpy")
    import pdhcg.cvxpy_backend  # noqa: F401

    u = cp.Variable(2)
    t = cp.Variable()
    fixed = u == np.array([3.0, 4.0])
    cone = cp.SOC(t, u)
    problem = cp.Problem(cp.Minimize(t), [fixed, cone])

    problem.solve(solver="PDHCG", eps=1e-7, verbose=False)

    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(fixed.dual_value, [-0.6, -0.8], atol=5e-5)
    np.testing.assert_allclose(cone.dual_value[0], [1.0], atol=5e-5)
    np.testing.assert_allclose(cone.dual_value[1].ravel(), [-0.6, -0.8], atol=5e-5)


@pytest.mark.gpu
def test_cvxpy_psd_primal_and_dual_svec_order() -> None:
    cp = pytest.importorskip("cvxpy")
    import pdhcg.cvxpy_backend  # noqa: F401

    matrix = cp.Variable((2, 2), symmetric=True)
    fixed = [matrix[0, 0] == 1.0, matrix[0, 1] == 2.0]
    cone = matrix >> 0
    problem = cp.Problem(cp.Minimize(matrix[1, 1]), [*fixed, cone])

    problem.solve(solver="PDHCG", eps=1e-6, verbose=False)

    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(matrix.value, [[1.0, 2.0], [2.0, 4.0]], atol=2e-3)
    np.testing.assert_allclose(cone.dual_value, [[4.0, -2.0], [-2.0, 1.0]], atol=3e-3)
