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
def test_model_native_affine_soc_columnar_input() -> None:
    model = _quiet_model(
        Model(
            objective_vector=np.array([1.0]),
            affine_cone_matrix=sp.csr_matrix([[0.0], [0.0], [1.0]]),
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
def test_cvxpy_constant_exp_rows_use_equalities_instead_of_fixed_slots() -> None:
    cp = pytest.importorskip("cvxpy")
    import pdhcg.cvxpy_backend  # noqa: F401

    z = cp.Variable()
    problem = cp.Problem(cp.Minimize(z), [cp.ExpCone(0.0, 1.0, z)])

    value = problem.solve(solver="PDHCG", eps=1e-6, verbose=False)

    assert problem.status == cp.OPTIMAL
    assert value == pytest.approx(1.0, abs=5e-4)
