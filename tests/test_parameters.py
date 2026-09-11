import numpy as np
import pytest
from pdhcg._core import get_default_params, validate_params

from pdhcg import Model


def test_parameter_defaults_pass_core_validation() -> None:
    defaults = get_default_params()

    assert "eps_infeasible" in defaults
    validate_params(defaults)


def test_set_param_rejects_unknown_name() -> None:
    model = Model(objective_vector=np.zeros(1))

    with pytest.raises(ValueError, match="Unknown parameter"):
        model.setParam("does_not_exist", 1)


def test_set_param_is_transactional() -> None:
    model = Model(objective_vector=np.zeros(1))
    original_frequency = model.getParam("TermCheckFreq")

    with pytest.raises(ValueError, match="termination_evaluation_frequency"):
        model.setParam("TermCheckFreq", 0)

    assert model.getParam("TermCheckFreq") == original_frequency


def test_set_params_is_transactional() -> None:
    model = Model(objective_vector=np.zeros(1))
    original_time_limit = model.getParam("TimeLimit")

    with pytest.raises(ValueError, match="eps_optimal_relative"):
        model.setParams(TimeLimit=2.0, OptimalityTol=0.0)

    assert model.getParam("TimeLimit") == original_time_limit


def test_infeasible_tolerance_alias() -> None:
    model = Model(objective_vector=np.zeros(1))

    model.setParam("InfeasibleTol", 1e-10)

    assert model.getParam("eps_infeasible") == pytest.approx(1e-10)
