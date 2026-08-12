from pathlib import Path

import numpy as np
import pytest
from pdhcg._core import read_problem_file

from pdhcg import ConeSpec, ConeType


def test_cone_spec_broadcasts_columnar_metadata() -> None:
    starts = 3 * np.arange(4, dtype=np.int32)
    cones = ConeSpec(ConeType.EXP, starts)

    assert len(cones) == 4
    assert cones.types.dtype == np.int32
    assert cones.starts.flags.c_contiguous
    np.testing.assert_array_equal(cones.types, np.full(4, ConeType.EXP, dtype=np.int32))
    np.testing.assert_array_equal(cones.v_dims, np.ones(4, dtype=np.int32))
    np.testing.assert_array_equal(cones.power_alphas, np.zeros(4))
    cones.validate_ambient(12, allow_fixed=True)


def test_cone_spec_stores_heterogeneous_columnar_metadata() -> None:
    cones = ConeSpec(
        np.array([ConeType.SOC, ConeType.EXP], dtype=np.int32),
        np.array([0, 3], dtype=np.int32),
        fixed_mask=np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8),
    )

    np.testing.assert_array_equal(cones.types, [ConeType.SOC, ConeType.EXP])
    np.testing.assert_array_equal(cones.starts, [0, 3])
    np.testing.assert_array_equal(cones.fixed_mask, [0, 0, 0, 0, 1, 0])


def test_cone_spec_validates_power_and_ambient_ranges() -> None:
    with pytest.raises(ValueError, match="alphas"):
        ConeSpec(ConeType.POWER, np.array([0], dtype=np.int32), power_alphas=np.nan)

    cones = ConeSpec(ConeType.SOC, np.array([1], dtype=np.int32), v_dims=2)
    with pytest.raises(ValueError, match="ambient"):
        cones.validate_ambient(4, allow_fixed=True)


def test_psd_cone_uses_matrix_order_and_rejects_fixed_slots() -> None:
    cones = ConeSpec("psd", np.array([0], dtype=np.int32), v_dims=3)
    cones.validate_ambient(6, allow_fixed=True, require_cover=True)

    fixed = ConeSpec(
        ConeType.PSD,
        np.array([0], dtype=np.int32),
        v_dims=2,
        fixed_mask=np.array([0, 1, 0], dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="PSD"):
        fixed.validate_ambient(3, allow_fixed=True)


def test_read_problem_file_returns_columnar_cones() -> None:
    problem = Path(__file__).parent / "data" / "cbf_q3_smoke.cbf"
    raw = read_problem_file(str(problem))
    cones = ConeSpec.from_columnar(raw["cones"])

    assert len(cones) == 1
    assert cones.types[0] == ConeType.SOC
    assert cones.starts[0] == 0

    with pytest.raises(TypeError):
        read_problem_file(str(problem), compact_cones=True)
