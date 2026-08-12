# Copyright 2026 Hongpei Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from enum import IntEnum
from numbers import Integral
from typing import Any, Mapping, Optional

import numpy as np


class ConeType(IntEnum):
    """Cone type codes shared with the C API."""

    RSOC = 0
    SOC = 1
    EXP = 2
    POWER = 3
    PSD = 4

    ROTATED_SOC = RSOC
    STANDARD_SOC = SOC
    EXPONENTIAL = EXP


_CONE_TYPE_NAMES = {
    "rsoc": ConeType.RSOC,
    "soc": ConeType.SOC,
    "exp": ConeType.EXP,
    "power": ConeType.POWER,
    "psd": ConeType.PSD,
}


def _cone_type_code(value: Any) -> int:
    if isinstance(value, str):
        try:
            return int(_CONE_TYPE_NAMES[value.lower()])
        except KeyError as exc:
            raise ValueError("cone type must be 'soc', 'rsoc', 'exp', 'power', or 'psd'") from exc
    if not isinstance(value, Integral):
        raise TypeError("cone type must be a ConeType, integer code, or string")
    code = int(value)
    if code < int(ConeType.RSOC) or code > int(ConeType.PSD):
        raise ValueError(f"invalid cone type code {code}")
    return code


def _as_i32_vector(value: Any, count: int, name: str, default: Optional[int] = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return np.full(count, default, dtype=np.int32)

    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    if array.ndim == 0:
        scalar = int(array)
        if scalar < np.iinfo(np.int32).min or scalar > np.iinfo(np.int32).max:
            raise OverflowError(f"{name} value {scalar} does not fit int32")
        return np.full(count, scalar, dtype=np.int32)
    if array.ndim != 1 or array.size != count:
        raise ValueError(f"{name} must be a scalar or a 1D array of length {count}")
    if array.size and (
        np.min(array) < np.iinfo(np.int32).min or np.max(array) > np.iinfo(np.int32).max
    ):
        raise OverflowError(f"{name} contains a value that does not fit int32")
    return np.ascontiguousarray(array, dtype=np.int32)


def _as_f64_vector(value: Any, count: int, name: str, default: float) -> np.ndarray:
    if value is None:
        return np.full(count, default, dtype=np.float64)
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(count, float(array), dtype=np.float64)
    if array.ndim != 1 or array.size != count:
        raise ValueError(f"{name} must be a scalar or a 1D array of length {count}")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_type_vector(value: Any, count: int) -> np.ndarray:
    if isinstance(value, (str, ConeType, int, np.integer)):
        return np.full(count, _cone_type_code(value), dtype=np.int32)
    array = np.asarray(value)
    if array.ndim != 1 or array.size != count:
        raise ValueError(f"types must be a scalar or a 1D array of length {count}")
    if array.dtype.kind in "iu":
        if array.size and (array.min() < int(ConeType.RSOC) or array.max() > int(ConeType.PSD)):
            raise ValueError("types contains an invalid cone type code")
        return np.ascontiguousarray(array, dtype=np.int32)
    return np.fromiter((_cone_type_code(item) for item in array), dtype=np.int32, count=count)


class ConeSpec:
    """Columnar description of one or more cone blocks.

    ``types``, ``v_dims``, and ``power_alphas`` may be scalars and are then
    broadcast to all entries in ``starts``. For variable cones, ``starts`` are
    variable indices. For affine cones, they are rows of the separately supplied
    affine map ``F``. For PSD cones, ``v_dims`` stores the matrix order and each
    block occupies lower-triangular column-major ``svec`` coordinates.
    """

    __slots__ = ("types", "starts", "v_dims", "power_alphas", "fixed_mask")

    def __init__(
        self,
        types: Any,
        starts: Any,
        v_dims: Any = 1,
        power_alphas: Any = 0.0,
        fixed_mask: Optional[Any] = None,
    ) -> None:
        starts_array = np.asarray(starts)
        if starts_array.ndim != 1:
            raise ValueError("starts must be a 1D array")
        count = int(starts_array.size)

        self.starts = _as_i32_vector(starts_array, count, "starts")
        self.types = _as_type_vector(types, count)
        self.v_dims = _as_i32_vector(v_dims, count, "v_dims", default=1)
        self.power_alphas = _as_f64_vector(power_alphas, count, "power_alphas", 0.0)

        if np.any(self.starts < 0):
            raise ValueError("starts must be nonnegative")
        if np.any(self.v_dims <= 0):
            raise ValueError("v_dims must be positive")
        three_dimensional = (self.types == int(ConeType.EXP)) | (self.types == int(ConeType.POWER))
        if np.any(self.v_dims[three_dimensional] != 1):
            raise ValueError("EXP and POWER cones require v_dim == 1")
        power = self.types == int(ConeType.POWER)
        if np.any(
            ~np.isfinite(self.power_alphas[power])
            | (self.power_alphas[power] <= 0.0)
            | (self.power_alphas[power] >= 1.0)
        ):
            raise ValueError("POWER cone alphas must lie in (0, 1)")

        if fixed_mask is None:
            self.fixed_mask = None
        else:
            mask = np.asarray(fixed_mask)
            if mask.ndim != 1:
                raise ValueError("fixed_mask must be a 1D ambient-coordinate mask")
            self.fixed_mask = np.ascontiguousarray(mask, dtype=np.uint8)

    def __len__(self) -> int:
        return int(self.starts.size)

    def validate_ambient(
        self,
        ambient_dimension: int,
        *,
        allow_fixed: bool,
        require_cover: bool = False,
    ) -> None:
        """Validate ranges against the variable or affine-row dimension."""
        ambient_dimension = int(ambient_dimension)
        if ambient_dimension < 0:
            raise ValueError("ambient_dimension must be nonnegative")
        v_dims_i64 = self.v_dims.astype(np.int64)
        lengths = v_dims_i64 + 2
        three_dimensional = (self.types == int(ConeType.EXP)) | (self.types == int(ConeType.POWER))
        lengths = np.where(three_dimensional, 3, lengths)
        psd = self.types == int(ConeType.PSD)
        lengths = np.where(psd, v_dims_i64 * (v_dims_i64 + 1) // 2, lengths)
        ends = self.starts.astype(np.int64) + lengths
        if np.any(ends > ambient_dimension):
            raise ValueError("a cone block extends beyond the ambient dimension")
        if require_cover and int(lengths.sum()) != ambient_dimension:
            raise ValueError("affine cones must cover every row of F")
        if self.fixed_mask is not None:
            if not allow_fixed:
                raise ValueError("affine cones do not support fixed slots")
            if self.fixed_mask.size != ambient_dimension:
                raise ValueError(
                    "fixed_mask length "
                    f"{self.fixed_mask.size} != ambient dimension {ambient_dimension}"
                )
            for start, length, is_psd in zip(self.starts, lengths, psd):
                if is_psd and np.any(self.fixed_mask[int(start) : int(start + length)]):
                    raise ValueError("PSD cones do not support fixed slots")

    @classmethod
    def from_columnar(cls, payload: Mapping[str, Any]) -> "ConeSpec":
        """Construct from the compact payload returned by ``read_problem_file``."""
        return cls(
            payload["types"],
            payload["starts"],
            payload["v_dims"],
            payload["power_alphas"],
            payload.get("fixed_mask"),
        )
