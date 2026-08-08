# Copyright 2025 Haihao Lu
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

import warnings
from typing import Any, Optional, Union

import numpy as np
import scipy.sparse as sp

from . import PDHCG
from ._core import get_default_params, read_problem_file, solve_once
from .cones import ConeSpec

# array-like type
ArrayLike = Union[np.ndarray, list, tuple]


def _as_dense_f64_c(a: ArrayLike) -> np.ndarray:
    """
    Convert input to a C-contiguous numpy array of float64.
    """
    arr = np.asarray(a, dtype=np.float64)
    # ensure C-contiguous
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr


def _as_csr_f64_i32(A: sp.spmatrix) -> sp.csr_matrix:
    """
    Convert input sparse matrix to CSR format with float64 values and int32 indices.
    """
    csr = A.tocsr().astype(np.float64, copy=False)
    # force int32 indices (common C/CUDA req)
    if csr.indptr.dtype != np.int32:
        csr.indptr = csr.indptr.astype(np.int32, copy=True)
    if csr.indices.dtype != np.int32:
        csr.indices = csr.indices.astype(np.int32, copy=True)
    csr.sort_indices()
    return csr


class _ParamsView:
    """
    A view of the model parameters that allows getting/setting via attributes or keys.
    """

    def __init__(self, model: Model):
        object.__setattr__(self, "_m", model)

    def __getattr__(self, name: str):
        key = PDHCG._PARAM_ALIAS.get(name, name)
        if key in self._m._params:
            return self._m._params[key]
        raise AttributeError(f"Unknown parameter '{name}'")

    def __setattr__(self, name: str, value):
        self._m.setParam(name, value)

    def __getitem__(self, name: str):
        return getattr(self, name)

    def __setitem__(self, name: str, value):
        self._m.setParam(name, value)

    def keys(self):
        return self._m._params.keys()


class Model:
    r"""
    A class representing a quadratic programming (QP) model for PDHCG.

    The quadratic programming problem is defined as:

    ```
    minimize      1/2 x^T (Q + R^T D R) x + c^T x
    subject to    l_c <= A x <= u_c
                  F x + g in K
                  l_v <= x <= u_v
    ```

    where D is an optional rank-by-rank middle matrix; if omitted, D defaults to
    the identity, recovering the standard Q + R^T R formulation.

    Solver Parameters:
        Parameters can be set via the `Params` attribute or `setParam()` method.
        Common parameters include:

        - TimeLimit: Time limit in seconds (default: 3600.0)
        - IterationLimit: Maximum number of iterations (default: INT_MAX)
        - OptimalityTol: Relative optimality tolerance (default: 1e-4)
        - FeasibilityTol: Relative feasibility tolerance (default: 1e-4)
        - Presolve: Enable/disable presolve (default: True)
        - RuizIters: Number of Ruiz rescaling iterations (default: 10)
        - LogLevel: Verbosity level 0-3 (default: 1)

        Use `model.Params["Presolve"] = False` to disable presolve.
    """

    def __init__(
        self,
        objective_vector: ArrayLike,
        constraint_matrix: Optional[Union[np.ndarray, sp.spmatrix]] = None,
        constraint_lower_bound: Optional[ArrayLike] = None,
        constraint_upper_bound: Optional[ArrayLike] = None,
        objective_matrix: Optional[Union[np.ndarray, sp.spmatrix]] = None,
        objective_matrix_low_rank: Optional[Union[np.ndarray, sp.spmatrix]] = None,
        objective_matrix_low_rank_middle: Optional[ArrayLike] = None,
        variable_lower_bound: Optional[ArrayLike] = None,
        variable_upper_bound: Optional[ArrayLike] = None,
        objective_constant: float = 0.0,
        affine_cone_matrix: Optional[Union[np.ndarray, sp.spmatrix]] = None,
        affine_cone_offset: Optional[ArrayLike] = None,
        affine_cones: Optional[ConeSpec] = None,
        variable_cones: Optional[ConeSpec] = None,
    ):
        """
        Initialize the Model with the given parameters.

        Args:
            objective_vector: Linear coefficients of the objective function (c).
            constraint_matrix: Coefficients of the linear constraints (A).
            constraint_lower_bound: Lower bounds for the linear constraints.
            constraint_upper_bound: Upper bounds for the linear constraints.
            objective_matrix: Quadratic coefficients of the objective function (Q).
            objective_matrix_low_rank: Low-rank quadratic coefficients of the objective (R).
            objective_matrix_low_rank_middle: Optional middle matrix D in Q + R^T D R.
                Accepts a 1-D array of length `rank` (treated as diag(D)) or a
                2-D `rank` x `rank` symmetric array. Defaults to identity.
            variable_lower_bound: Lower bounds for the decision variables.
            variable_upper_bound: Upper bounds for the decision variables.
            objective_constant: Constant term in the objective function.
            affine_cone_matrix: Matrix F in the native constraint F x + g in K.
            affine_cone_offset: Offset g, with one entry per row of F. Defaults to zero.
            affine_cones: Compact cone metadata covering all rows of F.
            variable_cones: Compact metadata for cone blocks in the variable vector.
                Must be a :class:`pdhcg.ConeSpec`.

        Note:
            If variable bounds are not provided, they default to -inf and +inf respectively.
        """
        # problem dimensions
        self.num_vars = 0
        self.num_constrs = 0
        self.num_affine_constrs = 0

        # Check A
        if constraint_matrix is not None:
            if not hasattr(constraint_matrix, "shape") or len(constraint_matrix.shape) != 2:
                raise ValueError(
                    "constraint_matrix must be a 2D numpy.ndarray or scipy.sparse matrix."
                )
            m, n = constraint_matrix.shape
            self.num_vars = int(n)
            self.num_constrs = int(m)

        # Check Q (if A was None, try to infer n from Q)
        if objective_matrix is not None:
            if not hasattr(objective_matrix, "shape") or len(objective_matrix.shape) != 2:
                raise ValueError(
                    "objective_matrix must be a 2D numpy.ndarray or scipy.sparse matrix."
                )
            if self.num_vars == 0:
                self.num_vars = int(objective_matrix.shape[1])
            elif objective_matrix.shape[1] != self.num_vars:
                raise ValueError(
                    f"objective_matrix dimensions mismatch variables ({self.num_vars})"
                )

        # Check R (if A and Q were None, try to infer n from R)
        if objective_matrix_low_rank is not None:
            if (
                not hasattr(objective_matrix_low_rank, "shape")
                or len(objective_matrix_low_rank.shape) != 2
            ):
                raise ValueError(
                    "objective_matrix_low_rank must be a 2D numpy.ndarray or scipy.sparse matrix."
                )
            if self.num_vars == 0:
                self.num_vars = int(objective_matrix_low_rank.shape[1])
            elif objective_matrix_low_rank.shape[1] != self.num_vars:
                raise ValueError(
                    f"objective_matrix_low_rank dimensions mismatch variables ({self.num_vars})"
                )

        # Check F (if A, Q, and R were None, infer n from the affine cone map).
        if affine_cone_matrix is not None:
            if not hasattr(affine_cone_matrix, "shape") or len(affine_cone_matrix.shape) != 2:
                raise ValueError(
                    "affine_cone_matrix must be a 2D numpy.ndarray or scipy.sparse matrix."
                )
            if self.num_vars == 0:
                self.num_vars = int(affine_cone_matrix.shape[1])
            elif affine_cone_matrix.shape[1] != self.num_vars:
                raise ValueError(
                    f"affine_cone_matrix dimensions mismatch variables ({self.num_vars})"
                )

        if self.num_vars == 0 and objective_vector is not None:
            objective_shape = np.asarray(objective_vector).shape
            if len(objective_shape) != 1:
                raise ValueError(f"objective_vector must be 1D, got shape {objective_shape}")
            self.num_vars = int(objective_shape[0])

        if (
            self.num_vars == 0
            and constraint_matrix is None
            and objective_matrix is None
            and objective_matrix_low_rank is None
            and affine_cone_matrix is None
        ):
            return None

        # sense
        self.ModelSense = PDHCG.MINIMIZE
        # always start from backend defaults PDHCG params
        self._params: dict[str, Any] = dict(get_default_params())
        self.Params = _ParamsView(self)
        # set coefficients and bounds
        self.setObjectiveVector(objective_vector)
        self.setObjectiveConstant(objective_constant)
        self.setObjectiveMatrix(objective_matrix)
        self.setObjectiveMatrixLowRank(objective_matrix_low_rank)
        self.setObjectiveMatrixLowRankMiddle(objective_matrix_low_rank_middle)
        self.setConstraintMatrix(constraint_matrix)
        self.setConstraintLowerBound(constraint_lower_bound)
        self.setConstraintUpperBound(constraint_upper_bound)
        self.setVariableLowerBound(variable_lower_bound)
        self.setVariableUpperBound(variable_upper_bound)
        self._variable_cones: Optional[ConeSpec] = None
        self.affine_F = None
        self.affine_g = None
        self._affine_cones: Optional[ConeSpec] = None
        self.setVariableCones(variable_cones)
        self.setAffineConeConstraints(affine_cone_matrix, affine_cone_offset, affine_cones)
        # initialize warm start values
        self._primal_start: Optional[np.ndarray] = None  # warm start primal solution
        self._dual_start: Optional[np.ndarray] = None  # warm start dual solution
        # initialize solution attributes
        self._x: Optional[np.ndarray] = None  # primal solution
        self._y: Optional[np.ndarray] = None  # dual solution
        self._objval: Optional[float] = None  # objective value
        self._dualobj: Optional[float] = None  # dual objective value
        self._gap: Optional[float] = None  # primal-dual gap
        self._rel_gap: Optional[float] = None  # relative gap
        self._status: Optional[str] = None  # solution status
        self._status_code: Optional[int] = None  # solution status code
        self._iter: Optional[int] = None  # number of iterations
        self._runtime: Optional[float] = None  # runtime
        self._rescale_time: Optional[float] = None  # rescale time
        self._rel_p_res: Optional[float] = None  # relative primal residual
        self._rel_d_res: Optional[float] = None  # relative dual residual
        self._max_p_ray: Optional[float] = None  # maximum primal ray
        self._max_d_ray: Optional[float] = None  # maximum dual ray
        self._p_ray_lin_obj: Optional[float] = None  # primal ray linear objective
        self._d_ray_obj: Optional[float] = None  # dual ray objective

    @classmethod
    def read_file(cls, path: str) -> Model:
        """
        Read a problem file (.mps/.mps.gz/.cbf/.cbf.gz) and construct a Model.
        Cones (SOC/RSOC/EXP/POWER) present in the file are preserved and passed
        through to the solver.
        """
        raw = read_problem_file(path)

        def _to_csr(d):
            if d is None:
                return None
            return sp.csr_matrix(
                (d["data"], d["indices"], d["indptr"]),
                shape=d["shape"],
            )

        Q = _to_csr(raw.get("Q"))
        A = _to_csr(raw.get("A"))
        affine_F = _to_csr(raw.get("affine_F"))
        m = cls(
            objective_vector=raw["c"],
            constraint_matrix=A,
            constraint_lower_bound=raw["constr_lb"] if A is not None else None,
            constraint_upper_bound=raw["constr_ub"] if A is not None else None,
            objective_matrix=Q,
            variable_lower_bound=raw["var_lb"],
            variable_upper_bound=raw["var_ub"],
            objective_constant=raw.get("obj_const", 0.0),
            affine_cone_matrix=affine_F,
            affine_cone_offset=raw.get("affine_g"),
            affine_cones=(
                ConeSpec.from_columnar(raw["affine_cones"])
                if raw.get("affine_cones") is not None
                else None
            ),
            variable_cones=(
                ConeSpec.from_columnar(raw["cones"]) if raw.get("cones") is not None else None
            ),
        )
        ps = raw.get("primal_start")
        if ps is not None:
            m._primal_start = np.asarray(ps, dtype=np.float64)
        return m

    def setObjectiveVector(self, c: ArrayLike) -> None:
        """
        Overwrite the linear objective vector c.
        """
        # store as float64
        self.c = _as_dense_f64_c(c)
        # check dimensions
        if self.c.ndim != 1:
            raise ValueError(f"setObjectiveVector: c must be 1D, got shape {self.c.shape}")
        if self.c.size != self.num_vars:
            raise ValueError(
                f"setObjectiveVector: length {self.c.size} != self.num_vars ({self.num_vars})"
            )
        # clear cached solution
        self._clear_solution_cache()

    def setObjectiveConstant(self, c0: float) -> None:
        """
        Overwrite objective constant term.
        """
        self.c0 = float(c0)
        # clear cached solution
        self._clear_solution_cache()

    def setObjectiveMatrix(self, Q_like: ArrayLike) -> None:
        """
        Overwrite the quadratic objective matrix Q.
        """
        if Q_like is None:
            self.Q = None
            self._clear_solution_cache()
            return
        if not isinstance(Q_like, (np.ndarray, sp.spmatrix)):
            raise TypeError("setObjectiveMatrix: Q must be a numpy.ndarray or scipy.sparse matrix")
        if len(Q_like.shape) != 2:
            raise ValueError(f"setObjectiveMatrix: Q must be 2D, got shape {Q_like.shape}")
        if Q_like.shape[1] != self.num_vars:
            raise ValueError(
                f"setObjectiveMatrix: Q shape {Q_like.shape} does not match number of variables ({self.num_vars})"
            )
        # store as float64
        if sp.issparse(Q_like):
            self.Q = _as_csr_f64_i32(Q_like)
        else:
            self.Q = _as_dense_f64_c(Q_like)
        # problem dimensions
        if not hasattr(self.Q, "shape") or len(self.Q.shape) != 2:
            raise ValueError("objective_matrix must be a 2D numpy.ndarray or scipy.sparse matrix.")

    def setObjectiveMatrixLowRank(self, R_like: ArrayLike) -> None:
        """
        Overwrite the low-rank quadratic objective matrix R.
        """
        if R_like is None:
            self.R = None
            self._clear_solution_cache()
            return

        if not isinstance(R_like, (np.ndarray, sp.spmatrix)):
            raise TypeError(
                "setObjectiveMatrixLowRank: R must be a numpy.ndarray or scipy.sparse matrix"
            )
        if len(R_like.shape) != 2:
            raise ValueError(f"setObjectiveMatrixLowRank: R must be 2D, got shape {R_like.shape}")
        if R_like.shape[1] != self.num_vars:
            raise ValueError(
                f"setObjectiveMatrixLowRank: R columns {R_like.shape[1]} must match variables ({self.num_vars})"
            )

        if sp.issparse(R_like):
            self.R = _as_csr_f64_i32(R_like)
        else:
            self.R = _as_dense_f64_c(R_like)

        self._clear_solution_cache()

    def setObjectiveMatrixLowRankMiddle(
        self, D_like: Optional[Union[np.ndarray, sp.spmatrix, ArrayLike]]
    ) -> None:
        """
        Overwrite the middle matrix D in Q + R^T D R.

        Accepts any of:
          - None -> D = I (identity, original behavior).
          - 1-D array of length `rank` -> diagonal D.
          - 2-D `rank` x `rank` numpy array -> dense (symmetric) D.
          - scipy.sparse `rank` x `rank` matrix -> sparse D.

        The backend's `preprocess_qp_problem` inspects the nonzero pattern and
        picks either a diagonal or a dense runtime representation.
        """
        if D_like is None:
            self.D = None
            self._clear_solution_cache()
            return
        if getattr(self, "R", None) is None:
            raise ValueError(
                "setObjectiveMatrixLowRankMiddle: D is only meaningful when R is set; "
                "call setObjectiveMatrixLowRank first."
            )
        rank = int(self.R.shape[0])
        if sp.issparse(D_like):
            if D_like.shape != (rank, rank):
                raise ValueError(
                    f"setObjectiveMatrixLowRankMiddle: sparse D shape {D_like.shape} must be ({rank}, {rank})"
                )
            self.D = _as_csr_f64_i32(D_like)
        else:
            d_arr = _as_dense_f64_c(D_like)
            if d_arr.ndim == 1:
                if d_arr.size != rank:
                    raise ValueError(
                        f"setObjectiveMatrixLowRankMiddle: diag D length {d_arr.size} must equal rank {rank}"
                    )
            elif d_arr.ndim == 2:
                if d_arr.shape != (rank, rank):
                    raise ValueError(
                        f"setObjectiveMatrixLowRankMiddle: dense D shape {d_arr.shape} must be ({rank}, {rank})"
                    )
            else:
                raise ValueError(
                    f"setObjectiveMatrixLowRankMiddle: D must be 1D, 2D, or scipy.sparse; got ndim={d_arr.ndim}"
                )
            self.D = d_arr
        self._clear_solution_cache()

    def setConstraintMatrix(self, A_like: ArrayLike) -> None:
        """
        Overwrite the linear constraint matrix A.
        """
        if A_like is None:
            self.A = None
            self.num_constrs = 0
            self.constr_lb = None
            self.constr_ub = None
            self._clear_solution_cache()
            return
        if not isinstance(A_like, (np.ndarray, sp.spmatrix)):
            raise TypeError("setConstraintMatrix: A must be a numpy.ndarray or scipy.sparse matrix")
        if len(A_like.shape) != 2:
            raise ValueError(f"setConstraintMatrix: A must be 2D, got shape {A_like.shape}")
        if A_like.shape[1] != self.num_vars:
            raise ValueError(
                f"setConstraintMatrix: A shape {A_like.shape} does not match number of variables ({self.num_vars})"
            )
        # store as float64
        if sp.issparse(A_like):
            self.A = _as_csr_f64_i32(A_like)
        else:
            self.A = _as_dense_f64_c(A_like)
        # problem dimensions
        if not hasattr(self.A, "shape") or len(self.A.shape) != 2:
            raise ValueError("constraint_matrix must be a 2D numpy.ndarray or scipy.sparse matrix.")
        m, _ = self.A.shape
        self.num_constrs = int(m)
        # check constraint bounds
        l = getattr(self, "constr_lb", None)
        if l is not None:
            l = np.asarray(l, dtype=np.float64).ravel()
            if l.size != self.num_constrs:
                raise ValueError(
                    f"setConstraintMatrix: constraint_lower_bound length {l.size} != rows {self.num_constrs}. "
                    f"Call setConstraintLowerBound(...) to update it."
                )
        u = getattr(self, "constr_ub", None)
        if u is not None:
            u = np.asarray(u, dtype=np.float64).ravel()
            if u.size != self.num_constrs:
                raise ValueError(
                    f"setConstraintMatrix: constraint_upper_bound length {u.size} != rows {self.num_constrs}. "
                    f"Call setConstraintUpperBound(...) to update it."
                )
        # clear cached solution
        self._clear_solution_cache()

    def setConstraintLowerBound(self, constr_lb: Optional[ArrayLike]) -> None:
        """
        Overwrite the linear constraint lower bounds.
        """
        # check if the input is None
        if constr_lb is None:
            self.constr_lb = None
            # clear cached solution
            self._clear_solution_cache()
            return
        # convert to numpy array
        constr_lb = _as_dense_f64_c(constr_lb).ravel()
        if constr_lb.size != self.num_constrs:
            raise ValueError(
                f"setConstraintLowerBound: length {constr_lb.size} != self.num_constrs ({self.num_constrs})"
            )
        self.constr_lb = constr_lb
        # clear cached solution
        self._clear_solution_cache()

    def setConstraintUpperBound(self, constr_ub: Optional[ArrayLike]) -> None:
        """
        Overwrite the linear constraint upper bounds.
        """
        # check if the input is None
        if constr_ub is None:
            self.constr_ub = None
            # clear cached solution
            self._clear_solution_cache()
            return
        # convert to numpy array
        constr_ub = _as_dense_f64_c(constr_ub).ravel()
        if constr_ub.size != self.num_constrs:
            raise ValueError(
                f"setConstraintUpperBound: length {constr_ub.size} != self.num_constrs ({self.num_constrs})"
            )
        self.constr_ub = constr_ub
        # clear cached solution
        self._clear_solution_cache()

    def setAffineConeConstraints(
        self,
        F_like: Optional[Union[np.ndarray, sp.spmatrix]],
        g: Optional[ArrayLike],
        cones: Optional[ConeSpec],
    ) -> None:
        """Set the native affine cone constraint ``F x + g in K``."""
        if F_like is None:
            if g is not None or cones is not None:
                raise ValueError(
                    "affine_cone_matrix is required when affine offset or cones are provided"
                )
            self.affine_F = None
            self.affine_g = None
            self._affine_cones = None
            self.num_affine_constrs = 0
            self._clear_solution_cache()
            return
        if not isinstance(F_like, (np.ndarray, sp.spmatrix)):
            raise TypeError(
                "setAffineConeConstraints: F must be a numpy.ndarray or scipy.sparse matrix"
            )
        if len(F_like.shape) != 2 or F_like.shape[1] != self.num_vars:
            raise ValueError(
                f"setAffineConeConstraints: F shape {F_like.shape} must have {self.num_vars} columns"
            )
        if not isinstance(cones, ConeSpec):
            raise TypeError("setAffineConeConstraints: cones must be a ConeSpec")
        if len(cones) == 0:
            raise ValueError("setAffineConeConstraints: cones must cover every row of F")
        affine_F = _as_csr_f64_i32(F_like) if sp.issparse(F_like) else _as_dense_f64_c(F_like)
        num_affine_constrs = int(F_like.shape[0])
        if g is None:
            affine_g = None
        else:
            affine_g = _as_dense_f64_c(g).ravel()
            if affine_g.size != num_affine_constrs:
                raise ValueError(
                    "setAffineConeConstraints: affine offset length "
                    f"{affine_g.size} != rows {num_affine_constrs}"
                )
        cones.validate_ambient(
            num_affine_constrs,
            allow_fixed=False,
            require_cover=True,
        )
        self.affine_F = affine_F
        self.affine_g = affine_g
        self.num_affine_constrs = num_affine_constrs
        self._affine_cones = cones
        self._clear_solution_cache()

    def setVariableCones(self, cones: Optional[ConeSpec]) -> None:
        """Set cone blocks embedded directly in the variable vector."""
        if cones is None:
            self._variable_cones = None
            self._clear_solution_cache()
            return
        if not isinstance(cones, ConeSpec):
            raise TypeError("setVariableCones: cones must be a ConeSpec")
        if len(cones) == 0:
            self._variable_cones = None
            self._clear_solution_cache()
            return
        cones.validate_ambient(self.num_vars, allow_fixed=True)
        self._variable_cones = cones
        self._clear_solution_cache()

    def setVariableLowerBound(self, lb: Optional[ArrayLike]) -> None:
        """
        Overwrite the decision variable lower bounds.
        """
        # check if the input is None
        if lb is None:
            self.lb = None
            # clear cached solution
            self._clear_solution_cache()
            return
        # convert to numpy array
        lb = _as_dense_f64_c(lb).ravel()
        if lb.size != self.num_vars:
            raise ValueError(
                f"setVariableLowerBound: length {lb.size} != self.num_vars ({self.num_vars})"
            )
        self.lb = lb
        # clear cached solution
        self._clear_solution_cache()

    def setVariableUpperBound(self, ub: Optional[ArrayLike]) -> None:
        """
        Overwrite the decision variable upper bounds.
        """
        # check if the input is None
        if ub is None:
            self.ub = None
            # clear cached solution
            self._clear_solution_cache()
            return
        # convert to numpy array
        ub = _as_dense_f64_c(ub).ravel()
        if ub.size != self.num_vars:
            raise ValueError(
                f"setVariableUpperBound: length {ub.size} != self.num_vars ({self.num_vars})"
            )
        self.ub = ub
        # clear cached solution
        self._clear_solution_cache()

    def setWarmStart(
        self, primal: Optional[ArrayLike] = None, dual: Optional[ArrayLike] = None
    ) -> None:
        """
        Set warm start values for primal and/or dual solutions.
        """
        # set primal warm start
        if primal is not None:
            primal_arr = _as_dense_f64_c(primal).ravel()
            if primal_arr.size == self.num_vars:  # otherwise default to None
                self._primal_start = primal_arr
            else:
                warnings.warn(
                    f"Warm start primal size mismatch (expected {self.num_vars}, got {primal_arr.size}).",
                    RuntimeWarning,
                )
        # clear primal warm start
        else:
            self._primal_start = None
        # set dual warm start
        if dual is not None:
            dual_arr = _as_dense_f64_c(dual).ravel()
            expected_dual_size = self.num_constrs + self.num_affine_constrs
            if dual_arr.size == expected_dual_size:  # otherwise default to None
                self._dual_start = dual_arr
            else:
                warnings.warn(
                    f"Warm start dual size mismatch (expected {expected_dual_size}, got {dual_arr.size}).",
                    RuntimeWarning,
                )
        # clear dual warm start
        else:
            self._dual_start = None

    def clearWarmStart(self) -> None:
        """
        Clear any existing warm start values.
        """
        self.setWarmStart(primal=None, dual=None)

    def setParam(self, name: str, value: Any) -> None:
        """
        Set the value of a solver parameter by name.
        """
        key = PDHCG._PARAM_ALIAS.get(name, name)
        self._params[key] = value

    def getParam(self, name: str) -> Any:
        """
        Get the value of a solver parameter by name.
        """
        key = PDHCG._PARAM_ALIAS.get(name, name)
        return self._params.get(key)

    def setParams(self, /, **kwargs) -> None:
        """
        Set multiple solver parameters by keyword arguments.
        """
        for k, v in kwargs.items():
            self.setParam(k, v)

    def optimize(self):
        """
        Solve the quadratic programming problem using the PDHCG solver.
        """
        # clear cached solution
        self._clear_solution_cache()
        # check model sense
        if self.ModelSense not in (PDHCG.MINIMIZE, PDHCG.MAXIMIZE):
            raise ValueError("model_sense must be PDHCG.MINIMIZE or PDHCG.MAXIMIZE")
        # determine sign
        sign = 1 if self.ModelSense == PDHCG.MINIMIZE else -1
        # effective objective based on sense
        c_eff = sign * self.c if self.c is not None else None
        c0_eff = sign * self.c0 if self.c0 is not None else None
        # call the core solver
        info = solve_once(
            self.Q,
            self.R,
            self.A,
            c_eff,
            c0_eff,
            self.lb,
            self.ub,
            self.constr_lb,
            self.constr_ub,
            zero_tolerance=0.0,
            params=self._params,
            primal_start=self._primal_start,
            dual_start=self._dual_start,
            D=getattr(self, "D", None),
            cones=self._variable_cones,
            affine_F=self.affine_F,
            affine_g=self.affine_g,
            affine_cones=self._affine_cones,
        )
        # solutions
        self._x = np.asarray(info.get("X")) if info.get("X") is not None else None
        self._y = np.asarray(info.get("Pi")) if info.get("Pi") is not None else None
        # objectives & gaps
        primal_obj_eff = info.get("PrimalObj")
        dual_obj_eff = info.get("DualObj")
        self._objval = sign * primal_obj_eff if primal_obj_eff is not None else None
        self._dualobj = sign * dual_obj_eff if dual_obj_eff is not None else None
        self._gap = info.get("ObjectiveGap")
        self._rel_gap = info.get("RelativeObjectiveGap")
        # status & counters
        self._status = str(info.get("Status")) if info.get("Status") is not None else None
        self._status_code = (
            int(info.get("StatusCode")) if info.get("StatusCode") is not None else None
        )
        self._iter = int(info.get("Iterations")) if info.get("Iterations") is not None else None
        self._runtime = info.get("RuntimeSec")
        self._rescale_time = info.get("RescalingTimeSec")
        # residuals
        self._rel_p_res = info.get("RelativePrimalResidual")
        self._rel_d_res = info.get("RelativeDualResidual")
        # rays
        self._max_p_ray = info.get("MaxPrimalRayInfeas")
        self._max_d_ray = info.get("MaxDualRayInfeas")
        p_ray_lin_eff = info.get("PrimalRayLinObj")
        d_ray_obj_eff = info.get("DualRayObj")
        self._p_ray_lin_obj = sign * p_ray_lin_eff if p_ray_lin_eff is not None else None
        self._d_ray_obj = sign * d_ray_obj_eff if d_ray_obj_eff is not None else None

    def _clear_solution_cache(self) -> None:
        """
        Clear cached solution attributes.
        """
        self._x = self._y = None
        self._objval = self._dualobj = None
        self._gap = self._rel_gap = None
        self._status = None
        self._status_code = None
        self._iter = None
        self._runtime = self._rescale_time = None
        self._rel_p_res = None
        self._rel_d_res = None
        self._max_p_ray = self._max_d_ray = None
        self._p_ray_lin_obj = self._d_ray_obj = None

    @property
    def X(self) -> Optional[np.ndarray]:
        return self._x

    @property
    def Pi(self) -> Optional[np.ndarray]:
        return self._y

    @property
    def ObjVal(self) -> Optional[float]:
        return self._objval

    @property
    def DualObj(self) -> Optional[float]:
        return self._dualobj

    @property
    def Gap(self) -> Optional[float]:
        return self._gap

    @property
    def RelGap(self) -> Optional[float]:
        return self._rel_gap

    @property
    def Status(self) -> Optional[str]:
        return self._status

    @property
    def StatusCode(self) -> Optional[int]:
        return self._status_code

    @property
    def IterCount(self) -> Optional[int]:
        return self._iter

    @property
    def Runtime(self) -> Optional[float]:
        return self._runtime

    @property
    def RescalingTime(self) -> Optional[float]:
        return self._rescale_time

    @property
    def RelPrimalResidual(self) -> Optional[float]:
        return self._rel_p_res

    @property
    def RelDualResidual(self) -> Optional[float]:
        return self._rel_d_res

    @property
    def MaxPrimalRayInfeas(self) -> Optional[float]:
        return self._max_p_ray

    @property
    def MaxDualRayInfeas(self) -> Optional[float]:
        return self._max_d_ray

    @property
    def PrimalRayLinObj(self) -> Optional[float]:
        return self._p_ray_lin_obj

    @property
    def DualRayObj(self) -> Optional[float]:
        return self._d_ray_obj

    @property
    def PrimalInfeas(self) -> Optional[float]:
        return self._rel_p_res

    @property
    def DualInfeas(self) -> Optional[float]:
        return self._rel_d_res
