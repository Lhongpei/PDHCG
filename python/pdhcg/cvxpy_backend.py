# Copyright 2026 Hongpei Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CVXPY conic-solver backend for PDHCG.

Import this module (``import pdhcg.cvxpy_backend``) once per process; it will
register ``PDHCG`` under ``cvxpy.settings.SOLVER_MAP_CONIC`` so that
``problem.solve(solver='PDHCG')`` works.

Supported CVXPY constraints: Zero, NonNeg, SOC, PSD, ExpCone, PowCone3D.
Not supported: integer variables.
"""

from __future__ import annotations

import warnings
from typing import Any

import cvxpy.settings as _cvx_s
import numpy as np
import scipy.sparse as sp
from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver

from ._core import solve_once
from .cones import ConeSpec, ConeType

_STATUS_MAP = {
    "OPTIMAL": _cvx_s.OPTIMAL,
    "PRIMAL_INFEASIBLE": _cvx_s.INFEASIBLE,
    "DUAL_INFEASIBLE": _cvx_s.UNBOUNDED,
    "INFEASIBLE_OR_UNBOUNDED": _cvx_s.INFEASIBLE_OR_UNBOUNDED,
    "TIME_LIMIT": _cvx_s.USER_LIMIT,
    "ITERATION_LIMIT": _cvx_s.USER_LIMIT,
    "FEAS_POLISH_SUCCESS": _cvx_s.OPTIMAL,
    "UNSPECIFIED": _cvx_s.SOLVER_ERROR,
}


class PDHCG(ConicSolver):
    """PDHCG conic-solver plugin for CVXPY."""

    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg, SOC, PSD, ExpCone, PowCone3D]

    # CVXPY's ExpCone convention is (x, y, z) with z >= y * exp(x/y), y > 0.
    # PDHCG's internal exp cone convention is (r1, r2, r3) with r3 >= r2 * exp(r1/r2).
    # Direct mapping: cvxpy(x,y,z) -> internal(r1,r2,r3).
    EXP_CONE_ORDER = [0, 1, 2]

    def name(self):
        return "PDHCG"

    def import_solver(self) -> None:
        import pdhcg  # noqa: F401

    def supports_quad_obj(self) -> bool:
        return True

    @staticmethod
    def psd_format_mat(constr):
        """Map a symmetric matrix to lower-triangular column-major svec."""
        order = constr.expr.shape[0]
        packed_length = order * (order + 1) // 2

        lower = np.tril_indices(order)
        columns = np.sort(np.ravel_multi_index(lower, (order, order), order="F"))
        values = np.zeros((order, order), dtype=np.float64)
        values[lower] = np.sqrt(2.0)
        np.fill_diagonal(values, 1.0)
        values = values.ravel(order="F")
        values = values[values != 0.0]
        packed = sp.csc_array(
            (values, (np.arange(packed_length), columns)),
            shape=(packed_length, order * order),
        )

        indices = np.arange(order * order)
        matrix_indices = indices.reshape((order, order))
        symmetrize = sp.csc_array(
            (
                np.full(2 * order * order, 0.5),
                (
                    np.concatenate((indices, matrix_indices.ravel(order="F"))),
                    np.concatenate((indices, matrix_indices.T.ravel(order="F"))),
                ),
            ),
            shape=(order * order, order * order),
        )
        return packed @ symmetrize

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """Expand a PSD svec dual before CVXPY restores constraint shapes."""
        if not isinstance(constraint, PSD):
            return utilities.extract_dual_value(result_vec, offset, constraint)

        order = constraint.shape[0]
        packed_length = order * (order + 1) // 2
        new_offset = offset + packed_length
        full = np.zeros((order, order), dtype=np.float64)
        full[np.triu_indices(order)] = result_vec[offset:new_offset]
        full += full.T
        full[np.diag_indices(order)] *= 0.5
        full[np.tril_indices(order, k=-1)] /= np.sqrt(2.0)
        full[np.triu_indices(order, k=1)] /= np.sqrt(2.0)
        return full.ravel(order="F"), new_offset

    def cite(self, data):
        return (
            "@misc{pdhcg,\n"
            "  title  = {PDHCG: GPU-accelerated Primal-Dual Hybrid Conjugate Gradient QP solver},\n"
            "  author = {Li, Hongpei and collaborators},\n"
            "  year   = {2026},\n"
            "  url    = {https://github.com/Lhongpei/PDHCG}\n"
            "}"
        )

    def invert(self, solution, inverse_data):
        status = _STATUS_MAP.get(solution.get("status", "UNSPECIFIED"), _cvx_s.SOLVER_ERROR)
        attr: dict[str, Any] = {
            _cvx_s.SOLVE_TIME: solution.get("solve_time", 0.0),
            _cvx_s.NUM_ITERS: solution.get("iterations", 0),
        }
        if status in _cvx_s.SOLUTION_PRESENT:
            opt_val = solution["value"] + inverse_data[_cvx_s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution["primal"]}
            eq_dual = utilities.get_dual_values(
                solution["eq_dual"],
                self.extract_dual_value,
                inverse_data[self.EQ_CONSTR],
            )
            ineq_dual = utilities.get_dual_values(
                solution["ineq_dual"],
                self.extract_dual_value,
                inverse_data[self.NEQ_CONSTR],
            )
            dual_vars = {**eq_dual, **ineq_dual}
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        return failure_solution(status, attr)

    def solve_via_data(
        self, data, warm_start: bool, verbose: bool, solver_opts: dict, solver_cache=None
    ):
        A_cvx = sp.csr_matrix(data[_cvx_s.A])
        b_cvx = np.asarray(data[_cvx_s.B], dtype=np.float64).ravel()
        c = np.asarray(data[_cvx_s.C], dtype=np.float64).ravel()
        P = data.get(_cvx_s.P, None)

        cone_dims = data[ConicSolver.DIMS]
        n_zero = int(cone_dims.zero)
        n_nonneg = int(cone_dims.nonneg)
        soc_dims = list(cone_dims.soc)
        psd_dims = list(cone_dims.psd)
        n_exp = int(cone_dims.exp)
        pow_alphas = list(cone_dims.p3d)

        n = c.size
        soc_total = sum(soc_dims)
        psd_total = sum(order * (order + 1) // 2 for order in psd_dims)
        exp_total = 3 * n_exp
        pow_total = 3 * len(pow_alphas)
        n_cone_rows = soc_total + psd_total + exp_total + pow_total
        n_total_rows = n_zero + n_nonneg + n_cone_rows

        assert A_cvx.shape == (n_total_rows, n), (
            f"A shape {A_cvx.shape} != expected ({n_total_rows}, {n})"
        )

        # Internal slack layout: one SOC needs (v_dim + 2) slots = (k - 1) + 2 = k + 1
        # (extra "phantom" w-slot pinned to 0). PSD uses lower-triangular svec;
        # EXP and POWER need 3 slots each.
        n_soc_blocks = len(soc_dims)
        n_psd_blocks = len(psd_dims)
        n_pow_blocks = len(pow_alphas)
        n_slack = soc_total + n_soc_blocks + psd_total + 3 * n_exp + 3 * n_pow_blocks
        n_vars_total = n + n_slack
        if n_vars_total > np.iinfo(np.int32).max or n_total_rows > np.iinfo(np.int32).max:
            raise ValueError("PDHCG dimensions must fit signed 32-bit indices.")

        # Every CVXPY cone row maps to exactly one internal slack slot. Store the
        # column map directly; row indices are simply arange(n_cone_rows).
        S_cols = np.empty(n_cone_rows, dtype=np.int64)

        n_cones = n_soc_blocks + n_psd_blocks + n_exp + n_pow_blocks
        cone_types = np.empty(n_cones, dtype=np.int32)
        cone_starts = np.empty(n_cones, dtype=np.int32)
        cone_v_dims = np.ones(n_cones, dtype=np.int32)
        cone_alphas = np.zeros(n_cones, dtype=np.float64)
        is_fixed_mask = np.zeros(n_slack, dtype=np.uint8)
        slack_off = 0
        cvx_row_off = 0
        cone_idx = 0

        # --- SOC blocks ---
        for k in soc_dims:
            # cvxpy layout: (top, tail_0..tail_{k-2}) at rows cvx_row_off..cvx_row_off+k-1
            # internal layout: [v_0..v_{k-2}, w, z] at slots slack_off..slack_off+k
            mapped_slots = S_cols[cvx_row_off : cvx_row_off + k]
            mapped_slots[0] = slack_off + k  # z
            mapped_slots[1:] = np.arange(slack_off, slack_off + k - 1, dtype=np.int64)
            is_fixed_mask[slack_off + (k - 1)] = 1  # phantom w always pinned

            cone_types[cone_idx] = int(ConeType.SOC)
            cone_starts[cone_idx] = n + slack_off
            cone_v_dims[cone_idx] = k - 1
            cone_idx += 1
            slack_off += k + 1
            cvx_row_off += k

        # --- PSD blocks ---
        # CVXPY uses lower-triangular column-major svec with sqrt(2)-scaled
        # off-diagonal entries, which is PDHCG's native PSD representation.
        for order in psd_dims:
            packed_length = order * (order + 1) // 2
            S_cols[cvx_row_off : cvx_row_off + packed_length] = np.arange(
                slack_off, slack_off + packed_length, dtype=np.int64
            )
            cone_types[cone_idx] = int(ConeType.PSD)
            cone_starts[cone_idx] = n + slack_off
            cone_v_dims[cone_idx] = order
            cone_idx += 1
            slack_off += packed_length
            cvx_row_off += packed_length

        # --- EXP blocks ---
        if n_exp:
            cone_slice = slice(cone_idx, cone_idx + n_exp)
            cone_types[cone_slice] = int(ConeType.EXP)
            cone_starts[cone_slice] = n + slack_off + 3 * np.arange(n_exp, dtype=np.int64)
            row_count = 3 * n_exp
            mapped_slots = np.arange(slack_off, slack_off + row_count, dtype=np.int64)
            S_cols[cvx_row_off : cvx_row_off + row_count] = mapped_slots
            cone_idx += n_exp
            slack_off += row_count
            cvx_row_off += row_count

        # --- POWER3D blocks ---
        # cvxpy PowCone3D: x^alpha * y^(1-alpha) >= |z|, x,y >= 0. Direct 1-1 mapping.
        if n_pow_blocks:
            cone_slice = slice(cone_idx, cone_idx + n_pow_blocks)
            cone_types[cone_slice] = int(ConeType.POWER)
            cone_starts[cone_slice] = n + slack_off + 3 * np.arange(n_pow_blocks, dtype=np.int64)
            cone_alphas[cone_slice] = np.asarray(pow_alphas, dtype=np.float64)
            row_count = 3 * n_pow_blocks
            mapped_slots = np.arange(slack_off, slack_off + row_count, dtype=np.int64)
            S_cols[cvx_row_off : cvx_row_off + row_count] = mapped_slots
            cone_idx += n_pow_blocks
            slack_off += row_count
            cvx_row_off += row_count

        assert cone_idx == n_cones
        assert slack_off == n_slack
        assert cvx_row_off == n_cone_rows

        # --- Assemble A_new = [A_x, mapping matrix M(cvx rows -> internal slack cols)] ---
        # Zero + nonneg rows: no slack (absorbed into row bounds).
        # Cone rows: A_cvx (x-part) + M (identity-like row-to-slot mapping).
        n_row_lp = n_zero + n_nonneg
        A_top_x = A_cvx[:n_row_lp, :]  # LP rows: x-part
        A_bot_x = A_cvx[n_row_lp:, :]  # cone rows: x-part

        M = sp.csr_matrix(
            (
                np.ones(n_cone_rows, dtype=np.float64),
                (np.arange(n_cone_rows, dtype=np.int64), S_cols),
            ),
            shape=(n_cone_rows, n_slack),
        )

        # LP rows have zero on slack columns; cone rows have M
        zeros_top = sp.csr_matrix((n_row_lp, n_slack))
        A_new = sp.vstack(
            [
                sp.hstack([A_top_x, zeros_top]),
                sp.hstack([A_bot_x, M]),
            ],
            format="csr",
        )

        # Row bounds
        row_lb = np.full(n_total_rows, -np.inf, dtype=np.float64)
        row_ub = np.full(n_total_rows, np.inf, dtype=np.float64)
        # Zero rows: A x = b_cvx  (interpret A_cvx*x + s = b_cvx with s = 0)
        row_lb[:n_zero] = b_cvx[:n_zero]
        row_ub[:n_zero] = b_cvx[:n_zero]
        # Nonneg rows: A x + s = b_cvx, s >= 0  =>  A x <= b_cvx
        row_ub[n_zero : n_zero + n_nonneg] = b_cvx[n_zero : n_zero + n_nonneg]
        # Cone rows: A x + M*s_slack = b_cvx (equality)
        row_lb[n_row_lp:] = b_cvx[n_row_lp:]
        row_ub[n_row_lp:] = b_cvx[n_row_lp:]

        # Objective: c and P are on x-part; slack has zero coeff.
        c_full = np.concatenate([c, np.zeros(n_slack, dtype=np.float64)])
        if P is not None and sp.issparse(P) and P.nnz > 0:
            P_full = sp.block_diag(
                [sp.csr_matrix(P), sp.csr_matrix((n_slack, n_slack))], format="csr"
            )
        else:
            P_full = None

        # Variable bounds: x is unbounded; slack is unbounded (cone-slot semantics).
        var_lb = np.full(n_vars_total, -np.inf, dtype=np.float64)
        var_ub = np.full(n_vars_total, np.inf, dtype=np.float64)

        # Assemble is_fixed / primal_start on the full [x; slack] vector.
        is_fixed_full = np.zeros(n_vars_total, dtype=np.uint8)
        primal_start_full = np.zeros(n_vars_total, dtype=np.float64)
        is_fixed_full[n:] = is_fixed_mask

        cones_spec = (
            ConeSpec(
                cone_types,
                cone_starts,
                cone_v_dims,
                cone_alphas,
                fixed_mask=is_fixed_full if n_soc_blocks else None,
            )
            if n_cones
            else None
        )

        # Merge solver_opts into params dict (accepted keys are the PDHCG params).
        params_dict = _translate_opts(solver_opts, verbose)

        info = solve_once(
            Q=P_full,
            R=None,
            A=A_new,
            objective_vector=c_full,
            objective_constant=None,
            variable_lower_bound=var_lb,
            variable_upper_bound=var_ub,
            constraint_lower_bound=row_lb,
            constraint_upper_bound=row_ub,
            zero_tolerance=0.0,
            params=params_dict,
            primal_start=primal_start_full if n_soc_blocks else None,
            dual_start=None,
            D=None,
            cones=cones_spec,
        )

        # Build the solution dict expected by our invert().
        status = info.get("Status", "UNSPECIFIED")
        x_full = np.asarray(info.get("X"), dtype=np.float64) if info.get("X") is not None else None
        y_full = (
            np.asarray(info.get("Pi"), dtype=np.float64) if info.get("Pi") is not None else None
        )

        # Extract primal for original x (first n entries).
        primal = x_full[:n] if x_full is not None else None
        # PDHCG's row multiplier convention is the negative of CVXPY's canonical
        # A*x + s = b convention. Convert once before splitting Zero and inequality
        # cone duals so equality, NonNeg, SOC, PSD, Exp, and Power duals agree with CVXPY.
        cvxpy_dual = -y_full if y_full is not None else None
        eq_dual = cvxpy_dual[:n_zero] if cvxpy_dual is not None else None
        ineq_dual = cvxpy_dual[n_zero:] if cvxpy_dual is not None else None

        return {
            "status": status,
            "value": float(info.get("PrimalObj", 0.0)),
            "primal": primal,
            "eq_dual": eq_dual,
            "ineq_dual": ineq_dual,
            "solve_time": float(info.get("RuntimeSec", 0.0)),
            "iterations": int(info.get("Iterations", 0)),
        }


# Map cvxpy solver_opts to pdhcg's params dict. Common cvxpy option names get
# translated; anything else is passed through if it matches a pdhcg param key.
_OPT_ALIASES = {
    "time_limit": "time_sec_limit",
    "max_iter": "iteration_limit",
    "iter_limit": "iteration_limit",
    "eps": "eps_optimal_relative",
    "eps_abs": "eps_optimal_relative",
    "eps_rel": "eps_optimal_relative",
    "feas_tol": "eps_feasible_relative",
    "opt_tol": "eps_optimal_relative",
    "verbose": "verbose",
}


def _translate_opts(solver_opts: dict, verbose: bool) -> dict:
    params = {"verbose": int(verbose)}
    # cvxpy inserts use_quad_obj to control canonicalization; ignore.
    solver_opts = dict(solver_opts or {})
    solver_opts.pop("use_quad_obj", None)
    for k, v in solver_opts.items():
        key = _OPT_ALIASES.get(k, k)
        params[key] = v
    return params


# --- Register with cvxpy at import time -----------------------------------
def _register() -> None:
    import contextlib

    from cvxpy.reductions.solvers import defines

    inst = PDHCG()
    try:
        _ = inst.is_installed()
    except Exception:
        contextlib.suppress(Exception)
    defines.SOLVER_MAP_CONIC[inst.name()] = inst
    if inst.name() not in defines.CONIC_SOLVERS:
        # Insert near the front so it's preferred when explicitly requested.
        defines.CONIC_SOLVERS.append(inst.name())
    if inst.name() not in defines.INSTALLED_CONIC_SOLVERS:
        defines.INSTALLED_CONIC_SOLVERS.append(inst.name())
    if inst.name() not in defines.INSTALLED_SOLVERS:
        defines.INSTALLED_SOLVERS.append(inst.name())
    # cvxpy.settings mirrors these; update if present.
    for attr in (
        "SOLVER_MAP_CONIC",
        "CONIC_SOLVERS",
        "INSTALLED_CONIC_SOLVERS",
        "INSTALLED_SOLVERS",
    ):
        if hasattr(_cvx_s, attr):
            setattr(_cvx_s, attr, getattr(defines, attr))
    # Add the solver constant string on cvxpy.settings so `cvxpy.PDHCG` works.
    if not hasattr(_cvx_s, "PDHCG"):
        _cvx_s.PDHCG = inst.name()


try:
    _register()
except Exception as exc:  # pragma: no cover — registration is best-effort
    warnings.warn(f"PDHCG cvxpy backend failed to register: {exc}")
