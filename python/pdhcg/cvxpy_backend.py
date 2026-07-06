# Copyright 2026 Hongpei Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CVXPY conic-solver backend for PDHCG.

Import this module (``import pdhcg.cvxpy_backend``) once per process; it will
register ``PDHCG`` under ``cvxpy.settings.SOLVER_MAP_CONIC`` so that
``problem.solve(solver='PDHCG')`` works.

Supported CVXPY constraints: Zero, NonNeg, SOC, ExpCone, PowCone3D.
Not supported: PSD, integer variables.
"""

from __future__ import annotations

import warnings
from typing import Any

import cvxpy.settings as _cvx_s
import numpy as np
import scipy.sparse as sp
from cvxpy.constraints import SOC, ExpCone, NonNeg, PowCone3D, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver

from ._core import solve_once

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
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg, SOC, ExpCone, PowCone3D]

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

    def cite(self, data):
        return (
            "@misc{pdhcg,\n"
            "  title  = {PDHCG: GPU-accelerated Primal-Dual Hybrid Conjugate Gradient QP solver},\n"
            "  author = {Li, Hongpei and collaborators},\n"
            "  year   = {2026},\n"
            "  url    = {https://github.com/Lhongpei/PDHCG-II}\n"
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
                utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR],
            )
            ineq_dual = utilities.get_dual_values(
                solution["ineq_dual"],
                utilities.extract_dual_value,
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
        n_exp = int(cone_dims.exp)
        pow_alphas = list(cone_dims.p3d)
        if cone_dims.psd:
            raise ValueError("PDHCG does not support PSD constraints.")

        n = c.size
        soc_total = sum(soc_dims)
        exp_total = 3 * n_exp
        pow_total = 3 * len(pow_alphas)
        n_cone_rows = soc_total + exp_total + pow_total
        n_total_rows = n_zero + n_nonneg + n_cone_rows

        assert A_cvx.shape == (n_total_rows, n), (
            f"A shape {A_cvx.shape} != expected ({n_total_rows}, {n})"
        )

        # Internal slack layout: one SOC needs (v_dim + 2) slots = (k - 1) + 2 = k + 1
        # (extra "phantom" w-slot pinned to 0). EXP and POWER need 3 slots each.
        n_soc_blocks = len(soc_dims)
        n_pow_blocks = len(pow_alphas)
        n_slack = soc_total + n_soc_blocks + 3 * n_exp + 3 * n_pow_blocks
        n_vars_total = n + n_slack

        # Build mapping matrix S: (n_cone_rows) x (n_slack). Each cvxpy cone row maps
        # to exactly one internal slack slot; phantom w-slots have no cvxpy row (S
        # column all zero, slot pinned via is_fixed / primal_start).
        S_rows, S_cols = [], []

        # Track internal cone-block starts (offset into slack vector).
        cones_specs: list[dict] = []
        is_fixed_mask = np.zeros(n_slack, dtype=np.int8)
        primal_start_slack = np.zeros(n_slack, dtype=np.float64)
        # Precompute which cone rows are structurally constant (empty A-row).
        # Such rows determine s_slot = b directly; pinning improves conditioning.
        A_lp = A_cvx[: n_zero + n_nonneg, :] if (n_zero + n_nonneg) > 0 else None  # noqa
        A_cone_dense_zero = (
            (np.abs(A_cvx[n_zero + n_nonneg :, :]).sum(axis=1).A1 == 0)
            if n_cone_rows > 0
            else np.zeros(0, dtype=bool)
        )
        b_cone = b_cvx[n_zero + n_nonneg :] if n_cone_rows > 0 else np.zeros(0)
        slack_off = 0
        cvx_row_off = 0

        cone_row_off = 0  # index into A_cone_dense_zero / b_cone

        def maybe_pin(cone_slot_idx: int, cone_row_idx: int) -> bool:
            """If cvxpy cone row `cone_row_idx` has all zeros in A_cvx, its s value
            is fully determined by b_cvx. Pin the corresponding internal slot to
            improve conditioning of the cone projection."""
            if not A_cone_dense_zero[cone_row_idx]:
                return False
            is_fixed_mask[cone_slot_idx] = 1
            primal_start_slack[cone_slot_idx] = float(b_cone[cone_row_idx])
            return True

        # --- SOC blocks ---
        for k in soc_dims:
            # cvxpy layout: (top, tail_0..tail_{k-2}) at rows cvx_row_off..cvx_row_off+k-1
            # internal layout: [v_0..v_{k-2}, w, z] at slots slack_off..slack_off+k
            S_rows.append(cvx_row_off + 0)
            S_cols.append(slack_off + k)  # z
            pin_flags = [False] * (k + 1)
            pin_flags[k - 1] = True  # phantom w
            if maybe_pin(slack_off + k, cone_row_off + 0):
                pin_flags[k] = True
            for i in range(1, k):
                S_rows.append(cvx_row_off + i)
                S_cols.append(slack_off + i - 1)  # v_{i-1}
                if maybe_pin(slack_off + i - 1, cone_row_off + i):
                    pin_flags[i - 1] = True
            is_fixed_mask[slack_off + (k - 1)] = 1  # phantom w always pinned

            cones_specs.append(
                {
                    "type": "soc",
                    "start_idx": n + slack_off,
                    "v_dim": k - 1,
                    "is_fixed": pin_flags,
                }
            )
            slack_off += k + 1
            cvx_row_off += k
            cone_row_off += k

        # --- EXP blocks ---
        for _ in range(n_exp):
            pin_flags = [False, False, False]
            for i in range(3):
                S_rows.append(cvx_row_off + i)
                S_cols.append(slack_off + i)
                if maybe_pin(slack_off + i, cone_row_off + i):
                    pin_flags[i] = True
            spec = {"type": "exp", "start_idx": n + slack_off, "v_dim": 1}
            if any(pin_flags):
                spec["is_fixed"] = pin_flags
            cones_specs.append(spec)
            slack_off += 3
            cvx_row_off += 3
            cone_row_off += 3

        # --- POWER3D blocks ---
        # cvxpy PowCone3D: x^alpha * y^(1-alpha) >= |z|, x,y >= 0. Direct 1-1 mapping.
        for alpha in pow_alphas:
            pin_flags = [False, False, False]
            for i in range(3):
                S_rows.append(cvx_row_off + i)
                S_cols.append(slack_off + i)
                if maybe_pin(slack_off + i, cone_row_off + i):
                    pin_flags[i] = True
            spec = {
                "type": "power",
                "start_idx": n + slack_off,
                "v_dim": 1,
                "alpha": float(alpha),
            }
            if any(pin_flags):
                spec["is_fixed"] = pin_flags
            cones_specs.append(spec)
            slack_off += 3
            cvx_row_off += 3
            cone_row_off += 3

        # --- Assemble A_new = [A_x, mapping matrix M(cvx rows -> internal slack cols)] ---
        # Zero + nonneg rows: no slack (absorbed into row bounds).
        # Cone rows: A_cvx (x-part) + M (identity-like, from S_rows/S_cols).
        n_row_lp = n_zero + n_nonneg
        A_top_x = A_cvx[:n_row_lp, :]  # LP rows: x-part
        A_bot_x = A_cvx[n_row_lp:, :]  # cone rows: x-part

        M = sp.csr_matrix(
            (
                np.ones(len(S_rows), dtype=np.float64),
                (np.asarray(S_rows, dtype=np.int64), np.asarray(S_cols, dtype=np.int64)),
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
        is_fixed_full = np.zeros(n_vars_total, dtype=np.int8)
        primal_start_full = np.zeros(n_vars_total, dtype=np.float64)
        is_fixed_full[n:] = is_fixed_mask
        primal_start_full[n:] = primal_start_slack

        # Convert cones list-of-dicts to solve_once's expected form with is_fixed
        # per cone (list of bool of length slot_count).
        # SOC needs is_fixed on w-slot; other cones don't get is_fixed here.
        # We already populated cones_specs with is_fixed above.

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
            primal_start=primal_start_full if is_fixed_mask.any() else None,
            dual_start=None,
            D=None,
            cones=cones_specs if cones_specs else None,
        )

        # Build the solution dict expected by our invert().
        status = info.get("Status", "UNSPECIFIED")
        x_full = np.asarray(info.get("X"), dtype=np.float64) if info.get("X") is not None else None
        y_full = (
            np.asarray(info.get("Pi"), dtype=np.float64) if info.get("Pi") is not None else None
        )

        # Extract primal for original x (first n entries).
        primal = x_full[:n] if x_full is not None else None
        # Duals: cvxpy expects eq_dual for Zero-cone rows (first n_zero) and ineq_dual
        # for the remaining (nonneg + soc + exp + pow) rows. We have y_full over all
        # n_total_rows rows.
        eq_dual = y_full[:n_zero] if y_full is not None else None
        ineq_dual = y_full[n_zero:] if y_full is not None else None

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
