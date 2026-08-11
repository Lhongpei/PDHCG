"""
pdhcg core bindings (auto-detect dense/CSR/CSC/COO; initialize default params here)
"""
from __future__ import annotations
import typing
__all__: list[str] = ['get_default_params', 'read_problem_file', 'solve_once']
def get_default_params() -> dict:
    """
    Return default PDHG parameters as a dict
    """
def read_problem_file(path: str) -> dict:
    """
    Read an MPS or CBF file (.mps/.mps.gz/.cbf/.cbf.gz) and return a dict with c, obj_const, Q, A, constr_lb, constr_ub, var_lb, var_ub, cones, affine_F, affine_g, affine_cones, and primal_start.
    """
def solve_once(Q: typing.Any, R: typing.Any, A: typing.Any, objective_vector: typing.Any, objective_constant: typing.Any = None, variable_lower_bound: typing.Any = None, variable_upper_bound: typing.Any = None, constraint_lower_bound: typing.Any = None, constraint_upper_bound: typing.Any = None, zero_tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, params: typing.Any = None, primal_start: typing.Any = None, dual_start: typing.Any = None, D: typing.Any = None, cones: typing.Any = None, affine_F: typing.Any = None, affine_g: typing.Any = None, affine_cones: typing.Any = None) -> dict:
    ...
