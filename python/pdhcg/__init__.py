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

"""
PDHCG: Primal-Dual Hybrid Conjugate Gradient Solver
===================================================

PDHCG is a high-performance, GPU-accelerated solver designed for
large-scale convex Quadratic Programming (QP) and structured optimization.
It fully leverages NVIDIA CUDA architectures for extreme-scale problems.

Main Classes
------------
Model
    The core solver interface for defining and optimizing QP subproblems.

Links
-----
* Repository: https://github.com/Lhongpei/PDHCG
"""

from importlib.metadata import PackageNotFoundError, version

from . import PDHCG
from .cones import ConeSpec, ConeType
from .model import Model

__all__ = ["ConeSpec", "ConeType", "Model", "PDHCG"]

# get version from package metadata (toml file)
try:
    __version__ = version("pdhcg")
except PackageNotFoundError:
    __version__ = "0.0.0"
