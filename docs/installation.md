# Installation

## Requirements

- **GPU**: NVIDIA GPU with CUDA 12.4+
- **Build Tools**: CMake (≥ 3.20), GCC, NVCC
- **Python**: Python 3.8+ (for Python bindings)
- **Distributed (Optional)**: MPI (e.g., OpenMPI) and NCCL for multi-GPU support

!!! note "CUDA Version and SpMVOp"
    PDHCG automatically detects your CUDA version at compile time:

    - **CUDA 13+**: Uses cuSPARSE **SpMVOp** for improved performance.
    - **CUDA 12.x**: Falls back to the standard **SpMV** API. No manual intervention is required.

## C++ Executable

### Build from Source

Clone the repository and compile the project using CMake:

```bash
git clone https://github.com/Lhongpei/PDHCG.git
cd PDHCG
cmake -S . -B build
cmake --build build --clean-first
```

This will create the solver binary at `./build/bin/pdhcg`.

### Specifying CUDA Compiler

If your system has multiple CUDA versions or the default nvcc is outdated, explicitly specify the path to your CUDA compiler:

```bash
# Replace '/your/path/to/nvcc' with the actual path, e.g., /usr/local/cuda-12.6/bin/nvcc
CUDACXX=/your/path/to/nvcc cmake -S . -B build
cmake --build build --clean-first
```

### Build with Multi-GPU Support

To enable distributed multi-GPU solving, turn on the `PDHCG_COMPILE_DISTRIBUTED` CMake option. This requires MPI and NCCL to be installed on your system.

```bash
cmake -S . -B build -DPDHCG_COMPILE_DISTRIBUTED=ON
cmake --build build --clean-first
```

When enabled, the solver binary automatically detects whether it is launched with multiple MPI ranks and switches to the distributed solver.

## Python Package

!!! note "Multi-GPU support"
    The Python interface currently supports single-GPU solving only. For multi-GPU distributed solving, build the C++ executable with `-DPDHCG_COMPILE_DISTRIBUTED=ON` and launch it via `mpirun`.

### From PyPI (Recommended)

```bash
pip install pdhcg
```

### From Source

```bash
git clone https://github.com/Lhongpei/PDHCG.git
cd PDHCG
pip install .
```

### Development Installation

For development with editable install:

```bash
git clone https://github.com/Lhongpei/PDHCG.git
cd PDHCG
pip install -e ".[test]"
```

If your system has multiple CUDA installations or the default nvcc (typically in `/usr/bin/nvcc`) is outdated, you must explicitly point to your modern CUDA compiler using environment variables:

```bash
# Replace '/your/path/to/nvcc' with your actual path
# Example: export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
export CUDACXX=/your/path/to/nvcc
export SKBUILD_CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/your/path/to/nvcc"

pip install pdhcg
```

## Verification

### C++ Executable

```bash
./build/bin/pdhcg --help
```

### Python Package

```python
import pdhcg
print(pdhcg.__version__)
```

## Pre-commit Hooks (For Contributors)

To ensure code quality before committing, install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This automatically formats Python (Ruff), C/C++ (clang-format), and checks spelling when you commit.
