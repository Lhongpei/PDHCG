# -----------------------------------------------------------------------------
# ROOT CMakeLists.txt for PDHCG (Unified Build System)
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

project(pdhcg LANGUAGES C CXX CUDA)

set(PDHCG_VERSION_MAJOR 0)
set(PDHCG_VERSION_MINOR 2)
set(PDHCG_VERSION_PATCH 0)

set(PDHCG_VERSION "${PDHCG_VERSION_MAJOR}.${PDHCG_VERSION_MINOR}.${PDHCG_VERSION_PATCH}")
add_compile_definitions(PDHCG_VERSION="${PDHCG_VERSION}")
add_compile_definitions(CUSPARSE_ENABLE_EXPERIMENTAL_API)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Set default build type to Release if not specified (Optimized builds)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
endif()

# Global Compile flags (corresponding to CFLAGS/NVCCFLAGS)
add_compile_options(-fPIC -O3 -Wall -Wextra -g)

# Windows compatibility
if (WIN32)
    add_definitions(-Dstrtok_r=strtok_s)
endif()

# CUDA standards and RDC (Relocatable Device Code)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# CONTROL OPTIONS
# -----------------------------------------------------------------------------
include(CMakeDependentOption)

option(PDHCG_BUILD_STATIC_LIB "Build the PDHCG static library" ON)
option(PDHCG_BUILD_SHARED_LIB "Build the PDHCG shared library" ON)
option(PDHCG_COMPILE_PSQP "Download and use PSQP for presolving" OFF)
option(PDHCG_COMPILE_DISTRIBUTED "Enable distributed computing with MPI" OFF)

# format: cmake_dependent_option(OPTION "docstring" DEFAULT_VALUE "DEPENDENCY_VARIABLE" FORCE_OFF_VALUE)
cmake_dependent_option(PDHCG_BUILD_PYTHON "Build the PDHCG Python bindings" OFF
                       "PDHCG_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(PDHCG_BUILD_CLI "Build the PDHCG command-line executable" ON
                       "PDHCG_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(PDHCG_BUILD_TESTS "Build the PDHCG test suite" OFF
                       "PDHCG_BUILD_STATIC_LIB" OFF)

# -----------------------------------------------------------------------------
# FIND DEPENDENCIES
# -----------------------------------------------------------------------------
# Core dependencies (required for Julia/Yggdrasil and Python)
find_package(CUDAToolkit REQUIRED)
find_package(ZLIB REQUIRED)

if (PDHCG_BUILD_PYTHON)
    # Dependencies required only for Python bindings
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED) # For versioning script and pybind11
endif()

include(FetchContent)

# -----------------------------------------------------------------------------
# PSQP (QP Presolver) Integration
# -----------------------------------------------------------------------------
if(PDHCG_COMPILE_PSQP)
    # Use FetchContent to automatically download PSQP from git repository
    set(PSQP_VERSION_TAG "main" CACHE STRING "PSQP git tag/branch to use")

    FetchContent_Declare(
      psqp
      GIT_REPOSITORY https://github.com/Lhongpei/PSQP.git
      GIT_TAG        ${PSQP_VERSION_TAG}
    )

    # Check if PSQP has already been populated
    FetchContent_GetProperties(psqp)
    if(NOT psqp_POPULATED)
        message(STATUS "Fetching PSQP from https://github.com/Lhongpei/PSQP.git (${PSQP_VERSION_TAG})")
        FetchContent_MakeAvailable(psqp)
        message(STATUS "PSQP populated at: ${psqp_SOURCE_DIR}")
    endif()

    # Set up PSQP include directories
    if(TARGET PSQP)
        target_include_directories(PSQP INTERFACE
            $<BUILD_INTERFACE:${psqp_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${psqp_SOURCE_DIR}/include/PSQP>
        )
        message(STATUS "PSQP target configured successfully")
        # Add version definition
        add_compile_definitions(PSQP_VERSION=\"${PSQP_VERSION_TAG}\")
    else()
        message(WARNING "PSQP target not found. Presolving features will be disabled.")
    endif()
else()
    message(STATUS "PSQP integration disabled by user (PDHCG_COMPILE_PSQP=OFF).")
endif()

# -----------------------------------------------------------------------------
# SOURCE DISCOVERY & TARGET DEFINITION
# -----------------------------------------------------------------------------
# Using file(GLOB) for convenience, but explicit lists are recommended for robust builds
file(GLOB C_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)
file(GLOB CU_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

# Exclude cli.c from library builds
list(REMOVE_ITEM C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cli.c")

# Set common include directories for the core libraries
set(CORE_INCLUDE_DIRS
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include     # Public API headers
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal    # Internal implementation headers
)

if(PDHCG_COMPILE_DISTRIBUTED)
    add_definitions(-DPDHCG_COMPILE_DISTRIBUTED)
    file(GLOB DIST_C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/distributed/*.c")
    file(GLOB DIST_CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/distributed/*.cu")
    list(REMOVE_ITEM DIST_C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/distributed/distributed_ops_stub.c")

    list(APPEND C_SOURCES ${DIST_C_SOURCES})
    list(APPEND CU_SOURCES ${DIST_CU_SOURCES})

    list(APPEND CORE_INCLUDE_DIRS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/distributed)
else()
    list(APPEND C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/distributed/distributed_ops_stub.c")
    list(APPEND CORE_INCLUDE_DIRS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/distributed)
endif()

# Set common link libraries
set(CORE_LINK_LIBS PUBLIC
  CUDA::cudart
  CUDA::cublas
  CUDA::cusparse
  ZLIB::ZLIB
)

if(PDHCG_COMPILE_PSQP AND TARGET PSQP)
    list(APPEND CORE_LINK_LIBS PUBLIC PSQP)
    set(PDHCG_OPTIONAL_DEFINES "PSQP_AVAILABLE")
else()
    set(PDHCG_OPTIONAL_DEFINES "")
endif()

if(PDHCG_COMPILE_DISTRIBUTED)
    find_package(MPI REQUIRED)
    add_compile_definitions(PDHCG_COMPILE_DISTRIBUTED)
    message(STATUS "PDHCG Distributed Mode: ENABLED (MPI/NCCL support activated)")
else()
    message(STATUS "PDHCG Distributed Mode: DISABLED (Building for single-node execution)")
endif()


if(PDHCG_COMPILE_DISTRIBUTED AND MPI_FOUND)
    list(APPEND CORE_LINK_LIBS MPI::MPI_CXX)
    include_directories(${MPI_CXX_INCLUDE_PATH})
    find_library(NCCL_LIB NAMES nccl
                 PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
                       $ENV{NCCL_ROOT}/lib
                       /usr/local/cuda/lib64
                       /usr/lib/x86_64-linux-gnu)

    if(NCCL_LIB)
        message(STATUS "Found NCCL: ${NCCL_LIB}")
        list(APPEND CORE_LINK_LIBS ${NCCL_LIB})
    else()
        message(WARNING "NCCL library not found explicitly, falling back to -lnccl")
        list(APPEND CORE_LINK_LIBS nccl)
    endif()
endif()
# -----------------------------------------------------------------------------
# 1. Core STATIC Library (pdhcg_core)
# -----------------------------------------------------------------------------
if(PDHCG_BUILD_STATIC_LIB)
    add_library(pdhcg_core STATIC
      ${C_SOURCES}
      ${CU_SOURCES}
    )
    target_include_directories(pdhcg_core ${CORE_INCLUDE_DIRS})
    target_link_libraries(pdhcg_core ${CORE_LINK_LIBS})

    # Add PSQP compile definition
    target_compile_definitions(pdhcg_core PUBLIC ${PDHCG_OPTIONAL_DEFINES})

    set_target_properties(pdhcg_core PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# -----------------------------------------------------------------------------
# 2. Shared Library (libPDHCG.so)
# -----------------------------------------------------------------------------
if(PDHCG_BUILD_SHARED_LIB)
    add_library(PDHCG_shared SHARED
      ${C_SOURCES}
      ${CU_SOURCES}
    )
    target_include_directories(PDHCG_shared ${CORE_INCLUDE_DIRS})
    target_link_libraries(PDHCG_shared ${CORE_LINK_LIBS})

    # Add PSQP compile definition
    target_compile_definitions(PDHCG_shared PUBLIC ${PDHCG_OPTIONAL_DEFINES})

    # Shared library must resolve device symbols as it is a final link point
    set_target_properties(PDHCG_shared PROPERTIES
        OUTPUT_NAME "pdhcg"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# -----------------------------------------------------------------------------
# 3. CLI Executable (pdhcg)
# -----------------------------------------------------------------------------
if(PDHCG_BUILD_CLI)
    if(NOT TARGET pdhcg_core)
        message(FATAL_ERROR "PDHCG_BUILD_CLI=ON requires PDHCG_BUILD_STATIC_LIB=ON.")
    endif()

    add_executable(PDHCG_cli src/cli.c)

    target_include_directories(PDHCG_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/internal
    )

    if(PDHCG_COMPILE_DISTRIBUTED)
        target_include_directories(PDHCG_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/distributed)
        target_compile_definitions(PDHCG_cli PRIVATE PDHCG_COMPILE_DISTRIBUTED)
    endif()

    # Link CLI to the static core library
    target_link_libraries(PDHCG_cli PRIVATE pdhcg_core)

    # CLI is a final executable, it must resolve device symbols
    # Set RPATH so that libPSQP.so can be found without LD_LIBRARY_PATH
    # PSQP is in _deps/psqp-build when using FetchContent
    set_target_properties(PDHCG_cli PROPERTIES
        OUTPUT_NAME "pdhcg"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        INSTALL_RPATH "${CMAKE_BINARY_DIR}/_deps/psqp-build"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# -----------------------------------------------------------------------------
# 4. Tests (CTest Integration)
# -----------------------------------------------------------------------------
if(PDHCG_BUILD_TESTS)
    if(NOT TARGET pdhcg_core)
        message(FATAL_ERROR "PDHCG_BUILD_TESTS=ON requires PDHCG_BUILD_STATIC_LIB=ON.")
    endif()

    enable_testing()
    file(GLOB TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cu"
    )

    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

        add_executable(${TEST_NAME} ${TEST_SRC})

        # Link tests to the core static library
        target_link_libraries(${TEST_NAME} PRIVATE pdhcg_core)

        # Set up test includes
        target_include_directories(${TEST_NAME}
          PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
          PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal
        )

        # Set up RPATH to find PSQP if available
        if(TARGET PSQP)
            set(TEST_RPATH "${CMAKE_BINARY_DIR}/_deps/psqp-build")
        else()
            set(TEST_RPATH "")
        endif()

        # Tests are final executables, they must resolve device symbols
        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests"
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            INSTALL_RPATH "${TEST_RPATH}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )

        # Register with CTest
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    endforeach()
endif()

# -----------------------------------------------------------------------------
# 5. Python Bindings (Conditional)
# -----------------------------------------------------------------------------
if (PDHCG_BUILD_PYTHON)
    if(NOT TARGET pdhcg_core)
        message(FATAL_ERROR "PDHCG_BUILD_PYTHON=ON requires PDHCG_BUILD_STATIC_LIB=ON.")
    endif()
    add_subdirectory(python_bindings)
endif()

# -----------------------------------------------------------------------------
# 6. Install Targets
# -----------------------------------------------------------------------------

if (PDHCG_BUILD_PYTHON)
    install(DIRECTORY include/
        DESTINATION include/
        FILES_MATCHING PATTERN "*.h"
    )

else()
    if(TARGET pdhcg_core)
        install(TARGETS pdhcg_core
            ARCHIVE DESTINATION lib
        )
    endif()

    if(TARGET PDHCG_shared)
        install(TARGETS PDHCG_shared
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin # 'bin' for DLLs on Windows, 'lib' for .so on Linux
        )
    endif()

    if(TARGET PDHCG_cli)
        install(TARGETS PDHCG_cli
            RUNTIME DESTINATION bin
        )
    endif()

    install(DIRECTORY include/
        DESTINATION include/
        FILES_MATCHING PATTERN "*.h"
    )
endif()
