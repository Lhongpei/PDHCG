# python_bindings/CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# sources
set(PYBIND_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/_core_bindings.cpp)

# build the pybind11 module
pybind11_add_module(_pdhcg_core MODULE ${PYBIND_SOURCES})

target_link_libraries(_pdhcg_core PRIVATE
    pdhcg_core
)
target_include_directories(_pdhcg_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../internal
)

# set rpath so that the module can find the shared libraries at runtime
# Include PSQP build directory in RPATH if PSQP is available
if(TARGET PSQP)
    set(_PDHCG_PYTHON_RPATH "$ORIGIN/../lib;${CMAKE_BINARY_DIR}/_deps/psqp-build")
else()
    set(_PDHCG_PYTHON_RPATH "$ORIGIN/../lib")
endif()

set_target_properties(_pdhcg_core PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${_PDHCG_PYTHON_RPATH}"
  BUILD_RPATH   "${_PDHCG_PYTHON_RPATH}"
)

# better error messages from pybind11
target_compile_definitions(_pdhcg_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

add_custom_command(TARGET _pdhcg_core POST_BUILD
    COMMAND ${Python3_EXECUTABLE} -m pybind11_stubgen _pdhcg_core -o ${CMAKE_CURRENT_SOURCE_DIR}/../pdhcg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Auto-generating _pdhcg_core.pyi type stubs for Python IDEs..."
)

install(TARGETS _pdhcg_core
  LIBRARY DESTINATION pdhcg
  RUNTIME DESTINATION pdhcg
  ARCHIVE DESTINATION pdhcg
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../pdhcg/_pdhcg_core.pyi
  DESTINATION pdhcg
  OPTIONAL
)
