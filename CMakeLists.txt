cmake_minimum_required(VERSION 3.10)
project(CudaKernelCompilation)

find_package(CUDA REQUIRED)

set(KERNELS_DIR "${CMAKE_SOURCE_DIR}/src/kernel")
set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/src/kernel/compiled_kernel")
set(CUDA_NVCC_EXECUTABLE "$ENV{CUDA_HOME}/bin/nvcc")
message(STATUS "Using CUDA NVCC executable: ${CUDA_NVCC_EXECUTABLE}")
# --- Ensure output directory exists ---
if(NOT EXISTS ${OUTPUT_DIR})
    message(STATUS "Output directory does not exist. Creating: ${OUTPUT_DIR}")
    file(MAKE_DIRECTORY ${OUTPUT_DIR})
endif()

# --- Find all CUDA source files ---
# Note: While functional, file(GLOB) does not automatically detect new files.
# You'll need to re-run cmake if you add new .cu files to KERNELS_DIR.
file(GLOB CUDA_SOURCES "${KERNELS_DIR}/*.cu")

# --- Compile each CUDA file to PTX and place it in the output directory ---
foreach(CUDA_FILE ${CUDA_SOURCES})
    get_filename_component(FILE_NAME ${CUDA_FILE} NAME_WE)
    message(STATUS "Compiling CUDA file: ${CUDA_FILE}")

    set(PTX_OUT_PATH "${OUTPUT_DIR}/${FILE_NAME}.ptx")

    # Use add_custom_command to directly call nvcc and specify the output file
    add_custom_command(
        OUTPUT ${PTX_OUT_PATH}
        COMMAND ${CUDA_NVCC_EXECUTABLE} # Path to the nvcc compiler
                -ptx                  # Tell nvcc to compile to PTX
                -o ${PTX_OUT_PATH}    # Specify the exact output path
                ${CUDA_NVCC_FLAGS}    # Include any global NVCC flags (e.g., arch flags)
                ${CUDA_FILE}          # The input .cu file
        DEPENDS ${CUDA_FILE}          # Declare the input file as a dependency
        COMMENT "Compiling ${FILE_NAME}.cu to PTX for ${PTX_OUT_PATH}"
    )

    # Create a custom target to ensure the custom command is executed.
    # Marking it with ALL ensures it runs when you build all (e.g., `make all`).
    add_custom_target(${FILE_NAME}_ptx_gen ALL DEPENDS ${PTX_OUT_PATH})
endforeach()