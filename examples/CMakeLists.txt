# Examples CMakeLists.txt
# Build examples for HPC-AI-Optimization-Lab

cmake_minimum_required(VERSION 3.18)

# Option to build examples
option(BUILD_EXAMPLES "Build example programs" ON)

if(NOT BUILD_EXAMPLES)
    return()
endif()

message(STATUS "Building examples...")

# Common include directories
set(EXAMPLE_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/common
)

# Helper function to add CUDA example
function(add_cuda_example name source_file)
    add_executable(${name} ${source_file})
    target_include_directories(${name} PRIVATE ${EXAMPLE_INCLUDE_DIRS})
    target_compile_features(${name} PRIVATE cxx_std_20)
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    # Link against project libraries
    target_link_libraries(${name} PRIVATE
        elementwise_ops
        reduction_ops
        gemm_ops
        attention_ops
    )
endfunction()

# Elementwise examples
add_cuda_example(relu_example 01_elementwise/relu_example.cu)

# GEMM examples
add_cuda_example(gemm_benchmark 03_gemm/gemm_benchmark.cu)
target_link_libraries(gemm_benchmark PRIVATE CUDA::cublas)

# Install examples
install(TARGETS relu_example gemm_benchmark
    RUNTIME DESTINATION bin/examples
)

# Install Python examples
install(DIRECTORY python/
    DESTINATION share/examples/python
    FILES_MATCHING PATTERN "*.py"
)
