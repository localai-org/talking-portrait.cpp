cmake_minimum_required(VERSION 3.24)
project(talking-portrait LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(TP_CUDA        "enable the CUDA backend (via GGML_CUDA)" OFF)
option(TP_VULKAN      "enable the Vulkan backend (via GGML_VULKAN)" OFF)
option(TP_FUZZ        "build libFuzzer targets (clang only)"    OFF)
option(TP_BUILD_TOOLS "build CLI / parity / bench tools"        ON)
option(TP_BUILD_TESTS "build ctest suite"                       ON)
option(TP_SHARED      "build tp as a shared library for FFI/demo use" OFF)

if (TP_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 120)
    endif()
endif()

if (TP_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

if (TP_FUZZ)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "TP_FUZZ requires clang (libFuzzer)")
    endif()
    set(TP_FUZZ_SANITIZERS "address,undefined" CACHE STRING "fuzz sanitizers")
    # global: instrument ggml + lib + harnesses end-to-end
    add_compile_options(-fsanitize=${TP_FUZZ_SANITIZERS} -fsanitize=fuzzer-no-link
                        -fno-sanitize-recover=undefined
                        -fno-omit-frame-pointer -g -O1)
    add_link_options(-fsanitize=${TP_FUZZ_SANITIZERS})
endif()

# The vendored ggml submodule carries a small local patch (see
# patches/ggml-tp-ops.patch): a fused bilinear-gather op (grid_sample) and a
# channels-first im2col fast path, both needed to hit the M7 performance bar.
# Applied automatically on fresh checkouts.
if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cuda/bilinear-gather.cu)
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-tp-ops.patch
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ggml
        RESULT_VARIABLE TP_GGML_PATCH_RC)
    if (NOT TP_GGML_PATCH_RC EQUAL 0)
        message(FATAL_ERROR "failed to apply patches/ggml-tp-ops.patch to ggml/")
    endif()
    message(STATUS "applied patches/ggml-tp-ops.patch to ggml/")
endif()

add_subdirectory(ggml)

if (TP_SHARED)
    set(TP_LIBRARY_TYPE SHARED)
else()
    set(TP_LIBRARY_TYPE STATIC)
endif()

add_library(tp ${TP_LIBRARY_TYPE}
    src/tp.cpp
    src/backend.cpp
    src/gguf_load.cpp
    src/json_min.cpp
    src/dump.cpp
    src/audio_enc.cpp
    src/motion_enc.cpp
    src/fmt.cpp
    src/motion_dec.cpp
    src/pipeline.cpp
    src/wav_io.cpp
    src/image_io.cpp
)
target_include_directories(tp PUBLIC include PRIVATE src third_party)
target_link_libraries(tp PUBLIC ggml)

if (TP_BUILD_TOOLS)
    foreach(tool tp-cli tp-parity)
        add_executable(${tool} tools/${tool}.cpp)
        target_link_libraries(${tool} PRIVATE tp)
        target_include_directories(${tool} PRIVATE src)
    endforeach()
    foreach(tool tp-bench tp-allocproof)
        add_executable(${tool} bench/${tool}.cpp)
        target_link_libraries(${tool} PRIVATE tp)
        target_include_directories(${tool} PRIVATE src)
    endforeach()
endif()

if (TP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (TP_FUZZ)
    add_subdirectory(fuzz)
endif()
