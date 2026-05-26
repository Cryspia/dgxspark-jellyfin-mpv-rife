cmake_minimum_required(VERSION 3.18)
project(dgxspark_split_dual_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find VapourSynth4 headers. Honour CONDA_PREFIX so devs in vsmpv env
# pick up the right ones automatically.
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()
find_path(VAPOURSYNTH_INCLUDE_DIR
    NAMES VapourSynth4.h
    PATH_SUFFIXES vapoursynth include/vapoursynth
    HINTS $ENV{CONDA_PREFIX}/include
          $ENV{CONDA_PREFIX}/include/vapoursynth
)
if(NOT VAPOURSYNTH_INCLUDE_DIR)
    message(FATAL_ERROR "VapourSynth4.h not found — activate the vsmpv "
                          "conda env or pass -DVAPOURSYNTH_INCLUDE_DIR=...")
endif()
message(STATUS "VapourSynth headers: ${VAPOURSYNTH_INCLUDE_DIR}")

# pybind11 — used to call Python compute callables from the C++ filter.
# Installed in the conda env via `pip install pybind11`; pull its CMake
# config from there.
if(NOT DEFINED pybind11_DIR AND DEFINED ENV{CONDA_PREFIX})
    file(GLOB _pyb11_dirs
        "$ENV{CONDA_PREFIX}/lib/python*/site-packages/pybind11/share/cmake/pybind11")
    if(_pyb11_dirs)
        list(GET _pyb11_dirs 0 pybind11_DIR)
    endif()
endif()
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "pybind11: ${pybind11_DIR}")

add_library(dgxspark_split_dual SHARED src/split_dual_filter.cpp)
target_include_directories(dgxspark_split_dual PRIVATE ${VAPOURSYNTH_INCLUDE_DIR})
target_link_libraries(dgxspark_split_dual PRIVATE pybind11::embed)
target_compile_options(dgxspark_split_dual PRIVATE -Wall -Wextra -O3)
set_target_properties(dgxspark_split_dual PROPERTIES
    OUTPUT_NAME dgxspark_split_dual
    PREFIX "lib"
)
