cmake_minimum_required(VERSION 3.12)
project(stratego_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER "gcc-13")
set(CMAKE_CXX_COMPILER "g++-13")

# Set compiler flags for Debug and Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto")
endif()

find_package(pybind11 REQUIRED)

# Core library
add_library(stratego_core SHARED
    core/detectors.cpp
    core/masked_multi_discrete.cpp
    core/player_state_handler.cpp
    core/stratego_config.cpp
    core/stratego_env.cpp
    core/primitives.h
)

# Apply ASan flags for Debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(stratego_core PRIVATE -fPIC -fsanitize=address)
    target_link_options(stratego_core PRIVATE -fsanitize=address)
endif()

# Python module
pybind11_add_module(stratego_cpp binding.cpp)
target_link_libraries(stratego_cpp PRIVATE stratego_core pybind11::module)

# Apply ASan flags for Python module in Debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(stratego_cpp PRIVATE -fsanitize=address)
    target_link_options(stratego_cpp PRIVATE -fsanitize=address)
endif()