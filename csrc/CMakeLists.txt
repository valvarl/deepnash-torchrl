cmake_minimum_required(VERSION 3.12)
project(stratego_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Добавляем эту строку

find_package(pybind11 REQUIRED)

# Собираем core библиотеку как SHARED
add_library(stratego_core SHARED  # Изменяем STATIC на SHARED
    core/detectors.cpp
    core/masked_multi_discrete.cpp
    core/player_state_handler.cpp
    core/stratego_config.cpp
    core/stratego_env.cpp
    core/primitives.h
)

# Добавляем флаги компиляции
target_compile_options(stratego_core PRIVATE -fPIC)

# Создаем Python модуль
pybind11_add_module(stratego_cpp binding.cpp)
target_link_libraries(stratego_cpp PRIVATE stratego_core pybind11::module)