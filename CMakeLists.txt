cmake_minimum_required(VERSION 3.21)

# Project name, C++17, and enable DPC++/SYCL support
project(IntelMaxEnergy LANGUAGES CXX)

# Require at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the oneAPI DPC++ compiler
# If using Intel oneAPI, the environment variables should be set automatically
# Otherwise, user must set CMAKE_CXX_COMPILER to dpcpp
# e.g., cmake -DCMAKE_CXX_COMPILER=dpcpp ..

# Add subdirectories
add_subdirectory(SYnergy)
add_subdirectory(src)
