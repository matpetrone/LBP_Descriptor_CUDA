cmake_minimum_required(VERSION 3.15)
project(LBP_Descriptor_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(LBP_Descriptor_CUDA src/main.cu src/Utils.cpp src/Utils.h)

set_target_properties(
        LBP_Descriptor_CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# -----------------------------------------------------
