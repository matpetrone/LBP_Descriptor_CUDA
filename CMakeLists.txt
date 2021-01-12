cmake_minimum_required(VERSION 3.17)
project(LBP_Descriptor_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(LBP_Descriptor_CUDA main.cu)

set_target_properties(
        LBP_Descriptor_CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)