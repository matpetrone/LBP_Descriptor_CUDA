cmake_minimum_required(VERSION 3.17)
project(LBP_Descriptor_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(LBP_Descriptor_CUDA src/main.cu src/LbpUtils.h src/Lbp.cu src/PPM.h src/PPM.cpp src/Image.h src/Image.cpp)

set_target_properties(
        LBP_Descriptor_CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# -----------------------------------------------------

#cmake_minimum_required(VERSION 3.17)
#project(LBP_Descriptor_CUDA)
#
#set(CMAKE_CXX_STANDARD 14)
#
#add_executable(LBP_Descriptor_CUDA src/LbpUtils.cpp src/LbpUtils.h)