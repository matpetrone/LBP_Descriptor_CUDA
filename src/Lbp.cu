#include<iostream>

__global__ void lbp_kernel(cv::Mat grayImage, cv::Mat outputImage){
    int neighboor = 3;
    int chunks = neighboor*neighboor;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


}