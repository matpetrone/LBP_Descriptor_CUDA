#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
int main() {
    std::cout << "Hello, World!" << std::endl;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
    std::cout << "Num SM: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock << std::endl;
    return 0;
}
