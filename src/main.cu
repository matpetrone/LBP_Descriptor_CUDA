#include <iostream>


#include "LbpUtils.h"




int main() {
//    std::cout << "Hello, World!" << std::endl;
//    cudaDeviceProp deviceProp;
//    cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
//    std::cout << "Num SM: " << deviceProp.multiProcessorCount << std::endl;
//    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
//    std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock << std::endl;

    std::string csv_filename = "res/csv_images/leopard.csv";
    std::vector<std::vector<int>> vec = getImageFromCsv(csv_filename);

    int rows = vec.size();
    int cols = vec[0].size();
    int image[rows][cols];  // store image as a 2d matrix
    for (int i = 0; i < vec.size(); i++) {
        for (int j = 0; j < vec[i].size(); j++){
            image[i][j] = vec[i][j];
//            std::cout << "vec: " << vec[i][j] << ", image: " << image[i][j] << std::endl;
        }
    }



    return 0;

}