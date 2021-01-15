#include <cstdio>
#include <cassert>
#include <iostream>
#include <math.h>
#include "LbpUtils.h"
#include "Image.h"
#include "PPM.h"
//#include "Lbp.cuh"

// Constant values for LBP kerel
#define MASK_WIDTH 3
#define neighborhood  (MASK_WIDTH * MASK_WIDTH - 1)
#define img_deep 255

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define BLOCK_DIM 32


static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
              << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

__global__ void LBPkernel(float *img, float *out_img, int width, int height){
    //    Naive cuda kernel to compute LBP descriptor
//    !! the pixel values are between 0 and 1
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    printf("Hello from lbp k, col_idx: %d, row_idx: %d\n", col, row);

    // Ensure that threads do not attempt illegal memory access (this can happen because there could be more threads than elements in an array)
    if (col < width && row < height){
        int pixVal = 0;
        int threshold_values[neighborhood];
//        std::vector<int> threshold_values;
        int N_start_col = col - (MASK_WIDTH / 2);
        int N_start_row = row - (MASK_WIDTH / 2);
        int arr_idx = 0;
//        iterate over mask pixel values
        for (int j = 0; j < MASK_WIDTH; j++){
            for (int k = 0; k < MASK_WIDTH; k++){
                int curRow = N_start_row + j;
                int curCol = N_start_col + k;

                // Verify we have a valid image pixel
                if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
//                    printf("pixel value at row: %d, col: %d : %f\n",curRow, curCol, img[curRow * width + curCol] );
                    if (curRow != row && curCol != col){  // no compute for mask center
//                        threshold_values.push_back(((int)img[curRow * width + curCol] >= (int)img[row * width + col]) ? 1 : 0);
                        threshold_values[arr_idx++] = (img[curRow * width + curCol] >= img[row * width + col]) ? 1 : 0;
                    }
//                    else
//                        printf("central pixel value: %f \n", img[row * width + col]);
                }
            }
        }
//        ---
//        for (auto th : threashold_values){
//            pixVal += th * (int)exp2f();
//        }

        for (int i=0; i<neighborhood; i++){  // vec[i] operation  has O(1) Complexity
            pixVal += threshold_values[i] * (int)exp2f(i);
        }

//        printf("pixval: %d\n", pixVal);

        out_img[row * width + col] = (float)pixVal / 255;
    }
}

__global__ void hello(){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    printf("Hello from lbp k, col_idx: %d, row_idx: %d \n", col, row);
}

int main() {
//    cudaDeviceProp deviceProp;
//    cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
//    std::cout << "Num SM: " << deviceProp.multiProcessorCount << std::endl;
//    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
//    std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock << std::endl;


// -----------------


    int imageChannels;
    int imageWidth;
    int imageHeight;
    Image_t* inputImage;
    Image_t* outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    float *deviceInputImageData;
    float *deviceOutputImageData;

    inputImage = PPM_import("res/images/ppm/computer_programming.ppm");
    if (Image_getChannels(inputImage) == 3){
//        If RGB image convert to grayscale
        Image_t* oi = PPMtoGrayscale(inputImage);
        inputImage = oi;
    }

    imageWidth = Image_getWidth(inputImage);
    imageHeight = Image_getHeight(inputImage);
    imageChannels = Image_getChannels(inputImage);
    assert(imageChannels == 1);
    outputImage = Image_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = Image_getData(inputImage);
    hostOutputImageData = Image_getData(outputImage);


    // allocate device buffers
    cudaMalloc((void **) &deviceInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float));


    // copy memory from host to device
    cudaMemcpy(deviceInputImageData, hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);


    dim3 dimGrid(ceil((float) imageWidth / BLOCK_DIM), ceil((float) imageHeight / BLOCK_DIM));
//    dim3 dimGrid(64 , 62);

//    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
//    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    printf("dimGrid {%d, %d, %d}, dimBlock: {%d, %d, %d}\n", dimGrid.x, dimGrid.y , dimGrid.z, dimBlock.x, dimBlock.y , dimBlock.z);

    LBPkernel<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);
    cudaError_t  error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "GPU assert: %s  in cudaDeviceSynchronize \n", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }

    // copy from device to host memory
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);


    PPM_export("res/images/ppm/processed_computer_programming.ppm", outputImage);

    // free device memory
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
//    cudaFree(deviceMaskData);

    Image_delete(outputImage);
    Image_delete(inputImage);

    return 0;

}
