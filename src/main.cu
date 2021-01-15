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

__global__ void LBPkernel(float *img, float *out_img, int width, int height, unsigned int *histogram_bins, int num_bins){
    //    Naive cuda kernel to compute LBP descriptor
//    !! the pixel values are between 0 and 1
    extern __shared__ unsigned int histogram_bins_sm[]; // dynamically allocated shared memory to compute histogram

    int center_col = blockIdx.x * blockDim.x + threadIdx.x;
    int center_row = blockIdx.y * blockDim.y + threadIdx.y;

//    TODO adjust here!!! Initialize histogram bins to 0
    if (threadIdx.x == 0 && threadIdx.y == 0){
        for (unsigned int binIdx = 0; binIdx < num_bins; binIdx++) {
            histogram_bins_sm[binIdx] = 0;
        }
    }
    __syncthreads();

    // Ensure that threads do not attempt illegal memory access (this can happen because there could be more threads than elements in an array)
    if (center_col < width && center_row < height){
        int pixVal = 0;
        int threshold_values[neighborhood];
//        std::vector<int> threshold_values;
        int N_start_col = center_col - 1;
//        int N_start_col = center_col - (MASK_WIDTH / 2);
        int N_start_row = center_row - 1;
//        int N_start_row = center_row - (MASK_WIDTH / 2);
        int arr_idx = 0;
//        iterate over mask pixel values
        for (int j = 0; j < MASK_WIDTH; j++){
            for (int k = 0; k < MASK_WIDTH; k++){
                int curRow = N_start_row + j;
                int curCol = N_start_col + k;
                // Verify we have a valid image pixel
                if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    if (curRow == center_row && curCol == center_col){

                    } else
                    {
                        threshold_values[arr_idx] = (img[curRow * width + curCol] >= img[center_row * width + center_col]) ? 1 : 0;
                        arr_idx ++;
                    }
//                    if (curRow != center_row && curCol != center_col){  // no compute for mask center
////                        threshold_values.push_back(((int)img[curRow * width + curCol] >= (int)img[center_row * width + center_col]) ? 1 : 0);
//
//                    }
                }else
                    arr_idx++;

            }
        }
//        ---
        if (arr_idx > 4)
            printf("ARR VALUE: %d\n", arr_idx);
        for (int i=0; i<8; i++){  // vec[i] operation  has O(1) Complexity
            pixVal += threshold_values[i] * (int)exp2f(i);
        }

        out_img[center_row * width + center_col] = (float)pixVal / 255;

        // Histogram
        atomicAdd(&(histogram_bins_sm[(unsigned int)pixVal]), 1);
        __syncthreads();

//        When all threads of the block have written o histogram bin sm (shared mem) the histogram is stored in global mem
        if (threadIdx.x == 0 && threadIdx.y == 0){
            for (unsigned int binIdx = 0; binIdx < num_bins; binIdx++) {
                atomicAdd(&(histogram_bins[binIdx]), histogram_bins_sm[binIdx]);
            }
//            printf("finish histogram to glob mem, blockIDX: %d, %d\n", blockIdx.x, blockIdx.y);
        }


    }



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
    unsigned int* deviceHistogram;
//    unsigned int* finalHostHistogram;


    int n_histogram_bins = 256; // pixel values from 0 to 255
    unsigned int histogram_bins[n_histogram_bins];
    for(int i = 0; i<n_histogram_bins; i++)
        histogram_bins[i] = 0;

    std::string colour[4] = { "sample_1920_1280", "computer_programming", "post_2", "borabora_1" };
//    std::string filename = "res/images/ppm/computer_programming.ppm";
    std::string ppm_dir = "res/images/ppm/";
    int image_idx = 1;

    std::string filename = ppm_dir + colour[image_idx] + ".ppm";
    inputImage = PPM_import(filename.c_str());
//    inputImage = PPM_import("res/images/ppm/sample_1920_1280.ppm");
    if (Image_getChannels(inputImage) == 3){
//        If RGB image convert to grayscale
        Image_t* oi = PPMtoGrayscale(inputImage);
        inputImage = oi;
        std::string gray_img_filename = ppm_dir + colour[image_idx] + "_gray" + ".ppm";
        PPM_export(gray_img_filename.c_str(), oi);
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
    cudaMalloc((void **) &deviceHistogram,
               n_histogram_bins * sizeof(unsigned int));

    // copy memory from host to device
    cudaMemcpy(deviceInputImageData, hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceHistogram, histogram_bins,
               n_histogram_bins * sizeof(unsigned int), cudaMemcpyHostToDevice);


    dim3 dimGrid(ceil((float) imageWidth / BLOCK_DIM), ceil((float) imageHeight / BLOCK_DIM));

//    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
//    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    printf("dimGrid {%d, %d, %d}, dimBlock: {%d, %d, %d}\n", dimGrid.x, dimGrid.y , dimGrid.z, dimBlock.x, dimBlock.y , dimBlock.z);
    printf("shared mem: %lu \n", n_histogram_bins * sizeof(unsigned int));
    LBPkernel<<<dimGrid, dimBlock, n_histogram_bins * sizeof(unsigned int)>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight, deviceHistogram, n_histogram_bins);
    cudaError_t  error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "GPU assert: %s  in cudaDeviceSynchronize \n", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }

    // copy from device to host memory
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(histogram_bins, deviceHistogram, n_histogram_bins * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);

    std::string lbp_filename = ppm_dir + colour[image_idx] + "_lbp.ppm";
    PPM_export(lbp_filename.c_str(), outputImage);

    unsigned  int sum =0;
    for (int i=0; i<n_histogram_bins; i++){
        sum += histogram_bins[i];
    }
    assert (sum == imageHeight * imageWidth);
    printf("sum: %d\n", sum);
    printf("image width x height %d x %d : %d \n", imageWidth, imageHeight, imageWidth * imageHeight);
    std::string hist_filename = "res/histograms/" + colour[image_idx] + "_hist.csv";
    saveHistogramToCsv(histogram_bins, n_histogram_bins, hist_filename);

    // free device memory
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceHistogram);

    Image_delete(outputImage);
    Image_delete(inputImage);

    return 0;

}
