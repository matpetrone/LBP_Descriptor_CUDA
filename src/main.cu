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
#define n_histogram_bins  256
#define img_deep 255

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define TILE_WIDTH 8
#define BLOCK_DIM TILE_WIDTH
//#define BLOCK_DIM (TILE_WIDTH + MASK_WIDTH - 1)
static_assert(BLOCK_DIM * BLOCK_DIM < 1024, "max number of threads per block exceeded");
#define ww (TILE_WIDTH + MASK_WIDTH - 1)
#define MASK_RADIUS (MASK_WIDTH / 2)

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
    __shared__ unsigned int histogram_bins_sm[n_histogram_bins]; // shared memory to compute histogram

    int  center_col = blockIdx.x * blockDim.x + threadIdx.x;
    int  center_row = blockIdx.y * blockDim.y + threadIdx.y;

//    Initialize histogram bins to 0
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
        histogram_bins_sm[i] = 0;
    __syncthreads();

    // Ensure that threads do not attempt illegal memory access (this can happen because there could be more threads than elements in an array)
    if (center_col < width && center_row < height){
        int pixVal = 0;
        int threshold_values[neighborhood];
//        std::vector<int> threshold_values;
        int N_start_col = center_col - (MASK_WIDTH / 2);
        int N_start_row = center_row - (MASK_WIDTH / 2);
        int arr_idx = 0;
//        iterate over mask pixel values
        for (int j = 0; j < MASK_WIDTH; j++){
            for (int k = 0; k < MASK_WIDTH; k++){
                int curRow = N_start_row + j;
                int curCol = N_start_col + k;

                // Verify we have a valid image pixel
                if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    if (curRow == center_row && curCol == center_col){ } else
                    {
                        threshold_values[arr_idx] = (img[curRow * width + curCol] >= img[center_row * width + center_col]) ? 1 : 0;
                        arr_idx ++;
                    }
                }else
                    arr_idx++;
            }
        }

        for (int i=0; i<8; i++){  // vec[i] operation  has O(1) Complexity
            pixVal += threshold_values[i] * (int)exp2f(i);
        }

        out_img[center_row * width + center_col] = (float)pixVal / 255;
//        printf("out img: %f \n",out_img[center_row * width + center_col] );

        // Histogram
        atomicAdd(&(histogram_bins_sm[(unsigned int)pixVal]), 1);
        __syncthreads();

//        When all threads of the block have written o histogram bin sm (shared mem) the histogram is stored in global mem
        if (threadIdx.x == 0 && threadIdx.y == 0){
            for (unsigned int binIdx = 0; binIdx < num_bins; binIdx++) {
                atomicAdd(&(histogram_bins[binIdx]), histogram_bins_sm[binIdx]);
            }
        }
    }
}

__global__ void LBPkernelTiling(float *img, float *out_img, const int width, const int height, unsigned int *histogram_bins,const int num_bins){
    //    Shared memory cuda kernel to compute LBP descriptor
    __shared__ unsigned int histogram_bins_sm[n_histogram_bins]; // shared memory to compute histogram
    __shared__ float sm_img[ww][ww];

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

//    Initialize histogram bins to 0
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
        histogram_bins_sm[i] = 0;
    __syncthreads();

//    for (int i=0; i< num_bins; i++)
//        assert(histogram_bins_sm[i] == 0);
//    __syncthreads();

    // First batch loading (Load TILE_WIDTH*TILE_WIDTH elements)
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destY = dest / ww;
    int destX = dest % ww;
    int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
    int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
    int src = srcY * width + srcX;

    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
        sm_img[destY][destX] = img[src];
    } else {
        sm_img[destY][destX] = 0;
    }

    // Second batch loading (Load the data outside the TILE_WIDTH*TILE_WIDTH)
    dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    destY = dest / ww;
    destX = dest % ww;
    srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
    srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
    src =  srcY * width + srcX ;
    if (destY < ww) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            sm_img[destY][destX] = img[src];
        } else {
            sm_img[destY][destX] = 0;
        }
    }
    __syncthreads();

//    printf("ThreadIdx.x :  %d done.\n", threadIdx.x + blockIdx.x * blockDim.x);
//    printf("loading pix val: %d == %d, srcX: %d, srcY: %d, width: %d, height: %d, destX: %d, destY: %d, blockIdx.x: %d, blockIdx.y: %d\n", sm_img[destY][destX], img[src], srcX, srcY, width, height,destX, destY, blockIdx.x, blockIdx.y);

    int center_col = threadIdx.x;
    int center_row = threadIdx.y;

    if ((tx < width ) && (ty < height)) {
        int pixVal = 0;
        int threshold_values[neighborhood];

        int N_start_col = center_col - (MASK_WIDTH / 2);
        int N_start_row = center_row - (MASK_WIDTH / 2);
        int arr_idx = 0;

//        Iterate over mask pixel values
        for (int j = 0; j < MASK_WIDTH; j++) {
            for (int k = 0; k < MASK_WIDTH; k++) {
                int curRow = N_start_row + j;
                int curCol = N_start_col + k;

                // Verify we have a valid image pixel
                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    if (curRow == center_row && curCol == center_col) {}
                    else {
                        threshold_values[arr_idx] = (sm_img[curRow][curCol] >=
                                                     sm_img[center_row][center_col]) ? 1 : 0;
                        arr_idx++;
                    }
                } else
                    arr_idx++;
            }
        }

        for (int i = 0; i < 8; i++) {  // vec[i] operation  has O(1) Complexity
            pixVal += threshold_values[i] * (int) exp2f(i);
        }

//        Final image pixel value
        out_img[ty * width + tx] = (float) pixVal / 255;
//        __syncthreads();

// Compute histogram
        atomicAdd(&(histogram_bins_sm[(unsigned int) pixVal]), 1);
        __syncthreads();

//        When all threads of the block have written o histogram bin sm (shared mem) the histogram is stored in global mem
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            for (unsigned int binIdx = 0; binIdx < num_bins; binIdx++) {
                atomicAdd(&(histogram_bins[binIdx]), histogram_bins_sm[binIdx]);
            }
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

//    int n_histogram_bins = 256; // pixel values from 0 to 255
    unsigned int histogram_bins[n_histogram_bins];
    for(int i = 0; i<n_histogram_bins; i++)
        histogram_bins[i] = 0;

    std::string colour[5] = { "sample_1920_1280", "computer_programming", "post_2", "borabora_1", "leopard" };
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

//    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
//    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);

    dim3 dimGrid(ceil((float) imageWidth / TILE_WIDTH), ceil((float) imageHeight / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    printf("dimGrid {%d, %d, %d}, dimBlock: {%d, %d, %d}\n", dimGrid.x, dimGrid.y , dimGrid.z, dimBlock.x, dimBlock.y , dimBlock.z);
//    LBPkernel<<<dimGrid, dimBlock, n_histogram_bins * sizeof(unsigned int)>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight, deviceHistogram, n_histogram_bins);
    LBPkernelTiling<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight, deviceHistogram, n_histogram_bins);
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
//    assert (sum == imageHeight * imageWidth);
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
