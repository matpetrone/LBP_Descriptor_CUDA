////#include<iostream>
////#include<cmath>
//#include "../Image.h"
//#include "../PPM.h"
//
//#include <cstdio>
//#include <cassert>
//#include <iostream>
//#include <math.h>
//
//// Constant values for LBP kerel
//#define MASK_WIDTH 3
//#define neighborhood  (MASK_WIDTH * MASK_WIDTH - 1)
//#define img_deep 255
//
//
//
//static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
//
//#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
//
///**
// * Check the return value of the CUDA runtime API call and exit
// * the application if the call has failed.
// */
//static void CheckCudaErrorAux(const char *file, unsigned line,
//                              const char *statement, cudaError_t err) {
//    if (err == cudaSuccess)
//        return;
//    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
//              << err << ") at " << file << ":" << line << std::endl;
//    exit(1);
//}
//
//__global__ void LBPkernel(float *img, float *out_img, int width, int height){
//    //    Naive cuda kernel to compute LBP descriptor
////    !! the pixel values are between 0 and 1
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    printf("Hello from lbp k, col_idx: %d, row_idx: %d\n", col, row);
//
//    // Ensure that threads do not attempt illegal memory access (this can happen because there could be more threads than elements in an array)
//    if (col < width && row < height){
//        int pixVal = 0;
//        int threshold_values[neighborhood];
////        std::vector<int> threshold_values;
//        int N_start_col = col - (MASK_WIDTH / 2);
//        int N_start_row = row - (MASK_WIDTH / 2);
//        int arr_idx = 0;
////        iterate over mask pixel values
//        for (int j = 0; j < MASK_WIDTH; j++){
//            for (int k = 0; k < MASK_WIDTH; k++){
//                int curRow = N_start_row + j;
//                int curCol = N_start_col + k;
//
//                // Verify we have a valid image pixel
//                if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
//                    printf("pixel value at row: %d, col: %d : %d\n",curRow, curCol, (int)img[curRow * width + curCol] );
//                    if (curRow != row && curCol != col){  // no compute for mask center
////                        threshold_values.push_back(((int)img[curRow * width + curCol] >= (int)img[row * width + col]) ? 1 : 0);
//                        threshold_values[arr_idx++] = (img[curRow * width + curCol] >= img[row * width + col]) ? 1 : 0;
//                    } else
//                        printf("central pixel value: %d \n", img[row * width + col]);
//                }
//            }
//        }
////        ---
////        for (auto th : threashold_values){
////            pixVal += th * (int)exp2f();
////        }
//
//        for (int i=0; i<neighborhood; i++){  // vec[i] operation  has O(1) Complexity
//            pixVal += threshold_values[i] * (int)exp2f(i);
//        }
//
//        printf("pixval: %d\n", pixVal);
//
//        out_img[row * width + col] = pixVal / 255;
//    }
//}
//
//// simple test to read/write PPM images, and process Image_t data
//void test_images() {
//    Image_t* inputImg = PPM_import("computer_programming.ppm");
//    for (int i = 0; i < 300; i++) {
////		Image_setPixel(inputImg, i, 100, 0, float(i) / 300);
//        Image_setPixel(inputImg, i, 100, 1, float(i) / 300);
////		Image_setPixel(inputImg, i, 100, 2, float(i) / 200);
//    }
//    PPM_export("test_output3.ppm", inputImg);
//    Image_t* newImg = PPM_import("test_output.ppm");
//    inputImg = PPM_import("computer_programming.ppm");
//    if (Image_is_same(inputImg, newImg))
//        printf("Img uguali\n");
//    else
//        printf("Img diverse\n");
//}
//
//
//
//
//// ------------------------------------
//
////__constant__ int MASK_WIDTH = 3;
//
//
//
////__global__ void lbpKernel(int* grayImage, int* outputImage, int width, int height){
//////    Naive cuda kernel to compute LBP descriptor
////    int col = blockIdx.x * blockDim.x + threadIdx.x;
////    int row = blockIdx.y * blockDim.y + threadIdx.y;
////    printf("Hello from lbp k, col_idx: %d, row_idx: %d", col, row);
////    if (col < width && row < height){ // Ensure that threads do not attempt illegal memory access (this can happen because there could be more threads than elements in an array)
////        int pix_val = 0;
////        int threshold_values[MASK_WIDTH * MASK_WIDTH - 1];
////        int N_start_col = col - (MASK_WIDTH / 2); // N_start_col = col - 1
////        int N_start_row = row - (MASK_WIDTH / 2);
////        int center_idx = int(floor(MASK_WIDTH / 2));  // index of the central pixel of the mask (in this case (1,1))
////        int arr_idx = 0;
//////        iterate over pixels in the mask
////        for (int j = 0; j < MASK_WIDTH; j++){
////            for (int k = 0; k < MASK_WIDTH; k++){
////                int cur_row = N_start_row + j;
////                int cur_col = N_start_col + k;
////                // Verify we have a valid image pixel
////                if(cur_row > -1 && cur_row < height && cur_col > -1 && cur_col < width) {
////                    if (cur_row != row && cur_col != col){  // dot compute for central pixel
////                        if (grayImage[cur_row * width + cur_col] >= grayImage[row * width + col])
////                            threshold_values[arr_idx++] = 1;
////                        else
////                            threshold_values[arr_idx++] = 0;
////                    }
////                }
////            }
////        }
////        for (int i = 0; i < sizeof(threshold_values)/sizeof(threshold_values[0]); i++){
////            if (threshold_values[i] == 1)
////                pix_val += 2 * *i;
////        }
////        outputImage[row * width + col] = pix_val;
////    }
////}
//
