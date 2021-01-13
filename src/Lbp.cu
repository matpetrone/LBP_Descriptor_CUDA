#include<iostream>
#include<cmath>

__constant__ int MASK_WIDTH = 3;

__global__ void lbpKernel(int* grayImage, int* outputImage, int width, int height){
//    Naive cuda kernel to compute LBP descriptor
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height){ // Ensure that threads do not attempt illegal memory access (this can happen because there could be more threads than elements in an array)
        int pixVal = 0;
        int threshold_values[MASK_WIDTH * MASK_WIDTH - 1];
        int N_start_col = col - (MASK_WIDTH / 2);
        int N_start_row = row - (MASK_WIDTH / 2);
        int center_idx = int(floor(MASK_WIDTH / 2));
        int arr_idx = 0;
        for (int j = 0; j < MASK_WIDTH; j++){
            for (int k = 0; k < MASK_WIDTH; k++){
                int curRow = N_start_row + j;
                int curCol = N_start_col + k;
                // Verify we have a valid image pixel
                if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    if (curRow != row && curCol != col){
                        if (grayImage[curRow * width + curCol] >= grayImage[row * width + col])
                            threshold_values[arr_idx++] = 1;
                        else
                            threshold_values[arr_idx++] = 0;
                    }
                }
            }
        }
        for (int i = 0; i < sizeof(threshold_values)/sizeof(threshold_values[0]); i++){
            if (threshold_values[i] == 1)
                pixVal += 2**i;
        }
        outputImage[row * width + col] = pixVal;
    }


}