#include<iostream>
#include<cmath>

__global__ void lbp_kernel(cv::Mat grayImage, cv::Mat outputImage){
    int maskwidth = 3;
    int w = grayImage.cols;
    int h = grayImage.rows;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h){
        int pixVal = 0;
        int threshold_values[maskwidth*maskwidth-1];
        N_start_col = col - (maskwidth/2);
        N_start_row = row - (maskwidth/2);
        int center_idx = int(floor(maskwidth/2));
        int arr_idx = 0;
        for (int j = 0; j < maskwidth; j++){
            for (int k = 0; k < maskwidth; k++){
                int curRow = N_Start_row + j;
                int curCol = N_start_col + k;
                // Verify we have a valid image pixel
                if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    if (curRow != row && curCol != col){
                        if (grayImage[curRow * w + curCol] >= grayImage[row * w + col])
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
        outputImage[row * w + col] = pixVal;
    }


}