#include "maxpooling.h"
#include <stdio.h>

void maxPooling(int *input, int *output, int height, int width, int filterSize, int strideSize) {
    for (int h = 0; h < height; h += strideSize) {
        for (int w = 0; w < width; w += strideSize) {
            int maxValue = input[h * width + w];
            for (int fh = 0; fh < filterSize; fh++) {
                for (int fw = 0; fw < filterSize; fw++) {
                    if (h + fh < height && w + fw < width) {
                        int currentValue = input[(h + fh) * width + (w + fw)];
                        if (currentValue > maxValue) {
                            maxValue = currentValue;
                        }
                    }
                }
            }
            int outHeight = height / filterSize;
            int outWidth = width / filterSize;
            int outH = h / filterSize;
            int outW = w / filterSize;
            output[outH * outWidth + outW] = maxValue;
        }
    }
}

float calculateRelu(float value) {
    return value < 0 ? 0 : value;
}

void printMatrix(int *mat, int size) {
    for (int h = 0; h < size; ++h) {
        for (int w = 0; w < size; ++w) {
            printf("%d ", mat[h * size + w]);
        }
        printf("\n");
    }
}
