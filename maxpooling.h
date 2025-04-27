#ifndef MAXPOOLING_H
#define MAXPOOLING_H

void maxPooling(int *input, int *output, int height, int width, int filterSize, int strideSize);
float calculateRelu(float value);
void printMatrix(int *mat, int size);

#endif // MAXPOOLING_H
