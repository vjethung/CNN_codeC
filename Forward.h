/*
 * Forward.h
 *
 *  Created on: Mar 30, 2025
 *      Author: TGPM
 */
#include <math.h>

#ifndef SRC_FORWARD_H_
#define SRC_FORWARD_H_

struct conv2d_shape
{
    int batch_size; // Số lượng mẫu trong một batch
    int channels;   // Số kênh (ví dụ: 3 cho ảnh RGB)
    int height;     // Chiều cao của tensor
    int width;      // Chiều rộng của tensor
};

struct kernel_shape
{
    int size[2]; // size[0]: Chiều cao, size[1]: chiều rộng kernel
};

struct conv2d_params
{
    float* weight;  //Con trỏ tới mảng trọng số của kernel
    float* bias;
};
struct pool_shape {
    int height;
    int width;
    int depth;
};

void forward(float* input, float* output,struct conv2d_shape in_shape, int output_channels,struct  kernel_shape kernel_sh,struct  conv2d_params params, int stride[2], int padding[2]);
void relu(float input[],struct conv2d_shape in_shape) ;
void maxPooling2D(float input[], float output[],struct  conv2d_shape in_shape, int stride[],struct pool_shape pool);
#endif /* SRC_FORWARD_H_ */
