/*
 * Forward.c
 *
 *  Created on: Mar 30, 2025
 *      Author: TGPM
 */

#include"Forward.h"
void z_score_normalization(float* input, conv2d_shape in_shape, float epsilon,float gamma, float beta) {
    int size = in_shape.height * in_shape.width;
    for (int ic = 0; ic < in_shape.channels; ic++) {

        float mean = 0.0, std_dev = 0.0, sum = 0;

        for (int in_h = 0; in_h < in_shape.height; in_h++) {
            for (int in_w = 0; in_w < in_shape.width; in_w++) {
                int input_idx = ic * in_shape.height * in_shape.width + in_h * in_shape.width + in_w;
                sum += input[input_idx];
            }
        }

        mean = sum / size;
        for (int in_h = 0; in_h < in_shape.height; in_h++) {
            for (int in_w = 0; in_w < in_shape.width; in_w++) {
                int input_idx = ic * in_shape.height * in_shape.width + in_h * in_shape.width + in_w;
                std_dev += (input[input_idx] - mean) * (input[input_idx] - mean);
            }
        }
        for (int in_h = 0; in_h < in_shape.height; in_h++) {
            for (int in_w = 0; in_w < in_shape.width; in_w++) {
                int input_idx = ic * in_shape.height * in_shape.width + in_h * in_shape.width + in_w;
                input[input_idx] = gamma * (((input[input_idx] - mean) / (sqrt((std_dev / size) + epsilon)))) + beta; 
            }
        }

    }
}

void relu(float* input, conv2d_shape in_shape) {
    for (int i = 0; i < in_shape.channels * in_shape.height * in_shape.width; i++) {
        input[i] = fmax(0.0, input[i]);
    }
}

struct pool_shape {
    int height;
    int width;
    int depth;
};

void maxPooling2D(float* input, float* output, conv2d_shape in_shape, int* stride, pool_shape pool) {
    int out_height = ceil((float)(in_shape.height - pool.height) / stride[0]) + 1;
    int out_width = ceil((float)(in_shape.width - pool.width) / stride[1]) + 1;
    int out_depth = in_shape.channels;

    for (int c = 0; c < out_depth; c++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                int in_start_h = h * stride[0];
                int in_start_w = w * stride[1];
                int max_h = fmin(in_start_h + pool.height, in_shape.height);
                int max_w = fmin(in_start_w + pool.width, in_shape.width);
                float max_val = input[c * in_shape.height * in_shape.width + in_start_h * in_shape.width + in_start_w];
                for (int ph = in_start_h; ph < max_h; ph++) {
                    for (int pw = in_start_w; pw < max_w; pw++) {
                        int input_idx = c * in_shape.height * in_shape.width + ph * in_shape.width + pw;
                        max_val = fmax(max_val, input[input_idx]);
                    }
                }
                int output_idx = c * out_height * out_width + h * out_width + w;
                output[output_idx] = max_val;
            }
        }
    }
}

