/*
 * Forward.c
 *
 *  Created on: Mar 30, 2025
 *      Author: TGPM
 */

#include"Forward.h"

void forward(float* input, float* output,struct conv2d_shape in_shape, int output_channels,struct kernel_shape kernel_sh,struct conv2d_params params, int stride[2], int padding[2]){
        // Output shape
	struct conv2d_shape out_shape;
        int out_height = (in_shape.height + 2 * padding[0] - kernel_sh.size[0]) / stride[0] + 1;
        int out_width = (in_shape.width + 2 * padding[1] - kernel_sh.size[1]) / stride[1] + 1;
        out_shape.height = out_height;
        out_shape.width = out_width;
        out_shape.batch_size = in_shape.batch_size;

        for (int b = 0; b < in_shape.batch_size; b++) {
            for (int c = 0; c < out_shape.channels; c++){
                for (int h = 0; h < out_shape.height; h++){
                    for (int w = 0; w < out_shape.width; w++){
                        float sum = 0;
                        for (int ic = 0; ic < in_shape.channels; ic++){
                            for (int kh = 0; kh < kernel_sh.size[0]; kh++){
                                for (int kw = 0; kw < kernel_sh.size[1]; kw++){
                                    int in_h = h * stride[0] + kh - padding[0];
                                    int in_w = w * stride[1] + kw - padding[1];
                                    if (in_h >= 0 && in_h < in_shape.height && in_w >= 0 && in_w < in_shape.width){
                                        // Khai báo chỉ số input
                                        int input_idx = b*in_shape.channels*in_shape.height*in_shape.width + ic*in_shape.height*in_shape.width + in_h*in_shape.width + in_w;
                                        // Khai báo chỉ số weight
                                        int weight_idx = c*in_shape.channels*kernel_sh.size[0]*kernel_sh.size[1] + ic*kernel_sh.size[0]*kernel_sh.size[1] + kh*kernel_sh.size[1] + kw;
                                        // Tính tổng tích
                                        sum +=  input[input_idx] * params.weight[weight_idx];
                                    }
                                }
                            }
                        }
                        int output_idx = b*out_shape.channels*out_shape.height*out_shape.width + c*out_shape.height*out_shape.width + h*out_shape.width + w;
                        output[output_idx] = sum + params.bias[c];
                }
            }
        }
    }
 }

void z_score_normalization(float input[],struct conv2d_shape in_shape, float epsilon) {
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
            for (int in_w = 0; in_w < in_shape.height; in_w++) {
                int input_idx = ic * in_shape.height * in_shape.width + in_h * in_shape.width + in_w;
                std_dev += (input[input_idx] - mean) * (input[input_idx] - mean);
            }
        }
        std_dev = sqrt(std_dev / size);

        for (int in_h = 0; in_h < in_shape.height; in_h++) {
            for (int in_w = 0; in_w < in_shape.height; in_w++) {
                int input_idx = ic * in_shape.height * in_shape.width + in_h * in_shape.width + in_w;
                input[input_idx] = (input[input_idx] - mean) / (std_dev + epsilon);
            }
        }

    }
}

void relu(float input[],struct conv2d_shape in_shape) {
    for (int i = 0; i < in_shape.channels * in_shape.height * in_shape.width; i++) {
        input[i] = fmax(0.0, input[i]);
    }
}

void maxPooling2D(float input[], float output[],struct conv2d_shape in_shape, int stride[],struct pool_shape pool) {
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


