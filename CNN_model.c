#include"CNN_model.h"

float relu(float x) {
    return (x > 0) ? x : 0;
}

// Hàm softmax
void softmax(float* input, int batch_size, int num_nodes, float* output) {
    for (int b = 0; b < batch_size; b++) {
        float* current_input = input + b * num_nodes;
        float* current_output = output + b * num_nodes;
        
        // Tính tổng lũy thừa e^z_i
        float sum_exp = 0.0;
        for (int i = 0; i < num_nodes; i++) {
            sum_exp += exp(current_input[i]);
        }
        
        // Tính softmax cho từng phần tử
        for (int i = 0; i < num_nodes; i++) {
            current_output[i] = exp(current_input[i]) / sum_exp;
        }
    }
}

//conv2d va maxpooling

// Forward function
void Conv2d_forward(Layer* conv, Feature_map_shape in_shape, int output_channels, Kernel_shape kernel_sh, Params params, int stride[2], int padding[2], bool use_bias, float* input, float* output) {
    conv->in_shape = in_shape;
    conv->out_shape.channels = output_channels;
    conv->kernel_sh = kernel_sh;
    conv->params = params;
    conv->stride[0] = stride[0]; 
    conv->stride[1] = stride[1]; 
    conv->padding[0] = padding[0];
    conv->padding[1] = padding[1];
    conv->use_bias = use_bias;
    int out_height = (conv->in_shape.height + 2 * conv->padding[0] - conv->kernel_sh.size[0]) / conv->stride[0] + 1;
    int out_width = (conv->in_shape.width + 2 * conv->padding[1] - conv->kernel_sh.size[1]) / conv->stride[1] + 1;
    conv->out_shape.height = out_height;     
    conv->out_shape.width = out_width;
    conv->out_shape.batch_size = conv->in_shape.batch_size; 

    for (int b = 0; b < conv->in_shape.batch_size; b++) {
        for (int c = 0; c < conv->out_shape.channels; c++) {
            for (int h = 0; h < conv->out_shape.height; h++) {
                for (int w = 0; w < conv->out_shape.width; w++) {
                    float sum = 0;
                    for (int ic = 0; ic < conv->in_shape.channels; ic++) {
                        for (int kh = 0; kh < conv->kernel_sh.size[0]; kh++) {
                            for (int kw = 0; kw < conv->kernel_sh.size[1]; kw++) {
                                int in_h = h * conv->stride[0] + kh - conv->padding[0];
                                int in_w = w * conv->stride[1] + kw - conv->padding[1];
                                if (in_h >= 0 && in_h < conv->in_shape.height && in_w >= 0 && in_w < conv->in_shape.width) {
                                    int input_idx = b * conv->in_shape.channels * conv->in_shape.height * conv->in_shape.width + 
                                                   ic * conv->in_shape.height * conv->in_shape.width + 
                                                   in_h * conv->in_shape.width + in_w;
                                    int weight_idx = c * conv->in_shape.channels * conv->kernel_sh.size[0] * conv->kernel_sh.size[1] + 
                                                    ic * conv->kernel_sh.size[0] * conv->kernel_sh.size[1] + 
                                                    kh * conv->kernel_sh.size[1] + kw;
                                    sum += input[input_idx] * conv->params.weight[weight_idx];
                                }
                            }
                        }
                    }
                    int output_idx = b * conv->out_shape.channels * conv->out_shape.height * conv->out_shape.width + 
                                    c * conv->out_shape.height * conv->out_shape.width + 
                                    h * conv->out_shape.width + w;
                    // output[output_idx] = sum + (conv->use_bias ? conv->params.bias[c] : 0);
                    output[output_idx] = relu(sum + (conv->use_bias ? conv->params.bias[c] : 0));
                }   
            }
        }
    }
}

// Maxpooling function
void Maxpool2d(Layer* maxpool, Feature_map_shape in_shape, Kernel_shape kernel_sh, int stride[2], int padding[2], float* input, float* output) {
    maxpool->in_shape = in_shape;
    maxpool->kernel_sh = kernel_sh;
    maxpool->padding[0] = padding[0];
    maxpool->padding[1] = padding[1];
    maxpool->stride[0] = stride[0]; 
    maxpool->stride[1] = stride[1];

    int out_height = (maxpool->in_shape.height + 2 * maxpool->padding[0] - maxpool->kernel_sh.size[0]) / maxpool->stride[0] + 1;
    int out_width = (maxpool->in_shape.width + 2 * maxpool->padding[1] - maxpool->kernel_sh.size[1]) / maxpool->stride[1] + 1;
    maxpool->out_shape.batch_size = maxpool->in_shape.batch_size;
    maxpool->out_shape.height = out_height;     
    maxpool->out_shape.width = out_width;
    maxpool->out_shape.channels = in_shape.channels;

    for (int b = 0; b < maxpool->in_shape.batch_size; b++) {
        for (int ic = 0; ic < maxpool->in_shape.channels; ic++) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -FLT_MAX;
                    for (int kh = 0; kh < kernel_sh.size[0]; ++kh) {
                        for (int kw = 0; kw < kernel_sh.size[1]; ++kw) {
                            int ih = oh * stride[0] + kh - padding[0];
                            int iw = ow * stride[1] + kw - padding[1];
                            if (ih >= 0 && ih < maxpool->in_shape.height && iw >= 0 && iw < maxpool->in_shape.width) {
                                int input_idx = b * maxpool->in_shape.channels * maxpool->in_shape.height * maxpool->in_shape.width + 
                                                ic * maxpool->in_shape.height * maxpool->in_shape.width  + ih * maxpool->in_shape.width + iw;
                                float val = input[input_idx];
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    int output_idx = b * maxpool->in_shape.channels * out_height * out_width + 
                                    ic * out_height * out_width + 
                                    oh * out_width + ow;
                    output[output_idx] = max_val;
                }
            }
        }
    }
}

Feature_map_shape Layer_get_output_shape(const Layer* layer) { 
    return layer->out_shape; 
}

//fullyConnected_forward
void fullyConnected_forward(int num_NodePreLayer, int num_NodeThisLayer, int batch_size, layer_params params, bool use_bias, float* input, float* output, ActivationType activation) {
    for (int b = 0; b < batch_size; b++) {
        float* current_input = input + b * num_NodePreLayer;
        float* current_output = output + b * num_NodeThisLayer;
        
        // Tính  linear output
        for (int i = 0; i < num_NodeThisLayer; i++) {
            current_output[i] = use_bias ? params.bias[i] : 0;
            for (int j = 0; j < num_NodePreLayer; j++) {
                int weight_idx = i * num_NodePreLayer + j;
                current_output[i] += current_input[j] * params.weight[weight_idx];
            }
        }
        // Áp dụng activation
        if (activation == RELU) {
            for (int i = 0; i < num_NodeThisLayer; i++) {
                current_output[i] = relu(current_output[i]);
            }
        } else if (activation == SOFTMAX) {
            // Tính softmax cho batch hiện tại
            float sum_exp = 0.0;
            for (int i = 0; i < num_NodeThisLayer; i++) {
                sum_exp += exp(current_output[i]);
            }
        
            // Tính softmax cho từng phần tử
            for (int i = 0; i < num_NodeThisLayer; i++) {
                current_output[i] = exp(current_output[i]) / sum_exp;
            }
        }
    }
}

void readData_Conv2D_FromFile(const char *filename, float** input) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return;
    }
    
    int size = 0;
    float value;
    while (fscanf(file, "%f", &value) == 1) {
        size++;
    }
    
    rewind(file);
    
    *input = (float*)malloc(size * sizeof(float));
    
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%f", &(*input)[i]) != 1) {
            break;
        }
    }
    
    fclose(file);
}

void saveOutput_Conv2D_ToFile(const char *filename, const float *output, int batch_size, int output_channels, int height, int width) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Failed to open file for writing: %s\n", filename);
        return;
    }
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < output_channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    fprintf(file, "%f\n", output[b * output_channels * height * width 
                                                + c * height * width 
                                                + h * width 
                                                + w]);
                }
            }
        }
    }
    
    fclose(file);
    printf("Done %s\n", filename);
}

void readData_FullConnect_FromFile(const char* filename, float** input, int* size) {
    FILE* file = fopen(filename, "r");
    float* temp = NULL;
    int count = 0;
    float value;
    while (fscanf(file, "%f", &value) == 1) {
        temp = (float*)realloc(temp, (count + 1) * sizeof(float));
        temp[count++] = value;
    }
    fclose(file);
    *input = temp;
    *size = count;
}

void saveOutput_FullConnect_ToFile(const char* filename, float* output, int batch_size, int num_nodes) {
    FILE* file = fopen(filename, "w");
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < num_nodes; i++) {
            fprintf(file, "%f\n", output[b * num_nodes + i]);
        }
    }
    fclose(file);
}