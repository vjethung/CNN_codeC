#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

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

void readDataFromFile(const char* filename, float** input, int* size) {
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

void saveOutputToFile(const char* filename, float* output, int batch_size, int num_nodes) {
    FILE* file = fopen(filename, "w");
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < num_nodes; i++) {
            fprintf(file, "%f\n", output[b * num_nodes + i]);
        }
    }
    fclose(file);
}

typedef enum {
    RELU,
    SOFTMAX
} ActivationType;

typedef struct {
    float* weight;
    float* bias;
} layer_params;

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

            // float max_val = current_output[0];
            // for (int i = 1; i < num_NodeThisLayer; i++) {
            //     if (current_output[i] > max_val) max_val = current_output[i];
            // }
            
            // float sum_exp = 0.0;
            // for (int i = 0; i < num_NodeThisLayer; i++) {
            //     sum_exp += exp(current_output[i] - max_val);
            // }
            
            // for (int i = 0; i < num_NodeThisLayer; i++) {
            //     current_output[i] = exp(current_output[i] - max_val) / sum_exp;
            // }
        }
    }
}



int main() {
    int num_NodePreLayer = 64;
    int num_NodeThisLayer = 10;
    int batch_size = 2;

    float* weights = NULL;
    float* biases = NULL;
    float* input = NULL;
    int weight_size, bias_size, input_size;

    readDataFromFile("weight_full.txt", &weights, &weight_size);
    readDataFromFile("bias_full.txt", &biases, &bias_size);
    readDataFromFile("input_full.txt", &input, &input_size);

    layer_params params = {weights, biases};
    float* output = (float*)malloc(batch_size * num_NodeThisLayer * sizeof(float));

    // Chọn activation là SOFTMAX (hoặc có thể thay bằng RELU)
    fullyConnected_forward(num_NodePreLayer, num_NodeThisLayer, batch_size, params, true, input, output, SOFTMAX);
    saveOutputToFile("output_full.txt", output, batch_size, num_NodeThisLayer);

    // float* fc_output = (float*)malloc(batch_size * num_NodeThisLayer * sizeof(float));
    // float* softmax_output = (float*)malloc(batch_size * num_NodeThisLayer * sizeof(float));

    // fullyConnected_forward(num_NodePreLayer, num_NodeThisLayer, batch_size, params, true, input, fc_output);
    // softmax(fc_output, batch_size, num_NodeThisLayer, softmax_output);
    // saveOutputToFile("output_full.txt", softmax_output, batch_size, num_NodeThisLayer);

    free(weights);
    free(biases);
    free(input);
    free(output);

    // free(fc_output);
    // free(softmax_output);

    return 0;
}