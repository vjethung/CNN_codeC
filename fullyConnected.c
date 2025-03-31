#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

void readDataFromFile(const char* filename, float** input, int* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }

    int count = 0;
    float value;
    float* temp = NULL;
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
    if (!file) {
        printf("Error: Could not open file %s for writing!\n", filename);
        return;
    }

    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < num_nodes; i++) {
            fprintf(file, "%f\n", output[b * num_nodes + i]);
        }
    }
    fclose(file);
}

typedef struct {
    float* weight;
    float* bias;
} layer_params;

typedef struct {
    int num_NodePreLayer;
    int num_NodeThisLayer;
    int batch_size;
    layer_params params;
    bool use_bias;
    float* last_input;
    /* last_input lưu trữ input của lớp fullyConnected để sử dụng 
    trong quá trình lan truyền ngược (backpropagation). */
} fullyConnected;

void fullyConnected_init(fullyConnected* fc, int num_NodePreLayer, int num_NodeThisLayer, int batch_size, layer_params params, bool use_bias) {
    fc->num_NodePreLayer = num_NodePreLayer;
    fc->num_NodeThisLayer = num_NodeThisLayer;
    fc->batch_size = batch_size;
    fc->params = params;
    fc->use_bias = use_bias;
    fc->last_input = (float*)malloc(batch_size * num_NodePreLayer * sizeof(float));
    // Cấp phát để lưu lại dữ liệu input của lớp fullyConnected
}

void fullyConnected_forward(fullyConnected* fc, float* input, float* output) {
    for (int b = 0; b < fc->batch_size; b++) 
    {
        float* current_input = input + b * fc->num_NodePreLayer;
        float* current_output = output + b * fc->num_NodeThisLayer;
        
        // copy input vào last_input để sử dụng trong backpropagation
        for (int j = 0; j < fc->num_NodePreLayer; j++) 
        {
            fc->last_input[b * fc->num_NodePreLayer + j] = current_input[j];
        }
        
        for (int i = 0; i < fc->num_NodeThisLayer; i++) 
        {
            current_output[i] = fc->use_bias ? fc->params.bias[i] : 0;
            for (int j = 0; j < fc->num_NodePreLayer; j++) 
            {
                int weight_idx = i * fc->num_NodePreLayer + j;
                current_output[i] += current_input[j] * fc->params.weight[weight_idx];
            }
        }
    }
}

void fullyConnected_backward(fullyConnected* fc, float* output_gradient, float* input_gradient, float learning_rate) {
    // output_gradient: dL/dY (kích thước: batch_size * num_NodeThisLayer)
    // input_gradient: dL/dX (kích thước: batch_size * num_NodePreLayer)
        
    // Mảng tạm để lưu gradient của trọng số (dL/dW)
    float* weights_gradient = (float*)calloc(fc->num_NodeThisLayer * fc->num_NodePreLayer, sizeof(float));
    
    for (int b = 0; b < fc->batch_size; b++) 
    {
        // di chuyển con trỏ đến vị trí bắt đầu của output_gradient, last_input, input_gradient của batch hiện tại
        float* current_output_grad = output_gradient + (b * fc->num_NodeThisLayer);
        float* current_input = fc->last_input + (b * fc->num_NodePreLayer);
        float* current_input_grad = input_gradient + (b * fc->num_NodePreLayer);
        
        // weights_gradient (dL/dW) = dL/dy * X^T
        for (int i = 0; i < fc->num_NodeThisLayer; i++) 
        {
            for (int j = 0; j < fc->num_NodePreLayer; j++) 
            {
                int weight_idx = i * fc->num_NodePreLayer + j;
                if (b == 0) {
                    weights_gradient[weight_idx] = current_output_grad[i] * current_input[j];
                } else {
                    weights_gradient[weight_idx] += current_output_grad[i] * current_input[j];
                }
            }
        }
        // input_gradient (dL/dX) = W^T * dL/dy
        for (int j = 0; j < fc->num_NodePreLayer; j++) 
        {
            current_input_grad[j] = 0;
            for (int i = 0; i < fc->num_NodeThisLayer; i++) 
            {
                int weight_idx = i * fc->num_NodePreLayer + j;
                current_input_grad[j] += fc->params.weight[weight_idx] * current_output_grad[i];
            }
        }
        // cập nhật bias
        if (fc->use_bias && b == 0) // Chỉ cập nhật một lần với batch đầu tiên
        {
            for (int i = 0; i < fc->num_NodeThisLayer; i++) 
            {
                fc->params.bias[i] -= learning_rate * current_output_grad[i];
            }
        }
    }
    // cập nhật weight
    for (int i = 0; i < fc->num_NodeThisLayer * fc->num_NodePreLayer; i++) {
        fc->params.weight[i] -= learning_rate * weights_gradient[i];// W = W - a * dL/dW
    }
    
    free(weights_gradient);
}

void fullyConnected_free(fullyConnected *fc) {
    free(fc->last_input);
}



int main() {
    int num_NodePreLayer = 64;  // Đầu vào từ Dense(64)
    int num_NodeThisLayer = 10; // Đầu ra của lớp cuối (10 lớp CIFAR-10)
    int batch_size = 2;         // 2 mẫu trong file input.txt

    float* weights = NULL;
    float* biases = NULL;
    float* input = NULL;
    int weight_size, bias_size, input_size;

    readDataFromFile("weight_full.txt", &weights, &weight_size);
    readDataFromFile("bias_full.txt", &biases, &bias_size);
    readDataFromFile("input_full.txt", &input, &input_size);

    if (!weights || !biases || !input) {
        printf("Error: Failed to read one or more files!\n");
        return 1;
    }

    layer_params params = {weights, biases};
    float* output = (float*)malloc(batch_size * num_NodeThisLayer * sizeof(float));
    fullyConnected fc;
    fullyConnected_init(&fc, num_NodePreLayer, num_NodeThisLayer, batch_size, params, true);

    fullyConnected_forward(&fc, input, output);

    printf("Output:\n");
    for (int b = 0; b < batch_size; b++) {
        printf("Batch %d: ", b);
        for (int i = 0; i < num_NodeThisLayer; i++) {
            printf("%f ", output[b * num_NodeThisLayer + i]);
        }
        printf("\n");
    }

    saveOutputToFile("output_full.txt", output, batch_size, num_NodeThisLayer);
    
    free(weights);
    free(biases);
    free(input);
    free(output);
    fullyConnected_free(&fc);

    return 0;
}
