#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>

float relu(float x);
void softmax(float* input, int batch_size, int num_nodes, float* output);

//conv2d va maxpooling
typedef struct {
    int batch_size;
    int channels;
    int height;
    int width;
} Feature_map_shape;

typedef struct {
    int size[2];
} Kernel_shape;

typedef struct {
    float* weight;
    float* bias;
} Params;

typedef struct {
    Feature_map_shape in_shape, out_shape;
    Kernel_shape kernel_sh;
    Params params;
    int stride[2];
    int padding[2];
    bool use_bias;
} Layer;

void Conv2d_forward(Layer* conv, Feature_map_shape in_shape, int output_channels, Kernel_shape kernel_sh, Params params, int stride[2], int padding[2], bool use_bias, float* input, float* output);
// Maxpooling function
void Maxpool2d(Layer* maxpool, Feature_map_shape in_shape, Kernel_shape kernel_sh, int stride[2], int padding[2], float* input, float* output);
Feature_map_shape Layer_get_output_shape(const Layer* layer);

//fullyConnected_forward
typedef enum {
    RELU,
    SOFTMAX
} ActivationType;

typedef struct {
    float* weight;
    float* bias;
} layer_params;

void fullyConnected_forward(int num_NodePreLayer, int num_NodeThisLayer, int batch_size, layer_params params, bool use_bias, float* input, float* output, ActivationType activation);
void readData_Conv2D_FromFile(const char *filename, float** input);
void saveOutput_Conv2D_ToFile(const char *filename, const float *output, int batch_size, int output_channels, int height, int width);
void readData_FullConnect_FromFile(const char* filename, float** input, int* size);
void saveOutput_FullConnect_ToFile(const char* filename, float* output, int batch_size, int num_nodes);