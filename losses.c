#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-9  // Tránh log(0)

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


// Hàm tính loss cho một batch từ mảng 1 chiều
float batch_sparse_categorical_crossentropy(float* predicted_output, int* true_output, int batch_size, int num_classes) {
    float total_loss = 0.0;

    for (int i = 0; i < batch_size; i++) {
        int true_class = true_output[i];  // Lớp đúng của mẫu i
        int index = i * num_classes + true_class;  // Tính vị trí trong mảng 1D
        float prob = predicted_output[index];  // Lấy xác suất của lớp đúng
        total_loss += -log(prob + EPSILON);  // Tính loss
    }

    return total_loss / batch_size;  // Trung bình loss
}

int main() {
    int batch_size = 2;
    int num_classes = 10;
    int true_output[2] = {3, 5};  // Ví dụ nhãn thực tế của 2 mẫu

    // Đọc dữ liệu dự đoán từ file
    float* predicted_output = NULL;
    int size = 0;
    readDataFromFile("output_full.txt", &predicted_output, &size);
    
    // Kiểm tra nếu số lượng giá trị trong file không khớp với batch_size * num_classes
    if (size != batch_size * num_classes) {
        printf("Error: File data size does not match batch_size * num_classes\n");
        free(predicted_output);
        return -1;
    }

    // Tính loss
    float loss = batch_sparse_categorical_crossentropy(predicted_output, true_output, batch_size, num_classes);
    printf("Batch Loss: %f\n", loss);

    // Giải phóng bộ nhớ
    free(predicted_output);

    return 0;
}
