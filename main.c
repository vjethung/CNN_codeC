#include "maxpooling.h"
#include <stdio.h>

int main() {
    int input[4][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
    };
    
    int output[2][2];
    maxPooling((int*)input, (int*)output, 4, 4, 2, 2);
    
    printf("Input:\n");
    printMatrix((int*)input, 4);
    printf("Output:\n");
    printMatrix((int*)output, 2);
    
    return 0;
}
