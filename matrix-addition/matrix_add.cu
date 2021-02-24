#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

    __global__
void add_matrix(int *matrixA, int *matrixB, int *matrixC, int size)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size)
    {
        printf("Operating index : %d\n", index);
        matrixC[index] = matrixA[index] + matrixB[index];
    }
}

int main(int argc, char *argv[])
{

    if (argc != 2)
        return 1;
    size_t side = atoi(argv[1]);

    size_t size = side * side * sizeof(int);

    //CPU copy of matrix
    int *cpu_a = (int*) malloc(size);
    int *cpu_b = (int*) malloc(size);
    int *cpu_c = (int*) malloc(size);

    //GPU copy of matrix
    int *gpu_a;
    int *gpu_b;
    int *gpu_c;

    cudaMalloc((void**) &gpu_a, size);
    cudaMalloc((void**) &gpu_b, size);
    cudaMalloc((void**) &gpu_c, size);

    //Filling the matrixes
    for (int i = 0; i < pow(side, 2); i++)
    {
        cpu_a[i] = rand() % 100;
        cpu_b[i] = rand() % 100;
    }

    //Copy cpu to gpu device
    cudaMemcpy(gpu_a, cpu_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, cpu_b, size, cudaMemcpyHostToDevice);

    //Launch matrix_add()
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(size) / thr_per_blk);
    if (blk_in_grid == 0)
        blk_in_grid = 1;

    add_matrix<<<blk_in_grid, thr_per_blk>>>(gpu_a, gpu_b, gpu_c, side * side);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    cudaMemcpy(cpu_c, gpu_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < side; i++)
    {
        if (cpu_c[i] != cpu_a[i] + cpu_b[i])
        {
            fprintf(stderr, "Something went wrong...\n mat[%d] is %d and should be %d\n",
                    i, cpu_c[i], cpu_a[i] + cpu_b[i]);
            return 1;
        }
    }
    
    free(cpu_a);
    free(cpu_b);
    free(cpu_c);
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    printf("This program is a success !\n");
    return 0;

}
