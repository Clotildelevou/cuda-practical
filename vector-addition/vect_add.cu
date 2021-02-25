#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"


const int N = 2000;

    __global__
void block_add(int *vec_a, int *vec_b, int *vec_c)
{
    printf("Gpu op\n");
    vec_c[blockIdx.x] = vec_a[blockIdx.x] + vec_b[blockIdx.x];
}

__global__
void thread_add(int *vec_a, int *vec_b, int *vec_c)
{
    vec_c[threadIdx.x] = vec_a[threadIdx.x] + vec_b[threadIdx.x];
}

void random_ints(int* a, int N)
{
   int i;
   for (i = 0; i < N; ++i)
    a[i] = rand();
}

int main(void)
{

    size_t size = N * sizeof(int);

    //CPU copy of matrix
    int *cpu_a = (int *) malloc(size);
    int *cpu_b = (int *) malloc(size);
    int *cpu_c = (int *) malloc(size);

    //GPU copy of matrix
    int *gpu_a;
    int *gpu_b;
    int *gpu_c;

    cudaMalloc((void **) &gpu_a, size);
    cudaMalloc((void **) &gpu_b, size);
    cudaMalloc((void **) &gpu_c, size);

    //Filling the matrixes
    random_ints(cpu_a, N);
    random_ints(cpu_b, N);

    //Copy cpu to gpu device
    cudaMemcpy(gpu_a, cpu_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, cpu_b, size, cudaMemcpyHostToDevice);

    //Launch block_add()
    block_add<<<N, 1>>>(gpu_a, gpu_b, gpu_c);
   
    //Catches an eventual error
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
    
    //Copy result back to host
    cudaMemcpy(cpu_c, gpu_c, size, cudaMemcpyDeviceToHost);

    //Checks if everything is ok
    for (int i = 0; i < N; i++)
    {
        if ((i % 40) == 0)
            printf("\n");
        printf("%d ", cpu_c[i]);
    }
    printf("\n");

    //Launch thread_add()
    block_add<<<1, N>>>(gpu_a, gpu_b, gpu_c);
   
    //Catches an eventual error
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
    
    //Copy result back to host
    cudaMemcpy(cpu_c, gpu_c, size, cudaMemcpyDeviceToHost);

    //Checks if everything is ok
    for (int i = 0; i < N; i++)
    {
        if ((i % 40) == 0)
            printf("\n");
        printf("%d ", cpu_c[i]);
    }
    printf("\n");

    //Cleanup
    free(cpu_a);
    free(cpu_b);
    free(cpu_c);
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    return 0;

}
