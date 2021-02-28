#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

#define THREADS_PER_BLOCK 32

__global__ void block_add(int *vec_a, int *vec_b, int *vec_c)
{
    vec_c[blockIdx.x] = vec_a[blockIdx.x] + vec_b[blockIdx.x];
}

__global__ void thread_add(int *vec_a, int *vec_b, int *vec_c)
{
    vec_c[threadIdx.x] = vec_a[threadIdx.x] + vec_b[threadIdx.x];
}

__global__ void thread_and_block_add(int *vec_a, int *vec_b, int *vec_c)
{
    int index = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    vec_c[index] = vec_a[index] + vec_b[index];
}

void random_ints(int *a, int N)
{
    int i;
    for (i = 0; i < N; ++i)
        a[i] = rand() % 100;
}

int main(int argc, char *argv[])
{
    /*INIT*/

    if (argc != 2 || argv[1] == "")
    {
        fprintf(stderr,
                "vect_add usage: ./out mat_size with mat_size <= 1024\n");
        return 1;
    }

    size_t N = atoi(argv[1]);

    // Can't have more than 1024 threads
    if (N > 1024)
    {
        fprintf(stderr, "mat_size must be < 1024\n");
        return 1;
    }

    size_t size = N * sizeof(int);
    bool success = true;

    // CPU copy of matrix
    int *cpu_a = (int *)malloc(size);
    int *cpu_b = (int *)malloc(size);
    int *cpu_c = (int *)malloc(size);

    // GPU copy of matrix
    int *gpu_a;
    int *gpu_b;
    int *gpu_c;

    cudaMalloc((void **)&gpu_a, size);
    cudaMalloc((void **)&gpu_b, size);
    cudaMalloc((void **)&gpu_c, size);

    // Filling the matrixes
    random_ints(cpu_a, N);
    random_ints(cpu_b, N);

    // Copy cpu to gpu device
    cudaMemcpy(gpu_a, cpu_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, cpu_b, size, cudaMemcpyHostToDevice);

    // BLOCK ADDITION

    // Launch block_add()
    printf("\nLaunch block addition\n");
    block_add<<<N, 1>>>(gpu_a, gpu_b, gpu_c);
    cudaDeviceSynchronize();

    // Catches an eventual error
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    // Copy result back to host
    cudaMemcpy(cpu_c, gpu_c, size, cudaMemcpyDeviceToHost);

    // Checks if everything is ok
    for (int i = 0; i < N; i++)
    {
        if (cpu_c[i] != cpu_a[i] + cpu_b[i])
        {
            fprintf(stderr, "Block addition did not work !\n");
            success = false;
            break;
        }
    }

    if (success == true)
        printf("Success !\n");

    success = true;

    /* THREAD ADDITIONS */

    // Filling the matrixes
    random_ints(cpu_a, N);
    random_ints(cpu_b, N);

    // Copy cpu to gpu device
    cudaMemcpy(gpu_a, cpu_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, cpu_b, size, cudaMemcpyHostToDevice);

    // Launch thread_add()
    printf("\nLaunch thread addition\n");
    thread_add<<<1, N>>>(gpu_a, gpu_b, gpu_c);
    cudaDeviceSynchronize();

    // Catches an eventual error
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    // Copy result back to host
    cudaMemcpy(cpu_c, gpu_c, size, cudaMemcpyDeviceToHost);

    // Checks if everything is ok
    for (int i = 0; i < N; i++)
    {
        if (cpu_c[i] != cpu_a[i] + cpu_b[i])
        {
            fprintf(stderr, "Block addition did not work !\n");
            success = false;
            break;
        }
    }
    if (success == true)
        printf("Success !\n");

    success = true;

    // THREAD AND BLOCK ADDITIONS

    // Filling the matrixes
    random_ints(cpu_a, N);
    random_ints(cpu_b, N);

    // Copy cpu to gpu device
    cudaMemcpy(gpu_a, cpu_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, cpu_b, size, cudaMemcpyHostToDevice);

    // Launch thread_and_block_add()
    printf("\nLaunch thread addition\n");
    thread_and_block_add<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        gpu_a, gpu_b, gpu_c);

    // Catches an eventual error
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    // Copy result back to host
    cudaMemcpy(cpu_c, gpu_c, size, cudaMemcpyDeviceToHost);

    // Checks if everything is ok
    for (int i = 0; i < N; i++)
    {
        if (cpu_c[i] != cpu_a[i] + cpu_b[i])
        {
            fprintf(stderr, "Block addition did not work !\n");
            success = false;
            break;
        }
    }
    if (success == true)
        printf("Success !\n");

    // Cleanup
    free(cpu_a);
    free(cpu_b);
    free(cpu_c);
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    return 0;
}
