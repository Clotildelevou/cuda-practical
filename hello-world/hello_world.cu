#include <cstdio>
#include "cuda.h"

__global__
void GPUFunction()
{
  printf("hello from the Gpu.\n");
}

int main()
{
  GPUFunction<<<1, 1>>>();

  cudaDeviceSynchronize();

  return EXIT_SUCCESS;
}
