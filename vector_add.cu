#include <stdio.h>

#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int num_elements) {
  int i = threadIdx.x;

  C[i] = A[i] + B[i];
}

int main(void) {

  int num_elements = 50000;
  size_t size = num_elements * sizeof(float);
  printf("[Vector addition of %d elements]\n", num_elements);

  float *h_A = (float *)malloc(size);

  float *h_B = (float *)malloc(size);

  float *h_C = (float *)malloc(size);

  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < num_elements; ++i) {
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
  }

  float *d_A = NULL;
  cudaMalloc((void **)&d_A, size);

  float *d_B = NULL;
  cudaMalloc((void **)&d_B, size);

  float *d_C = NULL;
  cudaMalloc((void **)&d_C, size);

  printf("Copy input data from the host memory to the CUDA device\n");
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  int threads_per_block = 256;
  int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block; // ??
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocks_per_grid, threads_per_block);
  vectorAdd<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, num_elements);

  printf("Copy output data from the CUDA device to the host memory\n");
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < num_elements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);

  cudaDeviceReset();

  printf("Done\n");
  return 0;
}
