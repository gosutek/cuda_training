#include <cstdint>
#include <cstdlib>
#include <iostream>

constexpr uint16_t N = 50000;

__global__ void VecAdd(float *A, float *B, float *C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

int main() {
  size_t size = N * sizeof(float);

  // Normal malloc in heap
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  for (int i = 0; i < N; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // GPU allocation
  float *d_A;
  cudaMalloc(&d_A, size);
  float *d_B;
  cudaMalloc(&d_B, size);
  float *d_C;
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  int threads_per_block = 256;
  int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

  std::cout << "Launching kernel with " << blocks_per_grid
            << " blocks per grid, " << threads_per_block
            << " threads per blocks" << std::endl;
  VecAdd<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);
  std::cout << "Kernel finished (?)" << std::endl;

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
