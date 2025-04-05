// clang-format off
#include <stdio.h>
#include "lib/mmio.h"
// clang-format on

// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
  int width;
  int height;
  float *elements;
} Matrix;

#define BLOCK_SIZE 3
#define N 3

__global__ void spmv(Matrix A, Matrix x, Matrix rs) {
  float rs_value = 0;
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < A.height) {
    for (int e = 0; e < A.width; ++e)
      rs_value += A.elements[row * A.width + e] * x.elements[e];
    rs.elements[row] = rs_value;
  }
}

void spmv_global(const Matrix A, const Matrix x, const Matrix rs) {

  // Load sparse
  Matrix d_A;
  d_A.width = A.width;
  d_A.height = A.height;
  size_t size = d_A.width * d_A.height * sizeof(float);
  cudaMalloc(&d_A.elements, size);
  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

  // Load vector
  Matrix d_x;
  d_x.width = 1;
  d_x.height = x.height;
  size = d_x.height * sizeof(float);
  cudaMalloc(&d_x.elements, size);
  cudaMemcpy(d_x.elements, x.elements, size, cudaMemcpyHostToDevice);

  // Allocate result
  Matrix d_rs;
  d_rs.width = 1;
  d_rs.height = x.height;
  cudaMalloc(&d_rs.elements, size);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((A.height + BLOCK_SIZE - 1) / BLOCK_SIZE);
  spmv<<<dimGrid, dimBlock>>>(d_A, d_x, d_rs);

  // Copy result to host
  cudaMemcpy(rs.elements, d_rs.elements, size, cudaMemcpyDeviceToHost);

  cudaFree(d_A.elements);
  cudaFree(d_x.elements);
  cudaFree(d_rs.elements);
}

void parse_matrices(const char *filename) {
  MM_typecode matcode;
  FILE *f = fopen(filename, "r");

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Couldn't parse matrix");
    fclose(f);
    exit(1);
  }

  fclose(f);

  printf("Matrix type: %s\n", mm_typecode_to_str(matcode));
}

// Driver
int main() {
  Matrix A, x, rs;

  A.width = N;
  A.height = N;
  A.elements = new float[9]{1, 2, 3, 4, 5, 6, 7, 8, 9};

  x.width = 1;
  x.height = N;
  x.elements = new float[3]{1, 2, 3};

  rs.width = 1;
  rs.height = N;
  rs.elements = new float[3];

  spmv_global(A, x, rs);

  parse_matrices("data/scircuit.mtx");

  delete[] A.elements;
  delete[] x.elements;
  delete[] rs.elements;
}
