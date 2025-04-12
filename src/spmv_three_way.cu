#include <stdio.h>
#include "../include/mmio.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include "../include/spmv_three_way.h"

#define BLOCK_SIZE 256

__global__ void spmv(const CSRMatrix A, const DenseMatrix x, DenseMatrix y)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < A.rows) {

    VAL_TYPE row_result_val = 0.0;

    for (uint32_t i = A.row_ptr[row]; i < A.row_ptr[row + 1]; ++i) {
      row_result_val += A.val[i] * x.data[A.col_idx[i]];
    }
    y.data[row] = row_result_val;
  }
}

DenseMatrix spmv_global(const CSRMatrix& A, const DenseMatrix& x)
{

  // Load A
  CSRMatrix d_A(A.rows, A.cols, A.nnz, CREATE_FOR_DEVICE);

  // Allocate for the 3 arrays
  cudaMalloc(&d_A.col_idx, d_A.col_idx_size);
  cudaMalloc(&d_A.row_ptr, d_A.row_ptr_size);
  cudaMalloc(&d_A.val, d_A.val_size);

  // Copy 3 arrays to device
  cudaMemcpy(d_A.col_idx, A.col_idx, d_A.col_idx_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A.row_ptr, A.row_ptr, d_A.row_ptr_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A.val, A.val, d_A.val_size, cudaMemcpyHostToDevice);
  // Load x
  DenseMatrix d_x(x.rows, x.cols, CREATE_FOR_DEVICE);

  cudaMalloc(&d_x.data, d_x.data_size);

  cudaMemcpy(d_x.data, x.data, d_x.data_size, cudaMemcpyHostToDevice);

  // Allocate y
  DenseMatrix d_y(x.rows, x.cols, CREATE_FOR_DEVICE);

  cudaMalloc(&d_y.data, d_y.data_size);

  dim3 dimBlock(BLOCK_SIZE);

  dim3 dimGrid((A.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

  spmv<<<dimGrid, dimBlock>>>(d_A, d_x, d_y);

  // Allocate host y
  DenseMatrix y = DenseMatrix(x.rows, x.cols);

  // Copy y to host
  cudaMemcpy(y.data, d_y.data, d_y.data_size, cudaMemcpyDeviceToHost);

  // Deallocate A
  cudaFree(d_A.col_idx);
  cudaFree(d_A.row_ptr);
  cudaFree(d_A.val);

  // Deallocate x
  cudaFree(d_x.data);

  // Deallocate y
  cudaFree(d_y.data);

  return y;
}

CSRMatrix parse_sparse_matrix(const char* filename)
{
  FILE* f;
  f = fopen(filename, "r");
  if (f == NULL) { throw std::runtime_error("Failed to open file"); }

  MM_typecode matcode;
  if (mm_read_banner(f, &matcode) != 0) { throw std::runtime_error("Couldn't parse matrix"); }

  if (!mm_is_sparse(matcode)) { throw std::runtime_error("CSRMatrix is non-sparse -> should be sparse"); }

  int rows, cols, nnz;
  if (mm_read_mtx_crd_size(f, &rows, &cols, &nnz) != 0) { throw std::runtime_error("Failed to read matrix size"); }

  CSRMatrix matrix = CSRMatrix(rows, cols, nnz);

  std::vector<COO_Element> coo_elements;
  coo_elements.reserve(matrix.nnz);
  for (int i = 0; i < matrix.nnz; ++i) {
    COO_Element e;
    fscanf(f, "%u %u %lg\n", &e.row, &e.col, &e.val);
    e.row--;
    e.col--;
    coo_elements.push_back(e);
  }
  fclose(f);

  std::sort(coo_elements.begin(), coo_elements.end(),
            [](const auto& a, const auto& b) { return std::tie(a.row, a.col) < std::tie(b.row, b.col); });

  for (size_t i = 0; i < coo_elements.size(); ++i) {
    const auto& e = coo_elements[i];
    matrix.row_ptr[e.row + 1]++; // Account for element ONLY on the previous row
    matrix.col_idx[i] = e.col;
    matrix.val[i] = e.val;
  }
  std::partial_sum(matrix.row_ptr, matrix.row_ptr + (matrix.rows + 1), matrix.row_ptr);

  return matrix;
}

DenseMatrix parse_dense_matrix(const char* filename)
{
  FILE* f;
  f = fopen(filename, "r");
  if (f == NULL) { throw std::runtime_error("Failed to open file"); }

  MM_typecode matcode;
  if (mm_read_banner(f, &matcode) != 0) { throw std::runtime_error("Couldn't parse matrix"); }

  if (!mm_is_dense(matcode)) { throw std::runtime_error("CSRMatrix is non-dense -> should be dense"); }

  int rows, cols;
  if (mm_read_mtx_array_size(f, &rows, &cols) != 0) { throw std::runtime_error("Failed to read matrix size"); }

  DenseMatrix matrix = DenseMatrix(rows, cols);

  for (int i = 0; i < matrix.rows; ++i) {
    VAL_TYPE e;
    fscanf(f, "%lg\n", &e);

    matrix.data[i] = e;
  }

  fclose(f);

  return matrix;
}

int main()
{

  try {
    CSRMatrix A = parse_sparse_matrix("data/scircuit.mtx");
    DenseMatrix x = parse_dense_matrix("data/scircuit_b.mtx");

    DenseMatrix y_global = spmv_global(A, x);

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
