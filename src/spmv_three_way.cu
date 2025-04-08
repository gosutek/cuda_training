#include <stdio.h>
#include "../include/mmio.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#define BLOCK_SIZE 3
#define VAL_TYPE double

struct COO_Element {
  uint32_t row, col;
  VAL_TYPE val;
};

struct Matrix {
  int rows = 0, cols = 0;
};

struct CSRMatrix : public Matrix {

  int nnz = 0;

  std::vector<uint32_t> col_idx;
  std::vector<uint32_t> row_ptr;
  std::vector<VAL_TYPE> val;

  CSRMatrix(int r, int c, int z) : col_idx(z), val(z), row_ptr(r + 1)
  {
    rows = r;
    cols = c;
  }
};

struct DenseMatrix : public Matrix {
  std::vector<VAL_TYPE> data;

  DenseMatrix(uint32_t r, uint32_t c) : data(r * c, 0)
  {
    rows = r;
    cols = c;
  }
};

/*
__global__ void spmv(CSRMatrix A, CSRMatrix x, CSRMatrix rs)
{

  float rs_value = 0;

  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < A.rows) {

    for (int e = 0; e < A.cols; ++e) rs_value += A.elements[row * A.cols + e] * x.elements[e];

    rs.elements[row] = rs_value;
  }
}
*/
// void spmv_global(const CSRMatrix A, const CSRMatrix x, const CSRMatrix rs)
// {
//
//   // Load sparse
//
//   CSRMatrix d_A;
//
//   d_A.cols = A.cols;
//
//   d_A.rows = A.rows;
//
//   uint32_t* col_idx_ptr = d_A.col_idx.data();
//   uint32_t* row_ptr_ptr = d_A.row_ptr.data();
//   VAL_TYPE* val_ptr = d_A.val.data();
//
//   // Allocate for the 3 arrays
//   cudaMalloc(&col_idx_ptr, A.col_idx.size() * sizeof(uint32_t));
//   cudaMalloc(&row_ptr_ptr, A.row_ptr.size() * sizeof(uint32_t));
//   cudaMalloc(&val_ptr, A.val.size() * sizeof(VAL_TYPE));
//
//   // Copy 3 arrays to device
//   cudaMemcpy(col_idx_ptr, A.col_idx.data(), A.col_idx.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
//   cudaMemcpy(row_ptr_ptr, A.row_ptr.data(), A.row_ptr.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
//   cudaMemcpy(val_ptr, A.val.data(), A.val.size() * sizeof(VAL_TYPE), cudaMemcpyHostToDevice);
//
//   // Load vector
//   // Allocate result
//   // dim3 dimBlock(BLOCK_SIZE);
//
//   // dim3 dimGrid((A.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
//
//   // spmv<<<dimGrid, dimBlock>>>(d_A, d_x, d_rs);
//
//   // Copy result to host
//   // cudaMemcpy(rs.elements, d_rs.elements, size, cudaMemcpyDeviceToHost);
//
//   // Deallocate A
//   cudaFree(col_idx_ptr);
//   cudaFree(row_ptr_ptr);
//   cudaFree(val_ptr);
//
//   // Deallocate x
//
//   // Deallocate result
// }

std::unique_ptr<CSRMatrix> parse_sparse_matrix(const char* filename)
{
  FILE* f;
  if (!(f = fopen(filename, "r"))) {
    fclose(f);
    throw std::runtime_error("Failed to open file");
  }

  MM_typecode matcode;
  if (mm_read_banner(f, &matcode) != 0) {
    fclose(f);
    throw std::runtime_error("Couldn't parse matrix");
  }

  if (!mm_is_sparse(matcode)) {
    fclose(f);
    throw std::runtime_error("CSRMatrix is non-sparse");
  }

  CSRMatrix matrix;
  if (mm_read_mtx_crd_size(f, &matrix.rows, &matrix.cols, &matrix.nnz) != 0) {
    fclose(f);
    throw std::runtime_error("Failed to read matrix size");
  }

  std::vector<COO_Element> coo_elements;
  coo_elements.reserve(matrix->nnz);
  for (int i = 0; i < matrix->nnz; ++i) {
    COO_Element e;
    fscanf(f, "%u %u %lg\n", &e.row, &e.col, &e.val);
    e.row--;
    e.col--;
    coo_elements.push_back(e);
  }
  fclose(f);

  convert_coo_to_csr(std::move(coo_elements),
                     matrix); // coo_elements no longer exists
  return matrix;
}

// Driver
int main()
{
  // spmv_global(A, x, rs);

  try {
    CSRMatrix A = parse_and_convert("data/scircuit.mtx");
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
