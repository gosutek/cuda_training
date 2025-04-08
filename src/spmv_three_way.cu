#include <stdio.h>
#include "../include/mmio.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <memory>
#include "../include/spmv_three_way.h"

#define BLOCK_SIZE 256

/*
__global__ void spmv(const CSRMatrix* A, const DenseMatrix* x, const DenseMatrix* rs)
{

  float rs_value = 0;

  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < A.rows) {

    for (int e = 0; e < A.cols; ++e) rs_value += A.elements[row * A.cols + e] * x.elements[e];

    rs.elements[row] = rs_value;
  }
}
*/

std::unique_ptr<DenseMatrix> spmv_global(const std::unique_ptr<CSRMatrix> A, const std::unique_ptr<DenseMatrix> x)
{

  // Load sparse
  std::unique_ptr<CSRMatrix> d_A = std::make_unique<CSRMatrix>(A->rows, A->cols, A->nnz);

  uint32_t* sparse_col_idx_ptr = d_A->col_idx.data();
  uint32_t* sparse_row_ptr_ptr = d_A->row_ptr.data();
  VAL_TYPE* sparse_val_ptr = d_A->val.data();

  // Allocate for the 3 arrays
  cudaMalloc(&sparse_col_idx_ptr, A->col_idx.size() * sizeof(uint32_t));
  cudaMalloc(&sparse_row_ptr_ptr, A->row_ptr.size() * sizeof(uint32_t));
  cudaMalloc(&sparse_val_ptr, A->val.size() * sizeof(VAL_TYPE));

  // Copy 3 arrays to device
  cudaMemcpy(sparse_col_idx_ptr, A->col_idx.data(), A->col_idx.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(sparse_row_ptr_ptr, A->row_ptr.data(), A->row_ptr.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(sparse_val_ptr, A->val.data(), A->val.size() * sizeof(VAL_TYPE), cudaMemcpyHostToDevice);

  // Load vector
  std::unique_ptr<DenseMatrix> d_x = std::make_unique<DenseMatrix>(x->rows, x->cols);

  VAL_TYPE* vector_val_ptr = d_x->data.data();

  cudaMalloc(&vector_val_ptr, x->data.size() * sizeof(VAL_TYPE));

  cudaMemcpy(vector_val_ptr, x->data.data(), x->data.size() * sizeof(VAL_TYPE), cudaMemcpyHostToDevice);

  // Allocate result
  std::unique_ptr<DenseMatrix> d_rs = std::make_unique<DenseMatrix>(x->rows, x->cols);

  VAL_TYPE* result_val_ptr = d_rs->data.data();

  cudaMalloc(&result_val_ptr, x->data.size() * sizeof(VAL_TYPE));

  dim3 dimBlock(BLOCK_SIZE);

  dim3 dimGrid((A->rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // spmv<<<dimGrid, dimBlock>>>(d_A.get(), d_x.get(), d_rs.get());

  // Copy result to host
  cudaMemcpy(d_rs->data.data(), result_val_ptr, x->data.size() * sizeof(VAL_TYPE), cudaMemcpyDeviceToHost);

  // Deallocate A
  cudaFree(sparse_col_idx_ptr);
  cudaFree(sparse_row_ptr_ptr);
  cudaFree(sparse_val_ptr);

  // Deallocate x
  cudaFree(vector_val_ptr);

  // Deallocate result
  cudaFree(result_val_ptr);

  // d_A, d_x, A, x freed as they go out of scope

  return d_rs;
}

std::unique_ptr<CSRMatrix> parse_sparse_matrix(const char* filename)
{
  FILE* f;
  f = fopen(filename, "r");
  if (f == NULL) { throw std::runtime_error("Failed to open file"); }

  MM_typecode matcode;
  if (mm_read_banner(f, &matcode) != 0) { throw std::runtime_error("Couldn't parse matrix"); }

  if (!mm_is_sparse(matcode)) { throw std::runtime_error("CSRMatrix is non-sparse -> should be sparse"); }

  int rows, cols, nnz;
  if (mm_read_mtx_crd_size(f, &rows, &cols, &nnz) != 0) { throw std::runtime_error("Failed to read matrix size"); }

  std::unique_ptr<CSRMatrix> matrix = std::make_unique<CSRMatrix>(rows, cols, nnz);

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

  std::sort(coo_elements.begin(), coo_elements.end(),
            [](const auto& a, const auto& b) { return std::tie(a.row, a.col) < std::tie(b.row, b.col); });

  for (size_t i = 0; i < coo_elements.size(); ++i) {
    const auto& e = coo_elements[i];
    matrix->row_ptr[e.row + 1]++; // Account for element ONLY on the previous row
    matrix->col_idx[i] = e.col;
    matrix->val[i] = e.val;
  }
  std::partial_sum(matrix->row_ptr.begin(), matrix->row_ptr.end(), matrix->row_ptr.begin());

  return matrix;
}

std::unique_ptr<DenseMatrix> parse_dense_matrix(const char* filename)
{
  FILE* f;
  f = fopen(filename, "r");
  if (f == NULL) { throw std::runtime_error("Failed to open file"); }

  MM_typecode matcode;
  if (mm_read_banner(f, &matcode) != 0) { throw std::runtime_error("Couldn't parse matrix"); }

  if (!mm_is_dense(matcode)) { throw std::runtime_error("CSRMatrix is non-dense -> should be dense"); }

  int rows, cols;
  if (mm_read_mtx_array_size(f, &rows, &cols) != 0) { throw std::runtime_error("Failed to read matrix size"); }

  std::unique_ptr<DenseMatrix> matrix = std::make_unique<DenseMatrix>(rows, cols);

  for (int i = 0; i < 1; ++i) {
    VAL_TYPE e;
    fscanf(f, "%lg\n", &e);
    matrix->data.push_back(e);
  }

  fclose(f);

  return matrix;
}

// Driver
int main()
{

  try {
    std::unique_ptr<CSRMatrix> A = parse_sparse_matrix("data/scircuit.mtx");
    std::unique_ptr<DenseMatrix> x = parse_dense_matrix("data/scircuit_b.mtx");

    std::unique_ptr<DenseMatrix> global_result = spmv_global(std::move(A), std::move(x));

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
