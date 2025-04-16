#pragma once

#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include <stdexcept>
#include "../include/mmio.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"

#define VAL_TYPE double

enum AllocTarget { Host, Device };

struct COO_Element {
  uint32_t row, col;
  VAL_TYPE val;
};

struct Matrix {

public:
  int rows = 0, cols = 0;

protected:
  AllocTarget _Target;

private:
  virtual void allocate_memory() = 0;
};

struct CSRMatrix : public Matrix {

public:
  cudaError_t cuda_error = cudaSuccess;

  int nnz = 0;

  uint32_t* col_idx = nullptr;
  uint32_t* row_ptr = nullptr;
  VAL_TYPE* val = nullptr;

  size_t col_idx_size;
  size_t row_ptr_size;
  size_t val_size;

public:
  CSRMatrix(const char* filepath, AllocTarget Target)
  {
    _Target = Target;
    FILE* f;
    f = fopen(filepath, "r");
    if (f == NULL) { throw std::runtime_error("Failed to open file"); }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) { throw std::runtime_error("Couldn't parse matrix"); }

    if (!mm_is_sparse(matcode)) { throw std::runtime_error("CSRMatrix is non-sparse -> should be sparse"); }

    if (mm_read_mtx_crd_size(f, &rows, &cols, &nnz) != 0) { throw std::runtime_error("Failed to read matrix size"); }

    col_idx_size = nnz * sizeof(uint32_t);
    row_ptr_size = (rows + 1) * sizeof(uint32_t);
    val_size = nnz * sizeof(VAL_TYPE);

    allocate_memory();

    std::vector<COO_Element> coo_elements;
    coo_elements.reserve(nnz);
    for (int i = 0; i < nnz; ++i) {
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
      row_ptr[e.row + 1]++; // Account for element ONLY on the previous row
      col_idx[i] = e.col;
      val[i] = e.val;
    }
    std::partial_sum(row_ptr, row_ptr + (rows + 1), row_ptr);
  }

  CSRMatrix(const CSRMatrix& src, AllocTarget Target)
      : nnz(src.nnz), col_idx_size(src.col_idx_size), row_ptr_size(src.row_ptr_size), val_size(src.val_size)
  {

    rows = src.rows;
    cols = src.cols;
    _Target = Target;

    allocate_memory();

    if (src._Target == AllocTarget::Host && _Target == AllocTarget::Host) { // Host to Host
      std::memcpy(col_idx, src.col_idx, col_idx_size);
      std::memcpy(row_ptr, src.row_ptr, row_ptr_size);
      std::memcpy(val, src.val, val_size);
    } else if (src._Target == AllocTarget::Host && _Target == AllocTarget::Device) { // Host to dev
      cuda_error = cudaMemcpy(col_idx, src.col_idx, col_idx_size, cudaMemcpyHostToDevice);

      if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Failed to copy CSRMatrix.col_idx (error code %s)\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
      }

      cuda_error = cudaMemcpy(row_ptr, src.row_ptr, row_ptr_size, cudaMemcpyHostToDevice);

      if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Failed to copy CSRMatrix.row_ptr (error code %s)\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
      }

      cuda_error = cudaMemcpy(val, src.val, val_size, cudaMemcpyHostToDevice);

      if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Failed to copy CSRMatrix.val (error code %s)\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
      }
    }
  }

  ~CSRMatrix()
  {
    if (_Target == AllocTarget::Host) {
      free(col_idx);
      free(row_ptr);
      free(val);
    } else {
      cuda_error = cudaFree(col_idx);

      if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Failed to free CSRMatrix.col_idx (error code %s)\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
      }

      cuda_error = cudaFree(row_ptr);

      if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Failed to free CSRMatrix.row_ptr (error code %s)\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
      }

      cuda_error = cudaFree(val);

      if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Failed to free CSRMatrix.val (error code %s)\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
      }
    }
  }

private:
  virtual void allocate_memory() override
  {
    if (_Target == AllocTarget::Host) {
      col_idx = (uint32_t*)malloc(nnz * sizeof(uint32_t));
      if (!col_idx) throw std::runtime_error("Failed to allocate col_idx");

      val = (VAL_TYPE*)malloc(nnz * sizeof(VAL_TYPE));
      if (!val) {
        free(col_idx);
        throw std::runtime_error("Failed to allocate val");
      }

      row_ptr = (uint32_t*)malloc((rows + 1) * sizeof(uint32_t));
      if (!row_ptr) {
        free(col_idx);
        free(val);
        throw std::runtime_error("Failed to allocate row_ptr");
      }
    } else if (_Target == AllocTarget::Device) {
      cuda_error = cudaMalloc(&col_idx, col_idx_size);

      if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate CSRMatrix.col_idx (error code %s)\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
      }

      cuda_error = cudaMalloc(&row_ptr, row_ptr_size);

      if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate CSRMatrix.row_ptr (error code %s)\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
      }

      cuda_error = cudaMalloc(&val, val_size);

      if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate CSRMatrix.val (error code %s)\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
      }
    }
  }
};

struct DenseMatrix : public Matrix {

public:
  cudaError_t cuda_error = cudaSuccess;

  VAL_TYPE* data = nullptr;

  size_t data_size;

public:
  DenseMatrix(const char* filepath, AllocTarget Target)
  {
    _Target = Target;
    FILE* f;
    f = fopen(filepath, "r");
    if (f == NULL) { throw std::runtime_error("Failed to open file"); }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) { throw std::runtime_error("Couldn't parse matrix"); }

    if (!mm_is_dense(matcode)) { throw std::runtime_error("Matrix is non-dense -> should be dense"); }

    if (mm_read_mtx_array_size(f, &rows, &cols) != 0) { throw std::runtime_error("Failed to read matrix size"); }

    data_size = (rows * cols) * sizeof(VAL_TYPE);

    allocate_memory();

    for (int i = 0; i < rows; ++i) {
      VAL_TYPE e;
      fscanf(f, "%lg\n", &e);

      data[i] = e;
    }

    fclose(f);
  }

  DenseMatrix(const DenseMatrix& src, AllocTarget Target) : data_size(src.data_size)
  {
    rows = src.rows;
    cols = src.cols;
    _Target = Target;
    allocate_memory();

    if (src._Target == AllocTarget::Host && _Target == AllocTarget::Host) { // Host to Host
      std::memcpy(data, src.data, data_size);
    } else if (src._Target == AllocTarget::Host && _Target == AllocTarget::Device) { // Host to Dev
      cuda_error = cudaMemcpy(data, src.data, data_size, cudaMemcpyHostToDevice);
    } else if (src._Target == AllocTarget::Device && _Target == AllocTarget::Host) { // Dev to host
      cuda_error = cudaMemcpy(data, src.data, data_size, cudaMemcpyDeviceToHost);
    } else {
      cuda_error = cudaMemcpy(data, src.data, data_size, cudaMemcpyDeviceToDevice);
    }

    if (cuda_error != cudaSuccess) {
      fprintf(stderr, "Failed to copy DenseMatrix (error code %s)\n", cudaGetErrorString(cuda_error));
      exit(EXIT_FAILURE);
    }
  }

  ~DenseMatrix()
  {
    if (_Target == AllocTarget::Host) free(data);
    else {
      cuda_error = cudaFree(data);
    }

    if (cuda_error != cudaSuccess) {
      fprintf(stderr, "Failed to free DenseMatrix (error code %s)\n", cudaGetErrorString(cuda_error));
      exit(EXIT_FAILURE);
    }
  }
  DenseMatrix(const DenseMatrix& other) {}

private:
  virtual void allocate_memory() override
  {
    if (_Target == AllocTarget::Host) {
      data = (VAL_TYPE*)malloc((rows * cols) * sizeof(VAL_TYPE));
      if (!data) { throw std::runtime_error("Failed to allocate data"); }
    } else if (_Target == AllocTarget::Device) {
      cuda_error = cudaMalloc(&data, data_size);
    }

    if (cuda_error != cudaSuccess) {
      fprintf(stderr, "Failed to allocate DenseMatrix (error code %s)\n", cudaGetErrorString(cuda_error));
      exit(EXIT_FAILURE);
    }
  }
};
