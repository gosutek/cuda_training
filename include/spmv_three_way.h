#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <stdexcept>

#define VAL_TYPE double
#define CREATE_FOR_DEVICE 1
#define CREATE_FOR_HOST 2

struct COO_Element {
  uint32_t row, col;
  VAL_TYPE val;
};

struct Matrix {
  int rows = 0, cols = 0;
};

struct CSRMatrix : public Matrix {

  int nnz = 0;

  uint32_t* col_idx = nullptr;
  uint32_t* row_ptr = nullptr;
  VAL_TYPE* val = nullptr;

  bool allocated_on_device = false;

  CSRMatrix(int r, int c, int z, int t = CREATE_FOR_HOST)
  {
    rows = r;
    cols = c;
    nnz = z;

    if (t == CREATE_FOR_HOST) {
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
    } else if (t == CREATE_FOR_DEVICE) {
      allocated_on_device = true;
    }
  }

  ~CSRMatrix()
  {
    if (!allocated_on_device) {
      free(col_idx);
      free(row_ptr);
      free(val);
    }
  }
};

struct DenseMatrix : public Matrix {

  VAL_TYPE* data = nullptr;

  bool allocated_on_device = false;

  DenseMatrix(int r, int c, int t = CREATE_FOR_HOST)
  {
    rows = r;
    cols = c;

    if (t == CREATE_FOR_HOST) {
      data = (VAL_TYPE*)malloc((rows * cols) * sizeof(VAL_TYPE));
      if (!data) { throw std::runtime_error("Failed to allocate data"); }
    } else if (t == CREATE_FOR_DEVICE) {
      allocated_on_device = true;
    }
  }

  ~DenseMatrix()
  {
    if (!allocated_on_device) free(data);
  }
};
