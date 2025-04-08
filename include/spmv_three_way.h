#pragma once

#include <cstdint>
#include <vector>

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

  CSRMatrix(int r, int c, int z) : col_idx(), val(), row_ptr()
  {
    rows = r;
    cols = c;
    nnz = z;

    col_idx.reserve(nnz);
    val.reserve(nnz);
    row_ptr.reserve(rows + 1);
  }
};

struct DenseMatrix : public Matrix {
  std::vector<VAL_TYPE> data;

  DenseMatrix(uint32_t r, uint32_t c) : data()
  {
    rows = r;
    cols = c;

    data.reserve(r * c);
  }
};
