#include <stdio.h>
#include "../include/mmio.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

struct COO_Element {
  uint32_t row, col;
  double val;
};

struct CSRMatrix {
  int rows = 0, cols = 0, nnz = 0;
  std::vector<uint32_t> col_idx;
  std::vector<uint32_t> row_ptr;
  std::vector<double> val;
};

#define BLOCK_SIZE 3

void convert_coo_to_csr(std::vector<COO_Element> coo, CSRMatrix& matrix)
{
  std::sort(coo.begin(), coo.end(),
            [](const auto& a, const auto& b) { return std::tie(a.row, a.col) < std::tie(b.row, b.col); });

  matrix.col_idx.resize(matrix.nnz);
  matrix.val.resize(matrix.nnz);
  matrix.row_ptr.resize(matrix.rows + 1, 0);

  for (size_t i = 0; i < coo.size(); ++i) {
    const auto& e = coo[i];
    matrix.row_ptr[e.row + 1]++; // Account for element ONLY on the previous row
    matrix.col_idx[i] = e.col;
    matrix.val[i] = e.val;
  }
  std::partial_sum(coo.begin(), coo.end(), coo.begin());
}

CSRMatrix parse_and_convert(const char* filename)
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
  coo_elements.reserve(matrix.nnz);
  for (int i = 0; i < matrix.nnz; ++i) {
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
