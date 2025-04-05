// clang-format off
#include <stdio.h>
#include "lib/mmio.h"
#include <cstdint>
// clang-format on

typedef struct {
  uint32_t *col_idx;
  uint32_t *row_ptr;
  double *val;
  int rows;
  int cols;
  int nnz;
} CSRMatrix;

#define BLOCK_SIZE 3

int parse_matrix(FILE *f, CSRMatrix *matrix) {
  MM_typecode matcode;
  uint32_t *I, *J;
  double *val;

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Couldn't parse matrix");
    return 1;
  }

  if (!mm_is_sparse(matcode)) {
    printf("CSRMatrix is non-sparse");
    return 1;
  }

  if (mm_read_mtx_crd_size(f, &matrix->rows, &matrix->cols, &matrix->nnz) !=
      0) {
    return 1;
  }

  // Allocate for COO
  I = new uint32_t[matrix->nnz];
  J = new uint32_t[matrix->nnz];
  val = new double[matrix->nnz];

  // Read matrix in COO
  for (int i = 0; i < matrix->nnz; ++i) {

    fscanf(f, "%u %u %lg\n", &I[i], &J[i], &val[i]);
    I[i]--;
    J[i]--;
  }

  // Allocate for CSR
  matrix->val = new double[matrix->nnz];
  matrix->col_idx = new uint32_t[matrix->nnz];
  matrix->row_ptr = new uint32_t[matrix->rows + 1];

  printf("%u %u %lg\n", matrix->row_ptr[0], matrix->col_idx[0], matrix->val[0]);

  delete[] I;
  delete[] J;
  delete[] val;

  return 0;
}

// Driver
int main() {
  CSRMatrix A;
  FILE *f = fopen("data/scircuit.mtx", "r");

  // spmv_global(A, x, rs);

  parse_matrix(f, &A);
  fclose(f);

  delete[] A.val;
  delete[] A.col_idx;
  delete[] A.row_ptr;
}
