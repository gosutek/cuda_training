#include <cstdlib>
#include <iostream>
#include "../include/spmv_three_way.h"

#define BLOCK_SIZE 32
#define DEVICE_ID 0

__global__ void spmv(const uint32_t* col_idx, const uint32_t* row_ptr, const VAL_TYPE* val, const int rows,
                     const VAL_TYPE* vector_data, VAL_TYPE* res)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows) {

    VAL_TYPE row_result_val = 0.0;

    for (uint32_t i = row_ptr[row]; i < row_ptr[row + 1]; ++i) { row_result_val += val[i] * vector_data[col_idx[i]]; }
    res[row] = row_result_val;
  }
}

DenseMatrix spmv_l2_window(const CSRMatrix& d_A, const DenseMatrix& d_x, DenseMatrix& d_y)
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, DEVICE_ID);

  size_t size = std::min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                     size); // cudaLimitPersistingL2CacheSize -> Global limit for all persistent data

  size_t window_size
    = std::min(prop.accessPolicyMaxWindowSize,
               (int)d_x.data_size); // accessPolicyMaxWindowSize -> Per-stream limit for a single persistent region

  cudaStreamAttrValue stream_attribute;
  stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_x.data);
  stream_attribute.accessPolicyWindow.num_bytes = window_size;
  stream_attribute.accessPolicyWindow.hitRatio = 0.6;
  stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

  dim3 dimBlock(BLOCK_SIZE);

  dim3 dimGrid((d_A.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

  spmv<<<dimGrid, dimBlock>>>(d_A.col_idx, d_A.row_ptr, d_A.val, d_A.rows, d_x.data, d_y.data);

  // Reset L2 Access to Normal
  stream_attribute.accessPolicyWindow.num_bytes = 0;
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
  cudaCtxResetPersistingL2Cache();

  return DenseMatrix(d_y, AllocTarget::Host);
}

DenseMatrix spmv_global(const CSRMatrix& d_A, const DenseMatrix& d_x, DenseMatrix& d_y)
{

  dim3 dimBlock(BLOCK_SIZE);

  dim3 dimGrid((d_A.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

  spmv<<<dimGrid, dimBlock>>>(d_A.col_idx, d_A.row_ptr, d_A.val, d_A.rows, d_x.data, d_y.data);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Error check after kernel launch ~ %s", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return DenseMatrix(d_y, AllocTarget::Host);
}

int main()
{

  try {

    CSRMatrix A("data/scircuit.mtx", AllocTarget::Host);
    DenseMatrix x("data/scircuit_b.mtx", AllocTarget::Host);

    // ================================ DEVICE ALLOCATION ================================
    CSRMatrix d_A(A, AllocTarget::Device);

    DenseMatrix d_x(x, AllocTarget::Device);

    DenseMatrix d_y(x, AllocTarget::Device);
    // ================================ KERNEL EXECUTION ================================
    DenseMatrix y_global = spmv_global(d_A, d_x, d_y);
    DenseMatrix y_l2_window = spmv_l2_window(d_A, d_x, d_y);

    std::cout << y_global.data[0] << std::endl;
    std::cout << y_l2_window.data[0] << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
