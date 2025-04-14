#include <iostream>
#include "../include/spmv_three_way.h"

#define BLOCK_SIZE 32
#define DEVICE_ID 0

__global__ void spmv(const CSRMatrix A, const DenseMatrix x, DenseMatrix y)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row == 0) {
    printf("x.cols = %d\nx.rows = %d\nx.data points to %p\n", x.cols, x.rows, x.data);
    printf("Thread %d: x.data[0] = %f\n", row, x.data[0]);
    printf("Thread %d: x.data[0] = %f\n", row, x.data[1]);
    printf("Thread %d: x.data[0] = %f\n", row, x.data[2]);
    printf("Thread %d: x.data[0] = %f\n", row, x.data[3]);
    printf("Thread %d: x.data[0] = %f\n", row, x.data[4]);
    printf("\nThread %d: A.val[0] = %lg\n", row, A.val[0]);
  }

  if (row < A.rows) {

    VAL_TYPE row_result_val = 0.0;

    for (uint32_t i = A.row_ptr[row]; i < A.row_ptr[row + 1]; ++i) {
      row_result_val += A.val[i] * x.data[A.col_idx[i]];
    }
    y.data[row] = row_result_val;
  }
}

// DenseMatrix spmv_l2_window(const CSRMatrix& d_A, const DenseMatrix& d_x, DenseMatrix& d_y)
// {
//   cudaStream_t stream;
//   cudaStreamCreate(&stream);
//
//   cudaDeviceProp prop;
//   cudaGetDeviceProperties(&prop, DEVICE_ID);
//
//   size_t size = std::min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
//   cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
//                      size); // cudaLimitPersistingL2CacheSize -> Global limit for all persistent data
//
//   size_t window_size
//     = std::min(prop.accessPolicyMaxWindowSize,
//                (int)d_x.data_size); // accessPolicyMaxWindowSize -> Per-stream limit for a single persistent region
//
//   cudaStreamAttrValue stream_attribute;
//   stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_x.data);
//   stream_attribute.accessPolicyWindow.num_bytes = window_size;
//   stream_attribute.accessPolicyWindow.hitRatio = 0.6;
//   stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
//   stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
//
//   cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
//
//   dim3 dimBlock(BLOCK_SIZE);
//
//   dim3 dimGrid((d_A.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
//
//   spmv<<<dimGrid, dimBlock>>>(d_A, d_x, d_y);
//
//   // Reset L2 Access to Normal
//   stream_attribute.accessPolicyWindow.num_bytes = 0;
//   cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
//   cudaCtxResetPersistingL2Cache();
//
//   // Allocate host y
//   DenseMatrix y = DenseMatrix(d_x.rows, d_x.cols);
//
//   // Copy y to host
//   cudaMemcpy(y.data, d_y.data, d_y.data_size, cudaMemcpyDeviceToHost);
//
//   return y;
// }

DenseMatrix spmv_global(const CSRMatrix& d_A, const DenseMatrix& d_x, DenseMatrix& d_y)
{

  dim3 dimBlock(BLOCK_SIZE);

  dim3 dimGrid((d_A.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

  std::cout << "data points to " << d_x.data << std::endl;
  VAL_TYPE* host_ptr = (VAL_TYPE*)malloc(d_x.data_size);
  std::cout << host_ptr[0] << std::endl;
  cudaMemcpy(host_ptr, d_x.data, d_x.data_size, cudaMemcpyDeviceToHost);
  std::cout << host_ptr[0] << std::endl;

  spmv<<<dimGrid, dimBlock>>>(d_A, d_x, d_y);

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
    // DenseMatrix y_l2_window = spmv_l2_window(d_A, d_x, d_y);

    std::cout << y_global.data[0] << std::endl;
    // std::cout << y_l2_window.data[0] << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
