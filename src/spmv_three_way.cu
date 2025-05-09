#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
//clang-format off
#include "../include/mmio.h"
//clang-format on

#define VAL_TYPE double
#define BLOCK_SIZE 32
#define DEVICE_ID 0
#define CHECK_CUDA(S)                                                                                                     \
	do {                                                                                                                  \
		cudaError_t err = S;                                                                                              \
		if (err != cudaSuccess) {                                                                                         \
			std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
			exit(EXIT_FAILURE);                                                                                           \
		}                                                                                                                 \
	} while (0)

enum AllocTarget
{
	Host,
	Device
};

__global__ void spmv_shared_mem(uint32_t* const col_idx, uint32_t* const row_ptr, VAL_TYPE* const val, const int rows,
	VAL_TYPE* const vector_data, VAL_TYPE* res)
{
	int block_row = blockIdx.y;
	int block_col = blockIdx.x;
}

__global__ void spmv(const uint32_t* const col_idx, const uint32_t* const row_ptr, const VAL_TYPE* const val,
	const int rows, const VAL_TYPE* const vector_data, VAL_TYPE* const res)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rows) {
		VAL_TYPE row_result_val = 0.0;

		for (uint32_t i = row_ptr[row]; i < row_ptr[row + 1]; ++i) {
			row_result_val += val[i] * vector_data[col_idx[i]];
		}
		res[row] = row_result_val;
	}
}

struct COO_Element
{
	uint32_t row, col;
	VAL_TYPE val;
};

struct Matrix
{
public:
	int rows = 0, cols = 0;

protected:
	AllocTarget _Target;

private:
	virtual void allocate_memory() = 0;
};

struct CSRMatrix : public Matrix
{
public:
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
		if (f == NULL) {
			throw std::runtime_error("Failed to open file");
		}

		MM_typecode matcode;
		if (mm_read_banner(f, &matcode) != 0) {
			throw std::runtime_error("Couldn't parse matrix");
		}

		if (!mm_is_sparse(matcode)) {
			throw std::runtime_error("CSRMatrix is non-sparse -> should be sparse");
		}

		if (mm_read_mtx_crd_size(f, &rows, &cols, &nnz) != 0) {
			throw std::runtime_error("Failed to read matrix size");
		}

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
			row_ptr[e.row + 1]++;  // Account for element ONLY on the previous row
			col_idx[i] = e.col;
			val[i] = e.val;
		}
		std::partial_sum(row_ptr, row_ptr + (rows + 1), row_ptr);
	}

	CSRMatrix(const CSRMatrix& src, AllocTarget Target) :
		nnz(src.nnz), col_idx_size(src.col_idx_size), row_ptr_size(src.row_ptr_size), val_size(src.val_size)
	{
		rows = src.rows;
		cols = src.cols;
		_Target = Target;

		allocate_memory();

		if (src._Target == AllocTarget::Host && _Target == AllocTarget::Host) {  // Host to Host
			std::memcpy(col_idx, src.col_idx, col_idx_size);
			std::memcpy(row_ptr, src.row_ptr, row_ptr_size);
			std::memcpy(val, src.val, val_size);
		} else if (src._Target == AllocTarget::Host && _Target == AllocTarget::Device) {  // Host to dev
			CHECK_CUDA(cudaMemcpy(col_idx, src.col_idx, col_idx_size, cudaMemcpyHostToDevice));
			CHECK_CUDA(cudaMemcpy(row_ptr, src.row_ptr, row_ptr_size, cudaMemcpyHostToDevice));
			CHECK_CUDA(cudaMemcpy(val, src.val, val_size, cudaMemcpyHostToDevice));
		}
	}

	__device__ void get_sub_matrix(uint32_t* const col_idx, uint32_t* const row_ptr, VAL_TYPE* const val, const int rows,
		const uint32_t row_offset, const uint32_t col_offset) const
	{
		SubMatrix d_A_sub(col_idx, row_ptr, val, rows);

		for (int i = row_ptr[row_offset]; i < row_ptr[row_offset + BLOCK_SIZE]; ++i) {  // TODO: Bounds checking
			if (col_idx[i] < col_offset) {
				d_A_sub.col_idx_size++;
			}
		}
	}

	virtual ~CSRMatrix()
	{
		if (_Target == AllocTarget::Host) {
			free(col_idx);
			free(row_ptr);
			free(val);
		} else {
			CHECK_CUDA(cudaFree(col_idx));
			CHECK_CUDA(cudaFree(row_ptr));
			CHECK_CUDA(cudaFree(val));
		}
	}

private:
	virtual void allocate_memory() override
	{
		if (_Target == AllocTarget::Host) {
			col_idx = (uint32_t*)malloc(nnz * sizeof(uint32_t));
			if (!col_idx)
				throw std::runtime_error("Failed to allocate col_idx");

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
			CHECK_CUDA(cudaMalloc(&col_idx, col_idx_size));
			CHECK_CUDA(cudaMalloc(&row_ptr, row_ptr_size));
			CHECK_CUDA(cudaMalloc(&val, val_size));
		}
	}
};

struct SubMatrix : public CSRMatrix
{
	__device__ SubMatrix(uint32_t* const _col_idx, uint32_t* const _row_ptr, VAL_TYPE* const _val, const int _rows)
	{
		col_idx = _col_idx, row_ptr = _row_ptr, val = _val, rows = _rows;
		col_idx_size = row_ptr_size = val_size = 0;
	}
};

struct DenseMatrix : public Matrix
{
public:
	VAL_TYPE* data = nullptr;

	size_t data_size;

public:
	DenseMatrix(const char* filepath, AllocTarget Target)
	{
		_Target = Target;
		FILE* f;
		f = fopen(filepath, "r");
		if (f == NULL) {
			throw std::runtime_error("Failed to open file");
		}

		MM_typecode matcode;
		if (mm_read_banner(f, &matcode) != 0) {
			throw std::runtime_error("Couldn't parse matrix");
		}

		if (!mm_is_dense(matcode)) {
			throw std::runtime_error("Matrix is non-dense -> should be dense");
		}

		if (mm_read_mtx_array_size(f, &rows, &cols) != 0) {
			throw std::runtime_error("Failed to read matrix size");
		}

		data_size = (rows * cols) * sizeof(VAL_TYPE);

		allocate_memory();

		for (int i = 0; i < rows; ++i) {
			VAL_TYPE e;
			fscanf(f, "%lg\n", &e);

			data[i] = e;
		}

		fclose(f);
	}

	DenseMatrix(const DenseMatrix& src, AllocTarget Target) :
		data_size(src.data_size)
	{
		rows = src.rows;
		cols = src.cols;
		_Target = Target;
		allocate_memory();

		if (src._Target == AllocTarget::Host && _Target == AllocTarget::Host) {  // Host to Host
			std::memcpy(data, src.data, data_size);
		} else if (src._Target == AllocTarget::Host && _Target == AllocTarget::Device) {  // Host to Dev
			CHECK_CUDA(cudaMemcpy(data, src.data, data_size, cudaMemcpyHostToDevice));
		} else if (src._Target == AllocTarget::Device && _Target == AllocTarget::Host) {  // Dev to host
			CHECK_CUDA(cudaMemcpy(data, src.data, data_size, cudaMemcpyDeviceToHost));
		} else {
			CHECK_CUDA(cudaMemcpy(data, src.data, data_size, cudaMemcpyDeviceToDevice));
		}
	}

	~DenseMatrix()
	{
		if (_Target == AllocTarget::Host)
			free(data);
		else {
			CHECK_CUDA(cudaFree(data));
		}
	}

private:
	virtual void allocate_memory() override
	{
		if (_Target == AllocTarget::Host) {
			data = (VAL_TYPE*)malloc((rows * cols) * sizeof(VAL_TYPE));
			if (!data) {
				throw std::runtime_error("Failed to allocate data");
			}
		} else if (_Target == AllocTarget::Device) {
			CHECK_CUDA(cudaMalloc(&data, data_size));
		}
	}
};

DenseMatrix spmv_shared(const CSRMatrix& d_A, const DenseMatrix& d_x, DenseMatrix& d_y)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((d_A.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_A.rows + BLOCK_SIZE - 1 / BLOCK_SIZE));

	spmv<<<dimGrid, dimBlock>>>(d_A.col_idx, d_A.row_ptr, d_A.val, d_A.rows, d_x.data, d_y.data);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Error check after kernel launch ~ %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	return DenseMatrix(d_y, AllocTarget::Host);
}

DenseMatrix spmv_l2_window(const CSRMatrix& d_A, const DenseMatrix& d_x, DenseMatrix& d_y)
{
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, DEVICE_ID);

	size_t size = std::min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
	cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
		size);  // cudaLimitPersistingL2CacheSize -> Global limit for all persistent data

	size_t window_size = std::min(prop.accessPolicyMaxWindowSize,
		(int)d_x.data_size);  // accessPolicyMaxWindowSize -> Per-stream limit for a single persistent region

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
	cudaDeviceSynchronize();

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
	cudaDeviceSynchronize();

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
		CSRMatrix   A("data/scircuit.mtx", AllocTarget::Host);
		DenseMatrix x("data/scircuit_b.mtx", AllocTarget::Host);

		// ================================ DEVICE ALLOCATION ================================
		CSRMatrix d_A(A, AllocTarget::Device);

		DenseMatrix d_x(x, AllocTarget::Device);

		DenseMatrix d_y(x, AllocTarget::Device);
		// ================================ KERNEL EXECUTION ================================
		DenseMatrix y_global = spmv_global(d_A, d_x, d_y);
		DenseMatrix y_l2_window = spmv_l2_window(d_A, d_x, d_y);
		DenseMatrix y_shared = spmv_shared(d_A, d_x, d_y);

		std::cout << y_global.data[0] << std::endl;
		std::cout << y_l2_window.data[0] << std::endl;

	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
}
