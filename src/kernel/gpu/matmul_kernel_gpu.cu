#include <cub/block/block_reduce.cuh>
#include "matmul_kernel_gpu.cuh"

namespace kernel {

/**
 * @brief CUDA kernel for matrix multiplication of float arrays
 *
 * This GPU kernel performs matrix multiplication between two matrices:
 * C = scale * (A * B)
 * Each thread computes one element of the output matrix.
 *
 * @param input Input matrix A of size M×K (row-major)
 * @param weight Weight matrix B of size K×N (row-major)
 * @param output Output matrix C of size M×N (row-major)
 * @param scale Scaling factor applied to the result
 * @param M Number of rows in A and C
 * @param K Number of columns in A and rows in B
 * @param N Number of columns in B and C
 */
__global__ void matmul_kernel_gpu_fp32(const float* input, const float* weight, float* output,
                                       float scale, int M, int K, int N) {
  // Calculate global row and column indices
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if thread is within bounds
  if (row < M && col < N) {
    float sum = 0.0f;

    // Compute dot product for this element
    for (int k = 0; k < K; k++) {
      sum += input[row * K + k] * weight[k * N + col];
    }

    // Write result to output
    output[row * N + col] = scale * sum;
  }
}

/**
 * @brief Performs matrix multiplication between input and weight tensors on GPU
 *
 * This function launches a CUDA kernel to multiply the input tensor by the weight tensor
 * and applies an optional scaling factor. It handles different input tensor dimensions
 * (1D or 2D) and configures the grid and block dimensions for efficient execution.
 *
 * @param input Input tensor (1D or 2D)
 * @param weight Weight tensor (must be 2D)
 * @param output Output tensor to store the result
 * @param scale Scaling factor applied to the multiplication result
 * @param stream Optional CUDA stream for asynchronous execution
 *
 * @note If input is 1D, it's treated as a single row. If 2D, each row is processed independently.
 *       The function checks dimension compatibility: input.dim1 must equal weight.dim0
 */
void matmul_kernel_gpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == core::DeviceType::GPU &&
        weight.device_type() == core::DeviceType::GPU &&
        output.device_type() == core::DeviceType::GPU);

  CHECK(input.dims_size() == 1 || input.dims_size() == 2);
  CHECK(weight.dims_size() == 2);

  int in_dim0 = 1;
  int in_dim1 = 1;

  if (input.dims_size() == 2) {
    in_dim0 = input.get_dim(0);
    in_dim1 = input.get_dim(1);
  } else {
    in_dim1 = input.get_dim(0);
  }
  // Extract weight dimensions
  int wei_dim0 = weight.get_dim(0);
  int wei_dim1 = weight.get_dim(1);

  CHECK_EQ(in_dim1, wei_dim0);
  CHECK_EQ(output.size(), in_dim0 * wei_dim1);

  const int M = in_dim0;
  const int K = in_dim1;
  const int N = wei_dim1;

  // Define block and grid dimensions
  dim3 block_dim(16, 16);
  dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (M + block_dim.y - 1) / block_dim.y);

  if (stream) {
    auto stream_ = static_cast<cudaStream_t>(stream);
    matmul_kernel_gpu_fp32<<<grid_dim, block_dim, 0, stream_>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), scale, M,
        K, N);

  } else {
    matmul_kernel_gpu_fp32<<<grid_dim, block_dim>>>(input.ptr<float>(), weight.ptr<float>(),
                                                    const_cast<float*>(output.ptr<float>()), scale,
                                                    M, K, N);
  }
}

}  // namespace kernel