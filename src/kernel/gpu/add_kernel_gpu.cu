#include "add_kernel_gpu.cuh"

namespace kernel {

/**
 * @brief CUDA kernel for element-wise addition of two float arrays
 *
 * This GPU kernel performs element-wise addition of two float arrays and stores
 * the result in the output array. Each thread processes one element.
 *
 * @param size The number of elements in the arrays
 * @param in1 Pointer to the first input array
 * @param in2 Pointer to the second input array
 * @param out Pointer to the output array
 */
__global__ void add_kernel_gpu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  float in_val1 = in1[tid];
  float in_val2 = in2[tid];
  out[tid] = in_val1 + in_val2;
}

/**
 * @brief Performs element-wise addition of two tensors on GPU
 *
 * This function launches a CUDA kernel to add two input tensors element-wise
 * and stores the result in the output tensor. It configures the grid and block
 * dimensions based on the input size and optionally uses a CUDA stream.
 *
 * @param input1 First input tensor
 * @param input2 Second input tensor
 * @param output Output tensor to store the result
 * @param stream Optional CUDA stream for asynchronous execution
 *
 * @note The function expects all tensors to be non-empty and of the same size
 */
void add_kernel_gpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  int32_t block_size = 512;
  int32_t grid_size = (size + block_size - 1) / block_size;
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    add_kernel_gpu_fp32<<<grid_size, block_size, 0, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    add_kernel_gpu_fp32<<<grid_size, block_size>>>(size, input1.ptr<float>(), input2.ptr<float>(),
                                                   const_cast<float*>(output.ptr<float>()));
  }
}
}  // namespace kernel