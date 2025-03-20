#include "swiglu_kernel_gpu.cuh"

namespace kernel {

/**
 * @brief CUDA kernel for applying SwiGLU activation function
 *
 * This kernel implements the SwiGLU (Swish-Gated Linear Unit) activation function,
 * which is a variant of GLU used in many recent transformer models like Llama.
 * For each element, it computes: output = input1 * sigmoid(input1) * input2
 * where the Swish function (x * sigmoid(x)) is applied to input1 before
 * multiplying element-wise with input2.
 *
 * @param size Number of elements to process
 * @param in1 First input array for Swish activation (gate value)
 * @param in2 Second input array to be gated
 * @param out Output array to store the result
 *
 * @note The sigmoid function is implemented as 1/(1+exp(-x))
 *       Each thread processes one element of the input arrays
 */
__global__ void swiglu_kernel_gpu_fp32(int size, const float* in1, const float* in2, float* out) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  float x = in1[idx];
  float y = in2[idx];
  out[idx] = x / (1.0f + expf(-x)) * y;
}

/**
 * @brief Applies SwiGLU activation function to input tensors on GPU
 *
 * This function launches a CUDA kernel to apply the SwiGLU activation function
 * to the input tensors. SwiGLU combines the Swish activation (x * sigmoid(x))
 * with a gating mechanism, which has been shown to be effective in transformer models
 * for language modeling tasks.
 *
 * @param input1 First input tensor for Swish activation (gate value)
 * @param input2 Second input tensor to be gated
 * @param output Output tensor to store the result
 * @param stream Optional CUDA stream for asynchronous execution
 *
 * @note The function expects:
 *       - All tensors must have the same shape and size
 *       - All tensors must be on GPU
 *       - The operation is performed element-wise
 */
void swiglu_kernel_gpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);

  CHECK(input1.device_type() == core::DeviceType::GPU);
  CHECK(input2.device_type() == core::DeviceType::GPU);
  CHECK(output.device_type() == core::DeviceType::GPU);

  int32_t size = static_cast<int32_t>(input1.size());
  int32_t block_size = 128;
  int32_t grid_size = (size + block_size - 1) / block_size;
  float* in1_ptr = const_cast<float*>(input1.ptr<float>());
  float* in2_ptr = const_cast<float*>(input2.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  if (stream) {
    auto stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_gpu_fp32<<<grid_size, block_size, 0, stream_>>>(size, in1_ptr, in2_ptr, out_ptr);
  } else {
    swiglu_kernel_gpu_fp32<<<grid_size, block_size>>>(size, in1_ptr, in2_ptr, out_ptr);
  }
}
}  // namespace kernel