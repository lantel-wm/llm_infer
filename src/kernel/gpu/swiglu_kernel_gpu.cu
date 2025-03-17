#include "swiglu_kernel_gpu.cuh"

namespace kernel {

__global__ void swiglu_kernel_gpu_fp32(int size, const float* in1, const float* in2, float* out) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  float x = in1[idx];
  float y = in2[idx];
  out[idx] = x / (1.0f + expf(-x)) * y;
}

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
  if (!stream) {
    swiglu_kernel_gpu_fp32<<<grid_size, block_size>>>(size, in1_ptr, in2_ptr, out_ptr);
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_gpu_fp32<<<grid_size, block_size, 0, stream_>>>(size, in1_ptr, in2_ptr, out_ptr);
  }
}
}  // namespace kernel