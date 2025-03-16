#include "add_kernel_gpu.cuh"

namespace kernel {

__global__ void add_kernel_gpu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  float in_val1 = in1[tid];
  float in_val2 = in2[tid];
  out[tid] = in_val1 + in_val2;
}

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
