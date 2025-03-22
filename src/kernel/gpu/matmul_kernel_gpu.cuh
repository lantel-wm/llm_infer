#ifndef MATMUL_KERNEL_GPU_CUH
#define MATMUL_KERNEL_GPU_CUH

#include "tensor.hpp"

namespace kernel {
void matmul_kernel_gpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, const tensor::Tensor& bias, float scale = 1.f,
                       void* stream = nullptr);
}  // namespace kernel

#endif  // MATMUL_KERNEL_GPU_CUH
