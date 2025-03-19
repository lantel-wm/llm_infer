#ifndef MATMUL_KERNEL_CPU_HPP
#define MATMUL_KERNEL_CPU_HPP

#include "tensor.hpp"

namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale = 1.f, void* stream = nullptr);
}  // namespace kernel
#endif  // MATMUL_KERNEL_CPU_HPP
