#ifndef SWIGLU_KERNEL_CPU_HPP
#define SWIGLU_KERNEL_CPU_HPP

#include "tensor.hpp"

namespace kernel {
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream);
}  // namespace kernel
#endif  // SWIGLU_KERNEL_CPU_HPP
