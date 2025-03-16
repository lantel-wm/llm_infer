#ifndef SOFTMAX_KERNEL_HPP
#define SOFTMAX_KERNEL_HPP

#include "tensor.hpp"

namespace kernel {
void softmax_kernel_cpu(const tensor::Tensor& input, void* stream = nullptr);
}  // namespace kernel

#endif  // SOFTMAX_KERNEL_HPP
