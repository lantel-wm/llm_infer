#ifndef ADD_KERNEL_CPU_HPP
#define ADD_KERNEL_CPU_HPP

#include <glog/logging.h>
#include "tensor.hpp"

namespace kernel {
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
#endif  // ADD_KERNEL_CPU_HPP