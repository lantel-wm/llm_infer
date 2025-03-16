#ifndef RMSNORM_KERNEL_CPU_HPP
#define RMSNORM_KERNEL_CPU_HPP

#include <glog/logging.h>
#include "tensor.hpp"

namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
#endif  // RMSNORM_KERNEL_CPU_HPP
