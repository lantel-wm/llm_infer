#ifndef EMBEDDING_KERNEL_CPU
#define EMBEDDING_KERNEL_CPU

#include "tensor.hpp"

namespace kernel {
void embedding_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, int32_t vocab_size, void* stream = nullptr);
}  // namespace kernel
#endif  // EMBEDDING_KERNEL_CPU
