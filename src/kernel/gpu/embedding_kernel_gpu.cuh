#ifndef EMBEDDING_KERNEL_HPP
#define EMBEDDING_KERNEL_HPP

#include "tensor.hpp"

namespace kernel {
void embedding_kernel_gpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, int32_t vocab_size, void* stream = nullptr);
}

#endif  // EMBEDDING_KERNEL_HPP
