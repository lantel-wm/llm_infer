#ifndef ROPE_KERNEL_CPU_HPP
#define ROPE_KERNEL_CPU_HPP

#include "tensor.hpp"

namespace kernel {
void sin_cos_cache_calc_cpu(float rope_theta, int head_size, int max_seq_len,
                            const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                            void* stream = nullptr);

void rope_kernel_cpu(int32_t hidden_size, int32_t key_value_size, int32_t head_size,
                     const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                     const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                     const tensor::Tensor& cos_cache, void* stream = nullptr);
}  // namespace kernel
#endif  // ROPE_KERNEL_CPU_HPP
