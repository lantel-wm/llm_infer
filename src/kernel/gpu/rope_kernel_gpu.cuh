#ifndef ROPE_KERNEL_GPU_CUH
#define ROPE_KERNEL_GPU_CUH

#include "tensor.hpp"

namespace kernel {
void rope_kernel_gpu(int32_t hidden_size, int32_t key_value_size, int32_t head_size,
                     const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                     const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                     const tensor::Tensor& cos_cache, void* stream);

void sin_cos_cache_calc_gpu(float rope_theta, int head_size, int max_seq_len,
                            const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                            void* stream);

}  // namespace kernel
#endif  // ROPE_KERNEL_GPU_CUH
