#ifndef MHA_KERNEL_GPU_HPP
#define MHA_KERNEL_GPU_HPP

#include "tensor.hpp"

namespace kernel {
void mha_kernel_gpu(int32_t layer_idx, int32_t num_layers, int32_t batch_size,
                    int32_t query_seq_len, int32_t kv_seq_len, tensor::Tensor& mha_output,
                    tensor::Tensor& query_tensor, tensor::Tensor& score_tensor,
                    const tensor::Tensor& key_cache_tensor,
                    const tensor::Tensor& value_cache_tensor, void* stream = nullptr);

void mha_softmax_kernel_gpu(int32_t num_heads, int32_t batch_size, tensor::Tensor& score);

}  // namespace kernel

#endif  // MHA_KERNEL_GPU_HPP
