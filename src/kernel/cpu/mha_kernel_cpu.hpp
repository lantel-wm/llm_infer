#ifndef MHA_KERNEL_CPU_HPP
#define MHA_KERNEL_CPU_HPP

#include "tensor.hpp"

namespace kernel {
void mha_qkT_kernel_cpu(const tensor::Tensor& query, const tensor::Tensor& key,
                        const tensor::Tensor& qkT, float scale, bool is_causal = true);

void mha_softmax_kernel_cpu(const tensor::Tensor& score);

void mha_scorev_kernel_cpu(const tensor::Tensor& score, const tensor::Tensor& key,
                           const tensor::Tensor& output);

void mha_kernel_cpu(int32_t layer_idx, int32_t num_layers, int32_t batch_size,
                    int32_t query_seq_len, int32_t kv_seq_len, tensor::Tensor& mha_output,
                    tensor::Tensor& query_tensor, tensor::Tensor& score_tensor,
                    const tensor::Tensor& key_cache_tensor,
                    const tensor::Tensor& value_cache_tensor, void* stream = nullptr);
}  // namespace kernel

#endif  // MHA_KERNEL_CPU_HPP
