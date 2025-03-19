#include "mha_kernel_cpu.hpp"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdint>
#include "matmul_kernel_cpu.hpp"
#include "memory_manager.hpp"

namespace kernel {
void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t kv_mul,
                const tensor::Tensor& mha_out, tensor::Tensor& query_tensor,
                const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                const tensor::Tensor& value_cache_tensor, void* stream) {
  // key_cache_tensor: (batch_size, num_layers, num_kv_heads, seq_len, head_size)
  // value_cache_tensor: (batch_size, num_layers, num_kv_heads, seq_len, head_size)
  CHECK(query_tensor.dims_size() == 4);
  CHECK(key_cache_tensor.dims_size() == 5);
  CHECK(value_cache_tensor.dims_size() == 5);

  const int32_t batch_size = query_tensor.get_dim(0);
  const int32_t seq_len = query_tensor.get_dim(1);
  const int32_t num_heads = query_tensor.get_dim(2);
  const int32_t num_kv_heads = key_cache_tensor.get_dim(2);
  const int32_t head_size = query_tensor.get_dim(3);
  const int32_t hidden_size = num_heads * head_size;
  const int32_t kv_size = num_kv_heads * head_size;
  const int32_t num_layers = key_cache_tensor.get_dim(1);

  std::shared_ptr<core::MemoryManager> memory_manager;
  memory_manager = core::CPUMemoryManagerFactory::get_instance();
  float scale = 1.f / std::sqrt(static_cast<float>(head_size));

  // [batch, seq_len, num_heads, head_size] -> [batch, num_heads, seq_len, head_size]
  query_tensor.transpose<float>(1, 2);

  for (int b = 0; b < batch_size; b++) {
    for (int n = 0; n < num_heads; n++) {
      // key_cache[b, layer_idx]: (num_kv_heads, seq_len, head_size)
      int32_t kv_cache_offset =
          b * num_layers * seq_len * kv_size + layer_index * seq_len * kv_size +
          ;
      int32_t q_offset = b * num_heads
    }
  }
}
}  // namespace kernel