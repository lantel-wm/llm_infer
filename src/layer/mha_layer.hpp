#ifndef MHA_LAYER_HPP
#define MHA_LAYER_HPP

#include <cstdint>
#include "layer.hpp"

namespace layer {
class MultiHeadAttention : public Layer {
 public:
  explicit MultiHeadAttention(core::DeviceType device_type, int32_t layer_index, int32_t num_layers,
                              int32_t max_position_embeddings, int32_t kv_mul, int32_t kv_size,
                              int32_t num_heads, int32_t head_size);

  core::Status check() const override;

  void set_layer_idx(int32_t layer_idx);

  void set_batch_size(int32_t batch_size);

  void set_query_seq_len(int32_t query_seq_len);

  void set_kv_seq_len(int32_t kv_seq_len);

  core::Status forward() override;

 private:
  int32_t m_layer_index = 0;
  int32_t m_num_layers = 0;
  int32_t m_max_position_embeddings = 0;
  int32_t m_batch_size = 0;
  int32_t m_kv_mul = 0;
  int32_t m_kv_size = 0;
  int32_t m_query_seq_len = 0;
  int32_t m_kv_seq_len = 0;
  int32_t m_num_heads = 0;
  int32_t m_head_size = 0;
};
}  // namespace layer
#endif  // MHA_LAYER_HPP
