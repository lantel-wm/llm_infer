#include "mha_layer.hpp"
#include "kernel.hpp"

namespace layer {
MultiHeadAttention::MultiHeadAttention(core::DeviceType device_type, int32_t layer_index,
                                       int32_t num_layers, int32_t batch_size, int32_t kv_mul,
                                       int32_t kv_size, int32_t query_seq_len, int32_t kv_seq_len,
                                       int32_t num_heads, int32_t head_size)
    : Layer(device_type, core::LayerType::MHA, "MultiHead"),
      m_layer_index(layer_index),
      m_num_layers(num_layers),
      m_batch_size(batch_size),
      m_kv_mul(kv_mul),
      m_kv_size(kv_size),
      m_query_seq_len(query_seq_len),
      m_kv_seq_len(kv_seq_len),
      m_num_heads(num_heads),
      m_head_size(head_size) {
  reset_input_size(5);
  reset_output_size(1);
}

core::Status MultiHeadAttention::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  tensor::Tensor& mha_output = this->get_output(0);
  tensor::Tensor& query_tensor = this->get_input(0);
  tensor::Tensor& score_tensor = this->get_input(1);
  const tensor::Tensor& key_cache_tensor = this->get_input(2);
  const tensor::Tensor& value_cache_tensor = this->get_input(3);

  if (m_device_type == core::DeviceType::GPU) {
    CHECK(m_cuda_config != nullptr);
  }
  kernel::get_mha_kernel(m_device_type)(m_layer_index, m_num_layers, m_batch_size, m_query_seq_len,
                                        m_kv_seq_len, mha_output, query_tensor, score_tensor,
                                        key_cache_tensor, value_cache_tensor,
                                        m_cuda_config ? m_cuda_config->stream : nullptr);
  return core::error::Success();
}

// void MultiHeadAttention::set_pos(int32_t pos) { this->m_pos = pos; }

void MultiHeadAttention::set_layer_idx(int32_t layer_idx) { this->m_layer_index = layer_idx; }

core::Status MultiHeadAttention::check() const {
  core::Status status;
  const int32_t input_tensor_num = 4;
  for (int32_t i = 0; i < input_tensor_num; ++i) {
    // mha score tensor
    status = check_tensor(get_input(i), m_device_type, m_data_type);
    if (!status) {
      LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the matmul layer.";
      return status;
    }
  }
  return check_tensor(get_output(0), m_device_type, m_data_type);
}

}  // namespace layer