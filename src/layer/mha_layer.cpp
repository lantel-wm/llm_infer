#include "mha_layer.hpp"
#include <cstdint>
#include "kernel.hpp"
#include "status.hpp"

namespace layer {
MultiHeadAttention::MultiHeadAttention(core::DeviceType device_type, int32_t layer_index,
                                       int32_t num_layers, int32_t max_position_embeddings,
                                       int32_t kv_mul, int32_t kv_size, int32_t num_heads,
                                       int32_t head_size)
    : Layer(device_type, core::LayerType::MHA, "MultiHead"),
      m_layer_index(layer_index),
      m_num_layers(num_layers),
      m_max_position_embeddings(max_position_embeddings),
      m_kv_mul(kv_mul),
      m_kv_size(kv_size),
      m_num_heads(num_heads),
      m_head_size(head_size) {
  reset_input_size(4);
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

void MultiHeadAttention::set_layer_idx(int32_t layer_idx) { this->m_layer_index = layer_idx; }

void MultiHeadAttention::set_batch_size(int32_t batch_size) { this->m_batch_size = batch_size; }

void MultiHeadAttention::set_query_seq_len(int32_t query_seq_len) {
  this->m_query_seq_len = query_seq_len;
}

void MultiHeadAttention::set_kv_seq_len(int32_t kv_seq_len) { this->m_kv_seq_len = kv_seq_len; }

core::Status MultiHeadAttention::check() const {
  core::Status status;
  const tensor::Tensor& query_tensor = this->get_input(0);
  const tensor::Tensor& score_tensor = this->get_input(1);
  const tensor::Tensor& key_cache_tensor = this->get_input(2);
  const tensor::Tensor& value_cache_tensor = this->get_input(3);
  const tensor::Tensor& mha_output = this->get_output(0);

  // query_tensor: (batch_size, query_seq_len, num_heads, head_size)
  status = check_tensor_with_dim(query_tensor, m_device_type, m_data_type, m_batch_size,
                                 m_query_seq_len, m_num_heads, m_head_size);
  if (!status) {
    LOG(ERROR) << "The input tensor 0 (query_tensor) error in the matmul layer.";
    return status;
  }

  // score_tensor: (batch_size, num_heads, query_seq_len, kv_seq_len)
  status = check_tensor_with_dim(score_tensor, m_device_type, m_data_type, m_batch_size,
                                 m_num_heads, m_query_seq_len, m_kv_seq_len);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 (score_tensor) error in the matmul layer.";
    return status;
  }

  // key_cache_tensor: (num_layers, batch_size, num_kv_heads, max_position_embeddings, head_size)
  status = check_tensor_with_dim(key_cache_tensor, m_device_type, m_data_type, m_num_layers,
                                 m_batch_size, m_num_heads / m_kv_mul, m_max_position_embeddings,
                                 m_head_size);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 (key_cache_tensor) error in the matmul layer.";
    return status;
  }

  // value_cache_tensor: (num_layers, batch_size, num_kv_heads, max_position_embeddings, head_size)
  status = check_tensor_with_dim(value_cache_tensor, m_device_type, m_data_type, m_num_layers,
                                 m_batch_size, m_num_heads / m_kv_mul, m_max_position_embeddings,
                                 m_head_size);
  if (!status) {
    LOG(ERROR) << "The input tensor 3 (value_cache_tensor) error in the matmul layer.";
    return status;
  }

  // mha_output: (batch_size, query_seq_len, num_heads, head_size)
  status = check_tensor_with_dim(mha_output, m_device_type, m_data_type, m_batch_size,
                                 m_query_seq_len, m_num_heads, m_head_size);
  if (!status) {
    LOG(ERROR) << "The output tensor (mha_output) error in the matmul layer.";
    return status;
  }

  return core::error::Success();
}

}  // namespace layer