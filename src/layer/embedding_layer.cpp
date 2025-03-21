#include "embedding_layer.hpp"
#include "kernel.hpp"
#include "type.hpp"

namespace layer {

EmbeddingLayer::EmbeddingLayer(core::DeviceType device_type, int32_t dim, int32_t seq_len,
                               int32_t vocab_size)
    : m_dim(dim),
      m_seq_len(seq_len),
      m_vocab_size(vocab_size),
      LayerParam(device_type, core::LayerType::Embedding, false, "Embedding") {
  reset_weight_size(1);
  reset_input_size(2);
  reset_output_size(1);
}

core::Status EmbeddingLayer::check() const {
  const auto& input_tensor = get_input(0);
  const auto& token_size = get_input(1).size();
  if (token_size > input_tensor.size()) {
    return core::error::InvalidArgument("The number of input tensor is greater than seq len.");
  }

  core::Status status =
      check_tensor_with_dim(input_tensor, m_device_type, core::DataType::INT32, token_size);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the embedding layer: " << status.get_err_msg();
    return status;
  }

  status = check_tensor_with_dim(get_weight(0), m_device_type, m_data_type, m_vocab_size, m_dim);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the embedding layer: " << status.get_err_msg();
    return status;
  }

  status = check_tensor_with_dim(get_output(0), m_device_type, m_data_type, token_size, m_dim);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the embedding layer: " << status.get_err_msg();
    return status;
  }
  return core::error::Success();
}

core::Status EmbeddingLayer::forward() {
  core::Status status = check();
  if (!status) {
    return status;
  }
  if (m_device_type == core::DeviceType::GPU) {
    CHECK(m_cuda_config != nullptr);
  }
  kernel::get_embedding_kernel(m_device_type)(get_input(0), get_weight(0), get_output(0),
                                              m_vocab_size,
                                              m_cuda_config ? m_cuda_config->stream : nullptr);
  return core::StatusCode::Success;
}
}  // namespace layer