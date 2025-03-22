#include "rope_layer.hpp"
#include <cmath>
#include <cstdint>
#include "kernel.hpp"

namespace layer {
RoPELayer::RoPELayer(core::DeviceType device_type, int32_t hidden_size, int32_t kv_size,
                     int32_t head_size)
    : Layer(device_type, core::LayerType::RoPe, "RoPe"),
      m_hidden_size(hidden_size),
      m_kv_size(kv_size),
      m_head_size(head_size) {
  reset_input_size(5);
  reset_output_size(1);
}

core::Status RoPELayer::forward() {
  core::Status status = check();
  if (!status) {
    return status;
  }

  tensor::Tensor input_q = this->get_input(0);
  tensor::Tensor input_k = this->get_input(1);
  tensor::Tensor input_pos = this->get_input(2);

  tensor::Tensor sin_cache = this->get_input(3);
  tensor::Tensor cos_cache = this->get_input(4);

  if (m_device_type == core::DeviceType::GPU) {
    CHECK(m_cuda_config != nullptr);
  }
  kernel::get_rope_kernel(m_device_type)(m_hidden_size, m_kv_size, m_head_size, input_q, input_k,
                                         input_pos, sin_cache, cos_cache,
                                         m_cuda_config ? m_cuda_config->stream : nullptr);
  return core::error::Success();
}

core::Status RoPELayer::check() const {
  // query
  auto status =
      check_tensor_with_dim(get_input(0), m_device_type, m_data_type, m_batch_size, m_hidden_size);
  if (!status) {
    LOG(ERROR) << "The input tensor 0 error in the rope layer.";
    return status;
  }

  // key
  status = check_tensor_with_dim(get_input(1), m_device_type, m_data_type, m_batch_size, m_kv_size);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the rope layer.";
    return status;
  }

  // pos tensor
  status = check_tensor_with_dim(get_input(2), core::DeviceType::CPU, core::DataType::INT32, -1);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 error in the rope layer.";
    return status;
  }

  return core::error::Success();
}

void RoPELayer::set_batch_size(int32_t batch_size) { m_batch_size = batch_size; }

}  // namespace layer