#include "rmsnorm_layer.hpp"
#include "kernel.hpp"

namespace layer {
RmsNormLayer::RmsNormLayer(core::DeviceType device_type, int32_t hidden_size)
    : LayerParam(device_type, core::LayerType::RMSNorm, false, "RMSNorm"),
      m_hidden_size(hidden_size) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
}

core::Status RmsNormLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  auto input = this->get_input(0);
  auto weight = this->get_weight(0);
  auto output = this->get_output(0);
  if (m_device_type == core::DeviceType::GPU) {
    CHECK(m_cuda_config != nullptr);
  }
  kernel::get_rmsnorm_kernel(m_device_type)(input, weight, output,
                                            m_cuda_config ? m_cuda_config->stream : nullptr);
  return core::error::Success();
}

core::Status RmsNormLayer::check() const {
  auto status = check_tensor_with_dim(get_input(0), m_device_type, m_data_type, -1, m_hidden_size);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the rmsnorm layer.";
    return status;
  }

  status = check_tensor_with_dim(get_weight(0), m_device_type, m_data_type, m_hidden_size);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the rmsnorm layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), m_device_type, m_data_type, -1, m_hidden_size);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the rmsnorm layer.";
    return status;
  }
  return core::error::Success();
}

}  // namespace layer