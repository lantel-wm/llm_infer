#include "swiglu_layer.hpp"
#include "kernel.hpp"

namespace layer {
SwiGLULayer::SwiGLULayer(core::DeviceType device_type, int32_t hidden_dim)
    : Layer(device_type, core::LayerType::SwiGLU, "SwiGLU"), m_hidden_size(hidden_dim) {
  reset_input_size(2);
  reset_output_size(1);
}

core::Status SwiGLULayer::check() const {
  core::Status status;
  const int32_t input_tensor_num = 2;
  for (int32_t i = 0; i < input_tensor_num; ++i) {
    status = check_tensor_with_dim(get_input(0), m_device_type, m_data_type, m_hidden_size);
    if (!status) {
      LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the swiglu layer.";
      return status;
    }
  }

  status = check_tensor_with_dim(get_output(0), m_device_type, m_data_type, m_hidden_size);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the swiglu layer.";
    return status;
  }
  return core::error::Success();
}

core::Status SwiGLULayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);
  if (m_device_type == core::DeviceType::GPU) {
    CHECK(m_cuda_config != nullptr);
  }
  kernel::get_swiglu_kernel(m_device_type)(input1, input2, output,
                                           m_cuda_config ? m_cuda_config->stream : nullptr);
  return core::error::Success();
}

}  // namespace layer
