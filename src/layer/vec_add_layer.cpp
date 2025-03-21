#include "vec_add_layer.hpp"
#include "kernel.hpp"
#include "type.hpp"

namespace layer {
VecAddLayer::VecAddLayer(core::DeviceType device_type)
    : Layer(device_type, core::LayerType::Add, "Add") {
  reset_input_size(2);
  reset_output_size(1);
}

core::Status VecAddLayer::check() const {
  tensor::Tensor input1 = this->get_input(0);
  tensor::Tensor input2 = this->get_input(1);
  int32_t size = input1.size();
  core::Status status;
  status = check_tensor_with_dim(input1, m_device_type, m_data_type, size);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the add layer.";
    return status;
  }

  status = check_tensor_with_dim(input2, m_device_type, m_data_type, size);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 error in the add layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), m_device_type, m_data_type, size);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the add layer.";
    return status;
  }
  return core::error::Success();
}

core::Status VecAddLayer::forward() {
  auto status = this->check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);
  if (m_device_type == core::DeviceType::GPU) {
    CHECK(m_cuda_config != nullptr);
  }
  kernel::get_add_kernel(m_device_type)(input1, input2, output,
                                        m_cuda_config ? m_cuda_config->stream : nullptr);
  return core::error::Success();
}

}  // namespace layer