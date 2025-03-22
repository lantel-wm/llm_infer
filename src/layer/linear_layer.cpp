#include "linear_layer.hpp"
#include <cstdint>
#include "kernel.hpp"
#include "status.hpp"
#include "tensor.hpp"

namespace layer {
LinearLayer::LinearLayer(core::DeviceType device_type, int32_t in_features, int32_t out_features,
                         bool is_quant_layer)
    : LayerParam(device_type, core::LayerType::Matmul, is_quant_layer, "Matmul"),
      m_in_features(in_features),
      m_out_features(out_features) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
  m_bias.resize(1);
}

core::Status LinearLayer::check() const {
  auto status = check_tensor_with_dim(get_input(0), m_device_type, m_data_type, -1, m_in_features);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the linear layer.";
    return status;
  }

  if (!m_quant_layer) {
    status = check_tensor_with_dim(get_weight(0), m_device_type, m_data_type, m_in_features,
                                   m_out_features);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the linear layer.";
      return status;
    }
  } else {
    return core::error::FunctionNotImplement("int8 quant is not implemented");
    // status =
    //     check_tensor_with_dim(get_weight(0), m_device_type, core::DataType::INT8, m_dim0,
    //     m_dim1);
    // if (!status) {
    //   LOG(ERROR) << "The weight tensor error in the linear layer.";
    //   return status;
    // }
  }

  if (m_quant_layer) {
    return core::error::FunctionNotImplement("int8 quant is not implemented");
    // status = check_tensor_with_dim(m_scales, m_device_type, core::DataType::FP32,
    // m_scales.size()); if (!status) {
    //   LOG(ERROR) << "The scale tensor error in the linear layer.";
    //   return status;
    // }
  }

  status = check_tensor_with_dim(get_output(0), m_device_type, m_data_type, -1, m_out_features);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the linear layer.";
    return status;
  }

  status = check_tensor_with_dim(get_bias(0), m_device_type, m_data_type, m_out_features);
  if (!status) {
    LOG(ERROR) << "The bias tensor error in the linear layer.";
    return status;
  }

  // check first dim of input and output
  int32_t input_dim0 = get_input(0).get_dim(0);
  int32_t output_dim0 = get_output(0).get_dim(0);
  if (input_dim0 != output_dim0) {
    LOG(ERROR) << "The first dim of input and output don't match";
    return core::error::InvalidArgument("The first dimension of input tensor (" +
                                        std::to_string(input_dim0) +
                                        ") does not match the first dimension of output tensor (" +
                                        std::to_string(output_dim0) + ")");
  }

  return core::error::Success();
}

core::Status LinearLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  if (m_device_type == core::DeviceType::GPU) {
    CHECK(m_cuda_config != nullptr);
  }
  if (m_quant_layer) {
    return core::error::FunctionNotImplement("int8 quant is not implemented");
    // kernel::get_matmul_kernel_quant8(m_device_type)(get_input(0), get_weight(0), get_output(0),
    //                                                 m_group_size, m_scales,
    //                                                 m_cuda_config ? m_cuda_config.get() :
    //                                                 nullptr);
  } else {
    kernel::get_matmul_kernel(m_device_type)(get_input(0), get_weight(0), get_output(0),
                                             get_bias(0), 1.f,
                                             m_cuda_config ? m_cuda_config->stream : nullptr);
  }

  return core::error::Success();
}

core::Status LinearLayer::set_bias(int32_t idx, const tensor::Tensor& bias) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_bias.size());

  if (!m_quant_layer) {
    m_bias.at(idx) = bias;
  } else {
    return core::error::FunctionNotImplement("int8 quant is not implemented");
    // // is quant layer
    // tensor::Tensor bias(core::DataType::INT8, dim);
    // bias.set_device_type(device_type);
    // CHECK(bias.assign(buffer));
    // bias_.at(idx) = bias;

    // const int32_t bias_size = static_cast<int32_t>(bias.size());
    // CHECK(bias_size % m_group_size == 0);

    // int32_t scale_nums = bias_size / m_group_size;
    // m_scales = tensor::Tensor{core::DataType::INT8, scale_nums, false, nullptr,
    //                           reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size)};
    // m_scales.set_device_type(device_type);
  }

  return core::error::Success();
}

tensor::Tensor& LinearLayer::get_bias(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_bias.size());

  return m_bias.at(idx);
}

const tensor::Tensor& LinearLayer::get_bias(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_bias.size());
  return m_bias.at(idx);
}

void LinearLayer::to_cuda() {
  LayerParam::to_cuda();
  for (auto& bias : m_bias) {
    bias.to_cuda(m_cuda_config ? m_cuda_config->stream : nullptr);
  }
}

}  // namespace layer