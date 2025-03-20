#include "layer.hpp"
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>

namespace op {
/**
 * @brief Constructs a base layer with specified parameters
 *
 * @param device_type Type of device (CPU/GPU) where the layer will operate
 * @param layer_type Type of the layer
 * @param data_type Data type used by the layer
 * @param layer_name Name of the layer
 */
LayerBase::LayerBase(core::DeviceType device_type, core::LayerType layer_type,
                     core::DataType data_type, std::string layer_name)
    : m_device_type(device_type),
      m_layer_type(layer_type),
      m_data_type(data_type),
      m_layer_name(std::move(layer_name)) {}

/**
 * @brief Gets the data type used by the layer
 *
 * @return Data type of the layer
 */
core::DataType LayerBase::data_type() const { return m_data_type; }

/**
 * @brief Gets the type of the layer
 *
 * @return Layer type
 */
core::LayerType LayerBase::layer_type() const { return m_layer_type; }

/**
 * @brief Sets a weight tensor for the layer
 *
 * @param idx Index of the weight to set
 * @param weight Weight tensor to set
 * @return Status indicating success or failure
 */
core::Status LayerBase::set_weight(int32_t idx, const tensor::Tensor& weight) {
  return core::error::FunctionNotImplement();
}

/**
 * @brief Sets a weight tensor using raw data
 *
 * @param idx Index of the weight to set
 * @param dims Dimensions of the weight tensor
 * @param weight_ptr Pointer to the weight data
 * @param device_type Device type where the weight data resides
 * @return Status indicating success or failure
 */
core::Status LayerBase::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                   const void* weight_ptr, core::DeviceType device_type) {
  return core::error::FunctionNotImplement();
}

/**
 * @brief Gets the name of the layer
 *
 * @return Name of the layer
 */
const std::string& LayerBase::get_layer_name() const { return m_layer_name; }

/**
 * @brief Sets the name of the layer
 *
 * @param layer_name New name for the layer
 */
void LayerBase::set_layer_name(const std::string& layer_name) { m_layer_name = layer_name; }

/**
 * @brief Gets the device type where the layer operates
 *
 * @return Device type of the layer
 */
core::DeviceType LayerBase::device_type() const { return m_device_type; }

/**
 * @brief Sets the device type for the layer
 *
 * @param device_type New device type for the layer
 */
void LayerBase::set_device_type(core::DeviceType device_type) { m_device_type = device_type; }

Layer::Layer(core::DeviceType device_type, core::LayerType layer_type, std::string layer_name)
    : LayerBase(device_type, layer_type, core::DataType::FP32, std::move(layer_name)) {}

/**
 * @brief Initializes the layer
 *
 * @return Status indicating success or failure
 */
core::Status Layer::init() { return core::error::Success(); }

/**
 * @brief Performs forward pass through the layer
 *
 * @return Status indicating success or failure
 */
core::Status Layer::forward() { return core::error::FunctionNotImplement(""); }

/**
 * @brief Checks if a tensor has the correct device type and data type
 *
 * @param tensor Tensor to check
 * @param device_type Expected device type
 * @param data_type Expected data type
 * @return Status indicating if tensor properties match expectations
 */
core::Status Layer::check_tensor(const tensor::Tensor& tensor, core::DeviceType device_type,
                                 core::DataType data_type) const {
  if (tensor.is_empty()) {
    return core::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return core::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return core::error::InvalidArgument("The tensor has a wrong data type.");
  }
  return core::error::Success();
}

/**
 * @brief Checks if a tensor has correct device type, data type, and dimensions
 *
 * @param tensor Tensor to check
 * @param device_type Expected device type
 * @param data_type Expected data type
 * @param ... Variable arguments specifying expected dimensions
 * @return Status indicating if tensor properties match expectations
 */
core::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor,
                                          core::DeviceType device_type, core::DataType data_type,
                                          ...) const {
  std::va_list args;
  if (tensor.is_empty()) {
    return core::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return core::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return core::error::InvalidArgument("The tensor has a wrong data type.");
  }

  va_start(args, data_type);
  int32_t dims = tensor.dims_size();
  for (int32_t i = 0; i < dims; ++i) {
    int32_t dim = va_arg(args, int32_t);
    if (dim != tensor.get_dim(i)) {
      return core::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
    }
  }
  va_end(args);
  return core::error::Success();
}

/**
 * @brief Sets an input tensor at the specified index
 *
 * @param idx Index of the input to set
 * @param input Input tensor to set
 */
void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_inputs.size());
  this->m_inputs.at(idx) = input;
}

/**
 * @brief Sets an output tensor at the specified index
 *
 * @param idx Index of the output to set
 * @param output Output tensor to set
 */
void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_outputs.size());
  this->m_outputs.at(idx) = output;
}

/**
 * @brief Gets a const reference to the input tensor at the specified index
 *
 * @param idx Index of the input to get
 * @return Const reference to the input tensor
 */
const tensor::Tensor& Layer::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_inputs.size());
  return m_inputs.at(idx);
}

/**
 * @brief Gets a reference to the input tensor at the specified index
 *
 * @param idx Index of the input to get
 * @return Reference to the input tensor
 */
tensor::Tensor& Layer::get_input(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_inputs.size());
  return m_inputs.at(idx);
}

/**
 * @brief Gets a reference to the output tensor at the specified index
 *
 * @param idx Index of the output to get
 * @return Reference to the output tensor
 */
tensor::Tensor& Layer::get_output(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_outputs.size());
  return m_outputs.at(idx);
}

/**
 * @brief Checks if the layer configuration is valid
 *
 * @return Status indicating if the layer configuration is valid
 */
core::Status Layer::check() const {
  return core::error::FunctionNotImplement("The check function is not implement yet");
}

/**
 * @brief Gets a const reference to the output tensor at the specified index
 *
 * @param idx Index of the output to get
 * @return Const reference to the output tensor
 */
const tensor::Tensor& Layer::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_outputs.size());
  return m_outputs.at(idx);
}

/**
 * @brief Resizes the input tensor vector
 *
 * @param size New size for the input vector
 */
void Layer::reset_input_size(size_t size) { m_inputs.resize(size); }

/**
 * @brief Resizes the output tensor vector
 *
 * @param size New size for the output vector
 */
void Layer::reset_output_size(size_t size) { m_outputs.resize(size); }

/**
 * @brief Transfers all tensors to CUDA device
 */
void Layer::to_cuda() {
  for (auto& input : m_inputs) {
    if (!input.is_empty()) {
      input.to_cuda(m_cuda_config ? m_cuda_config->stream : nullptr);
    }
  }
  for (auto& output : m_outputs) {
    if (!output.is_empty()) {
      output.to_cuda(m_cuda_config ? m_cuda_config->stream : nullptr);
    }
  }
}

/**
 * @brief Sets the CUDA configuration for the layer
 *
 * @param config CUDA configuration to set
 */
void Layer::set_cuda_config(std::shared_ptr<core::CudaConfig> config) {
  if (!config) {
    return;
  }
  this->m_cuda_config = config;
}

/**
 * @brief Gets the current CUDA configuration
 *
 * @return Current CUDA configuration
 */
std::shared_ptr<core::CudaConfig> Layer::cuda_config() const { return m_cuda_config; }

/**
 * @brief Gets the number of input tensors
 *
 * @return Number of input tensors
 */
size_t Layer::input_size() const { return m_inputs.size(); }

/**
 * @brief Gets the number of output tensors
 *
 * @return Number of output tensors
 */
size_t Layer::output_size() const { return m_outputs.size(); }

LayerParam::LayerParam(core::DeviceType device_type, core::LayerType layer_type,
                       bool is_quant_layer, std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)), m_quant_layer(is_quant_layer) {}

/**
 * @brief Sets a weight tensor for the layer
 *
 * @param idx Index of the weight to set
 * @param weight Weight tensor to set
 * @return Status indicating success or failure
 */
core::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_weights.size());
  CHECK(weight.data_type() == core::DataType::FP32);
  if (!weight.is_empty()) {
    CHECK(weight.device_type() == m_device_type);
  }
  m_weights.at(idx) = weight;
  return core::error::Success();
}

/**
 * @brief Gets a const reference to the weight tensor at the specified index
 *
 * @param idx Index of the weight to get
 * @return Const reference to the weight tensor
 */
const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_weights.size());
  return m_weights.at(idx);
}

/**
 * @brief Transfers all tensors and weights to CUDA device
 */
void LayerParam::to_cuda() {
  Layer::to_cuda();
  for (auto& weight : m_weights) {
    weight.to_cuda(m_cuda_config ? m_cuda_config->stream : nullptr);
  }
  if (!m_scales.is_empty()) {
    m_scales.to_cuda(m_cuda_config ? m_cuda_config->stream : nullptr);
  }
}

/**
 * @brief Sets a weight tensor using raw data
 *
 * @param idx Index of the weight to set
 * @param dims Dimensions of the weight tensor
 * @param weight_ptr Pointer to the weight data
 * @param device_type Device type where the weight data resides
 * @return Status indicating success or failure
 */
core::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr, core::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_weights.size());
  CHECK_NE(weight_ptr, nullptr);

  size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
  std::shared_ptr<core::Buffer> buffer =
      std::make_shared<core::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
  if (device_type != core::DeviceType::Unknown) {
    buffer->set_device_type(device_type);
  }

  if (!m_quant_layer) {
    tensor::Tensor weight(core::DataType::FP32, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    m_weights.at(idx) = weight;
  } else {
    // is quant layer
    tensor::Tensor weight(core::DataType::INT8, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    m_weights.at(idx) = weight;

    const int32_t weight_size = static_cast<int32_t>(weight.size());
    CHECK(weight_size % m_group_size == 0);

    int32_t scale_nums = weight_size / m_group_size;
    m_scales = tensor::Tensor{core::DataType::FP32, scale_nums, false, nullptr,
                              reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size)};
    m_scales.set_device_type(device_type);
  }

  return core::error::Success();
}

/**
 * @brief Sets the scale tensor for quantized layers
 *
 * @param scales Scale tensor to set
 */
void LayerParam::set_scales(const tensor::Tensor& scales) {
  CHECK(!scales.is_empty());
  this->m_scales = scales;
}

/**
 * @brief Sets the group size for quantization
 *
 * @param group_size Size of quantization groups
 */
void LayerParam::set_group_size(int32_t group_size) { this->m_group_size = group_size; }

/**
 * @brief Gets the number of scale values
 *
 * @return Number of scale values
 */
int32_t LayerParam::get_scale_num() const {
  CHECK(!m_scales.is_empty());
  return static_cast<int32_t>(m_scales.size());
}

/**
 * @brief Resizes the weight tensor vector
 *
 * @param size New size for the weight vector
 */
void LayerParam::reset_weight_size(size_t size) { m_weights.resize(size); }

/**
 * @brief Gets the number of weight tensors
 *
 * @return Number of weight tensors
 */
size_t LayerParam::weight_size() const { return m_weights.size(); }

/**
 * @brief Forward pass with one input and one output
 *
 * @param input1 Input tensor
 * @param output1 Output tensor
 * @return Status indicating success or failure
 */
core::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->forward();
}

/**
 * @brief Forward pass with two inputs and one output
 *
 * @param input1 First input tensor
 * @param input2 Second input tensor
 * @param output1 Output tensor
 * @return Status indicating success or failure
 */
core::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);

  this->set_output(0, output1);
  return this->forward();
}

/**
 * @brief Forward pass with three inputs and one output
 *
 * @param input1 First input tensor
 * @param input2 Second input tensor
 * @param input3 Third input tensor
 * @param output1 Output tensor
 * @return Status indicating success or failure
 */
core::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);

  this->set_output(0, output1);
  return this->forward();
}

/**
 * @brief Forward pass with four inputs and one output
 *
 * @param input1 First input tensor
 * @param input2 Second input tensor
 * @param input3 Third input tensor
 * @param input4 Fourth input tensor
 * @param output1 Output tensor
 * @return Status indicating success or failure
 */
core::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);

  this->set_output(0, output1);
  return this->forward();
}

/**
 * @brief Forward pass with five inputs and one output
 *
 * @param input1 First input tensor
 * @param input2 Second input tensor
 * @param input3 Third input tensor
 * @param input4 Fourth input tensor
 * @param input5 Fifth input tensor
 * @param output1 Output tensor
 * @return Status indicating success or failure
 */
core::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& input5, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_input(4, input5);

  this->set_output(0, output1);
  return this->forward();
}

/**
 * @brief Gets a reference to the weight tensor at the specified index
 *
 * @param idx Index of the weight to get
 * @return Reference to the weight tensor
 */
tensor::Tensor& LayerParam::get_weight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, m_weights.size());
  return m_weights.at(idx);
}

}  // namespace op