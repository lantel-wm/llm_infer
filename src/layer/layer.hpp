#ifndef LAYER_HPP
#define LAYER_HPP
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "status.hpp"
#include "tensor.hpp"
#include "type.hpp"

namespace op {

struct CudaConfig {
  cudaStream_t stream = nullptr;
  ~CudaConfig() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
};

class Layer;
enum class LayerType : uint8_t {
  Unknown = 0,
  Linear = 1,
  Encode = 2,
  Embedding = 3,
  RMSNorm = 4,
  Matmul = 5,
  RoPe = 6,
  MHA = 7,
  Softmax = 8,
  Add = 9,
  SwiGLU = 10,
};

class LayerBase {
 public:
  explicit LayerBase(core::DeviceType device_type, LayerType layer_type, core::DataType data_type,
                     std::string layer_name = "");

  core::DataType data_type() const;

  LayerType layer_type() const;

  virtual core::Status init() = 0;

  virtual core::Status forward() = 0;

  virtual core::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;

  virtual core::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& output1) = 0;

  virtual core::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;

  virtual core::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& output1) = 0;

  virtual core::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

  virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

  virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

  virtual size_t input_size() const = 0;

  virtual size_t output_size() const = 0;

  virtual core::Status check() const = 0;

  virtual tensor::Tensor& get_input(int32_t idx) = 0;

  virtual tensor::Tensor& get_output(int32_t idx) = 0;

  virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

  virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

  virtual core::Status set_weight(int32_t idx, const tensor::Tensor& weight);

  virtual core::Status set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                  const void* weight_ptr,
                                  core::DeviceType device_type = core::DeviceType::Unknown);

  const std::string& get_layer_name() const;

  void set_layer_name(const std::string& layer_name);

  core::DeviceType device_type() const;

  void set_device_type(core::DeviceType device_type);

 protected:
  std::string m_layer_name;
  LayerType m_layer_type = LayerType::Unknown;
  core::DataType m_data_type = core::DataType::Unknown;
  core::DeviceType m_device_type = core::DeviceType::Unknown;
};

class Layer : public LayerBase {
 public:
  explicit Layer(core::DeviceType device_type, LayerType layer_type, std::string layer_name = "");

  core::Status init() override;

  core::Status check_tensor(const tensor::Tensor& tensor, core::DeviceType device_type,
                            core::DataType data_type) const;

  core::Status check_tensor_with_dim(const tensor::Tensor& tensor, core::DeviceType device_type,
                                     core::DataType data_type, ...) const;

  core::Status check() const override;

  core::Status forward() override;

  core::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;

  core::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output1) override;

  core::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& output1) override;

  core::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& output1) override;

  core::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& input5, const tensor::Tensor& output1) override;

  void set_input(int32_t idx, const tensor::Tensor& input) override;

  void set_output(int32_t idx, const tensor::Tensor& output) override;

  const tensor::Tensor& get_input(int32_t idx) const override;

  const tensor::Tensor& get_output(int32_t idx) const override;

  tensor::Tensor& get_input(int32_t idx) override;

  tensor::Tensor& get_output(int32_t idx) override;

  size_t input_size() const override;

  size_t output_size() const override;

  void reset_input_size(size_t size);

  void reset_output_size(size_t size);

  virtual void to_cuda();

  void set_cuda_config(std::shared_ptr<CudaConfig> config);

  std::shared_ptr<CudaConfig> cuda_config() const;

 protected:
  std::vector<tensor::Tensor> m_inputs;
  std::vector<tensor::Tensor> m_outputs;
  std::shared_ptr<CudaConfig> m_cuda_config;
};

class LayerParam : public Layer {
 public:
  explicit LayerParam(core::DeviceType device_type, LayerType layer_type,
                      bool is_quant_layer = false, std::string layer_name = "");

  size_t weight_size() const;

  void reset_weight_size(size_t size);

  tensor::Tensor& get_weight(int32_t idx);

  const tensor::Tensor& get_weight(int32_t idx) const;

  void to_cuda() override;

  core::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;

  core::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
                          core::DeviceType device_type = core::DeviceType::Unknown) override;

  void set_scales(const tensor::Tensor& scales);

  void set_group_size(int32_t group_size);

  int32_t get_scale_num() const;

 protected:
  int32_t m_group_size = 0;
  bool m_quant_layer = false;
  tensor::Tensor m_scales;
  std::vector<tensor::Tensor> m_weights;
};
}  // namespace op
#endif  // LAYER_HPP
