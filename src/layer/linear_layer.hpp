#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include "layer.hpp"

namespace layer {

class LinearLayer : public LayerParam {
 public:
  explicit LinearLayer(core::DeviceType device_type, int32_t dim0, int32_t dim1,
                       bool is_quant_layer = false);

  core::Status check() const override;

  core::Status forward() override;

  core::Status set_bias(int32_t idx, const tensor::Tensor& bias);

  tensor::Tensor& get_bias(int32_t idx);

  const tensor::Tensor& get_bias(int32_t idx) const;

  void to_cuda() override;

 private:
  int32_t m_in_features = 0;
  int32_t m_out_features = 0;
  std::vector<tensor::Tensor> m_bias;
};

}  // namespace layer
#endif  // LINEAR_LAYER_HPP
