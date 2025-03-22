#ifndef RMSNORM_LAYER_HPP
#define RMSNORM_LAYER_HPP

#include "layer.hpp"

namespace layer {
class RmsNormLayer : public LayerParam {
 public:
  explicit RmsNormLayer(core::DeviceType device_type, int32_t hidden_size);

  core::Status check() const override;

  core::Status forward() override;

 private:
  int32_t m_hidden_size = 0;
};
}  // namespace layer
#endif  // RMSNORM_LAYER_HPP
