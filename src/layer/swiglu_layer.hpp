#ifndef SWIGLU_LAYER_HPP
#define SWIGLU_LAYER_HPP

#include "layer.hpp"

namespace layer {
class SwiGLULayer : public Layer {
 public:
  explicit SwiGLULayer(core::DeviceType device_type, int32_t hidden_dim);

  core::Status check() const override;

  core::Status forward() override;

 private:
  int32_t m_hidden_size = 0;
};
}  // namespace layer

#endif  // SWIGLU_LAYER_HPP
