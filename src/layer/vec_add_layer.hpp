#ifndef VEC_ADD_LAYER_HPP
#define VEC_ADD_LAYER_HPP

#include "layer.hpp"

namespace layer {
class VecAddLayer : public Layer {
 public:
  explicit VecAddLayer(core::DeviceType device_type);

  core::Status check() const override;

  core::Status forward() override;
};
}  // namespace layer
#endif  // VEC_ADD_HPP