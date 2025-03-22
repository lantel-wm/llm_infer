#ifndef ROPE_LAYER_HPP
#define ROPE_LAYER_HPP

#include <cstdint>
#include "layer.hpp"

namespace layer {
class RoPELayer : public Layer {
 public:
  explicit RoPELayer(core::DeviceType device_type, int32_t hidden_size, int32_t kv_size,
                     int32_t head_size);

  core::Status check() const override;

  core::Status forward() override;

  void set_batch_size(int32_t batch_size);

 private:
  int32_t m_hidden_size = 0;
  int32_t m_kv_size = 0;
  int32_t m_head_size = 0;
  int32_t m_batch_size = 0;
};
}  // namespace layer
#endif  // ROPE_LAYER_HPP
