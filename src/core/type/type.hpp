#ifndef TYPE_HPP
#define TYPE_HPP

#include <cstdint>

namespace core {

enum class DeviceType : uint8_t {
  Unknown = 0,
  CPU = 1,
  GPU = 2,
};

}  // namespace core

#endif  // TYPE_HPP