#ifndef TYPE_HPP
#define TYPE_HPP

#include <cstdint>

namespace core {

class NoCopyable {
 protected:
  NoCopyable() = default;

  ~NoCopyable() = default;

  NoCopyable(const NoCopyable&) = delete;

  NoCopyable& operator=(const NoCopyable&) = delete;
};

enum class DeviceType : uint8_t {
  Unknown = 0,
  CPU = 1,
  GPU = 2,
};

}  // namespace core

#endif  // TYPE_HPP