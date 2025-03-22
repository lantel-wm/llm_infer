#ifndef TYPE_HPP
#define TYPE_HPP

#include <cstddef>
#include <cstdint>
#include <ostream>

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

enum class DataType : uint8_t {
  Unknown = 0,
  FP32 = 1,
  INT8 = 2,
  INT32 = 3,
};

inline std::ostream& operator<<(std::ostream& os, const DataType& dt) {
  switch (dt) {
    case DataType::Unknown:
      os << "DataType::Unknown";
      break;
    case DataType::FP32:
      os << "DataType::FP32";
      break;
    case DataType::INT8:
      os << "DataType::INT8";
      break;
    case DataType::INT32:
      os << "DataType::INT32";
      break;
    default:
      os << "DataType::Invalid";
      break;
  }
  return os;
}

inline size_t DataTypeSize(DataType data_type) {
  if (data_type == DataType::FP32) {
    return sizeof(float);
  } else if (data_type == DataType::INT8) {
    return sizeof(int8_t);
  } else if (data_type == DataType::INT32) {
    return sizeof(int32_t);
  } else {
    return 0;
  }
}

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

}  // namespace core

#endif  // TYPE_HPP