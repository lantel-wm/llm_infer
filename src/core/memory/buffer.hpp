#ifndef BUFFER_HPP
#define BUFFER_HPP

#include <memory>
#include "memory_manager.hpp"
#include "type.hpp"

namespace core {

class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 private:
  size_t m_size = 0;
  void* m_ptr = nullptr;
  bool m_use_external = false;
  DeviceType m_device_type = DeviceType::Unknown;
  std::shared_ptr<MemoryManager> m_memory_manager;

 public:
  explicit Buffer() = default;

  explicit Buffer(size_t size, std::shared_ptr<MemoryManager> memory_manager = nullptr,
                  void* ptr = nullptr, bool use_external = false);

  virtual ~Buffer();

  bool allocate();

  void copy_from(const Buffer& buffer) const;

  void copy_from(const Buffer* buffer) const;

  void* ptr();

  const void* ptr() const;

  size_t size() const;

  std::shared_ptr<MemoryManager> memory_manager() const;

  DeviceType device_type() const;

  void set_device_type(DeviceType device_type);

  std::shared_ptr<Buffer> get_shared_from_this();

  bool is_external() const;
};
}  // namespace core

#endif  // BUFFER_HPP