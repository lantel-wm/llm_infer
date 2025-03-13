#ifndef MEMORY_MANAGER_HPP
#define MEMORY_MANAGER_HPP

#include <glog/logging.h>
#include <cstddef>
#include <map>
#include <memory>
#include <vector>
#include "type.hpp"

namespace core {
enum class MemcpyKind : int {
  MemcpyCPU2CPU = 0,
  MemcpyCPU2GPU = 1,
  MemcpyGPU2CPU = 2,
  MemcpyGPU2GPU = 3,
};

class MemoryManager {
 public:
  explicit MemoryManager(DeviceType device_type) : m_device_type(device_type) {}

  virtual DeviceType device_type() const { return m_device_type; }

  virtual void deallocate(void* ptr) const = 0;

  virtual void* allocate(size_t size) const = 0;

  virtual void memcpy(const void* src, void* dst, size_t count, MemcpyKind kind,
                      void* stream = nullptr, bool sync = false) const;

  virtual void memset0(void* ptr, size_t size, void* stream, bool sync = false);

 private:
  DeviceType m_device_type = DeviceType::Unknown;
};

class CPUMemoryManager : public MemoryManager {
 public:
  explicit CPUMemoryManager();

  void* allocate(size_t size) const override;

  void deallocate(void* ptr) const override;
};

struct GPUMemoryBuffer {
  void* data;
  size_t size;
  bool busy;

  GPUMemoryBuffer() = default;

  GPUMemoryBuffer(void* data, size_t size, bool busy) : data(data), size(size), busy(busy) {}
};

class GPUMemoryManager : public MemoryManager {
 public:
  explicit GPUMemoryManager();

  void* allocate(size_t byte_size) const override;

  void deallocate(void* ptr) const override;

 private:
  mutable std::map<int, size_t> m_fragments_size;
  mutable std::map<int, std::vector<GPUMemoryBuffer>> m_big_buffers_map;
  mutable std::map<int, std::vector<GPUMemoryBuffer>> m_cuda_buffers_map;
};

class CPUMemoryManagerFactory {
 public:
  static std::shared_ptr<CPUMemoryManager> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUMemoryManager>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUMemoryManager> instance;
};

class GPUMemoryManagerFactory {
 public:
  static std::shared_ptr<GPUMemoryManager> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<GPUMemoryManager>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<GPUMemoryManager> instance;
};

}  // namespace core

#endif  // MEMORY_MANAGER_HPP