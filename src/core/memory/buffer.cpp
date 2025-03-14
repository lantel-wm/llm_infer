#include "buffer.hpp"
#include <glog/logging.h>
#include "memory_manager.hpp"

namespace core {
Buffer::Buffer(size_t size, std::shared_ptr<MemoryManager> memeory_manager, void* ptr,
               bool use_external)
    : m_size(size), m_memory_manager(memeory_manager), m_ptr(ptr), m_use_external(use_external) {
  if (m_ptr == nullptr && memeory_manager) {
    m_device_type = memeory_manager->device_type();
    m_use_external = false;
    m_ptr = memeory_manager->allocate(size);
  }
}

Buffer::~Buffer() {
  if (!m_use_external) {
    if (m_ptr && m_memory_manager) {
      m_memory_manager->deallocate(m_ptr);
      m_ptr = nullptr;
    }
  }
}

void* Buffer::ptr() { return m_ptr; }

const void* Buffer::ptr() const { return m_ptr; }

size_t Buffer::size() const { return m_size; }

bool Buffer::allocate() {
  if (m_memory_manager && m_size != 0) {
    m_use_external = false;
    m_ptr = m_memory_manager->allocate(m_size);

    if (m_ptr == nullptr) {
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

std::shared_ptr<MemoryManager> Buffer::memory_manager() const { return m_memory_manager; }

void Buffer::copy_from(const Buffer& buffer) const {
  CHECK(m_memory_manager != nullptr);
  CHECK(buffer.m_ptr != nullptr);

  size_t dest_size = m_size;
  size_t src_size = buffer.m_size;
  size_t size = src_size < dest_size ? src_size : dest_size;

  const DeviceType& buffer_device = buffer.device_type();
  const DeviceType& current_device = this->device_type();
  CHECK(buffer_device != DeviceType::Unknown && current_device != DeviceType::Unknown);

  if (buffer_device == DeviceType::CPU && current_device == DeviceType::CPU) {
    return m_memory_manager->memcpy(buffer.ptr(), this->m_ptr, size, MemcpyKind::MemcpyCPU2CPU);
  } else if (buffer_device == DeviceType::GPU && current_device == DeviceType::CPU) {
    return m_memory_manager->memcpy(buffer.ptr(), this->m_ptr, size, MemcpyKind::MemcpyGPU2CPU);
  } else if (buffer_device == DeviceType::CPU && current_device == DeviceType::GPU) {
    return m_memory_manager->memcpy(buffer.ptr(), this->m_ptr, size, MemcpyKind::MemcpyCPU2GPU);
  } else {
    return m_memory_manager->memcpy(buffer.ptr(), this->m_ptr, size, MemcpyKind::MemcpyGPU2GPU);
  }
}

void Buffer::copy_from(const Buffer* buffer) const {
  CHECK(m_memory_manager != nullptr);
  CHECK(buffer != nullptr || buffer->m_ptr != nullptr);

  size_t dest_size = m_size;
  size_t src_size = buffer->m_size;
  size_t size = src_size < dest_size ? src_size : dest_size;

  const DeviceType& buffer_device = buffer->device_type();
  const DeviceType& current_device = this->device_type();
  CHECK(buffer_device != DeviceType::Unknown && current_device != DeviceType::Unknown);

  if (buffer_device == DeviceType::CPU && current_device == DeviceType::CPU) {
    return m_memory_manager->memcpy(buffer->m_ptr, this->m_ptr, size, MemcpyKind::MemcpyCPU2CPU);
  } else if (buffer_device == DeviceType::GPU && current_device == DeviceType::CPU) {
    return m_memory_manager->memcpy(buffer->m_ptr, this->m_ptr, size, MemcpyKind::MemcpyGPU2CPU);
  } else if (buffer_device == DeviceType::CPU && current_device == DeviceType::GPU) {
    return m_memory_manager->memcpy(buffer->m_ptr, this->m_ptr, size, MemcpyKind::MemcpyCPU2GPU);
  } else {
    return m_memory_manager->memcpy(buffer->m_ptr, this->m_ptr, size, MemcpyKind::MemcpyGPU2GPU);
  }
}

DeviceType Buffer::device_type() const { return m_device_type; }

void Buffer::set_device_type(DeviceType device_type) { m_device_type = device_type; }

std::shared_ptr<Buffer> Buffer::get_shared_from_this() { return shared_from_this(); }

bool Buffer::is_external() const { return this->m_use_external; }

}  // namespace core