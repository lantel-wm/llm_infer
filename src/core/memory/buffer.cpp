#include "buffer.hpp"
#include <glog/logging.h>
#include "memory_manager.hpp"

namespace core {
/**
 * @brief Constructs a buffer with specified size and memory manager
 *
 * This constructor creates a buffer with the given size and memory manager.
 * If a pointer is provided and use_external is true, the buffer will use external memory.
 * Otherwise, it will allocate new memory using the memory manager.
 *
 * @param size Size of the buffer in bytes
 * @param memory_manager Memory manager to use for allocation
 * @param ptr External memory pointer (if provided)
 * @param use_external Whether to use external memory
 */
Buffer::Buffer(size_t size, std::shared_ptr<MemoryManager> memeory_manager, void* ptr,
               bool use_external)
    : m_byte_size(size),
      m_memory_manager(memeory_manager),
      m_ptr(ptr),
      m_use_external(use_external) {
  if (m_ptr == nullptr && memeory_manager) {
    m_device_type = memeory_manager->device_type();
    m_use_external = false;
    m_ptr = memeory_manager->allocate(size);
  }
}

/**
 * @brief Destructor for Buffer class
 *
 * Deallocates memory if the buffer owns it (not using external memory)
 * and a memory manager is available.
 */
Buffer::~Buffer() {
  if (!m_use_external) {
    if (m_ptr && m_memory_manager) {
      m_memory_manager->deallocate(m_ptr);
      m_ptr = nullptr;
    }
  }
}

/**
 * @brief Gets the pointer to the buffer's memory
 *
 * @return Pointer to the buffer's memory
 */
void* Buffer::ptr() { return m_ptr; }

/**
 * @brief Gets the const pointer to the buffer's memory
 *
 * @return Const pointer to the buffer's memory
 */
const void* Buffer::ptr() const { return m_ptr; }

/**
 * @brief Gets the size of the buffer in bytes
 *
 * @return Size of the buffer in bytes
 */
size_t Buffer::byte_size() const { return m_byte_size; }

/**
 * @brief Allocates memory for the buffer
 *
 * This method allocates memory using the buffer's memory manager.
 * The allocation will only succeed if a memory manager is available
 * and the buffer size is non-zero.
 *
 * @return True if allocation was successful, false otherwise
 */
bool Buffer::allocate() {
  if (m_memory_manager && m_byte_size != 0) {
    m_use_external = false;
    m_ptr = m_memory_manager->allocate(m_byte_size);

    if (m_ptr == nullptr) {
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

/**
 * @brief Gets the memory manager associated with the buffer
 *
 * @return Shared pointer to the memory manager
 */
std::shared_ptr<MemoryManager> Buffer::memory_manager() const { return m_memory_manager; }

/**
 * @brief Copies data from another buffer to this buffer
 *
 * This method copies data from the source buffer to this buffer.
 * The copy operation handles different device types (CPU/GPU) appropriately.
 * If the source buffer is larger than this buffer, only the amount that fits
 * will be copied.
 *
 * @param buffer Source buffer to copy from
 */
void Buffer::copy_from(const Buffer& buffer) const {
  CHECK(m_memory_manager != nullptr);
  CHECK(buffer.m_ptr != nullptr);

  size_t dest_size = m_byte_size;
  size_t src_size = buffer.m_byte_size;
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

/**
 * @brief Copies data from another buffer pointer to this buffer
 *
 * This method copies data from the source buffer pointer to this buffer.
 * The copy operation handles different device types (CPU/GPU) appropriately.
 * If the source buffer is larger than this buffer, only the amount that fits
 * will be copied.
 *
 * @param buffer Pointer to source buffer to copy from
 */
void Buffer::copy_from(const Buffer* buffer) const {
  CHECK(m_memory_manager != nullptr);
  CHECK(buffer != nullptr || buffer->m_ptr != nullptr);

  size_t dest_size = m_byte_size;
  size_t src_size = buffer->m_byte_size;
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

/**
 * @brief Gets the device type where the buffer is stored
 *
 * @return Device type (CPU, GPU, or Unknown) where the buffer is stored
 */
DeviceType Buffer::device_type() const { return m_device_type; }

/**
 * @brief Sets the device type for the buffer
 *
 * @param device_type New device type to set
 */
void Buffer::set_device_type(DeviceType device_type) { m_device_type = device_type; }

/**
 * @brief Gets a shared pointer to this buffer
 *
 * @return Shared pointer to this buffer
 */
std::shared_ptr<Buffer> Buffer::get_shared_from_this() { return shared_from_this(); }

/**
 * @brief Checks if the buffer uses external memory
 *
 * @return True if the buffer uses external memory, false otherwise
 */
bool Buffer::is_external() const { return this->m_use_external; }

}  // namespace core