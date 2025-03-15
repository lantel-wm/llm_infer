#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "buffer.hpp"
#include "memory_manager.hpp"
#include "type.hpp"

namespace tensor {

class Tensor {
 public:
  explicit Tensor() = default;

  explicit Tensor(core::DataType data_type, int32_t dim0, bool need_alloc = false,
                  std::shared_ptr<core::MemoryManager> alloc = nullptr, void* ptr = nullptr);

  explicit Tensor(core::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                  std::shared_ptr<core::MemoryManager> alloc = nullptr, void* ptr = nullptr);

  explicit Tensor(core::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                  bool need_alloc = false, std::shared_ptr<core::MemoryManager> alloc = nullptr,
                  void* ptr = nullptr);

  explicit Tensor(core::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                  bool need_alloc = false, std::shared_ptr<core::MemoryManager> alloc = nullptr,
                  void* ptr = nullptr);

  explicit Tensor(core::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                  std::shared_ptr<core::MemoryManager> alloc = nullptr, void* ptr = nullptr);

  void to_cpu();

  void to_cuda(cudaStream_t stream = nullptr);

  bool is_empty() const;

  void init_buffer(std::shared_ptr<core::MemoryManager> alloc, core::DataType data_type,
                   bool need_alloc, void* ptr);

  template <typename T>
  T* ptr();

  template <typename T>
  const T* ptr() const;

  void reshape(const std::vector<int32_t>& dims);

  std::shared_ptr<core::Buffer> get_buffer() const;

  size_t size() const;

  size_t byte_size() const;

  int32_t dims_size() const;

  core::DataType data_type() const;

  int32_t get_dim(int32_t idx) const;

  const std::vector<int32_t>& dims() const;

  std::vector<size_t> strides() const;

  bool assign(std::shared_ptr<core::Buffer> buffer);

  void reset(core::DataType data_type, const std::vector<int32_t>& dims);

  void set_device_type(core::DeviceType device_type) const;

  core::DeviceType device_type() const;

  bool allocate(std::shared_ptr<core::MemoryManager> allocator, bool need_realloc = false);

  template <typename T>
  T* ptr(int64_t index);

  template <typename T>
  const T* ptr(int64_t index) const;

  template <typename T>
  T& index(int64_t offset);

  template <typename T>
  const T& index(int64_t offset) const;

  tensor::Tensor clone() const;

 private:
  size_t m_size = 0;
  std::vector<int32_t> m_dims;
  std::shared_ptr<core::Buffer> m_buffer;
  core::DataType m_data_type = core::DataType::Unknown;
};

template <typename T>
T& Tensor::index(int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  T& val = *(reinterpret_cast<T*>(m_buffer->ptr()) + offset);
  return val;
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  const T& val = *(reinterpret_cast<T*>(m_buffer->ptr()) + offset);
  return val;
}

template <typename T>
const T* Tensor::ptr() const {
  if (!m_buffer) {
    return nullptr;
  }
  return const_cast<const T*>(reinterpret_cast<T*>(m_buffer->ptr()));
}

template <typename T>
T* Tensor::ptr() {
  if (!m_buffer) {
    return nullptr;
  }
  return reinterpret_cast<T*>(m_buffer->ptr());
}

template <typename T>
T* Tensor::ptr(int64_t index) {
  CHECK(m_buffer != nullptr && m_buffer->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return const_cast<T*>(reinterpret_cast<const T*>(m_buffer->ptr())) + index;
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
  CHECK(m_buffer != nullptr && m_buffer->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return reinterpret_cast<const T*>(m_buffer->ptr()) + index;
}

};  // namespace tensor

#endif  // TENSOR_HPP