#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cuda_runtime.h>
#include <cstdint>
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
                  std::shared_ptr<core::MemoryManager> memory_manager = nullptr,
                  void* ptr = nullptr);

  explicit Tensor(core::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                  std::shared_ptr<core::MemoryManager> memory_manager = nullptr,
                  void* ptr = nullptr);

  explicit Tensor(core::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                  bool need_alloc = false, std::shared_ptr<core::MemoryManager> alloc = nullptr,
                  void* ptr = nullptr);

  explicit Tensor(core::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                  bool need_alloc = false,
                  std::shared_ptr<core::MemoryManager> memory_manager = nullptr,
                  void* ptr = nullptr);

  explicit Tensor(core::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                  std::shared_ptr<core::MemoryManager> memory_manager = nullptr,
                  void* ptr = nullptr);

  void to_cpu();

  void to_cuda(cudaStream_t stream = nullptr);

  bool is_empty() const;

  void init_buffer(std::shared_ptr<core::MemoryManager> memory_manager, core::DataType data_type,
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

  void update_strides();

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

  template <typename T, typename... Dims>
  T& at(Dims... dims);

  template <typename T, typename... Dims>
  const T& at(Dims... dims) const;

  template <typename... Dims>
  size_t get_offset(Dims... dims) const;

  template <typename T>
  void transpose(int32_t axis0, int32_t axis1);

  tensor::Tensor clone() const;

 private:
  size_t m_size = 0;
  std::vector<int32_t> m_dims;
  std::vector<size_t> m_strides;
  std::shared_ptr<core::Buffer> m_buffer;
  core::DataType m_data_type = core::DataType::Unknown;
};

/**
 * @brief Gets a typed pointer to the tensor data
 *
 * This method returns a const typed pointer to the beginning of the tensor data.
 * Returns nullptr if the tensor has no buffer.
 *
 * @tparam T Type to cast the data pointer to
 * @return Const typed pointer to the tensor data
 */
template <typename T>
const T* Tensor::ptr() const {
  if (!m_buffer) {
    return nullptr;
  }
  return const_cast<const T*>(reinterpret_cast<T*>(m_buffer->ptr()));
}

/**
 * @brief Gets a typed pointer to the tensor data
 *
 * This method returns a typed pointer to the beginning of the tensor data.
 * Returns nullptr if the tensor has no buffer.
 *
 * @tparam T Type to cast the data pointer to
 * @return Typed pointer to the tensor data
 */
template <typename T>
T* Tensor::ptr() {
  if (!m_buffer) {
    return nullptr;
  }
  return reinterpret_cast<T*>(m_buffer->ptr());
}

/**
 * @brief Gets a typed pointer to the tensor data at the specified index
 *
 * This method returns a typed pointer to the tensor data at the specified linear index.
 *
 * @tparam T Type to cast the data pointer to
 * @param index Linear index into the tensor data
 * @return Typed pointer to the tensor data at the specified index
 */
template <typename T>
T* Tensor::ptr(int64_t index) {
  CHECK(m_buffer != nullptr && m_buffer->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return const_cast<T*>(reinterpret_cast<const T*>(m_buffer->ptr())) + index;
}

/**
 * @brief Gets a const typed pointer to the tensor data at the specified index
 *
 * This method returns a const typed pointer to the tensor data at the specified linear index.
 *
 * @tparam T Type to cast the data pointer to
 * @param index Linear index into the tensor data
 * @return Const typed pointer to the tensor data at the specified index
 */
template <typename T>
const T* Tensor::ptr(int64_t index) const {
  CHECK(m_buffer != nullptr && m_buffer->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return reinterpret_cast<const T*>(m_buffer->ptr()) + index;
}

/**
 * @brief Accesses the tensor element at the specified linear offset
 *
 * This method returns a reference to the tensor element at the specified linear offset.
 *
 * @tparam T Type of the tensor element
 * @param offset Linear offset into the tensor data
 * @return Reference to the tensor element at the specified offset
 */
template <typename T>
T& Tensor::index(int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  T& val = *(reinterpret_cast<T*>(m_buffer->ptr()) + offset);
  return val;
}

/**
 * @brief Accesses the tensor element at the specified linear offset (const version)
 *
 * This method returns a const reference to the tensor element at the specified linear offset.
 *
 * @tparam T Type of the tensor element
 * @param offset Linear offset into the tensor data
 * @return Const reference to the tensor element at the specified offset
 */
template <typename T>
const T& Tensor::index(int64_t offset) const {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  const T& val = *(reinterpret_cast<T*>(m_buffer->ptr()) + offset);
  return val;
}

/**
 * @brief Accesses the tensor element at the specified coordinates
 *
 * This variadic template method allows accessing tensor elements using multidimensional indices.
 * For example, a 3D tensor can be accessed with tensor.at<float>(i, j, k).
 *
 * @tparam T Type of the tensor element
 * @tparam Dims Types of the dimension indices (should be convertible to int32_t)
 * @param dims Indices for each dimension of the tensor
 * @return Reference to the tensor element at the specified coordinates
 */
template <typename T, typename... Dims>
T& Tensor::at(Dims... dims) {
  // Convert parameter pack to array for easier handling
  const std::array<int32_t, sizeof...(Dims)> indices{dims...};

  // Check number of dimensions matches
  CHECK_EQ(sizeof...(Dims), dims_size()) << "Number of indices doesn't match tensor dimensions";

  // Check bounds for each dimension
  for (size_t i = 0; i < sizeof...(Dims); ++i) {
    CHECK_GE(indices[i], 0) << "Index out of bounds at dimension " << i;
    CHECK_LT(indices[i], m_dims[i]) << "Index out of bounds at dimension " << i;
  }

  // Calculate offset using strides
  std::vector<size_t> stride = strides();
  int64_t offset = 0;
  for (size_t i = 0; i < sizeof...(Dims); ++i) {
    offset += indices[i] * stride[i];
  }

  return index<T>(offset);
}

/**
 * @brief Accesses the tensor element at the specified coordinates (const version)
 *
 * This variadic template method allows accessing tensor elements using multidimensional indices.
 * For example, a 3D tensor can be accessed with tensor.at<float>(i, j, k).
 *
 * @tparam T Type of the tensor element
 * @tparam Dims Types of the dimension indices (should be convertible to int32_t)
 * @param dims Indices for each dimension of the tensor
 * @return Const reference to the tensor element at the specified coordinates
 */
template <typename T, typename... Dims>
const T& Tensor::at(Dims... dims) const {
  // Convert parameter pack to array for easier handling
  const std::array<int32_t, sizeof...(Dims)> indices{dims...};

  // Check number of dimensions matches
  CHECK_EQ(sizeof...(Dims), dims_size()) << "Number of indices doesn't match tensor dimensions";

  // Check bounds for each dimension
  for (size_t i = 0; i < sizeof...(Dims); ++i) {
    CHECK_GE(indices[i], 0) << "Index out of bounds at dimension " << i;
    CHECK_LT(indices[i], m_dims[i]) << "Index out of bounds at dimension " << i;
  }

  // Calculate offset using strides
  std::vector<size_t> stride = strides();
  int64_t offset = 0;
  for (size_t i = 0; i < sizeof...(Dims); ++i) {
    offset += indices[i] * stride[i];
  }

  return index<T>(offset);
}

template <typename... Dims>
size_t Tensor::get_offset(Dims... dims) const {
  // Convert parameter pack to array for easier handling
  const std::array<int32_t, sizeof...(Dims)> indices{dims...};

  // Check number of dimensions matches
  CHECK_EQ(sizeof...(Dims), dims_size()) << "Number of indices doesn't match tensor dimensions";

  // Check bounds for each dimension
  for (size_t i = 0; i < sizeof...(Dims); ++i) {
    CHECK_GE(indices[i], 0) << "Index out of bounds at dimension " << i;
    CHECK_LT(indices[i], m_dims[i]) << "Index out of bounds at dimension " << i;
  }

  // Calculate offset using strides
  std::vector<size_t> stride = strides();
  size_t offset = 0;
  for (size_t i = 0; i < sizeof...(Dims); ++i) {
    offset += indices[i] * stride[i];
  }

  return offset;
}

/**
 * @brief Transposes the tensor by swapping two dimensions
 *
 * This method performs an in-place transpose operation by swapping the specified axes.
 * For example, transposing a matrix with dimensions [3, 4] along axes 0 and 1
 * results in a matrix with dimensions [4, 3].
 *
 * Currently only implemented for CPU tensors.
 *
 * @tparam T Type of the tensor elements
 * @param axis0 First dimension to swap
 * @param axis1 Second dimension to swap
 */
template <typename T>
void Tensor::transpose(int32_t axis0, int32_t axis1) {
  CHECK(device_type() == core::DeviceType::CPU) << "Transpose only implemented on CPU tensor";
  CHECK_GE(axis0, 0) << "axis0 must be non-negative";
  CHECK_GE(axis1, 0) << "axis1 must be non-negative";
  CHECK_LT(axis0, dims_size()) << "axis0 out of bounds";
  CHECK_LT(axis1, dims_size()) << "axis1 out of bounds";

  if (axis0 == axis1) {
    return;  // No change needed
  }

  // Swap dimensions
  std::vector<int32_t> new_dims = m_dims;
  std::swap(new_dims[axis0], new_dims[axis1]);

  // Create a new tensor with the transposed dimensions
  Tensor transposed(m_data_type, new_dims, true, m_buffer->memory_manager());

  // Get sizes for iteration
  const size_t total_size = size();
  std::vector<int32_t> indices(dims_size(), 0);
  std::vector<size_t> src_strides = strides();
  std::vector<size_t> dst_strides = transposed.strides();

  // Iterate through all elements and copy with transposed indices
  for (size_t i = 0; i < total_size; ++i) {
    // Calculate source offset
    size_t src_offset = 0;
    for (int32_t d = 0; d < dims_size(); ++d) {
      src_offset += indices[d] * src_strides[d];
    }

    // Calculate destination indices (swap the specified axes)
    std::vector<int32_t> dst_indices = indices;
    std::swap(dst_indices[axis0], dst_indices[axis1]);

    // Calculate destination offset
    size_t dst_offset = 0;
    for (int32_t d = 0; d < dims_size(); ++d) {
      dst_offset += dst_indices[d] * dst_strides[d];
    }

    // Copy the value
    transposed.index<T>(dst_offset) = index<T>(src_offset);

    // Increment indices
    for (int32_t d = dims_size() - 1; d >= 0; --d) {
      indices[d]++;
      if (indices[d] < m_dims[d]) {
        break;
      }
      indices[d] = 0;
    }
  }

  // Move transposed data back to this tensor
  m_dims = new_dims;
  update_strides();
  m_buffer->copy_from(transposed.get_buffer().get());
}

};  // namespace tensor

#endif  // TENSOR_HPP