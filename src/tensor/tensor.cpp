#include "tensor.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cstdint>
#include <numeric>
#include "type.hpp"

namespace tensor {
/**
 * @brief Helper function to reduce dimensions to calculate the total size
 *
 * This template function multiplies all dimensions from begin to end iterators
 * to calculate the total number of elements in a tensor.
 *
 * @tparam T Iterator type for dimensions
 * @tparam Tp Initial value type
 * @param begin Begin iterator for dimensions
 * @param end End iterator for dimensions
 * @param init Initial value for multiplication
 * @return Total number of elements
 */
template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init) {
  if (begin >= end) {
    return 0;
  }
  size_t size = std::accumulate(begin, end, init, std::multiplies<>());
  return size;
}

/**
 * @brief Gets the size in bytes for a given data type
 *
 * This function returns the number of bytes required to store a single
 * element of the specified data type.
 *
 * @param data_type The data type to get the size for
 * @return Size in bytes for the data type
 */
static size_t data_type_size(core::DataType data_type) {
  switch (data_type) {
    case core::DataType::FP32: {
      return 4;
    }
    case core::DataType::INT8: {
      return 1;
    }
    case core::DataType::INT32: {
      return 4;
    }
    default: {
      LOG(FATAL) << "Unknown data type size for " << int(data_type);
      return 0;
    }
  }
}

/**
 * @brief Constructs a 1D tensor with specified data type
 *
 * This constructor creates a 1D tensor with the given data type
 * If need_alloc is true and a memory manager is provided, memory will be allocated.
 * If ptr is provided and need_alloc is false, the tensor will use the external memory.
 *
 * @param data_type The data type of the tensor elements
 * @param dim0 The dimension of the tensor
 * @param need_alloc Whether memory needs to be allocated
 * @param memory_manager Memory manager for allocation
 * @param ptr External memory pointer (if provided)
 */
Tensor::Tensor(core::DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<core::MemoryManager> memory_manager, void* ptr)
    : m_data_type(data_type) {
  m_dims.push_back(dim0);
  m_size = dim0;
  update_strides();
  if (need_alloc && memory_manager) {
    allocate(memory_manager);
  } else {
    if (ptr != nullptr) {
      CHECK(need_alloc == false)
          << "The need_alloc is is true when ptr parameter is not a null pointer.";
      init_buffer(memory_manager, m_data_type, need_alloc, ptr);
    }
  }
}

/**
 * @brief Constructs a 2D tensor with specified data type
 *
 * This constructor creates a 2D tensor with the given data type and dimensions.
 * If need_alloc is true and a memory manager is provided, memory will be allocated.
 * If ptr is provided and need_alloc is false, the tensor will use the external memory.
 *
 * @param data_type The data type of the tensor elements
 * @param dim0 The first dimension of the tensor
 * @param dim1 The second dimension of the tensor
 * @param need_alloc Whether memory needs to be allocated
 * @param memory_manager Memory manager for allocation
 * @param ptr External memory pointer (if provided)
 */
Tensor::Tensor(core::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
               std::shared_ptr<core::MemoryManager> memory_manager, void* ptr)
    : m_data_type(data_type) {
  m_dims.push_back(dim0);
  m_dims.push_back(dim1);
  m_size = dim0 * dim1;
  update_strides();
  if (need_alloc && memory_manager) {
    allocate(memory_manager);
  } else {
    init_buffer(memory_manager, m_data_type, need_alloc, ptr);
  }
}

/**
 * @brief Constructs a 3D tensor with specified data type
 *
 * This constructor creates a 3D tensor with the given data type and dimensions.
 * If need_alloc is true and a memory manager is provided, memory will be allocated.
 * If ptr is provided and need_alloc is false, the tensor will use the external memory.
 *
 * @param data_type The data type of the tensor elements
 * @param dim0 The first dimension of the tensor
 * @param dim1 The second dimension of the tensor
 * @param dim2 The third dimension of the tensor
 * @param need_alloc Whether memory needs to be allocated
 * @param memory_manager Memory manager for allocation
 * @param ptr External memory pointer (if provided)
 */
Tensor::Tensor(core::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
               std::shared_ptr<core::MemoryManager> memeory_manager, void* ptr)
    : m_data_type(data_type) {
  m_dims.push_back(dim0);
  m_dims.push_back(dim1);
  m_dims.push_back(dim2);
  m_size = dim0 * dim1 * dim2;
  update_strides();
  if (need_alloc && memeory_manager) {
    allocate(memeory_manager);
  } else {
    init_buffer(memeory_manager, m_data_type, need_alloc, ptr);
  }
}

/**
 * @brief Constructs a 4D tensor with specified data type
 *
 * This constructor creates a 4D tensor with the given data type and dimensions.
 * If need_alloc is true and a memory manager is provided, memory will be allocated.
 * If ptr is provided and need_alloc is false, the tensor will use the external memory.
 *
 * @param data_type The data type of the tensor elements
 * @param dim0 The first dimension of the tensor
 * @param dim1 The second dimension of the tensor
 * @param dim2 The third dimension of the tensor
 * @param dim3 The fourth dimension of the tensor
 * @param need_alloc Whether memory needs to be allocated
 * @param memory_manager Memory manager for allocation
 * @param ptr External memory pointer (if provided)
 */
Tensor::Tensor(core::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
               bool need_alloc, std::shared_ptr<core::MemoryManager> memory_manager, void* ptr)
    : m_data_type(data_type) {
  m_dims.push_back(dim0);
  m_dims.push_back(dim1);
  m_dims.push_back(dim2);
  m_dims.push_back(dim3);
  m_size = dim0 * dim1 * dim2 * dim3;
  update_strides();
  if (need_alloc && memory_manager) {
    allocate(memory_manager);
  } else {
    init_buffer(memory_manager, m_data_type, need_alloc, ptr);
  }
}

/**
 * @brief Constructs a tensor with arbitrary dimensions and specified data type
 *
 * This constructor creates a tensor with the given data type and arbitrary dimensions.
 * If need_alloc is true and a memory manager is provided, memory will be allocated.
 * If ptr is provided and need_alloc is false, the tensor will use the external memory.
 *
 * @param data_type The data type of the tensor elements
 * @param dims Vector containing the dimensions of the tensor
 * @param need_alloc Whether memory needs to be allocated
 * @param memory_manager Memory manager for allocation
 * @param ptr External memory pointer (if provided)
 */
Tensor::Tensor(core::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
               std::shared_ptr<core::MemoryManager> memory_manager, void* ptr)
    : m_dims(std::move(dims)), m_data_type(data_type) {
  m_size = reduce_dimension(m_dims.begin(), m_dims.end(), 1);
  update_strides();
  if (need_alloc && memory_manager) {
    allocate(memory_manager);
  } else {
    init_buffer(memory_manager, m_data_type, need_alloc, ptr);
  }
}

/**
 * @brief Constructs a scalar tensor with specified data type
 *
 * This constructor creates a scalar tensor (rank-0 tensor) with the given data type.
 * A scalar tensor has exactly one element and no dimensions.
 * If need_alloc is true and a memory manager is provided, memory will be allocated.
 * If ptr is provided and need_alloc is false, the tensor will use the external memory.
 *
 * @param data_type The data type of the scalar value
 * @param need_alloc Whether memory needs to be allocated
 * @param memory_manager Memory manager for allocation
 * @param ptr External memory pointer (if provided)
 */
Tensor::Tensor(core::DataType data_type, bool need_alloc,
               std::shared_ptr<core::MemoryManager> memory_manager, void* ptr)
    : m_data_type(data_type) {
  // A scalar has size 1 but no dimensions
  m_size = 1;
  update_strides();
  if (need_alloc && memory_manager) {
    allocate(memory_manager);
  } else {
    if (ptr != nullptr) {
      CHECK(need_alloc == false)
          << "The need_alloc is true when ptr parameter is not a null pointer.";
      init_buffer(memory_manager, m_data_type, need_alloc, ptr);
    }
  }
}

/**
 * @brief Transfers tensor data from CPU to GPU memory
 *
 * This method transfers the tensor data from CPU memory to GPU memory using
 * the specified CUDA stream for asynchronous operations.
 *
 * @param stream CUDA stream for asynchronous memory operations
 */
void Tensor::to_cuda(cudaStream_t stream) {
  CHECK_NE(m_buffer, nullptr);
  const core::DeviceType device_type = this->device_type();
  if (device_type == core::DeviceType::Unknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
  } else if (device_type == core::DeviceType::CPU) {
    size_t byte_size = this->byte_size();
    auto gpu_manager = core::GPUMemoryManagerFactory::get_instance();
    auto gpu_buffer = std::make_shared<core::Buffer>(byte_size, gpu_manager);
    gpu_manager->memcpy(m_buffer->ptr(), gpu_buffer->ptr(), byte_size,
                        core::MemcpyKind::MemcpyCPU2GPU, stream);
    this->m_buffer = gpu_buffer;
  } else {
    LOG(INFO) << "The device type of the tensor is already cuda.";
  }
}

/**
 * @brief Transfers tensor data from GPU to CPU memory
 *
 * This method transfers the tensor data from GPU memory to CPU memory.
 */
void Tensor::to_cpu() {
  CHECK_NE(m_buffer, nullptr);
  const core::DeviceType device_type = this->device_type();

  if (device_type == core::DeviceType::Unknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
  } else if (device_type == core::DeviceType::GPU) {
    size_t byte_size = this->byte_size();
    auto cpu_manager = core::CPUMemoryManagerFactory::get_instance();
    auto cpu_buffer = std::make_shared<core::Buffer>(byte_size, cpu_manager);
    cpu_manager->memcpy(m_buffer->ptr(), cpu_buffer->ptr(), byte_size,
                        core::MemcpyKind::MemcpyGPU2CPU);
    this->m_buffer = cpu_buffer;
  } else {
    LOG(INFO) << "The device type of the tensor is already cpu.";
  }
}

/**
 * @brief Gets the total number of elements in the tensor
 *
 * @return Total number of elements in the tensor
 */
size_t Tensor::size() const { return this->m_size; }

/**
 * @brief Gets the dimension size at the specified index
 *
 * @param idx Index of the dimension to retrieve
 * @return Size of the dimension at the specified index
 */
int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->m_dims.size());
  return this->m_dims.at(idx);
}

/**
 * @brief Gets the device type where the tensor data is stored
 *
 * @return Device type (CPU, GPU, or Unknown) where the tensor is stored
 */
core::DeviceType Tensor::device_type() const {
  if (!m_buffer) {
    return core::DeviceType::Unknown;
  }
  return m_buffer->device_type();
}

/**
 * @brief Assigns a new buffer to the tensor
 *
 * This method replaces the current buffer with the provided buffer.
 * The buffer must be large enough to hold the tensor data.
 *
 * @param buffer New buffer to assign to the tensor
 * @return True if assignment was successful, false otherwise
 */
bool Tensor::assign(std::shared_ptr<core::Buffer> buffer) {
  if (!buffer) {
    LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
    return false;
  }
  if (m_buffer) {
    if (m_buffer->device_type() != buffer->device_type()) {
      LOG(ERROR) << "The device type of the new buffer is different from the original one.";
    }
  }

  size_t byte_size = this->byte_size();
  if (byte_size > buffer->byte_size()) {
    LOG(ERROR) << "The size of buffer is too small for the tensor!";
    return false;
  }
  m_buffer = buffer;
  return true;
}

/**
 * @brief Allocates memory for the tensor
 *
 * This method allocates memory for the tensor using the provided memory manager.
 * If the tensor already has a buffer and it's large enough, reallocation may be skipped.
 *
 * @param memory_manager Memory manager to use for allocation
 * @param need_realloc Whether to force reallocation even if a buffer exists
 * @return True if allocation was successful, false otherwise
 */
bool Tensor::allocate(std::shared_ptr<core::MemoryManager> memory_manager, bool need_realloc) {
  if (!memory_manager) {
    LOG(ERROR) << "The memory parameter in the allocate function is null "
                  "pointer!";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (!byte_size) {
    LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
    return false;
  }

  if (m_buffer && byte_size <= m_buffer->byte_size()) {
    if (!need_realloc) {
      return true;
    }
  }

  m_buffer = std::make_shared<core::Buffer>(byte_size, memory_manager, nullptr);
  if (!m_buffer->ptr()) {
    LOG(ERROR) << "The memory allocated is a null pointer!";
    return false;
  }
  return true;
}

/**
 * @brief Gets the dimensions of the tensor
 *
 * @return Vector containing the dimensions of the tensor
 */
const std::vector<int32_t>& Tensor::dims() const { return this->m_dims; }

/**
 * @brief Sets the device type for the tensor's buffer
 *
 * @param device_type New device type to set
 */
void Tensor::set_device_type(core::DeviceType device_type) const {
  if (m_buffer) {
    m_buffer->set_device_type(device_type);
  }
}

/**
 * @brief Resets the tensor with new data type and dimensions
 *
 * This method changes the tensor's data type and dimensions,
 * and clears the buffer.
 *
 * @param data_type New data type for the tensor
 * @param dims New dimensions for the tensor
 */
void Tensor::reset(core::DataType data_type, const std::vector<int32_t>& dims) {
  this->m_data_type = data_type;
  this->m_dims = dims;
  this->m_size = reduce_dimension(dims.begin(), dims.end(), 1);
  this->m_buffer = nullptr;
  update_strides();
}

/**
 * @brief Gets the number of dimensions in the tensor
 *
 * @return Number of dimensions in the tensor
 */
int32_t Tensor::dims_size() const { return static_cast<int32_t>(m_dims.size()); }

/**
 * @brief Gets the data type of the tensor
 *
 * @return Data type of the tensor
 */
core::DataType Tensor::data_type() const { return m_data_type; }

/**
 * @brief Reshapes the tensor to new dimensions
 *
 * This method changes the tensor's dimensions while preserving the data.
 * If the new shape requires more memory, a new buffer is allocated.
 *
 * @param dims New dimensions for the tensor
 */
void Tensor::reshape(const std::vector<int32_t>& dims) {
  size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
  if (!m_buffer) {
    this->m_dims = dims;
    this->m_size = size;
    return;
  }

  if (size > m_size) {
    auto new_buffer = std::make_shared<core::Buffer>(size * core::DataTypeSize(this->m_data_type),
                                                     m_buffer->memory_manager());
    CHECK(new_buffer->allocate());
    new_buffer->copy_from(m_buffer.get());
    this->m_buffer = new_buffer;
  }
  this->m_dims = dims;
  this->m_size = size;
  update_strides();
}

/**
 * @brief Gets the buffer storing the tensor data
 *
 * @return Shared pointer to the buffer
 */
std::shared_ptr<core::Buffer> Tensor::get_buffer() const { return m_buffer; }

/**
 * @brief Creates a deep copy of the tensor
 *
 * This method creates a new tensor with the same data type, dimensions,
 * and a copy of the data.
 *
 * @return New tensor that is a deep copy of this tensor
 */
Tensor Tensor::clone() const {
  Tensor new_tensor = *this;
  size_t byte_size = this->byte_size();

  auto allocator = m_buffer->memory_manager();
  new_tensor.m_buffer = std::make_shared<core::Buffer>(byte_size, allocator);
  new_tensor.m_buffer->copy_from(m_buffer.get());
  return new_tensor;
}

/**
 * @brief Gets the total size in bytes of the tensor
 *
 * @return Total size in bytes occupied by the tensor
 */
size_t Tensor::byte_size() const { return this->size() * DataTypeSize(m_data_type); }

/**
 * @brief Gets the strides for each dimension of the tensor
 *
 * Strides represent the number of elements to skip to move to the next position
 * in each dimension. For a tensor with dimensions [d0, d1, d2, ..., dn], the stride
 * for dimension i is the product of all dimensions after it.
 *
 * The strides are used for efficient indexing into the tensor's linear memory.
 * Call update_strides() to recalculate if the tensor dimensions change.
 *
 * @return Vector containing the current strides for each dimension
 */
std::vector<size_t> Tensor::strides() const { return m_strides; }

/**
 * @brief Updates the strides for each dimension of the tensor
 *
 * This method calculates and updates the internal stride values used for
 * indexing into the tensor. Strides represent the number of elements to skip
 * to move to the next position in each dimension.
 *
 * For a tensor with dimensions [d0, d1, d2, ..., dn], the stride for dimension i
 * is the product of all dimensions after it: stride[i] = d[i+1] * d[i+2] * ... * d[n].
 * The last dimension always has a stride of 1.
 *
 * If the tensor has no dimensions, the strides vector will be empty.
 */
void Tensor::update_strides() {
  m_strides.clear();
  if (!m_dims.empty()) {
    for (int32_t i = 0; i < m_dims.size() - 1; ++i) {
      size_t stride = reduce_dimension(m_dims.begin() + i + 1, m_dims.end(), 1);
      m_strides.push_back(stride);
    }
    m_strides.push_back(1);
  }
}

/**
 * @brief Checks if the tensor is empty
 *
 * A tensor is considered empty if it has zero size, no buffer,
 * or the buffer pointer is null.
 *
 * @return True if the tensor is empty, false otherwise
 */
bool Tensor::is_empty() const {
  return m_size == 0 || m_buffer == nullptr || m_buffer->ptr() == nullptr;
}

/**
 * @brief Initializes the tensor buffer
 *
 * This method initializes the tensor buffer with either a provided pointer or
 * by allocating new memory.
 *
 * @param memory_manager Memory manager to use for allocation
 * @param data_type Data type of the tensor elements
 * @param need_alloc Whether memory needs to be allocated
 * @param ptr External memory pointer (if provided)
 */
void Tensor::init_buffer(std::shared_ptr<core::MemoryManager> memory_manager,
                         core::DataType data_type, bool need_alloc, void* ptr) {
  if (!memory_manager && !need_alloc) {
    std::shared_ptr<core::Buffer> buffer =
        std::make_shared<core::Buffer>(data_type_size(data_type) * m_size, nullptr, ptr, true);
    this->m_buffer = buffer;
  } else {
    allocate(memory_manager, true);
  }
}
}  // namespace tensor