#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <vector>
#include "core/memory/buffer.hpp"
#include "core/memory/memory_manager.hpp"
#include "tensor/tensor.hpp"

namespace tensor {
namespace {

// Test fixture for Tensor tests
class TensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance();
    gpu_memory_manager = core::GPUMemoryManagerFactory::get_instance();
  }

  std::shared_ptr<core::CPUMemoryManager> cpu_memory_manager;
  std::shared_ptr<core::GPUMemoryManager> gpu_memory_manager;
};

// Test default constructor
TEST_F(TensorTest, DefaultConstructor) {
  Tensor tensor;
  EXPECT_EQ(tensor.size(), 0);
  EXPECT_EQ(tensor.dims_size(), 0);
  EXPECT_EQ(tensor.data_type(), core::DataType::Unknown);
  EXPECT_TRUE(tensor.is_empty());
}

// Test 1D tensor constructor
TEST_F(TensorTest, Constructor1D) {
  const int32_t dim0 = 10;
  Tensor tensor(core::DataType::FP32, dim0, true, cpu_memory_manager);

  EXPECT_EQ(tensor.data_type(), core::DataType::FP32);
  EXPECT_EQ(tensor.size(), dim0);
  EXPECT_EQ(tensor.dims_size(), 1);
  EXPECT_EQ(tensor.get_dim(0), dim0);
  EXPECT_EQ(tensor.byte_size(), dim0 * 4);  // FP32 is 4 bytes
  EXPECT_FALSE(tensor.is_empty());
  EXPECT_EQ(tensor.device_type(), core::DeviceType::CPU);
}

// Test 2D tensor constructor
TEST_F(TensorTest, Constructor2D) {
  const int32_t dim0 = 5, dim1 = 10;
  Tensor tensor(core::DataType::FP32, dim0, dim1, true, cpu_memory_manager);

  EXPECT_EQ(tensor.data_type(), core::DataType::FP32);
  EXPECT_EQ(tensor.size(), dim0 * dim1);
  EXPECT_EQ(tensor.dims_size(), 2);
  EXPECT_EQ(tensor.get_dim(0), dim0);
  EXPECT_EQ(tensor.get_dim(1), dim1);
  EXPECT_EQ(tensor.byte_size(), dim0 * dim1 * 4);  // FP32 is 4 bytes
  EXPECT_FALSE(tensor.is_empty());
  EXPECT_EQ(tensor.device_type(), core::DeviceType::CPU);
}

// Test 3D tensor constructor
TEST_F(TensorTest, Constructor3D) {
  const int32_t dim0 = 2, dim1 = 3, dim2 = 4;
  Tensor tensor(core::DataType::INT8, dim0, dim1, dim2, true, cpu_memory_manager);

  EXPECT_EQ(tensor.data_type(), core::DataType::INT8);
  EXPECT_EQ(tensor.size(), dim0 * dim1 * dim2);
  EXPECT_EQ(tensor.dims_size(), 3);
  EXPECT_EQ(tensor.get_dim(0), dim0);
  EXPECT_EQ(tensor.get_dim(1), dim1);
  EXPECT_EQ(tensor.get_dim(2), dim2);
  EXPECT_EQ(tensor.byte_size(), dim0 * dim1 * dim2 * 1);  // INT8 is 1 byte
  EXPECT_FALSE(tensor.is_empty());
  EXPECT_EQ(tensor.device_type(), core::DeviceType::CPU);
}

// Test 4D tensor constructor
TEST_F(TensorTest, Constructor4D) {
  const int32_t dim0 = 2, dim1 = 3, dim2 = 4, dim3 = 5;
  Tensor tensor(core::DataType::INT32, dim0, dim1, dim2, dim3, true, cpu_memory_manager);

  EXPECT_EQ(tensor.data_type(), core::DataType::INT32);
  EXPECT_EQ(tensor.size(), dim0 * dim1 * dim2 * dim3);
  EXPECT_EQ(tensor.dims_size(), 4);
  EXPECT_EQ(tensor.get_dim(0), dim0);
  EXPECT_EQ(tensor.get_dim(1), dim1);
  EXPECT_EQ(tensor.get_dim(2), dim2);
  EXPECT_EQ(tensor.get_dim(3), dim3);
  EXPECT_EQ(tensor.byte_size(), dim0 * dim1 * dim2 * dim3 * 4);  // INT32 is 4 bytes
  EXPECT_FALSE(tensor.is_empty());
  EXPECT_EQ(tensor.device_type(), core::DeviceType::CPU);
}

// Test constructor with vector of dimensions
TEST_F(TensorTest, ConstructorWithDims) {
  std::vector<int32_t> dims = {2, 3, 4, 5};
  Tensor tensor(core::DataType::FP32, dims, true, cpu_memory_manager);

  EXPECT_EQ(tensor.data_type(), core::DataType::FP32);
  EXPECT_EQ(tensor.size(), 2 * 3 * 4 * 5);
  EXPECT_EQ(tensor.dims_size(), 4);
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(tensor.get_dim(i), dims[i]);
  }
  EXPECT_EQ(tensor.byte_size(), 2 * 3 * 4 * 5 * 4);  // FP32 is 4 bytes
  EXPECT_FALSE(tensor.is_empty());
}

// Test tensor with external memory
TEST_F(TensorTest, ExternalMemory1) {
  const int32_t dim0 = 10;
  const size_t byte_size = dim0 * 4;  // FP32 is 4 bytes
  float* external_data = new float[dim0];

  // Initialize external data
  for (int i = 0; i < dim0; i++) {
    external_data[i] = static_cast<float>(i);
  }

  // Create tensor with external data
  Tensor tensor(core::DataType::FP32, dim0, false, nullptr, external_data);

  // Verify tensor properties
  EXPECT_EQ(tensor.data_type(), core::DataType::FP32);
  EXPECT_EQ(tensor.size(), dim0);
  EXPECT_EQ(tensor.byte_size(), byte_size);

  // Verify tensor data
  const float* tensor_data = tensor.ptr<float>();
  for (int i = 0; i < dim0; i++) {
    EXPECT_EQ(tensor_data[i], static_cast<float>(i));
  }

  // Free external memory (tensor doesn't own it)
  delete[] external_data;
}

// Test tensor with external memory
TEST_F(TensorTest, ExternalMemory2) {
  const int32_t dim0 = 10;
  const int32_t dim1 = 8;
  const int32_t size = dim0 * dim1;
  const size_t byte_size = dim0 * dim1 * 4;  // FP32 is 4 bytes
  float* external_data = new float[dim0 * dim1];

  // Initialize external data
  for (int i = 0; i < size; i++) {
    external_data[i] = static_cast<float>(i);
  }

  // Create tensor with external data
  Tensor tensor(core::DataType::FP32, dim0, dim1, false, nullptr, external_data);

  // Verify tensor properties
  EXPECT_EQ(tensor.data_type(), core::DataType::FP32);
  EXPECT_EQ(tensor.size(), size);
  EXPECT_EQ(tensor.byte_size(), byte_size);

  // Verify tensor data
  const float* tensor_data = tensor.ptr<float>();
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(tensor_data[i], static_cast<float>(i));
  }

  // Free external memory (tensor doesn't own it)
  delete[] external_data;
}

// Test reset method
TEST_F(TensorTest, Reset) {
  // Create initial tensor
  Tensor tensor(core::DataType::FP32, 10, true, cpu_memory_manager);
  EXPECT_EQ(tensor.data_type(), core::DataType::FP32);
  EXPECT_EQ(tensor.size(), 10);

  // Reset tensor with new data type and dimensions
  std::vector<int32_t> new_dims = {2, 3, 4};
  tensor.reset(core::DataType::INT8, new_dims);

  // Verify reset worked
  EXPECT_EQ(tensor.data_type(), core::DataType::INT8);
  EXPECT_EQ(tensor.size(), 2 * 3 * 4);
  EXPECT_EQ(tensor.dims_size(), 3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tensor.get_dim(i), new_dims[i]);
  }
}

// Test reshape method
TEST_F(TensorTest, Reshape) {
  // Create initial tensor
  Tensor tensor(core::DataType::FP32, 24, true, cpu_memory_manager);
  EXPECT_EQ(tensor.size(), 24);
  EXPECT_EQ(tensor.dims_size(), 1);

  // Reshape to 2x3x4 (same total size)
  std::vector<int32_t> new_shape = {2, 3, 4};
  tensor.reshape(new_shape);

  // Verify reshape worked
  EXPECT_EQ(tensor.size(), 24);  // Total size remains the same
  EXPECT_EQ(tensor.dims_size(), 3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tensor.get_dim(i), new_shape[i]);
  }
}

// Test data access methods
TEST_F(TensorTest, DataAccess) {
  const int32_t dim0 = 10;
  Tensor tensor(core::DataType::FP32, dim0, true, cpu_memory_manager);

  // Write data using ptr
  float* data_ptr = tensor.ptr<float>();
  for (int i = 0; i < dim0; i++) {
    data_ptr[i] = static_cast<float>(i);
  }

  // Read data using index
  for (int i = 0; i < dim0; i++) {
    EXPECT_EQ(tensor.index<float>(i), static_cast<float>(i));
  }

  // Test ptr with offset
  float* offset_ptr = tensor.ptr<float>(5);
  EXPECT_EQ(*offset_ptr, 5.0f);

  // Test const ptr
  const Tensor& const_tensor = tensor;
  const float* const_ptr = const_tensor.ptr<float>();
  for (int i = 0; i < dim0; i++) {
    EXPECT_EQ(const_ptr[i], static_cast<float>(i));
  }
}

// Test 2D data access methods
TEST_F(TensorTest, DataAccess2D) {
  const int32_t dim0 = 3;
  const int32_t dim1 = 4;
  Tensor tensor(core::DataType::FP32, dim0, dim1, true, cpu_memory_manager);

  // Fill data in row-major order
  float* data_ptr = tensor.ptr<float>();
  for (int i = 0; i < dim0 * dim1; i++) {
    data_ptr[i] = static_cast<float>(i);
  }

  // Access data using linear indexing
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      int linear_idx = i * dim1 + j;
      EXPECT_EQ(tensor.index<float>(linear_idx), static_cast<float>(linear_idx));
    }
  }

  // Access specific elements
  EXPECT_EQ(tensor.index<float>(0), 0.0f);                 // First element
  EXPECT_EQ(tensor.index<float>(dim1 - 1), 3.0f);          // End of first row
  EXPECT_EQ(tensor.index<float>(dim1), 4.0f);              // Start of second row
  EXPECT_EQ(tensor.index<float>(dim0 * dim1 - 1), 11.0f);  // Last element

  // Access using ptr with offset
  EXPECT_EQ(*tensor.ptr<float>(5), 5.0f);
  EXPECT_EQ(*tensor.ptr<float>(dim0 * dim1 - 1), 11.0f);
}

// Test 3D data access methods
TEST_F(TensorTest, DataAccess3D) {
  const int32_t dim0 = 2;
  const int32_t dim1 = 3;
  const int32_t dim2 = 4;
  Tensor tensor(core::DataType::FP32, dim0, dim1, dim2, true, cpu_memory_manager);

  // Fill data in row-major order
  float* data_ptr = tensor.ptr<float>();
  for (int i = 0; i < dim0 * dim1 * dim2; i++) {
    data_ptr[i] = static_cast<float>(i);
  }

  // Access data using linear indexing
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      for (int k = 0; k < dim2; k++) {
        int linear_idx = (i * dim1 + j) * dim2 + k;
        EXPECT_EQ(tensor.index<float>(linear_idx), static_cast<float>(linear_idx));
      }
    }
  }

  // Test accessing specific elements
  const int slice_size = dim1 * dim2;

  // First element
  EXPECT_EQ(tensor.index<float>(0), 0.0f);

  // First element of second slice
  EXPECT_EQ(tensor.index<float>(slice_size), static_cast<float>(slice_size));

  // Last element of first row in first slice
  EXPECT_EQ(tensor.index<float>(dim2 - 1), static_cast<float>(dim2 - 1));

  // First element of second row in first slice
  EXPECT_EQ(tensor.index<float>(dim2), static_cast<float>(dim2));

  // Last element
  EXPECT_EQ(tensor.index<float>(dim0 * dim1 * dim2 - 1),
            static_cast<float>(dim0 * dim1 * dim2 - 1));

  // Test ptr with offset for 3D
  EXPECT_EQ(*tensor.ptr<float>(slice_size), static_cast<float>(slice_size));
  EXPECT_EQ(*tensor.ptr<float>(dim0 * dim1 * dim2 - 1), static_cast<float>(dim0 * dim1 * dim2 - 1));
}

// Test 4D data access methods
TEST_F(TensorTest, DataAccess4D) {
  const int32_t dim0 = 2;
  const int32_t dim1 = 3;
  const int32_t dim2 = 4;
  const int32_t dim3 = 5;
  Tensor tensor(core::DataType::FP32, dim0, dim1, dim2, dim3, true, cpu_memory_manager);

  // Fill data in row-major order
  float* data_ptr = tensor.ptr<float>();
  for (int i = 0; i < dim0 * dim1 * dim2 * dim3; i++) {
    data_ptr[i] = static_cast<float>(i);
  }

  // Access data using linear indexing computed from 4D coordinates
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      for (int k = 0; k < dim2; k++) {
        for (int l = 0; l < dim3; l++) {
          int linear_idx = ((i * dim1 + j) * dim2 + k) * dim3 + l;
          EXPECT_EQ(tensor.index<float>(linear_idx), static_cast<float>(linear_idx));
        }
      }
    }
  }

  // Test accessing specific elements
  const int volume3d = dim1 * dim2 * dim3;
  const int area = dim2 * dim3;

  // First element
  EXPECT_EQ(tensor.index<float>(0), 0.0f);

  // First element of second 3D block
  EXPECT_EQ(tensor.index<float>(volume3d), static_cast<float>(volume3d));

  // First element of second slice in first 3D block
  EXPECT_EQ(tensor.index<float>(area), static_cast<float>(area));

  // First element of second row in first slice of first 3D block
  EXPECT_EQ(tensor.index<float>(dim3), static_cast<float>(dim3));

  // Last element
  const int last_idx = dim0 * dim1 * dim2 * dim3 - 1;
  EXPECT_EQ(tensor.index<float>(last_idx), static_cast<float>(last_idx));

  // Test ptr with offset for 4D
  EXPECT_EQ(*tensor.ptr<float>(volume3d), static_cast<float>(volume3d));
  EXPECT_EQ(*tensor.ptr<float>(last_idx), static_cast<float>(last_idx));
}

// Test ND data access methods with arbitrary dimensions
TEST_F(TensorTest, DataAccessND) {
  // Create a 5D tensor with dimensions 2x3x2x3x2
  std::vector<int32_t> dims = {2, 3, 2, 3, 2};
  Tensor tensor(core::DataType::FP32, dims, true, cpu_memory_manager);

  // Calculate total size
  int total_size = 1;
  for (const auto& dim : dims) {
    total_size *= dim;
  }

  // Fill data
  float* data_ptr = tensor.ptr<float>();
  for (int i = 0; i < total_size; i++) {
    data_ptr[i] = static_cast<float>(i);
  }

  // Verify data using linear indexing
  for (int i = 0; i < total_size; i++) {
    EXPECT_EQ(tensor.index<float>(i), static_cast<float>(i));
  }

  // Create vector to hold multidimensional indices
  std::vector<int> indices(dims.size(), 0);

  // Test a specific coordinate: [1,1,1,1,1]
  indices = {1, 1, 1, 1, 1};

  // Calculate linear index from multidimensional indices
  int linear_idx = 0;
  int stride = 1;
  for (int i = dims.size() - 1; i >= 0; i--) {
    linear_idx += indices[i] * stride;
    stride *= dims[i];
  }

  // Verify element at specific coordinate
  EXPECT_EQ(tensor.index<float>(linear_idx), static_cast<float>(linear_idx));

  // Test another specific coordinate: [1,2,1,2,1]
  indices = {1, 2, 1, 2, 1};

  // Recalculate linear index
  linear_idx = 0;
  stride = 1;
  for (int i = dims.size() - 1; i >= 0; i--) {
    linear_idx += indices[i] * stride;
    stride *= dims[i];
  }

  // Verify element
  EXPECT_EQ(tensor.index<float>(linear_idx), static_cast<float>(linear_idx));

  // Test ptr with offset for ND
  int half_idx = total_size / 2;
  EXPECT_EQ(*tensor.ptr<float>(half_idx), static_cast<float>(half_idx));
  EXPECT_EQ(*tensor.ptr<float>(total_size - 1), static_cast<float>(total_size - 1));
}

// Test tensor assign method
TEST_F(TensorTest, Assign) {
  // Create a tensor
  Tensor tensor(core::DataType::FP32, 10, true, cpu_memory_manager);
  float* data = tensor.ptr<float>();
  for (int i = 0; i < 10; i++) {
    data[i] = static_cast<float>(i);
  }

  // Create a new buffer
  auto new_buffer = std::make_shared<core::Buffer>(10 * sizeof(float), cpu_memory_manager);
  float* new_data = static_cast<float*>(new_buffer->ptr());
  for (int i = 0; i < 10; i++) {
    new_data[i] = static_cast<float>(i * 2);
  }

  // Assign new buffer to tensor
  EXPECT_TRUE(tensor.assign(new_buffer));

  // Verify tensor now has new data
  float* tensor_data = tensor.ptr<float>();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(tensor_data[i], static_cast<float>(i * 2));
  }
}

// Test tensor clone method
TEST_F(TensorTest, Clone) {
  // Create and initialize a tensor
  Tensor original(core::DataType::FP32, 10, true, cpu_memory_manager);
  float* data = original.ptr<float>();
  for (int i = 0; i < 10; i++) {
    data[i] = static_cast<float>(i);
  }

  // Clone the tensor
  Tensor cloned = original.clone();

  // Verify cloned tensor has same properties
  EXPECT_EQ(cloned.data_type(), original.data_type());
  EXPECT_EQ(cloned.size(), original.size());
  EXPECT_EQ(cloned.dims_size(), original.dims_size());
  EXPECT_EQ(cloned.get_dim(0), original.get_dim(0));

  // Verify cloned tensor has same data
  float* cloned_data = cloned.ptr<float>();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(cloned_data[i], static_cast<float>(i));
  }

  // Modify original data
  data[0] = 99.0f;

  // Verify cloned data is not affected
  EXPECT_EQ(cloned_data[0], 0.0f);
}

// Test device transfer methods if CUDA is available
TEST_F(TensorTest, DeviceTransfer) {
  // Skip test if CUDA is not available
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "Skipping GPU tensor tests because no CUDA device is available";
  }

  // Create a CPU tensor with data
  const int32_t size = 10;
  Tensor cpu_tensor(core::DataType::FP32, size, true, cpu_memory_manager);
  float* cpu_data = cpu_tensor.ptr<float>();
  for (int i = 0; i < size; i++) {
    cpu_data[i] = static_cast<float>(i);
  }

  EXPECT_EQ(cpu_tensor.device_type(), core::DeviceType::CPU);

  // Transfer to GPU
  cpu_tensor.to_cuda();

  EXPECT_EQ(cpu_tensor.device_type(), core::DeviceType::GPU);

  // Create a new CPU tensor for verification
  Tensor verify_tensor(core::DataType::FP32, size, true, cpu_memory_manager);

  // Transfer data back to CPU for verification
  cpu_tensor.to_cpu();

  EXPECT_EQ(cpu_tensor.device_type(), core::DeviceType::CPU);

  // Verify data survived the round trip
  float* verify_data = cpu_tensor.ptr<float>();
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(verify_data[i], static_cast<float>(i));
  }
}

// Test strides method
TEST_F(TensorTest, Strides) {
  // Create a 3D tensor
  std::vector<int32_t> dims = {2, 3, 4};
  Tensor tensor(core::DataType::FP32, dims, true, cpu_memory_manager);

  // Calculate expected strides
  std::vector<size_t> expected_strides = {3 * 4, 4, 1};

  // Get actual strides
  std::vector<size_t> actual_strides = tensor.strides();

  // Verify strides
  ASSERT_EQ(actual_strides.size(), expected_strides.size());
  for (size_t i = 0; i < actual_strides.size(); i++) {
    EXPECT_EQ(actual_strides[i], expected_strides[i]);
  }
}

// Test 1D tensor at() method
TEST_F(TensorTest, At1D) {
  const int32_t dim0 = 5;
  Tensor tensor(core::DataType::FP32, dim0, true, cpu_memory_manager);
  float* data = tensor.ptr<float>();
  for (int i = 0; i < dim0; i++) {
    data[i] = static_cast<float>(i);
  }

  // Test non-const access
  for (int i = 0; i < dim0; i++) {
    EXPECT_EQ(tensor.at<float>(i), static_cast<float>(i));
  }

  // Test const access
  const Tensor& const_tensor = tensor;
  for (int i = 0; i < dim0; i++) {
    EXPECT_EQ(const_tensor.at<float>(i), static_cast<float>(i));
  }

  // Test bounds checking
  EXPECT_DEATH(tensor.at<float>(-1), ".*");
  EXPECT_DEATH(tensor.at<float>(dim0), ".*");
}

// Test 2D tensor at() method
TEST_F(TensorTest, At2D) {
  const int32_t dim0 = 3, dim1 = 4;
  Tensor tensor(core::DataType::FP32, dim0, dim1, true, cpu_memory_manager);
  float* data = tensor.ptr<float>();
  for (int i = 0; i < dim0 * dim1; i++) {
    data[i] = static_cast<float>(i);
  }

  // Test non-const access
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      EXPECT_EQ(tensor.at<float>(i, j), data[i * dim1 + j]);
    }
  }

  // Test const access
  const Tensor& const_tensor = tensor;
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      EXPECT_EQ(const_tensor.at<float>(i, j), data[i * dim1 + j]);
    }
  }

  // Test bounds checking
  EXPECT_DEATH(tensor.at<float>(dim0, 0), ".*");
  EXPECT_DEATH(tensor.at<float>(0, dim1), ".*");

  // Test wrong number of indices
  EXPECT_DEATH(tensor.at<float>(1), ".*");        // Too few indices
  EXPECT_DEATH(tensor.at<float>(1, 2, 3), ".*");  // Too many indices
}

// Test 3D tensor at() method
TEST_F(TensorTest, At3D) {
  const int32_t dim0 = 2, dim1 = 3, dim2 = 4;
  Tensor tensor(core::DataType::FP32, dim0, dim1, dim2, true, cpu_memory_manager);
  float* data = tensor.ptr<float>();
  for (int i = 0; i < dim0 * dim1 * dim2; i++) {
    data[i] = static_cast<float>(i);
  }

  // Test non-const access
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      for (int k = 0; k < dim2; k++) {
        int idx = (i * dim1 + j) * dim2 + k;
        EXPECT_EQ(tensor.at<float>(i, j, k), data[idx]);
      }
    }
  }

  // Test const access
  const Tensor& const_tensor = tensor;
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      for (int k = 0; k < dim2; k++) {
        int idx = (i * dim1 + j) * dim2 + k;
        EXPECT_EQ(const_tensor.at<float>(i, j, k), data[idx]);
      }
    }
  }

  // Test bounds checking
  EXPECT_DEATH(tensor.at<float>(dim0, 0, 0), ".*");
  EXPECT_DEATH(tensor.at<float>(0, dim1, 0), ".*");
  EXPECT_DEATH(tensor.at<float>(0, 0, dim2), ".*");

  // Test wrong number of indices
  EXPECT_DEATH(tensor.at<float>(1, 2), ".*");        // Too few indices
  EXPECT_DEATH(tensor.at<float>(1, 2, 3, 4), ".*");  // Too many indices
}

// Test 4D tensor at() method
TEST_F(TensorTest, At4D) {
  const int32_t dim0 = 2, dim1 = 2, dim2 = 2, dim3 = 2;
  Tensor tensor(core::DataType::FP32, dim0, dim1, dim2, dim3, true, cpu_memory_manager);
  float* data = tensor.ptr<float>();
  for (int i = 0; i < dim0 * dim1 * dim2 * dim3; i++) {
    data[i] = static_cast<float>(i);
  }

  // Test non-const access
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      for (int k = 0; k < dim2; k++) {
        for (int l = 0; l < dim3; l++) {
          int idx = ((i * dim1 + j) * dim2 + k) * dim3 + l;
          EXPECT_EQ(tensor.at<float>(i, j, k, l), data[idx]);
        }
      }
    }
  }

  // Test const access
  const Tensor& const_tensor = tensor;
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      for (int k = 0; k < dim2; k++) {
        for (int l = 0; l < dim3; l++) {
          int idx = ((i * dim1 + j) * dim2 + k) * dim3 + l;
          EXPECT_EQ(const_tensor.at<float>(i, j, k, l), data[idx]);
        }
      }
    }
  }

  // Test bounds checking
  EXPECT_DEATH(tensor.at<float>(dim0, 0, 0, 0), ".*");
  EXPECT_DEATH(tensor.at<float>(0, dim1, 0, 0), ".*");
  EXPECT_DEATH(tensor.at<float>(0, 0, dim2, 0), ".*");
  EXPECT_DEATH(tensor.at<float>(0, 0, 0, dim3), ".*");

  // Test wrong number of indices
  EXPECT_DEATH(tensor.at<float>(1, 2, 3), ".*");        // Too few indices
  EXPECT_DEATH(tensor.at<float>(1, 2, 3, 4, 5), ".*");  // Too many indices
}

// Test at() method after reshape operations
TEST_F(TensorTest, AtAfterReshape) {
  // Create initial 1D tensor
  const int32_t initial_size = 24;
  Tensor tensor(core::DataType::FP32, initial_size, true, cpu_memory_manager);
  float* data = tensor.ptr<float>();
  for (int i = 0; i < initial_size; i++) {
    data[i] = static_cast<float>(i);
  }

  // Test reshape to 3D (1D -> 3D)
  {
    std::vector<int32_t> new_shape = {2, 3, 4};
    tensor.reshape(new_shape);

    // Verify data access with at() after reshape
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 4; k++) {
          int idx = (i * 3 + j) * 4 + k;
          EXPECT_EQ(tensor.at<float>(i, j, k), static_cast<float>(idx));
        }
      }
    }

    // Test bounds checking after reshape
    EXPECT_DEATH(tensor.at<float>(2, 0, 0), ".*");
    EXPECT_DEATH(tensor.at<float>(0, 3, 0), ".*");
    EXPECT_DEATH(tensor.at<float>(0, 0, 4), ".*");

    // Test wrong number of indices
    EXPECT_DEATH(tensor.at<float>(1), ".*");           // Too few indices
    EXPECT_DEATH(tensor.at<float>(1, 2), ".*");        // Too few indices
    EXPECT_DEATH(tensor.at<float>(1, 2, 3, 4), ".*");  // Too many indices
  }

  // Test reshape to 2D (3D -> 2D)
  {
    std::vector<int32_t> new_shape_2d = {4, 6};
    tensor.reshape(new_shape_2d);

    // Verify data access with at() after second reshape
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 6; j++) {
        int idx = i * 6 + j;
        EXPECT_EQ(tensor.at<float>(i, j), static_cast<float>(idx));
      }
    }

    // Test bounds checking after reshape
    EXPECT_DEATH(tensor.at<float>(4, 0), ".*");
    EXPECT_DEATH(tensor.at<float>(0, 6), ".*");

    // Test wrong number of indices
    EXPECT_DEATH(tensor.at<float>(1), ".*");        // Too few indices
    EXPECT_DEATH(tensor.at<float>(1, 2, 3), ".*");  // Too many indices
  }

  // Test reshape to 1D (2D -> 1D)
  {
    std::vector<int32_t> new_shape_1d = {24};
    tensor.reshape(new_shape_1d);

    // Verify data access with at() after third reshape
    for (int i = 0; i < 24; i++) {
      EXPECT_EQ(tensor.at<float>(i), static_cast<float>(i));
    }

    // Test bounds checking after reshape
    EXPECT_DEATH(tensor.at<float>(-1), ".*");
    EXPECT_DEATH(tensor.at<float>(24), ".*");

    // Test wrong number of indices
    EXPECT_DEATH(tensor.at<float>(1, 2), ".*");  // Too many indices
  }
}

// Test at() method after reset operations
TEST_F(TensorTest, AtAfterReset) {
  // Test reset from 2D to 3D
  {
    // Create initial 2D tensor
    Tensor tensor(core::DataType::FP32, 3, 4, true, cpu_memory_manager);
    float* data = tensor.ptr<float>();
    for (int i = 0; i < 12; i++) {
      data[i] = static_cast<float>(i);
    }

    // Reset to 3D tensor
    std::vector<int32_t> new_dims = {2, 2, 3};
    tensor.reset(core::DataType::FP32, new_dims);

    // Allocate and fill new data
    tensor.allocate(cpu_memory_manager);
    float* new_data = tensor.ptr<float>();
    for (int i = 0; i < 12; i++) {
      new_data[i] = static_cast<float>(i * 2);  // Different pattern to verify reset
    }

    // Verify at() access after reset
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 3; k++) {
          int idx = (i * 2 + j) * 3 + k;
          EXPECT_EQ(tensor.at<float>(i, j, k), static_cast<float>(idx * 2));
        }
      }
    }

    // Test bounds checking after reset
    EXPECT_DEATH(tensor.at<float>(2, 0, 0), ".*");
    EXPECT_DEATH(tensor.at<float>(0, 2, 0), ".*");
    EXPECT_DEATH(tensor.at<float>(0, 0, 3), ".*");

    // Test wrong number of indices after reset
    EXPECT_DEATH(tensor.at<float>(1, 1), ".*");        // Too few indices
    EXPECT_DEATH(tensor.at<float>(1, 1, 1, 1), ".*");  // Too many indices
  }

  // Test reset from 3D to 1D
  {
    // Create initial 3D tensor
    Tensor tensor(core::DataType::FP32, 2, 3, 2, true, cpu_memory_manager);
    float* data = tensor.ptr<float>();
    for (int i = 0; i < 12; i++) {
      data[i] = static_cast<float>(i);
    }

    // Reset to 1D tensor
    std::vector<int32_t> new_dims = {12};
    tensor.reset(core::DataType::FP32, new_dims);

    // Allocate and fill new data
    tensor.allocate(cpu_memory_manager);
    float* new_data = tensor.ptr<float>();
    for (int i = 0; i < 12; i++) {
      new_data[i] = static_cast<float>(i * 3);  // Different pattern
    }

    // Verify at() access after reset
    for (int i = 0; i < 12; i++) {
      EXPECT_EQ(tensor.at<float>(i), static_cast<float>(i * 3));
    }

    // Test bounds checking after reset
    EXPECT_DEATH(tensor.at<float>(-1), ".*");
    EXPECT_DEATH(tensor.at<float>(12), ".*");

    // Test wrong number of indices after reset
    EXPECT_DEATH(tensor.at<float>(1, 1), ".*");  // Too many indices
  }
}

// Test transpose method
TEST_F(TensorTest, Transpose) {
  // Create a 2D tensor (3x4)
  const int32_t dim0 = 3, dim1 = 4;
  Tensor tensor(core::DataType::FP32, dim0, dim1, true, cpu_memory_manager);

  // Fill with test pattern
  float* data = tensor.ptr<float>();
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      data[i * dim1 + j] = static_cast<float>(i * 10 + j);
    }
  }

  // Transpose dimensions 0 and 1
  tensor.transpose<float>(0, 1);

  // Verify dimensions are swapped
  EXPECT_EQ(tensor.dims_size(), 2);
  EXPECT_EQ(tensor.get_dim(0), dim1);  // Was dim0
  EXPECT_EQ(tensor.get_dim(1), dim0);  // Was dim1

  // Verify data has been correctly transposed
  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim0; j++) {
      // Element at (i,j) in transposed tensor should be element (j,i) in original
      EXPECT_EQ(tensor.at<float>(i, j), static_cast<float>(j * 10 + i));
    }
  }

  // Test 3D tensor transpose (2x3x4)
  const int32_t tdim0 = 2, tdim1 = 3, tdim2 = 4;
  Tensor tensor3d(core::DataType::FP32, tdim0, tdim1, tdim2, true, cpu_memory_manager);

  // Fill 3D tensor with test pattern
  data = tensor3d.ptr<float>();
  for (int i = 0; i < tdim0; i++) {
    for (int j = 0; j < tdim1; j++) {
      for (int k = 0; k < tdim2; k++) {
        data[(i * tdim1 + j) * tdim2 + k] = static_cast<float>(i * 100 + j * 10 + k);
      }
    }
  }

  // Transpose dimensions 0 and 2
  tensor3d.transpose<float>(0, 2);

  // Verify dimensions are swapped
  EXPECT_EQ(tensor3d.dims_size(), 3);
  EXPECT_EQ(tensor3d.get_dim(0), tdim2);  // Was tdim0
  EXPECT_EQ(tensor3d.get_dim(1), tdim1);  // Unchanged
  EXPECT_EQ(tensor3d.get_dim(2), tdim0);  // Was tdim2

  // Verify data has been correctly transposed
  for (int i = 0; i < tdim2; i++) {
    for (int j = 0; j < tdim1; j++) {
      for (int k = 0; k < tdim0; k++) {
        // Element at (i,j,k) in transposed tensor should be element (k,j,i) in original
        EXPECT_EQ(tensor3d.at<float>(i, j, k), static_cast<float>(k * 100 + j * 10 + i));
      }
    }
  }

  // Test 4D tensor transpose (2x3x4x5)
  const int32_t fdim0 = 2, fdim1 = 3, fdim2 = 4, fdim3 = 5;
  Tensor tensor4d(core::DataType::FP32, fdim0, fdim1, fdim2, fdim3, true, cpu_memory_manager);

  // Fill 4D tensor with test pattern
  data = tensor4d.ptr<float>();
  for (int i = 0; i < fdim0; i++) {
    for (int j = 0; j < fdim1; j++) {
      for (int k = 0; k < fdim2; k++) {
        for (int l = 0; l < fdim3; l++) {
          data[((i * fdim1 + j) * fdim2 + k) * fdim3 + l] =
              static_cast<float>(i * 1000 + j * 100 + k * 10 + l);
        }
      }
    }
  }

  // Transpose dimensions 1 and 2
  tensor4d.transpose<float>(1, 2);

  // Verify dimensions are swapped
  EXPECT_EQ(tensor4d.dims_size(), 4);
  EXPECT_EQ(tensor4d.get_dim(0), fdim0);  // Unchanged
  EXPECT_EQ(tensor4d.get_dim(1), fdim2);  // Was fdim1
  EXPECT_EQ(tensor4d.get_dim(2), fdim1);  // Was fdim2
  EXPECT_EQ(tensor4d.get_dim(3), fdim3);  // Unchanged

  // Verify data has been correctly transposed
  for (int i = 0; i < fdim0; i++) {
    for (int j = 0; j < fdim2; j++) {
      for (int k = 0; k < fdim1; k++) {
        for (int l = 0; l < fdim3; l++) {
          // Element at (i,j,k,l) in transposed tensor should be element (i,k,j,l) in original
          EXPECT_EQ(tensor4d.at<float>(i, j, k, l),
                    static_cast<float>(i * 1000 + k * 100 + j * 10 + l));
        }
      }
    }
  }

  // Test no-op transpose (same axis)
  Tensor tensor_copy = tensor.clone();
  tensor_copy.transpose<float>(0, 0);

  // Dimensions should be unchanged
  EXPECT_EQ(tensor_copy.dims_size(), tensor.dims_size());
  EXPECT_EQ(tensor_copy.get_dim(0), tensor.get_dim(0));
  EXPECT_EQ(tensor_copy.get_dim(1), tensor.get_dim(1));

  // Data should be unchanged
  for (size_t i = 0; i < tensor.size(); i++) {
    EXPECT_EQ(tensor_copy.index<float>(i), tensor.index<float>(i));
  }

  // Test error conditions
  Tensor error_tensor(core::DataType::FP32, 2, 3, true, cpu_memory_manager);

  // Invalid axis should trigger crash in debug mode
  EXPECT_DEATH(error_tensor.transpose<float>(-1, 0), ".*");
  EXPECT_DEATH(error_tensor.transpose<float>(0, -1), ".*");
  EXPECT_DEATH(error_tensor.transpose<float>(2, 0), ".*");
  EXPECT_DEATH(error_tensor.transpose<float>(0, 2), ".*");
}

// Test get_offset method
TEST_F(TensorTest, GetOffset) {
  // Test 1D tensor offset calculation
  const int32_t dim0 = 5;
  Tensor tensor1d(core::DataType::FP32, dim0, true, cpu_memory_manager);

  // Offset should match the index in 1D
  for (int i = 0; i < dim0; i++) {
    EXPECT_EQ(tensor1d.get_offset(i), i);
  }

  // Test 2D tensor offset calculation
  const int32_t dim1 = 3, dim2 = 4;
  Tensor tensor2d(core::DataType::FP32, dim1, dim2, true, cpu_memory_manager);

  // Offsets should match row-major layout
  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      EXPECT_EQ(tensor2d.get_offset(i, j), i * dim2 + j);
    }
  }

  // Test 3D tensor offset calculation
  const int32_t dim3d_0 = 2, dim3d_1 = 3, dim3d_2 = 4;
  Tensor tensor3d(core::DataType::FP32, dim3d_0, dim3d_1, dim3d_2, true, cpu_memory_manager);

  // Offsets should match row-major layout in 3D
  for (int i = 0; i < dim3d_0; i++) {
    for (int j = 0; j < dim3d_1; j++) {
      for (int k = 0; k < dim3d_2; k++) {
        EXPECT_EQ(tensor3d.get_offset(i, j, k), (i * dim3d_1 + j) * dim3d_2 + k);
      }
    }
  }

  // Test 4D tensor offset calculation
  std::vector<int32_t> dims4d = {2, 3, 4, 5};
  Tensor tensor4d(core::DataType::FP32, dims4d, true, cpu_memory_manager);

  // Calculate expected stride for 4D tensor
  std::vector<size_t> expected_strides = {3 * 4 * 5, 4 * 5, 5, 1};

  // Verify a few specific offsets
  EXPECT_EQ(tensor4d.get_offset(0, 0, 0, 0), 0);
  EXPECT_EQ(tensor4d.get_offset(0, 0, 0, 4), 4);
  EXPECT_EQ(tensor4d.get_offset(0, 0, 1, 0), 5);
  EXPECT_EQ(tensor4d.get_offset(0, 1, 0, 0), 20);
  EXPECT_EQ(tensor4d.get_offset(1, 0, 0, 0), 60);
  EXPECT_EQ(tensor4d.get_offset(1, 2, 3, 4), 119);  // Last element

  // Test get_offset after reshape
  Tensor reshaped_tensor = tensor4d.clone();
  std::vector<int32_t> new_shape = {6, 20};
  reshaped_tensor.reshape(new_shape);

  // Verify offsets in reshaped tensor
  EXPECT_EQ(reshaped_tensor.get_offset(0, 0), 0);
  EXPECT_EQ(reshaped_tensor.get_offset(0, 19), 19);
  EXPECT_EQ(reshaped_tensor.get_offset(1, 0), 20);
  EXPECT_EQ(reshaped_tensor.get_offset(5, 19), 119);  // Last element

  // Test bounds checking
  EXPECT_DEATH(tensor2d.get_offset(dim1, 0), ".*");
  EXPECT_DEATH(tensor2d.get_offset(0, dim2), ".*");

  // Test wrong number of indices
  EXPECT_DEATH(tensor2d.get_offset(1), ".*");        // Too few indices
  EXPECT_DEATH(tensor2d.get_offset(1, 1, 1), ".*");  // Too many indices
}

// Test scalar tensor constructor and methods
TEST_F(TensorTest, ScalarTensor) {
  // Test scalar constructor
  Tensor scalar_tensor(core::DataType::FP32, true, cpu_memory_manager);

  EXPECT_EQ(scalar_tensor.data_type(), core::DataType::FP32);
  EXPECT_EQ(scalar_tensor.size(), 1);
  EXPECT_EQ(scalar_tensor.dims_size(), 0);
  EXPECT_TRUE(scalar_tensor.is_scalar());
  EXPECT_FALSE(scalar_tensor.is_empty());
  EXPECT_EQ(scalar_tensor.byte_size(), 4);  // FP32 is 4 bytes
  EXPECT_EQ(scalar_tensor.device_type(), core::DeviceType::CPU);

  // Test setting and getting scalar values
  scalar_tensor.set_scalar_value<float>(3.14f);
  EXPECT_FLOAT_EQ(scalar_tensor.scalar_value<float>(), 3.14f);

  // Test alternative scalar detection with 1D tensor of size 1
  Tensor tensor_1d(core::DataType::FP32, 1, true, cpu_memory_manager);
  EXPECT_TRUE(tensor_1d.is_scalar());
  EXPECT_EQ(tensor_1d.size(), 1);
  EXPECT_EQ(tensor_1d.dims_size(), 1);

  // Test is_scalar_compatible utility function
  EXPECT_TRUE(is_scalar_compatible(scalar_tensor));
  EXPECT_TRUE(is_scalar_compatible(tensor_1d));

  // Test make_scalar utility function
  Tensor made_scalar = make_scalar<float>(2.71f, cpu_memory_manager);
  EXPECT_TRUE(made_scalar.is_scalar());
  EXPECT_FLOAT_EQ(made_scalar.scalar_value<float>(), 2.71f);

  // Test int scalar
  Tensor int_scalar = make_scalar<int32_t>(42, cpu_memory_manager);
  EXPECT_TRUE(int_scalar.is_scalar());
  EXPECT_EQ(int_scalar.scalar_value<int32_t>(), 42);

  // Test cast operator
  float float_val = scalar_tensor;
  EXPECT_FLOAT_EQ(float_val, 3.14f);

  int32_t int_val = int_scalar;
  EXPECT_EQ(int_val, 42);

  // Test clone for scalar tensor
  Tensor cloned = scalar_tensor.clone();
  EXPECT_TRUE(cloned.is_scalar());
  EXPECT_FLOAT_EQ(cloned.scalar_value<float>(), 3.14f);

  // Change the original
  scalar_tensor.set_scalar_value<float>(1.23f);

  // Verify clone is independent
  EXPECT_FLOAT_EQ(cloned.scalar_value<float>(), 3.14f);
  EXPECT_FLOAT_EQ(scalar_tensor.scalar_value<float>(), 1.23f);

  // Test device transfers for scalar tensor
  // Skip test if CUDA is not available
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
    scalar_tensor.to_cuda();
    EXPECT_EQ(scalar_tensor.device_type(), core::DeviceType::GPU);

    scalar_tensor.to_cpu();
    EXPECT_EQ(scalar_tensor.device_type(), core::DeviceType::CPU);
    EXPECT_FLOAT_EQ(scalar_tensor.scalar_value<float>(), 1.23f);
  }
}

// Test scalar tensor in operations (basic arithmetic and comparison)
TEST_F(TensorTest, ScalarTensorOperations) {
  // Create scalar tensors with different types
  Tensor float_scalar = make_scalar<float>(3.5f, cpu_memory_manager);
  Tensor int_scalar = make_scalar<int32_t>(7, cpu_memory_manager);

  // Test implicit conversion in arithmetic operations
  float float_val = float_scalar;
  int32_t int_val = int_scalar;

  EXPECT_FLOAT_EQ(float_val + 1.0f, 4.5f);
  EXPECT_EQ(int_val * 2, 14);

  // Verify conversion between scalar tensor types
  EXPECT_FLOAT_EQ(static_cast<float>(int_scalar.scalar_value<int32_t>()), 7.0f);
  EXPECT_EQ(static_cast<int32_t>(float_scalar.scalar_value<float>()), 3);

  // Test comparison operators with explicit conversion
  EXPECT_TRUE(static_cast<float>(float_scalar) > 3.0f);
  EXPECT_TRUE(static_cast<int32_t>(int_scalar) < 10);
  EXPECT_FALSE(static_cast<float>(float_scalar) == 3.0f);
  EXPECT_TRUE(static_cast<int32_t>(int_scalar) != 8);

  // Create additional tensors for more tests
  Tensor scalar1 = make_scalar<float>(1.0f, cpu_memory_manager);
  Tensor scalar2 = make_scalar<float>(1.0f, cpu_memory_manager);

  // Test equality between scalar tensors with same value
  EXPECT_FLOAT_EQ(static_cast<float>(scalar1), static_cast<float>(scalar2));
}

// Test error conditions for scalar tensors
TEST_F(TensorTest, ScalarTensorErrors) {
  // Create scalar tensor
  Tensor scalar_tensor(core::DataType::FP32, true, cpu_memory_manager);
  scalar_tensor.set_scalar_value<float>(3.14f);

  // Create non-scalar tensor
  Tensor non_scalar(core::DataType::FP32, 2, true, cpu_memory_manager);

  // Test accessing scalar value of non-scalar tensor
  EXPECT_DEATH(non_scalar.scalar_value<float>(), ".*");

  // Test setting scalar value on non-scalar tensor
  EXPECT_DEATH(non_scalar.set_scalar_value<float>(1.0f), ".*");

  // Create scalar tensor without allocation
  Tensor unallocated_scalar(core::DataType::FP32, false, nullptr);

  // Test accessing/setting scalar value without allocation
  EXPECT_DEATH(unallocated_scalar.scalar_value<float>(), ".*");
  EXPECT_DEATH(unallocated_scalar.set_scalar_value<float>(1.0f), ".*");
}

// Test zeros function
TEST_F(TensorTest, ZerosFunction) {
  // Test 1D zeros tensor
  {
    std::vector<int32_t> dims = {5};
    Tensor zeros_tensor = zeros(core::DataType::FP32, dims, cpu_memory_manager);

    // Verify tensor properties
    EXPECT_EQ(zeros_tensor.data_type(), core::DataType::FP32);
    EXPECT_EQ(zeros_tensor.size(), 5);
    EXPECT_EQ(zeros_tensor.dims_size(), 1);
    EXPECT_EQ(zeros_tensor.get_dim(0), 5);
    EXPECT_FALSE(zeros_tensor.is_empty());
    EXPECT_EQ(zeros_tensor.device_type(), core::DeviceType::CPU);

    // Verify all elements are zero
    float* data = zeros_tensor.ptr<float>();
    for (int i = 0; i < 5; i++) {
      EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
  }

  // Test 2D zeros tensor
  {
    std::vector<int32_t> dims = {3, 4};
    Tensor zeros_tensor = zeros(core::DataType::FP32, dims, cpu_memory_manager);

    // Verify tensor properties
    EXPECT_EQ(zeros_tensor.data_type(), core::DataType::FP32);
    EXPECT_EQ(zeros_tensor.size(), 12);
    EXPECT_EQ(zeros_tensor.dims_size(), 2);
    EXPECT_EQ(zeros_tensor.get_dim(0), 3);
    EXPECT_EQ(zeros_tensor.get_dim(1), 4);

    // Verify all elements are zero
    float* data = zeros_tensor.ptr<float>();
    for (int i = 0; i < 12; i++) {
      EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
  }

  // Test 3D zeros tensor with INT32 data type
  {
    std::vector<int32_t> dims = {2, 3, 4};
    Tensor zeros_tensor = zeros(core::DataType::INT32, dims, cpu_memory_manager);

    // Verify tensor properties
    EXPECT_EQ(zeros_tensor.data_type(), core::DataType::INT32);
    EXPECT_EQ(zeros_tensor.size(), 24);
    EXPECT_EQ(zeros_tensor.dims_size(), 3);

    // Verify all elements are zero
    int32_t* data = zeros_tensor.ptr<int32_t>();
    for (int i = 0; i < 24; i++) {
      EXPECT_EQ(data[i], 0);
    }
  }

  // Test zeros tensor with INT8 data type
  {
    std::vector<int32_t> dims = {10};
    Tensor zeros_tensor = zeros(core::DataType::INT8, dims, cpu_memory_manager);

    // Verify tensor properties
    EXPECT_EQ(zeros_tensor.data_type(), core::DataType::INT8);
    EXPECT_EQ(zeros_tensor.size(), 10);

    // Verify all elements are zero
    int8_t* data = zeros_tensor.ptr<int8_t>();
    for (int i = 0; i < 10; i++) {
      EXPECT_EQ(data[i], 0);
    }
  }

  // Test zeros with GPU memory manager if CUDA is available
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
    std::vector<int32_t> dims = {5};
    Tensor zeros_tensor = zeros(core::DataType::FP32, dims, gpu_memory_manager);

    // Verify tensor properties
    EXPECT_EQ(zeros_tensor.data_type(), core::DataType::FP32);
    EXPECT_EQ(zeros_tensor.size(), 5);
    EXPECT_EQ(zeros_tensor.device_type(), core::DeviceType::GPU);

    // Transfer to CPU for verification
    zeros_tensor.to_cpu();

    // Verify all elements are zero
    float* data = zeros_tensor.ptr<float>();
    for (int i = 0; i < 5; i++) {
      EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
  }
}

}  // namespace
}  // namespace tensor

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Use threadsafe death test style to avoid warnings and potential issues
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  return RUN_ALL_TESTS();
}
