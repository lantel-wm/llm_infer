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

}  // namespace
}  // namespace tensor

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Use threadsafe death test style to avoid warnings and potential issues
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  return RUN_ALL_TESTS();
}
