#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "core/memory/buffer.hpp"
#include "core/memory/memory_manager.hpp"

namespace core {
namespace {

// Test fixture for Buffer tests
class BufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cpu_memory_manager = CPUMemoryManagerFactory::get_instance();
    gpu_memory_manager = GPUMemoryManagerFactory::get_instance();
  }

  std::shared_ptr<CPUMemoryManager> cpu_memory_manager;
  std::shared_ptr<GPUMemoryManager> gpu_memory_manager;
};

// Test default constructor
TEST_F(BufferTest, DefaultConstructor) {
  Buffer buffer;
  EXPECT_EQ(buffer.byte_size(), 0);
  EXPECT_EQ(buffer.ptr(), nullptr);
  EXPECT_EQ(buffer.device_type(), DeviceType::Unknown);
  EXPECT_EQ(buffer.memory_manager(), nullptr);
  EXPECT_FALSE(buffer.is_external());
}

// Test parameterized constructor with CPU memory manager
TEST_F(BufferTest, ParameterizedConstructorCPU) {
  const size_t buffer_size = 1024;
  Buffer buffer(buffer_size, cpu_memory_manager);

  EXPECT_EQ(buffer.byte_size(), buffer_size);
  EXPECT_NE(buffer.ptr(), nullptr);
  EXPECT_EQ(buffer.device_type(), DeviceType::CPU);
  EXPECT_EQ(buffer.memory_manager(), cpu_memory_manager);
  EXPECT_FALSE(buffer.is_external());
}

// // Test parameterized constructor with external pointer
// TEST_F(BufferTest, ParameterizedConstructorWithExternalPointer) {
//   const size_t buffer_size = 1024;
//   void* external_ptr = malloc(buffer_size);

//   Buffer buffer(buffer_size, cpu_memory_manager, external_ptr, true);

//   EXPECT_EQ(buffer.size(), buffer_size);
//   EXPECT_EQ(buffer.ptr(), external_ptr);
//   EXPECT_EQ(buffer.device_type(), DeviceType::CPU);
//   EXPECT_EQ(buffer.memory_manager(), cpu_memory_manager);
//   EXPECT_TRUE(buffer.is_external());

//   // Clean up external pointer (since buffer won't deallocate it)
//   free(external_ptr);
// }

// Test allocate method
TEST_F(BufferTest, Allocate) {
  const size_t buffer_size = 1024;
  void* external_ptr = malloc(buffer_size);

  // Create buffer with no initial allocation
  Buffer buffer(buffer_size, cpu_memory_manager, external_ptr, true);
  EXPECT_NE(buffer.ptr(), nullptr);
  EXPECT_TRUE(buffer.is_external());

  // free external ptr
  free(external_ptr);

  // Allocate memory
  EXPECT_TRUE(buffer.allocate());
  EXPECT_NE(buffer.ptr(), nullptr);
  EXPECT_FALSE(buffer.is_external());

  // Test allocate with no memory manager
  Buffer empty_buffer(buffer_size);
  EXPECT_FALSE(empty_buffer.allocate());

  // Test allocate with zero size
  Buffer zero_buffer(0, cpu_memory_manager);
  EXPECT_FALSE(zero_buffer.allocate());
}

// Test copy_from method (reference version)
TEST_F(BufferTest, CopyFromReference) {
  const size_t buffer_size = 1024;

  // Create source buffer and fill with test data
  Buffer src_buffer(buffer_size, cpu_memory_manager);
  char* src_data = static_cast<char*>(src_buffer.ptr());
  for (size_t i = 0; i < buffer_size; ++i) {
    src_data[i] = static_cast<char>(i % 256);
  }

  // Create destination buffer
  Buffer dst_buffer(buffer_size, cpu_memory_manager);

  // Copy data from source to destination
  dst_buffer.copy_from(src_buffer);

  // Verify the copy
  char* dst_data = static_cast<char*>(dst_buffer.ptr());
  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(dst_data[i], static_cast<char>(i % 256));
  }
}

// Test copy_from method (pointer version)
TEST_F(BufferTest, CopyFromPointer) {
  const size_t buffer_size = 1024;

  // Create source buffer and fill with test data
  Buffer src_buffer(buffer_size, cpu_memory_manager);
  char* src_data = static_cast<char*>(src_buffer.ptr());
  for (size_t i = 0; i < buffer_size; ++i) {
    src_data[i] = static_cast<char>(i % 256);
  }

  // Create destination buffer
  Buffer dst_buffer(buffer_size, cpu_memory_manager);

  // Copy data from source to destination
  dst_buffer.copy_from(&src_buffer);

  // Verify the copy
  char* dst_data = static_cast<char*>(dst_buffer.ptr());
  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(dst_data[i], static_cast<char>(i % 256));
  }
}

// Test copy_from with different sizes
TEST_F(BufferTest, CopyFromDifferentSizes) {
  const size_t src_size = 1024;
  const size_t dst_size = 512;  // Smaller than source

  // Create source buffer and fill with test data
  Buffer src_buffer(src_size, cpu_memory_manager);
  char* src_data = static_cast<char*>(src_buffer.ptr());
  for (size_t i = 0; i < src_size; ++i) {
    src_data[i] = static_cast<char>(i % 256);
  }

  // Create destination buffer (smaller than source)
  Buffer dst_buffer(dst_size, cpu_memory_manager);

  // Copy data from source to destination (should only copy dst_size bytes)
  dst_buffer.copy_from(src_buffer);

  // Verify the copy (only dst_size bytes should be copied)
  char* dst_data = static_cast<char*>(dst_buffer.ptr());
  for (size_t i = 0; i < dst_size; ++i) {
    EXPECT_EQ(dst_data[i], static_cast<char>(i % 256));
  }
}

// Test device_type and set_device_type
TEST_F(BufferTest, DeviceTypeOperations) {
  Buffer buffer;
  EXPECT_EQ(buffer.device_type(), DeviceType::Unknown);

  buffer.set_device_type(DeviceType::CPU);
  EXPECT_EQ(buffer.device_type(), DeviceType::CPU);

  buffer.set_device_type(DeviceType::GPU);
  EXPECT_EQ(buffer.device_type(), DeviceType::GPU);
}

// Test GPU buffer operations if CUDA is available
TEST_F(BufferTest, GPUBufferOperations) {
  // Skip test if CUDA is not available
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "Skipping GPU buffer tests because no CUDA device is available";
  }

  const size_t buffer_size = 1024;

  // Create GPU buffer
  Buffer gpu_buffer(buffer_size, gpu_memory_manager);
  EXPECT_EQ(gpu_buffer.device_type(), DeviceType::GPU);
  EXPECT_NE(gpu_buffer.ptr(), nullptr);

  // Create CPU buffer with test data
  Buffer cpu_buffer(buffer_size, cpu_memory_manager);
  char* cpu_data = static_cast<char*>(cpu_buffer.ptr());
  for (size_t i = 0; i < buffer_size; ++i) {
    cpu_data[i] = static_cast<char>(i % 256);
  }

  // Copy from CPU to GPU
  gpu_buffer.copy_from(cpu_buffer);

  // Create another CPU buffer for verification
  Buffer verify_buffer(buffer_size, cpu_memory_manager);

  // Copy from GPU back to CPU
  verify_buffer.copy_from(gpu_buffer);

  // Verify the round-trip copy
  char* verify_data = static_cast<char*>(verify_buffer.ptr());
  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(verify_data[i], static_cast<char>(i % 256));
  }
}

}  // namespace
}  // namespace core

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
