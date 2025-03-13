#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "core/memory/memory_manager.hpp"

namespace core {
namespace {

// Test fixture for memory manager tests
class MemoryManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cpu_memory_manager = std::make_shared<CPUMemoryManager>();
    gpu_memory_manager = std::make_shared<GPUMemoryManager>();
  }

  std::shared_ptr<CPUMemoryManager> cpu_memory_manager;
  std::shared_ptr<GPUMemoryManager> gpu_memory_manager;
};

// Test CPU Memory Manager
TEST_F(MemoryManagerTest, CPUMemoryManagerDeviceType) {
  EXPECT_EQ(cpu_memory_manager->device_type(), DeviceType::CPU);
}

TEST_F(MemoryManagerTest, CPUMemoryManagerAllocateAndDeallocate) {
  // Test allocation of different sizes
  const size_t sizes[] = {0, 1, 128, 1024, 1024 * 1024};

  for (const auto& size : sizes) {
    void* ptr = cpu_memory_manager->allocate(size);

    if (size == 0) {
      // For size 0, allocate should return nullptr
      EXPECT_EQ(ptr, nullptr);
    } else {
      // For non-zero sizes, allocate should return a valid pointer
      EXPECT_NE(ptr, nullptr);

      // Test that we can write to the allocated memory
      if (ptr) {
        std::memset(ptr, 0xAB, size);
        // Check first byte was set correctly
        EXPECT_EQ(*static_cast<unsigned char*>(ptr), 0xAB);
      }
    }

    // Test deallocation (should not crash)
    cpu_memory_manager->deallocate(ptr);
  }

  // Test deallocating nullptr (should not crash)
  cpu_memory_manager->deallocate(nullptr);
}

// Test CPU Memory Manager Factory
TEST_F(MemoryManagerTest, CPUMemoryManagerFactory) {
  auto instance1 = CPUMemoryManagerFactory::get_instance();
  auto instance2 = CPUMemoryManagerFactory::get_instance();

  // Test that the factory returns the same instance
  EXPECT_EQ(instance1, instance2);

  // Test that the instance is a valid CPU memory manager
  EXPECT_EQ(instance1->device_type(), DeviceType::CPU);
}

// Test GPU Memory Manager
TEST_F(MemoryManagerTest, GPUMemoryManagerDeviceType) {
  EXPECT_EQ(gpu_memory_manager->device_type(), DeviceType::GPU);
}

TEST_F(MemoryManagerTest, GPUMemoryManagerAllocateAndDeallocate) {
  // Skip test if CUDA is not available
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "Skipping GPU tests because no CUDA device is available";
  }

  // Test allocation of different sizes
  const size_t sizes[] = {0, 1, 128, 1024, 1024 * 1024, 2 * 1024 * 1024};

  for (const auto& size : sizes) {
    void* ptr = gpu_memory_manager->allocate(size);

    if (size == 0) {
      // For size 0, allocate should return nullptr
      EXPECT_EQ(ptr, nullptr);
    } else {
      // For non-zero sizes, allocate should return a valid pointer
      EXPECT_NE(ptr, nullptr);
    }

    // Test deallocation (should not crash)
    gpu_memory_manager->deallocate(ptr);
  }

  // Test deallocating nullptr (should not crash)
  gpu_memory_manager->deallocate(nullptr);
}

// Test GPU Memory Manager Factory
TEST_F(MemoryManagerTest, GPUMemoryManagerFactory) {
  auto instance1 = GPUMemoryManagerFactory::get_instance();
  auto instance2 = GPUMemoryManagerFactory::get_instance();

  // Test that the factory returns the same instance
  EXPECT_EQ(instance1, instance2);

  // Test that the instance is a valid GPU memory manager
  EXPECT_EQ(instance1->device_type(), DeviceType::GPU);
}

// Test memory copy functionality
TEST_F(MemoryManagerTest, MemcpyTest) {
  // Skip test if CUDA is not available
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "Skipping memcpy tests because no CUDA device is available";
  }

  const size_t size = 1024;

  // Allocate CPU memory
  void* cpu_src = cpu_memory_manager->allocate(size);
  void* cpu_dst = cpu_memory_manager->allocate(size);
  ASSERT_NE(cpu_src, nullptr);
  ASSERT_NE(cpu_dst, nullptr);

  // Initialize source memory
  for (size_t i = 0; i < size; ++i) {
    static_cast<char*>(cpu_src)[i] = static_cast<char>(i % 256);
  }

  // Test CPU to CPU copy
  cpu_memory_manager->memcpy(cpu_src, cpu_dst, size, MemcpyKind::MemcpyCPU2CPU);

  // Verify the copy
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(static_cast<char*>(cpu_dst)[i], static_cast<char>(i % 256));
  }

  // Allocate GPU memory
  void* gpu_src = gpu_memory_manager->allocate(size);
  void* gpu_dst = gpu_memory_manager->allocate(size);
  ASSERT_NE(gpu_src, nullptr);
  ASSERT_NE(gpu_dst, nullptr);

  // Test CPU to GPU copy
  gpu_memory_manager->memcpy(cpu_src, gpu_src, size, MemcpyKind::MemcpyCPU2GPU);

  // Test GPU to GPU copy
  gpu_memory_manager->memcpy(gpu_src, gpu_dst, size, MemcpyKind::MemcpyGPU2GPU);

  // Test GPU to CPU copy
  void* cpu_result = cpu_memory_manager->allocate(size);
  ASSERT_NE(cpu_result, nullptr);
  gpu_memory_manager->memcpy(gpu_dst, cpu_result, size, MemcpyKind::MemcpyGPU2CPU);

  // Verify the round-trip copy
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(static_cast<char*>(cpu_result)[i], static_cast<char>(i % 256));
  }

  // Clean up
  cpu_memory_manager->deallocate(cpu_src);
  cpu_memory_manager->deallocate(cpu_dst);
  cpu_memory_manager->deallocate(cpu_result);
  gpu_memory_manager->deallocate(gpu_src);
  gpu_memory_manager->deallocate(gpu_dst);
}

// Test memset0 functionality
TEST_F(MemoryManagerTest, Memset0Test) {
  const size_t size = 1024;

  // Test CPU memset0
  void* cpu_ptr = cpu_memory_manager->allocate(size);
  ASSERT_NE(cpu_ptr, nullptr);

  // Fill with non-zero data
  std::memset(cpu_ptr, 0xFF, size);

  // Use memset0 to clear
  cpu_memory_manager->memset0(cpu_ptr, size, nullptr);

  // Verify all bytes are zero
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(static_cast<char*>(cpu_ptr)[i], 0);
  }

  // Clean up
  cpu_memory_manager->deallocate(cpu_ptr);

  // Skip GPU test if CUDA is not available
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "Skipping GPU memset0 test because no CUDA device is available";
  }

  // Test GPU memset0
  void* gpu_ptr = gpu_memory_manager->allocate(size);
  ASSERT_NE(gpu_ptr, nullptr);

  // Use memset0 to clear GPU memory
  gpu_memory_manager->memset0(gpu_ptr, size, nullptr);

  // Copy back to CPU to verify
  void* cpu_check = cpu_memory_manager->allocate(size);
  ASSERT_NE(cpu_check, nullptr);

  gpu_memory_manager->memcpy(gpu_ptr, cpu_check, size, MemcpyKind::MemcpyGPU2CPU);

  // Verify all bytes are zero
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(static_cast<char*>(cpu_check)[i], 0);
  }

  // Clean up
  gpu_memory_manager->deallocate(gpu_ptr);
  cpu_memory_manager->deallocate(cpu_check);
}

}  // namespace
}  // namespace core

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
