#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include "kernel/cpu/swiglu_kernel_cpu.hpp"
#include "swiglu_kernel_gpu.cuh"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class SwiGLUKernelGPUTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance();
    gpu_memory_manager = core::GPUMemoryManagerFactory::get_instance();

    // Initialize random number generator
    rng.seed(42);  // Fixed seed for reproducibility

    // Check if CUDA is available and initialize CUDA context
    int device_count = 0;
    cuda_available = (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0);
  }

  void TearDown() override {
    if (cuda_available) {
      // Ensure all CUDA operations are complete
      cudaDeviceSynchronize();

      // Check for any errors that occurred during the test
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::cerr << "CUDA error during test: " << cudaGetErrorString(err) << std::endl;
      }
    }
  }

  std::shared_ptr<core::CPUMemoryManager> cpu_memory_manager;
  std::shared_ptr<core::GPUMemoryManager> gpu_memory_manager;
  std::mt19937 rng;
  bool cuda_available = false;

  // Helper to generate random values
  void generate_random_values(tensor::Tensor& tensor) {
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int32_t i = 0; i < tensor.size(); i++) {
      tensor.at<float>(i) = dist(rng);
    }
  }
};

// Test basic SwiGLU functionality with random data
TEST_F(SwiGLUKernelGPUTest, BasicRandomSwiGLU) {
  if (!cuda_available) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  // Define test parameters
  const int32_t size = 1024;

  // Create input tensors
  tensor::Tensor input1_cpu(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor input2_cpu(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor input1_gpu;
  tensor::Tensor input2_gpu;

  // Create output tensors for CPU and GPU
  tensor::Tensor output_cpu(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, size, true, gpu_memory_manager);

  // // Initialize with random data
  generate_random_values(input1_cpu);
  generate_random_values(input2_cpu);
  input1_gpu = input1_cpu.clone();
  input2_gpu = input2_cpu.clone();

  // Run CPU kernel as ground truth
  swiglu_kernel_cpu(input1_cpu, input2_cpu, output_cpu, nullptr);

  // Transfer to GPU
  input1_gpu.to_cuda();
  input2_gpu.to_cuda();

  // Run GPU kernel
  swiglu_kernel_gpu(input1_gpu, input2_gpu, output_gpu, nullptr);

  // Copy GPU results back to CPU for comparison
  output_gpu.to_cpu();

  // Verify results
  for (int32_t i = 0; i < size; i++) {
    EXPECT_NEAR(output_gpu.at<float>(i), output_cpu.at<float>(i), 1e-5f)
        << "Mismatch at position [" << i << "]";
  }
}

// Test with larger size
TEST_F(SwiGLUKernelGPUTest, LargeSize) {
  if (!cuda_available) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  // Define test parameters
  const int32_t size = 1024 * 1024;  // 1M elements

  // Create input tensors
  tensor::Tensor input1_cpu(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor input2_cpu(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor input1_gpu;
  tensor::Tensor input2_gpu;

  // Create output tensors for CPU and GPU
  tensor::Tensor output_cpu(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, size, true, gpu_memory_manager);

  // Initialize with random data
  generate_random_values(input1_cpu);
  generate_random_values(input2_cpu);

  // Clone CPU tensors to GPU tensors
  input1_gpu = input1_cpu.clone();
  input2_gpu = input2_cpu.clone();

  // Run CPU kernel as ground truth
  swiglu_kernel_cpu(input1_cpu, input2_cpu, output_cpu, nullptr);

  // Transfer to GPU
  input1_gpu.to_cuda();
  input2_gpu.to_cuda();

  // Run GPU kernel
  swiglu_kernel_gpu(input1_gpu, input2_gpu, output_gpu, nullptr);

  // Copy GPU results back to CPU for comparison
  output_gpu.to_cpu();

  // Verify results for specific positions to avoid excessive checking
  for (int32_t i = 0; i < size; i += 1024) {
    EXPECT_NEAR(output_gpu.at<float>(i), output_cpu.at<float>(i), 1e-5f)
        << "Mismatch at position [" << i << "]";
  }
}

// Test with CUDA stream
TEST_F(SwiGLUKernelGPUTest, StreamSwiGLU) {
  if (!cuda_available) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  // Create CUDA stream
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  const int32_t size = 1024;

  // Create input tensors
  tensor::Tensor input1_cpu(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor input2_cpu(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor input1_gpu;
  tensor::Tensor input2_gpu;

  // Create output tensors for CPU and GPU
  tensor::Tensor output_cpu(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, size, true, gpu_memory_manager);

  // Initialize with random data
  generate_random_values(input1_cpu);
  generate_random_values(input2_cpu);

  // Clone CPU tensors to GPU tensors
  input1_gpu = input1_cpu.clone();
  input2_gpu = input2_cpu.clone();

  // Run CPU kernel as ground truth
  swiglu_kernel_cpu(input1_cpu, input2_cpu, output_cpu, nullptr);

  // Transfer to GPU using stream
  input1_gpu.to_cuda(stream);
  input2_gpu.to_cuda(stream);

  // Run GPU kernel with stream
  swiglu_kernel_gpu(input1_gpu, input2_gpu, output_gpu, stream);

  // Check for errors
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "CUDA error in swiglu_kernel_gpu";

  // Synchronize stream
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  // Copy GPU results back to CPU for comparison
  output_gpu.to_cpu();

  // Verify results
  for (int32_t i = 0; i < size; i++) {
    EXPECT_NEAR(output_gpu.at<float>(i), output_cpu.at<float>(i), 1e-5f)
        << "Mismatch at position [" << i << "]";
  }

  // Clean up stream
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Test with specific edge cases
TEST_F(SwiGLUKernelGPUTest, EdgeCases) {
  if (!cuda_available) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  const int32_t size = 8;

  // Create input tensors
  tensor::Tensor input1_cpu(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor input2_cpu(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor input1_gpu;
  tensor::Tensor input2_gpu;

  // Create output tensors for CPU and GPU
  tensor::Tensor output_cpu(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, size, true, gpu_memory_manager);

  // Set specific test cases
  // Case 1: x = 0, y = 1 (sigmoid(0) = 0.5)
  input1_cpu.at<float>(0) = 0.0f;
  input2_cpu.at<float>(0) = 1.0f;

  // Case 2: x = large positive, y = 1 (sigmoid approaches 1)
  input1_cpu.at<float>(1) = 10.0f;
  input2_cpu.at<float>(1) = 1.0f;

  // Case 3: x = large negative, y = 1 (sigmoid approaches 0)
  input1_cpu.at<float>(2) = -10.0f;
  input2_cpu.at<float>(2) = 1.0f;

  // Case 4: x = 1, y = 0
  input1_cpu.at<float>(3) = 1.0f;
  input2_cpu.at<float>(3) = 0.0f;

  // Case 5: x = 0, y = 0
  input1_cpu.at<float>(4) = 0.0f;
  input2_cpu.at<float>(4) = 0.0f;

  // Case 6: x = -1, y = -1
  input1_cpu.at<float>(5) = -1.0f;
  input2_cpu.at<float>(5) = -1.0f;

  // Case 7: x = 1, y = 1
  input1_cpu.at<float>(6) = 1.0f;
  input2_cpu.at<float>(6) = 1.0f;

  // Case 8: x = NaN, y = 1 (should handle gracefully)
  input1_cpu.at<float>(7) = std::numeric_limits<float>::quiet_NaN();
  input2_cpu.at<float>(7) = 1.0f;

  // Clone CPU tensors to GPU tensors
  input1_gpu = input1_cpu.clone();
  input2_gpu = input2_cpu.clone();

  // Run CPU kernel
  swiglu_kernel_cpu(input1_cpu, input2_cpu, output_cpu, nullptr);

  // Transfer to GPU
  input1_gpu.to_cuda();
  input2_gpu.to_cuda();

  // Run GPU kernel
  swiglu_kernel_gpu(input1_gpu, input2_gpu, output_gpu, nullptr);

  // Copy GPU results back to CPU for comparison
  output_gpu.to_cpu();

  // Verify results (skip NaN case)
  for (int32_t i = 0; i < size - 1; i++) {
    EXPECT_NEAR(output_gpu.at<float>(i), output_cpu.at<float>(i), 1e-5f)
        << "Mismatch at edge case [" << i << "]";
  }
}

}  // namespace
}  // namespace kernel

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
