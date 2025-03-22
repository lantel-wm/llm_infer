#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>
#include "kernel/cpu/rmsnorm_kernel_cpu.hpp"
#include "rmsnorm_kernel_gpu.cuh"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class RMSNormKernelGPUTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance();
    gpu_memory_manager = core::GPUMemoryManagerFactory::get_instance();
  }

  std::shared_ptr<core::CPUMemoryManager> cpu_memory_manager;
  std::shared_ptr<core::GPUMemoryManager> gpu_memory_manager;
};

// Test basic RMSNorm functionality
TEST_F(RMSNormKernelGPUTest, BasicNormalization) {
  // Skip test if CUDA is not available
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f};  // Identity weights
  const int hidden_size = 4;
  const int batch_size = 2;

  // Create tensors for both CPU and GPU tests
  tensor::Tensor input_cpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                           cpu_memory_manager);
  tensor::Tensor weight_cpu(core::DataType::FP32, hidden_size, true, cpu_memory_manager);
  tensor::Tensor output_cpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                            cpu_memory_manager);
  tensor::Tensor input_gpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                           cpu_memory_manager);
  tensor::Tensor weight_gpu(core::DataType::FP32, hidden_size, true, cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                            gpu_memory_manager);

  // Initialize weight tensors
  for (int i = 0; i < hidden_size; i++) {
    weight_cpu.at<float>(i) = weight_data[i];
    weight_gpu.at<float>(i) = weight_data[i];
  }

  // Initialize input tensors with same data for all batches
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < hidden_size; i++) {
      input_cpu.at<float>(b, i) = input_data[i];
      input_gpu.at<float>(b, i) = input_data[i];
    }
  }

  // Run CPU kernel for reference
  rmsnorm_kernel_cpu(input_cpu, weight_cpu, output_cpu, nullptr);

  // Run GPU kernel
  input_gpu.to_cuda();
  weight_gpu.to_cuda();
  rmsnorm_kernel_gpu(input_gpu, weight_gpu, output_gpu, nullptr);
  output_gpu.to_cpu();

  // Verify results
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < hidden_size; i++) {
      EXPECT_NEAR(output_gpu.at<float>(b, i), output_cpu.at<float>(b, i), 1e-6f)
          << "Mismatch at batch " << b << ", index " << i;
    }
  }
}

// Test different tensor sizes
TEST_F(RMSNormKernelGPUTest, DifferentSizes) {
  // Skip test if CUDA is not available
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  const std::vector<int> hidden_sizes = {4, 16, 256, 1024};  // Must be multiples of 4 due to float4
  const int batch_size = 3;

  for (const auto& hidden_size : hidden_sizes) {
    // Create tensors for both CPU and GPU tests
    tensor::Tensor input_cpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                             cpu_memory_manager);
    tensor::Tensor weight_cpu(core::DataType::FP32, hidden_size, true, cpu_memory_manager);
    tensor::Tensor output_cpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                              cpu_memory_manager);
    tensor::Tensor input_gpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                             cpu_memory_manager);
    tensor::Tensor weight_gpu(core::DataType::FP32, hidden_size, true, cpu_memory_manager);
    tensor::Tensor output_gpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                              gpu_memory_manager);

    // Initialize with random numbers
    std::srand(42);
    // Initialize weights first
    for (int i = 0; i < hidden_size; i++) {
      weight_cpu.at<float>(i) =
          0.5f + static_cast<float>(std::rand()) / RAND_MAX;  // Random weights between 0.5 and 1.5
      weight_gpu.at<float>(i) = weight_cpu.at<float>(i);
    }

    // Then initialize inputs for each batch
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < hidden_size; i++) {
        float val = static_cast<float>(std::rand()) / RAND_MAX * 2.0f -
                    1.0f;  // Random value between -1.0 and 1.0
        input_cpu.at<float>(b, i) = val;
        input_gpu.at<float>(b, i) = val;
      }
    }

    // Run CPU kernel for reference
    rmsnorm_kernel_cpu(input_cpu, weight_cpu, output_cpu, nullptr);

    // Run GPU kernel
    input_gpu.to_cuda();
    weight_gpu.to_cuda();
    rmsnorm_kernel_gpu(input_gpu, weight_gpu, output_gpu, nullptr);
    output_gpu.to_cpu();

    // Verify results
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < hidden_size; i++) {
        EXPECT_NEAR(output_gpu.at<float>(b, i), output_cpu.at<float>(b, i), 1e-6f)
            << "Size: " << hidden_size << ", Mismatch at batch " << b << ", index " << i;
      }
    }
  }
}

// Test RMSNorm with custom weights
TEST_F(RMSNormKernelGPUTest, CustomWeights) {
  // Skip test if CUDA is not available
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> weight_data = {0.5f, 1.0f, 1.5f, 2.0f};
  const int hidden_size = 4;
  const int batch_size = 2;

  // Create tensors for both CPU and GPU tests
  tensor::Tensor input_cpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                           cpu_memory_manager);
  tensor::Tensor weight_cpu(core::DataType::FP32, hidden_size, true, cpu_memory_manager);
  tensor::Tensor output_cpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                            cpu_memory_manager);
  tensor::Tensor input_gpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                           cpu_memory_manager);
  tensor::Tensor weight_gpu(core::DataType::FP32, hidden_size, true, cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                            gpu_memory_manager);

  // Initialize weight tensors
  for (int i = 0; i < hidden_size; i++) {
    weight_cpu.at<float>(i) = weight_data[i];
    weight_gpu.at<float>(i) = weight_data[i];
  }

  // Initialize input tensors with same data for all batches
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < hidden_size; i++) {
      input_cpu.at<float>(b, i) = input_data[i];
      input_gpu.at<float>(b, i) = input_data[i];
    }
  }

  // Run CPU kernel for reference
  rmsnorm_kernel_cpu(input_cpu, weight_cpu, output_cpu, nullptr);

  // Run GPU kernel
  input_gpu.to_cuda();
  weight_gpu.to_cuda();
  rmsnorm_kernel_gpu(input_gpu, weight_gpu, output_gpu, nullptr);
  output_gpu.to_cpu();

  // Verify results
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < hidden_size; i++) {
      EXPECT_NEAR(output_gpu.at<float>(b, i), output_cpu.at<float>(b, i), 1e-6f)
          << "Mismatch at batch " << b << ", index " << i;
    }
  }
}

// Test RMSNorm with CUDA stream
TEST_F(RMSNormKernelGPUTest, StreamNormalization) {
  // Skip test if CUDA is not available
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  // Create CUDA stream
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  const int hidden_size = 1024;  // Must be multiple of 4 due to float4
  const int batch_size = 2;

  // Create tensors for both CPU and GPU tests
  tensor::Tensor input_cpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                           cpu_memory_manager);
  tensor::Tensor weight_cpu(core::DataType::FP32, hidden_size, true, cpu_memory_manager);
  tensor::Tensor output_cpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                            cpu_memory_manager);
  tensor::Tensor input_gpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                           cpu_memory_manager);
  tensor::Tensor weight_gpu(core::DataType::FP32, hidden_size, true, cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, {batch_size, hidden_size}, true,
                            gpu_memory_manager);

  // Initialize with random numbers
  std::srand(42);
  // Initialize weights first
  for (int i = 0; i < hidden_size; i++) {
    weight_cpu.at<float>(i) =
        0.5f + static_cast<float>(std::rand()) / RAND_MAX;  // Random weights between 0.5 and 1.5
    weight_gpu.at<float>(i) = weight_cpu.at<float>(i);
  }

  // Then initialize inputs for each batch
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < hidden_size; i++) {
      float val = static_cast<float>(std::rand()) / RAND_MAX * 2.0f -
                  1.0f;  // Random value between -1.0 and 1.0
      input_cpu.at<float>(b, i) = val;
      input_gpu.at<float>(b, i) = val;
    }
  }

  // Run CPU kernel for reference
  rmsnorm_kernel_cpu(input_cpu, weight_cpu, output_cpu, nullptr);

  // Run GPU kernel with stream
  input_gpu.to_cuda(stream);
  weight_gpu.to_cuda(stream);
  rmsnorm_kernel_gpu(input_gpu, weight_gpu, output_gpu, stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  output_gpu.to_cpu();

  // Verify results
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < hidden_size; i++) {
      EXPECT_NEAR(output_gpu.at<float>(b, i), output_cpu.at<float>(b, i), 1e-6f)
          << "Mismatch at batch " << b << ", index " << i;
    }
  }

  // Clean up stream
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace
}  // namespace kernel

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
