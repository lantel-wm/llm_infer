#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include "kernel/cpu/matmul_kernel_cpu.hpp"
#include "matmul_kernel_gpu.cuh"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class MatmulKernelGPUTest : public ::testing::Test {
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
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int32_t i = 0; i < tensor.size(); i++) {
      tensor.index<float>(i) = dist(rng);
    }
  }
};

// Test basic matrix multiplication functionality with simple values
TEST_F(MatmulKernelGPUTest, BasicOperation) {
  // Setup input tensor (2x3)
  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  // Setup weight tensor (3x2)
  const std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // Create tensors
  tensor::Tensor input_cpu(core::DataType::FP32, 2, 3, true, cpu_memory_manager);
  tensor::Tensor weight_cpu(core::DataType::FP32, 3, 2, true, cpu_memory_manager);
  tensor::Tensor input_gpu(core::DataType::FP32, 2, 3, true, cpu_memory_manager);
  tensor::Tensor weight_gpu(core::DataType::FP32, 3, 2, true, cpu_memory_manager);
  tensor::Tensor output_cpu(core::DataType::FP32, 2, 2, true, cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, 2, 2, true, gpu_memory_manager);

  // Initialize input tensors
  for (int i = 0; i < input_data.size(); i++) {
    input_cpu.index<float>(i) = input_data[i];
    input_gpu.index<float>(i) = input_data[i];
  }
  for (int i = 0; i < weight_data.size(); i++) {
    weight_cpu.index<float>(i) = weight_data[i];
    weight_gpu.index<float>(i) = weight_data[i];
  }
  // Run the cpu matmul kernel
  matmul_kernel_cpu(input_cpu, weight_cpu, output_cpu);

  input_gpu.to_cuda();
  weight_gpu.to_cuda();

  matmul_kernel_gpu(input_gpu, weight_gpu, output_gpu);

  output_gpu.to_cpu();

  // Verify results
  for (int i = 0; i < output_cpu.size(); i++) {
    EXPECT_NEAR(output_cpu.index<float>(i), output_gpu.index<float>(i), 1e-6f)
        << "Mismatch at index " << i;
  }
}

// Test with scale parameter
TEST_F(MatmulKernelGPUTest, ScaleOperation) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available, skipping GPU test";
  }

  // Setup input tensor (2x3)
  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  // Setup weight tensor (3x2)
  const std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  const float scale = 0.5f;

  // Create tensors
  tensor::Tensor input_cpu(core::DataType::FP32, 2, 3, true, cpu_memory_manager);
  tensor::Tensor weight_cpu(core::DataType::FP32, 3, 2, true, cpu_memory_manager);
  tensor::Tensor input_gpu(core::DataType::FP32, 2, 3, true, cpu_memory_manager);
  tensor::Tensor weight_gpu(core::DataType::FP32, 3, 2, true, cpu_memory_manager);
  tensor::Tensor output_cpu(core::DataType::FP32, 2, 2, true, cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, 2, 2, true, gpu_memory_manager);

  // Initialize input tensors
  for (int i = 0; i < input_data.size(); i++) {
    input_cpu.index<float>(i) = input_data[i];
    input_gpu.index<float>(i) = input_data[i];
  }
  for (int i = 0; i < weight_data.size(); i++) {
    weight_cpu.index<float>(i) = weight_data[i];
    weight_gpu.index<float>(i) = weight_data[i];
  }

  // Run the cpu matmul kernel with scale
  matmul_kernel_cpu(input_cpu, weight_cpu, output_cpu, scale);

  input_gpu.to_cuda();
  weight_gpu.to_cuda();

  // Run the GPU matmul kernel with scale
  matmul_kernel_gpu(input_gpu, weight_gpu, output_gpu, scale);

  output_gpu.to_cpu();

  // Verify results
  for (int i = 0; i < output_cpu.size(); i++) {
    EXPECT_NEAR(output_cpu.index<float>(i), output_gpu.index<float>(i), 1e-6f)
        << "Mismatch at index " << i;
  }
}

// Test with 1D input (vector) and 2D weight
TEST_F(MatmulKernelGPUTest, Vector1DInput) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available, skipping GPU test";
  }

  // Setup input tensor (1D vector of size 3)
  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  // Setup weight tensor (3x2)
  const std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // Create tensors
  tensor::Tensor input_cpu(core::DataType::FP32, 3, true, cpu_memory_manager);
  tensor::Tensor weight_cpu(core::DataType::FP32, 3, 2, true, cpu_memory_manager);
  tensor::Tensor input_gpu(core::DataType::FP32, 3, true, cpu_memory_manager);
  tensor::Tensor weight_gpu(core::DataType::FP32, 3, 2, true, cpu_memory_manager);
  tensor::Tensor output_cpu(core::DataType::FP32, 2, true, cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, 2, true, gpu_memory_manager);

  // Initialize input tensors
  for (int i = 0; i < input_data.size(); i++) {
    input_cpu.index<float>(i) = input_data[i];
    input_gpu.index<float>(i) = input_data[i];
  }
  for (int i = 0; i < weight_data.size(); i++) {
    weight_cpu.index<float>(i) = weight_data[i];
    weight_gpu.index<float>(i) = weight_data[i];
  }

  // Run the cpu matmul kernel
  matmul_kernel_cpu(input_cpu, weight_cpu, output_cpu);

  input_gpu.to_cuda();
  weight_gpu.to_cuda();

  // Run the GPU matmul kernel
  matmul_kernel_gpu(input_gpu, weight_gpu, output_gpu);

  output_gpu.to_cpu();

  // Verify results
  for (int i = 0; i < output_cpu.size(); i++) {
    EXPECT_NEAR(output_cpu.index<float>(i), output_gpu.index<float>(i), 1e-6f)
        << "Mismatch at index " << i;
  }
}

// Test with different matrix sizes
TEST_F(MatmulKernelGPUTest, DifferentSizes) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available, skipping GPU test";
  }

  const std::vector<std::pair<int, int>> sizes = {
      {1, 1},   // 1x1 matrix * 1x1 matrix
      {4, 4},   // 4x4 matrix * 4x4 matrix
      {16, 8},  // 16x8 matrix * 8x16 matrix
      {32, 64}  // 32x64 matrix * 64x32 matrix
  };

  for (const auto& size_pair : sizes) {
    const int M = size_pair.first;
    const int K = size_pair.second;
    const int N = size_pair.first;  // Output will be MxN

    // Create tensors
    tensor::Tensor input_cpu(core::DataType::FP32, M, K, true, cpu_memory_manager);
    tensor::Tensor weight_cpu(core::DataType::FP32, K, N, true, cpu_memory_manager);
    tensor::Tensor output_cpu(core::DataType::FP32, M, N, true, cpu_memory_manager);

    tensor::Tensor input_gpu(core::DataType::FP32, M, K, true, cpu_memory_manager);
    tensor::Tensor weight_gpu(core::DataType::FP32, K, N, true, cpu_memory_manager);
    tensor::Tensor output_gpu(core::DataType::FP32, M, N, true, gpu_memory_manager);

    // Initialize with simple pattern
    for (int i = 0; i < M * K; i++) {
      float val = static_cast<float>(i % 5);
      input_cpu.index<float>(i) = val;
      input_gpu.index<float>(i) = val;
    }
    for (int i = 0; i < K * N; i++) {
      float val = static_cast<float>(i % 7);
      weight_cpu.index<float>(i) = val;
      weight_gpu.index<float>(i) = val;
    }

    // Run the CPU matmul kernel
    matmul_kernel_cpu(input_cpu, weight_cpu, output_cpu);

    // Run the GPU matmul kernel
    input_gpu.to_cuda();
    weight_gpu.to_cuda();
    matmul_kernel_gpu(input_gpu, weight_gpu, output_gpu);
    output_gpu.to_cpu();

    // Verify results
    for (int i = 0; i < M * N; i++) {
      EXPECT_NEAR(output_cpu.index<float>(i), output_gpu.index<float>(i), 1e-5f)
          << "Mismatch at index " << i << " for size " << M << "x" << K << " * " << K << "x" << N;
    }
  }
}

// Test with zero inputs
TEST_F(MatmulKernelGPUTest, ZeroInput) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available, skipping GPU test";
  }

  const int M = 4;
  const int K = 4;
  const int N = 4;

  // Create tensors
  tensor::Tensor input_cpu(core::DataType::FP32, M, K, true, cpu_memory_manager);
  tensor::Tensor weight_cpu(core::DataType::FP32, K, N, true, cpu_memory_manager);
  tensor::Tensor output_cpu(core::DataType::FP32, M, N, true, cpu_memory_manager);

  tensor::Tensor input_gpu(core::DataType::FP32, M, K, true, cpu_memory_manager);
  tensor::Tensor weight_gpu(core::DataType::FP32, K, N, true, cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, M, N, true, gpu_memory_manager);

  // Initialize with zeros
  for (int i = 0; i < M * K; i++) {
    input_cpu.index<float>(i) = 0.0f;
    input_gpu.index<float>(i) = 0.0f;
  }
  for (int i = 0; i < K * N; i++) {
    weight_cpu.index<float>(i) = 0.0f;
    weight_gpu.index<float>(i) = 0.0f;
  }

  // Run the CPU matmul kernel
  matmul_kernel_cpu(input_cpu, weight_cpu, output_cpu);

  // Run the GPU matmul kernel
  input_gpu.to_cuda();
  weight_gpu.to_cuda();
  matmul_kernel_gpu(input_gpu, weight_gpu, output_gpu);
  output_gpu.to_cpu();

  // Verify all results are zeros
  for (int i = 0; i < M * N; i++) {
    EXPECT_NEAR(output_gpu.index<float>(i), 0.0f, 1e-6f)
        << "Zero input should produce zero output at index " << i;
    EXPECT_NEAR(output_cpu.index<float>(i), output_gpu.index<float>(i), 1e-6f)
        << "CPU and GPU results should match for zero input at index " << i;
  }
}

// Test identity matrix multiplication
TEST_F(MatmulKernelGPUTest, IdentityMatrix) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available, skipping GPU test";
  }

  const int size = 4;

  // Create tensors
  tensor::Tensor input_cpu(core::DataType::FP32, size, size, true, cpu_memory_manager);
  tensor::Tensor identity_cpu(core::DataType::FP32, size, size, true, cpu_memory_manager);
  tensor::Tensor output_cpu(core::DataType::FP32, size, size, true, cpu_memory_manager);

  tensor::Tensor input_gpu(core::DataType::FP32, size, size, true, cpu_memory_manager);
  tensor::Tensor identity_gpu(core::DataType::FP32, size, size, true, cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, size, size, true, gpu_memory_manager);

  // Initialize input with values
  for (int i = 0; i < size * size; i++) {
    float val = static_cast<float>(i);
    input_cpu.index<float>(i) = val;
    input_gpu.index<float>(i) = val;
  }

  // Initialize identity matrix
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      float val = (i == j) ? 1.0f : 0.0f;
      identity_cpu.at<float>(i, j) = val;
      identity_gpu.at<float>(i, j) = val;
    }
  }

  // Run the CPU matmul kernel
  matmul_kernel_cpu(input_cpu, identity_cpu, output_cpu);

  // Run the GPU matmul kernel
  input_gpu.to_cuda();
  identity_gpu.to_cuda();

  matmul_kernel_gpu(input_gpu, identity_gpu, output_gpu);

  output_gpu.to_cpu();
  input_gpu.to_cpu();

  // Verify A*I = A
  for (int i = 0; i < size * size; i++) {
    EXPECT_NEAR(output_cpu.index<float>(i), input_cpu.index<float>(i), 1e-6f)
        << "CPU result should be equal to input for identity multiplication at index " << i;
    EXPECT_NEAR(output_gpu.index<float>(i), input_gpu.index<float>(i), 1e-6f)
        << "GPU result should be equal to input for identity multiplication at index " << i;
    EXPECT_NEAR(output_cpu.index<float>(i), output_gpu.index<float>(i), 1e-6f)
        << "CPU and GPU results should match for identity multiplication at index " << i;
  }
}

// Test with large random matrices to stress the GPU implementation
TEST_F(MatmulKernelGPUTest, LargeRandomMatrices) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available, skipping GPU test";
  }
  // DeepSeek-R1-Distill-Qwen-1.5B config.json
  // https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/main/config.json
  // "hidden_size": 1536,
  // "intermediate_size": 8960,
  const int M = 512;   // seq_len
  const int K = 1536;  // hidden_size
  const int N = 8960;  // intermediate_size

  // Create tensors
  tensor::Tensor input_cpu(core::DataType::FP32, M, K, true, cpu_memory_manager);
  tensor::Tensor weight_cpu(core::DataType::FP32, K, N, true, cpu_memory_manager);
  tensor::Tensor output_cpu(core::DataType::FP32, M, N, true, cpu_memory_manager);

  tensor::Tensor input_gpu, weight_gpu;
  tensor::Tensor output_gpu(core::DataType::FP32, M, N, true, gpu_memory_manager);

  // Initialize with random values
  generate_random_values(input_cpu);
  generate_random_values(weight_cpu);

  // Run the CPU matmul kernel
  matmul_kernel_cpu(input_cpu, weight_cpu, output_cpu, 1.0f, nullptr);

  // Run the GPU matmul kernel
  input_gpu = input_cpu.clone();
  weight_gpu = weight_cpu.clone();
  input_gpu.to_cuda();
  weight_gpu.to_cuda();

  matmul_kernel_gpu(input_gpu, weight_gpu, output_gpu);

  output_gpu.to_cpu();

  // Verify results
  for (int i = 0; i < M * N; i++) {
    EXPECT_NEAR(output_cpu.index<float>(i), output_gpu.index<float>(i), 1e-4f)
        << "Mismatch at index " << i << " for large random matrices";
  }
}

}  // namespace
}  // namespace kernel