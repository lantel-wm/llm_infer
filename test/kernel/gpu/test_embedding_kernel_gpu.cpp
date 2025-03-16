#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include "kernel/cpu/embedding_kernel_cpu.hpp"
#include "kernel/gpu/embedding_kernel_gpu.cuh"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class EmbeddingKernelGPUTest : public ::testing::Test {
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

  // Helper to generate random token IDs
  void generate_random_tokens(tensor::Tensor& input, int32_t vocab_size) {
    std::uniform_int_distribution<int32_t> dist(0, vocab_size - 1);
    for (int32_t i = 0; i < input.size(); i++) {
      input.at<int32_t>(i) = dist(rng);
    }
  }

  // Helper to generate random embedding weights
  void generate_random_weights(tensor::Tensor& weight, int32_t vocab_size, int32_t embedding_dim) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int32_t i = 0; i < vocab_size; i++) {
      for (int32_t j = 0; j < embedding_dim; j++) {
        weight.at<float>(i, j) = dist(rng);
      }
    }
  }
};

// Test basic embedding lookup functionality with random data
TEST_F(EmbeddingKernelGPUTest, BasicRandomEmbedding) {
  if (!cuda_available) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  // Define test parameters - use smaller sizes to avoid memory issues
  const int32_t vocab_size = 2048;
  const int32_t embedding_dim = 1024;
  const int32_t seq_len = 512;

  // Create input tokens tensor
  tensor::Tensor input(core::DataType::INT32, seq_len, true, cpu_memory_manager);

  // Create embedding weight matrix
  tensor::Tensor weight(core::DataType::FP32, {vocab_size, embedding_dim}, true,
                        cpu_memory_manager);

  // Create output tensors for CPU and GPU
  tensor::Tensor output_cpu(core::DataType::FP32, {seq_len, embedding_dim}, true,
                            cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, {seq_len, embedding_dim}, true,
                            gpu_memory_manager);

  // Initialize with random data
  generate_random_tokens(input, vocab_size);
  generate_random_weights(weight, vocab_size, embedding_dim);

  embedding_kernel_cpu(input, weight, output_cpu, vocab_size);

  input.to_cuda();
  weight.to_cuda();
  embedding_kernel_gpu(input, weight, output_gpu, vocab_size, nullptr);
  output_gpu.to_cpu();

  for (int32_t i = 0; i < seq_len; i++) {
    for (int32_t j = 0; j < embedding_dim; j++) {
      EXPECT_NEAR(output_gpu.at<float>(i, j), output_cpu.at<float>(i, j), 1e-5f)
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }
}

// Test with larger embedding dimensions
TEST_F(EmbeddingKernelGPUTest, LargeEmbeddingDimension) {
  if (!cuda_available) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  // Define test parameters - reduced size to avoid memory issues
  const int32_t vocab_size = 1024;
  const int32_t embedding_dim = 512;
  const int32_t seq_len = 16;

  // Create input tokens tensor
  tensor::Tensor input(core::DataType::INT32, seq_len, true, cpu_memory_manager);

  // Create embedding weight matrix
  tensor::Tensor weight(core::DataType::FP32, {vocab_size, embedding_dim}, true,
                        cpu_memory_manager);

  // Create output tensors for CPU and GPU
  tensor::Tensor output_cpu(core::DataType::FP32, {seq_len, embedding_dim}, true,
                            cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, {seq_len, embedding_dim}, true,
                            gpu_memory_manager);

  // Initialize with random data
  generate_random_tokens(input, vocab_size);
  generate_random_weights(weight, vocab_size, embedding_dim);

  // Run CPU kernel as ground truth
  embedding_kernel_cpu(input, weight, output_cpu, vocab_size);

  // Transfer to GPU with error checking
  input.to_cuda();
  weight.to_cuda();

  // Run GPU kernel with error checking
  embedding_kernel_gpu(input, weight, output_gpu, vocab_size, nullptr);

  // Copy GPU results back to CPU for comparison
  output_gpu.to_cpu();

  // Verify results
  for (int32_t i = 0; i < seq_len; i++) {
    for (int32_t j = 0; j < embedding_dim; j++) {
      EXPECT_NEAR(output_gpu.at<float>(i, j), output_cpu.at<float>(i, j), 1e-5f)
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }
}

// Test with CUDA stream
TEST_F(EmbeddingKernelGPUTest, StreamEmbedding) {
  if (!cuda_available) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  // Create CUDA stream
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  const int32_t vocab_size = 1024;
  const int32_t embedding_dim = 512;
  const int32_t seq_len = 16;

  // Create input tokens tensor
  tensor::Tensor input(core::DataType::INT32, seq_len, true, cpu_memory_manager);

  // Create embedding weight matrix
  tensor::Tensor weight(core::DataType::FP32, {vocab_size, embedding_dim}, true,
                        cpu_memory_manager);

  // Create output tensors for CPU and GPU
  tensor::Tensor output_cpu(core::DataType::FP32, {seq_len, embedding_dim}, true,
                            cpu_memory_manager);
  tensor::Tensor output_gpu(core::DataType::FP32, {seq_len, embedding_dim}, true,
                            gpu_memory_manager);

  // Initialize with random data
  generate_random_tokens(input, vocab_size);
  generate_random_weights(weight, vocab_size, embedding_dim);

  // Run CPU kernel as ground truth
  embedding_kernel_cpu(input, weight, output_cpu, vocab_size);

  // Transfer to GPU with error checking using stream
  input.to_cuda(stream);
  weight.to_cuda(stream);

  // Run GPU kernel with stream
  embedding_kernel_gpu(input, weight, output_gpu, vocab_size, stream);

  // Check for errors
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "CUDA error in embedding_kernel_gpu";

  // Synchronize stream
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  // Copy GPU results back to CPU for comparison
  output_gpu.to_cpu();

  // Verify results
  for (int32_t i = 0; i < seq_len; i++) {
    for (int32_t j = 0; j < embedding_dim; j++) {
      EXPECT_NEAR(output_gpu.at<float>(i, j), output_cpu.at<float>(i, j), 1e-5f)
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }

  // Clean up stream
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Test with long sequence
TEST_F(EmbeddingKernelGPUTest, QwQ32B) {
  if (!cuda_available) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  // Define test parameters - reduced size to avoid memory issues
  const int32_t vocab_size = 152064;
  const int32_t embedding_dim = 5120;
  const int32_t seq_len = 16;

  // Create input tokens tensor
  tensor::Tensor input(core::DataType::INT32, seq_len, true, cpu_memory_manager);

  // Create embedding weight matrix
  tensor::Tensor weight(core::DataType::FP32, {vocab_size, embedding_dim}, true,
                        cpu_memory_manager);

  // Create output tensors for GPU
  tensor::Tensor output(core::DataType::FP32, {seq_len, embedding_dim}, true, gpu_memory_manager);

  // for (int32_t i = 0; i < seq_len; i++) {
  //   input.at<int32_t>(i) = i;
  // }

  // for (int i = 0; i < vocab_size; i++) {
  //   for (int j = 0; j < embedding_dim; j++) {
  //     weight.at<float>(i, j) = 1.0f;
  //   }
  // }

  // Transfer to GPU with error checking
  input.to_cuda();
  weight.to_cuda();
  // Run GPU kernel with error checking
  embedding_kernel_gpu(input, weight, output, vocab_size, nullptr);
  // Copy GPU results back to CPU for comparison
  output.to_cpu();

  // // Verify results for specific positions
  // for (int32_t i = 0; i < seq_len; i += 64) {
  //   for (int32_t j = 0; j < embedding_dim; j += 64) {
  //     EXPECT_NEAR(output.at<float>(i, j), 1.0f, 1e-5f)
  //         << "Mismatch at position [" << i << ", " << j << "]";
  //   }
  // }
}

}  // namespace
}  // namespace kernel

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
