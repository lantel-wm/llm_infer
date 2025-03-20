#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include "kernel/cpu/mha_kernel_cpu.hpp"
#include "mha_kernel_gpu.cuh"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class MHAKernelGPUTest : public ::testing::Test {
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
  void generate_random_tensor(tensor::Tensor& tensor) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int32_t i = 0; i < tensor.size(); i++) {
      tensor.index<float>(i) = dist(rng);
    }
  }
};

TEST_F(MHAKernelGPUTest, MhaSoftmax) {
  // Skip test if CUDA is not available
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available, skipping GPU test";
  }

  const int32_t num_heads = 1;
  const int32_t batch_size = 1;
  const int32_t seq_len = 512;

  // Create tensor for scores
  tensor::Tensor score_cpu(core::DataType::FP32, seq_len, seq_len, true, cpu_memory_manager);
  tensor::Tensor score_gpu;

  generate_random_tensor(score_cpu);
  score_gpu = score_cpu.clone();

  // Apply softmax operation
  mha_softmax_kernel_cpu(score_cpu);

  score_gpu.to_cuda();

  mha_softmax_kernel_gpu(num_heads, batch_size, score_gpu);

  score_gpu.to_cpu();

  float rmse_diff = 0.0f;
  float max_diff = 0.0f;

  // Verify that the CPU and GPU implementations produce similar results
  for (int32_t i = 0; i < score_cpu.size(); i++) {
    float diff = std::abs(score_cpu.index<float>(i) - score_gpu.index<float>(i));
    rmse_diff += diff * diff;
    max_diff = std::max(max_diff, diff);
  }
  rmse_diff = std::sqrt(rmse_diff / score_cpu.size());

  // Print statistics about the differences
  std::cout << "RMSE difference between CPU and GPU softmax: " << rmse_diff << std::endl;
  std::cout << "Maximum absolute difference: " << max_diff << std::endl;

  // Check that the differences are within acceptable tolerance
  EXPECT_LT(rmse_diff, 1e-5f) << "RMSE difference between CPU and GPU softmax is too large";
  EXPECT_LT(max_diff, 1e-5f) << "Maximum difference between CPU and GPU softmax is too large";
}

TEST_F(MHAKernelGPUTest, MhaPrefill) {
  // Skip test if CUDA is not available
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available, skipping GPU test";
  }

  // Define different test parameter groups
  struct TestParams {
    int32_t batch_size;
    int32_t seq_len;
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t head_size;
    std::string description;
  };

  std::vector<TestParams> test_configurations = {
      // Small model configuration
      {1, 8, 4, 4, 64, "Small model (1 batch, 8 tokens, 4 heads)"},

      // Medium model configuration
      {2, 16, 8, 8, 128, "Medium model (2 batch, 16 tokens, 8 heads)"},

      // Large model configuration with GQA (grouped query attention)
      {4, 32, 16, 4, 128, "Large model with GQA (4 batch, 32 tokens, 16 heads, 4 KV heads)"},

      // Edge case: Single token
      {8, 128, 8, 8, 64, "Long sequence (128)"}};

  const int32_t layer_idx = 0;
  const int32_t num_layers = 1;

  for (const auto& params : test_configurations) {
    std::cout << "\nTesting " << params.description << std::endl;

    const int32_t batch_size = params.batch_size;
    const int32_t seq_len = params.seq_len;
    const int32_t num_heads = params.num_heads;
    const int32_t num_kv_heads = params.num_kv_heads;
    const int32_t head_size = params.head_size;
    const int32_t max_position_embedding =
        std::max(64, seq_len * 2);  // Make sure it's larger than seq_len

    // Create tensors for MHA operation
    tensor::Tensor query_tensor(core::DataType::FP32, {batch_size, seq_len, num_heads, head_size},
                                true, cpu_memory_manager);

    tensor::Tensor score_tensor(core::DataType::FP32, {batch_size, num_heads, seq_len, seq_len},
                                true, cpu_memory_manager);

    tensor::Tensor key_cache_tensor(
        core::DataType::FP32,
        {num_layers, batch_size, num_kv_heads, max_position_embedding, head_size}, true,
        cpu_memory_manager);

    tensor::Tensor value_cache_tensor(
        core::DataType::FP32,
        {num_layers, batch_size, num_kv_heads, max_position_embedding, head_size}, true,
        cpu_memory_manager);

    tensor::Tensor mha_output_cpu(core::DataType::FP32, {batch_size, seq_len, num_heads, head_size},
                                  true, cpu_memory_manager);

    tensor::Tensor mha_output_gpu(core::DataType::FP32, {batch_size, seq_len, num_heads, head_size},
                                  true, gpu_memory_manager);

    // Generate random data
    generate_random_tensor(query_tensor);
    generate_random_tensor(key_cache_tensor);
    generate_random_tensor(value_cache_tensor);

    // Clone tensors for GPU computation
    tensor::Tensor query_tensor_gpu = query_tensor.clone();
    tensor::Tensor key_cache_tensor_gpu = key_cache_tensor.clone();
    tensor::Tensor value_cache_tensor_gpu = value_cache_tensor.clone();
    tensor::Tensor score_tensor_gpu(core::DataType::FP32, {batch_size, num_heads, seq_len, seq_len},
                                    true, gpu_memory_manager);

    // Run CPU implementation
    mha_kernel_cpu(layer_idx, num_layers, batch_size, seq_len, seq_len, mha_output_cpu,
                   query_tensor, score_tensor, key_cache_tensor, value_cache_tensor);

    // Transfer tensors to GPU
    query_tensor_gpu.to_cuda();
    key_cache_tensor_gpu.to_cuda();
    value_cache_tensor_gpu.to_cuda();

    // Run GPU implementation
    mha_kernel_gpu(layer_idx, num_layers, batch_size, seq_len, seq_len, mha_output_gpu,
                   query_tensor_gpu, score_tensor_gpu, key_cache_tensor_gpu,
                   value_cache_tensor_gpu);

    // Transfer results back to CPU for comparison
    mha_output_gpu.to_cpu();

    // Calculate differences
    float rmse_diff = 0.0f;
    float max_diff = 0.0f;
    int max_diff_idx = -1;

    for (int32_t i = 0; i < mha_output_cpu.size(); i++) {
      float diff = std::abs(mha_output_cpu.index<float>(i) - mha_output_gpu.index<float>(i));
      rmse_diff += diff * diff;
      if (diff > max_diff) {
        max_diff = diff;
        max_diff_idx = i;
      }
    }
    rmse_diff = std::sqrt(rmse_diff / mha_output_cpu.size());

    // Print statistics about the differences
    std::cout << "RMSE difference: " << rmse_diff << std::endl;
    std::cout << "Maximum difference: " << max_diff;
    if (max_diff_idx >= 0) {
      std::cout << " at index " << max_diff_idx
                << " (CPU: " << mha_output_cpu.index<float>(max_diff_idx)
                << ", GPU: " << mha_output_gpu.index<float>(max_diff_idx) << ")";
    }
    std::cout << std::endl;

    // Check that the differences are within acceptable tolerance
    EXPECT_LT(rmse_diff, 1e-5f) << "RMSE difference between CPU and GPU MHA is too large for "
                                << params.description;
    EXPECT_LT(max_diff, 1e-5f) << "Maximum difference between CPU and GPU MHA is too large for "
                               << params.description;
  }
}

TEST_F(MHAKernelGPUTest, MhaDecode) {
  // Skip test if CUDA is not available
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available, skipping GPU test";
  }

  // Define different test parameter groups
  struct TestParams {
    int32_t batch_size;
    int32_t kv_seq_len;
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t head_size;
    std::string description;
  };

  std::vector<TestParams> test_configurations = {
      // Small model configuration
      {1, 8, 4, 4, 8, "Small model (1 batch, 8 tokens in context, 4 heads)"},

      // Medium model configuration
      {2, 16, 8, 8, 128, "Medium model (2 batch, 16 tokens in context, 8 heads)"},

      // Large model configuration with GQA (grouped query attention)
      {4, 32, 16, 4, 128,
       "Large model with GQA (4 batch, 32 tokens in context, 16 heads, 4 KV heads)"},

      // Large context length
      {1, 1024, 8, 8, 64, "Long context (1024 tokens)"}};

  const int32_t layer_idx = 0;
  const int32_t num_layers = 1;
  const int32_t query_seq_len = 1;  // In decoding, we generate one token at a time

  for (const auto& params : test_configurations) {
    std::cout << "\nTesting Decode: " << params.description << std::endl;

    const int32_t batch_size = params.batch_size;
    const int32_t kv_seq_len = params.kv_seq_len;
    const int32_t num_heads = params.num_heads;
    const int32_t num_kv_heads = params.num_kv_heads;
    const int32_t head_size = params.head_size;
    const int32_t max_position_embedding =
        std::max(1024, kv_seq_len * 2);  // Make sure it's larger than kv_seq_len

    // Create tensors for MHA operation
    // For decoding, query is a single token
    tensor::Tensor query_tensor(core::DataType::FP32,
                                {batch_size, query_seq_len, num_heads, head_size}, true,
                                cpu_memory_manager);

    // Score tensor for CPU implementation
    tensor::Tensor score_tensor_cpu(core::DataType::FP32,
                                    {batch_size, num_heads, query_seq_len, kv_seq_len}, true,
                                    cpu_memory_manager);

    // Key/value cache tensors (accumulated context)
    tensor::Tensor key_cache_tensor(
        core::DataType::FP32,
        {num_layers, batch_size, num_kv_heads, max_position_embedding, head_size}, true,
        cpu_memory_manager);

    tensor::Tensor value_cache_tensor(
        core::DataType::FP32,
        {num_layers, batch_size, num_kv_heads, max_position_embedding, head_size}, true,
        cpu_memory_manager);

    // Output tensors for CPU and GPU
    tensor::Tensor mha_output_cpu(core::DataType::FP32,
                                  {batch_size, query_seq_len, num_heads, head_size}, true,
                                  cpu_memory_manager);

    tensor::Tensor mha_output_gpu(core::DataType::FP32,
                                  {batch_size, query_seq_len, num_heads, head_size}, true,
                                  gpu_memory_manager);

    // Score tensor for GPU implementation - different shape from CPU
    tensor::Tensor score_tensor_gpu(core::DataType::FP32,
                                    {batch_size, num_heads, query_seq_len, kv_seq_len}, true,
                                    gpu_memory_manager);

    // Generate random data
    generate_random_tensor(query_tensor);
    generate_random_tensor(key_cache_tensor);
    generate_random_tensor(value_cache_tensor);

    // Clone tensors for GPU computation
    tensor::Tensor query_tensor_gpu = query_tensor.clone();
    tensor::Tensor key_cache_tensor_gpu = key_cache_tensor.clone();
    tensor::Tensor value_cache_tensor_gpu = value_cache_tensor.clone();

    // Run CPU implementation
    mha_kernel_cpu(layer_idx, num_layers, batch_size, query_seq_len, kv_seq_len, mha_output_cpu,
                   query_tensor, score_tensor_cpu, key_cache_tensor, value_cache_tensor);

    // Transfer tensors to GPU
    query_tensor_gpu.to_cuda();
    key_cache_tensor_gpu.to_cuda();
    value_cache_tensor_gpu.to_cuda();

    // Run GPU implementation
    mha_kernel_gpu(layer_idx, num_layers, batch_size, query_seq_len, kv_seq_len, mha_output_gpu,
                   query_tensor_gpu, score_tensor_gpu, key_cache_tensor_gpu,
                   value_cache_tensor_gpu);

    // Transfer results back to CPU for comparison
    mha_output_gpu.to_cpu();

    // Calculate differences
    float rmse_diff = 0.0f;
    float max_diff = 0.0f;
    int max_diff_idx = -1;

    for (int32_t i = 0; i < mha_output_cpu.size(); i++) {
      float diff = std::abs(mha_output_cpu.index<float>(i) - mha_output_gpu.index<float>(i));
      rmse_diff += diff * diff;
      if (diff > max_diff) {
        max_diff = diff;
        max_diff_idx = i;
      }
    }
    rmse_diff = std::sqrt(rmse_diff / mha_output_cpu.size());

    // Print statistics about the differences
    std::cout << "RMSE difference: " << rmse_diff << std::endl;
    std::cout << "Maximum difference: " << max_diff;
    if (max_diff_idx >= 0) {
      std::cout << " at index " << max_diff_idx
                << " (CPU: " << mha_output_cpu.index<float>(max_diff_idx)
                << ", GPU: " << mha_output_gpu.index<float>(max_diff_idx) << ")";
    }
    std::cout << std::endl;

    // Check that the differences are within acceptable tolerance
    EXPECT_LT(rmse_diff, 1e-5f)
        << "RMSE difference between CPU and GPU MHA decode is too large for " << params.description;
    EXPECT_LT(max_diff, 1e-5f)
        << "Maximum difference between CPU and GPU MHA decode is too large for "
        << params.description;
  }
}

}  // namespace
}  // namespace kernel