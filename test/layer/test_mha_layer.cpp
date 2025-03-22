#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "config.hpp"
#include "core/memory/memory_manager.hpp"
#include "core/status/status.hpp"
#include "layer/mha_layer.hpp"
#include "tensor/tensor.hpp"
#include "type.hpp"

namespace layer {
namespace {

class MhaLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance();

    // Check if CUDA is available
    int device_count = 0;
    cuda_available = (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0);

    if (cuda_available) {
      gpu_memory_manager = core::GPUMemoryManagerFactory::get_instance();
    }
  }

  std::shared_ptr<core::CPUMemoryManager> cpu_memory_manager;
  std::shared_ptr<core::GPUMemoryManager> gpu_memory_manager;
  bool cuda_available = false;

  // Test parameters for MHA layer
  const int32_t layer_idx = 0;
  const int32_t num_layers = 8;
  const int32_t batch_size = 1;
  const int32_t kv_mul = 1;  // For grouped-query attention, GQA=1 means regular attention
  const int32_t num_heads = 4;
  const int32_t head_size = 16;
  const int32_t hidden_size = num_heads * head_size;
  const int32_t kv_size = num_heads * head_size;  // Same as hidden_size for non-MQA/GQA
  const int32_t seq_len = 8;
  const int32_t query_seq_len = seq_len;
  const int32_t kv_seq_len = seq_len;
};

TEST_F(MhaLayerTest, CPUForward) {
  // Create MHA layer
  MultiHeadAttention mha_layer(core::DeviceType::CPU, layer_idx, num_layers, batch_size, kv_mul,
                               kv_size, query_seq_len, kv_seq_len, num_heads, head_size);

  // Create query tensor: [batch_size, seq_len, num_heads, head_size]
  tensor::Tensor query_tensor(core::DataType::FP32,
                              std::vector<int32_t>{batch_size, query_seq_len, num_heads, head_size},
                              true, cpu_memory_manager);

  // Create score tensor: [batch_size, num_heads, query_seq_len, kv_seq_len]
  tensor::Tensor score_tensor(
      core::DataType::FP32, std::vector<int32_t>{batch_size, num_heads, query_seq_len, kv_seq_len},
      true, cpu_memory_manager);

  // Create key cache tensor: [num_layers, batch_size, num_heads, kv_seq_len, head_size]
  tensor::Tensor key_cache_tensor(
      core::DataType::FP32,
      std::vector<int32_t>{num_layers, batch_size, num_heads, kv_seq_len, head_size}, true,
      cpu_memory_manager);

  // Create value cache tensor: [num_layers, batch_size, num_heads, kv_seq_len, head_size]
  tensor::Tensor value_cache_tensor(
      core::DataType::FP32,
      std::vector<int32_t>{num_layers, batch_size, num_heads, kv_seq_len, head_size}, true,
      cpu_memory_manager);

  // Create position tensor: [kv_seq_len]
  tensor::Tensor pos_tensor(core::DataType::FP32, kv_seq_len, true, cpu_memory_manager);

  // Create output tensor: [batch_size, query_seq_len, num_heads, head_size]
  tensor::Tensor output_tensor(
      core::DataType::FP32, std::vector<int32_t>{batch_size, query_seq_len, num_heads, head_size},
      true, cpu_memory_manager);

  // Initialize tensors with test values
  // Initialize query with simple pattern
  for (int i = 0; i < batch_size * query_seq_len * num_heads * head_size; i++) {
    query_tensor.index<float>(i) = static_cast<float>(i % 10) * 0.1f;
  }

  // Initialize key cache with a pattern
  for (int i = 0; i < num_layers * batch_size * num_heads * kv_seq_len * head_size; i++) {
    key_cache_tensor.index<float>(i) = static_cast<float>((i + 3) % 10) * 0.1f;
  }

  // Initialize value cache with a pattern
  for (int i = 0; i < num_layers * batch_size * num_heads * kv_seq_len * head_size; i++) {
    value_cache_tensor.index<float>(i) = static_cast<float>((i + 5) % 10) * 0.1f;
  }

  // Init position tensor
  for (int i = 0; i < kv_seq_len; i++) {
    pos_tensor.index<float>(i) = static_cast<float>(i);
  }

  // Set layer inputs and outputs
  mha_layer.set_input(0, query_tensor);
  mha_layer.set_input(1, score_tensor);
  mha_layer.set_input(2, key_cache_tensor);
  mha_layer.set_input(3, value_cache_tensor);
  mha_layer.set_input(4, pos_tensor);
  mha_layer.set_output(0, output_tensor);

  // Run checks
  ASSERT_EQ(core::error::Success(), mha_layer.check());

  // Run forward pass
  ASSERT_EQ(core::error::Success(), mha_layer.forward());

  // We're not testing specific values, but the function should execute without errors
  // and produce non-zero output
  bool has_non_zero = false;
  for (int i = 0; i < batch_size * query_seq_len * num_heads * head_size; i++) {
    if (std::abs(output_tensor.index<float>(i)) > 1e-6) {
      has_non_zero = true;
      break;
    }
  }
  EXPECT_TRUE(has_non_zero) << "Output tensor contains only zeros";
}

TEST_F(MhaLayerTest, SetLayerIdx) {
  // Create MHA layer
  MultiHeadAttention mha_layer(core::DeviceType::CPU, layer_idx, num_layers, batch_size, kv_mul,
                               kv_size, query_seq_len, kv_seq_len, num_heads, head_size);

  // Test set_layer_idx functionality
  const int32_t new_layer_idx = 2;
  mha_layer.set_layer_idx(new_layer_idx);

  // We can't directly check private member values,
  // but we can verify layer works correctly after the update

  // Create tensors
  tensor::Tensor query_tensor(core::DataType::FP32,
                              std::vector<int32_t>{batch_size, query_seq_len, num_heads, head_size},
                              true, cpu_memory_manager);
  tensor::Tensor score_tensor(
      core::DataType::FP32, std::vector<int32_t>{batch_size, num_heads, query_seq_len, kv_seq_len},
      true, cpu_memory_manager);
  tensor::Tensor key_cache_tensor(
      core::DataType::FP32,
      std::vector<int32_t>{num_layers, batch_size, num_heads, kv_seq_len, head_size}, true,
      cpu_memory_manager);
  tensor::Tensor value_cache_tensor(
      core::DataType::FP32,
      std::vector<int32_t>{num_layers, batch_size, num_heads, kv_seq_len, head_size}, true,
      cpu_memory_manager);
  tensor::Tensor pos_tensor(core::DataType::FP32, kv_seq_len, true, cpu_memory_manager);
  tensor::Tensor output_tensor(
      core::DataType::FP32, std::vector<int32_t>{batch_size, query_seq_len, num_heads, head_size},
      true, cpu_memory_manager);

  // Initialize tensors with test values
  // Initialize query with simple pattern
  for (int i = 0; i < batch_size * query_seq_len * num_heads * head_size; i++) {
    query_tensor.index<float>(i) = static_cast<float>(i % 10) * 0.1f;
  }

  // Initialize key cache with a pattern
  for (int i = 0; i < num_layers * batch_size * num_heads * kv_seq_len * head_size; i++) {
    key_cache_tensor.index<float>(i) = static_cast<float>((i + 3) % 10) * 0.1f;
  }

  // Initialize value cache with a pattern
  for (int i = 0; i < num_layers * batch_size * num_heads * kv_seq_len * head_size; i++) {
    value_cache_tensor.index<float>(i) = static_cast<float>((i + 5) % 10) * 0.1f;
  }

  // Set layer inputs and outputs
  mha_layer.set_input(0, query_tensor);
  mha_layer.set_input(1, score_tensor);
  mha_layer.set_input(2, key_cache_tensor);
  mha_layer.set_input(3, value_cache_tensor);
  mha_layer.set_input(4, pos_tensor);
  mha_layer.set_output(0, output_tensor);

  // Verify checks pass after layer_idx update
  ASSERT_EQ(core::error::Success(), mha_layer.check());
  // Verify forward works after layer_idx update
  ASSERT_EQ(core::error::Success(), mha_layer.forward());
}

TEST_F(MhaLayerTest, GPU) {
  // Skip GPU test if CUDA is not available
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA is not available, skipping GPU test";
    return;
  }

  // Create MHA layer for GPU
  MultiHeadAttention mha_layer(core::DeviceType::GPU, layer_idx, num_layers, batch_size, kv_mul,
                               kv_size, query_seq_len, kv_seq_len, num_heads, head_size);

  // Set CUDA config
  auto cuda_config = std::make_shared<core::CudaConfig>();
  mha_layer.set_cuda_config(cuda_config);

  // Create tensors in CPU memory first
  tensor::Tensor query_tensor(core::DataType::FP32,
                              std::vector<int32_t>{batch_size, query_seq_len, num_heads, head_size},
                              true, cpu_memory_manager);
  tensor::Tensor score_tensor(
      core::DataType::FP32, std::vector<int32_t>{batch_size, num_heads, query_seq_len, kv_seq_len},
      true, cpu_memory_manager);
  tensor::Tensor key_cache_tensor(
      core::DataType::FP32,
      std::vector<int32_t>{num_layers, batch_size, num_heads, kv_seq_len, head_size}, true,
      cpu_memory_manager);
  tensor::Tensor value_cache_tensor(
      core::DataType::FP32,
      std::vector<int32_t>{num_layers, batch_size, num_heads, kv_seq_len, head_size}, true,
      cpu_memory_manager);
  tensor::Tensor pos_tensor(core::DataType::FP32, kv_seq_len, true, cpu_memory_manager);
  tensor::Tensor output_tensor(
      core::DataType::FP32, std::vector<int32_t>{batch_size, query_seq_len, num_heads, head_size},
      true, gpu_memory_manager);

  // Initialize tensors with test values
  // Initialize query with simple pattern
  for (int i = 0; i < batch_size * query_seq_len * num_heads * head_size; i++) {
    query_tensor.index<float>(i) = static_cast<float>(i % 10) * 0.1f;
  }

  // Initialize key cache with a pattern
  for (int i = 0; i < num_layers * batch_size * num_heads * kv_seq_len * head_size; i++) {
    key_cache_tensor.index<float>(i) = static_cast<float>((i + 3) % 10) * 0.1f;
  }

  // Initialize value cache with a pattern
  for (int i = 0; i < num_layers * batch_size * num_heads * kv_seq_len * head_size; i++) {
    value_cache_tensor.index<float>(i) = static_cast<float>((i + 5) % 10) * 0.1f;
  }

  // Move tensors to GPU
  query_tensor.to_cuda();
  score_tensor.to_cuda();
  key_cache_tensor.to_cuda();
  value_cache_tensor.to_cuda();
  pos_tensor.to_cuda();

  // Set layer inputs and outputs
  mha_layer.set_input(0, query_tensor);
  mha_layer.set_input(1, score_tensor);
  mha_layer.set_input(2, key_cache_tensor);
  mha_layer.set_input(3, value_cache_tensor);
  mha_layer.set_input(4, pos_tensor);
  mha_layer.set_output(0, output_tensor);

  // Run checks
  ASSERT_EQ(core::error::Success(), mha_layer.check());

  // Run forward pass
  ASSERT_EQ(core::error::Success(), mha_layer.forward());

  // Copy GPU output back to CPU for verification
  tensor::Tensor output_gpu_cpu = output_tensor.clone();
  output_gpu_cpu.to_cpu();

  // We're not testing specific values, but the function should execute without errors
  // and produce non-zero output
  bool has_non_zero = false;
  for (int i = 0; i < batch_size * query_seq_len * num_heads * head_size; i++) {
    if (std::abs(output_gpu_cpu.index<float>(i)) > 1e-6) {
      has_non_zero = true;
      break;
    }
  }
  EXPECT_TRUE(has_non_zero) << "Output tensor contains only zeros";
}

}  // namespace
}  // namespace layer
