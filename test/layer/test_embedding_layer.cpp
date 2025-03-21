#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "config.hpp"
#include "core/memory/memory_manager.hpp"
#include "core/status/status.hpp"
#include "layer/embedding_layer.hpp"
#include "tensor/tensor.hpp"
#include "type.hpp"

namespace layer {
namespace {

class EmbeddingLayerTest : public ::testing::Test {
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

  // Test parameters
  const int32_t dim = 4;
  const int32_t seq_len = 3;
  const int32_t vocab_size = 10;

  // Input token indices
  std::vector<int32_t> input_tokens{2, 5, 7};

  // Expected embedding output for the tokens
  // Each token maps to a row in the weight matrix
  std::vector<float> expected_output{// Token 2 embedding
                                     0.3f, 0.4f, 0.5f, 0.6f,
                                     // Token 5 embedding
                                     0.6f, 0.7f, 0.8f, 0.9f,
                                     // Token 7 embedding
                                     0.8f, 0.9f, 1.0f, 1.1f};
};

TEST_F(EmbeddingLayerTest, CPU) {
  // Create embedding layer
  EmbeddingLayer cpu_embedding_layer(core::DeviceType::CPU, dim, seq_len, vocab_size);

  // Create input tensor with token indices
  tensor::Tensor input_tokens_tensor(core::DataType::INT32, input_tokens.size(), true,
                                     cpu_memory_manager, nullptr);
  tensor::Tensor input_token_num = tensor::make_scalar<int32_t>(seq_len, cpu_memory_manager);

  // Create weight tensor (embedding table)
  tensor::Tensor weight_tensor(core::DataType::FP32, vocab_size, dim, true, cpu_memory_manager,
                               nullptr);

  // Initialize weight tensor with values
  // We'll use a simple pattern where token i has embedding values starting at i/10
  for (int32_t i = 0; i < vocab_size; i++) {
    for (int32_t j = 0; j < dim; j++) {
      weight_tensor.at<float>(i, j) = (i / 10.0f) + ((j + 1) / 10.0f);
    }
  }

  // Create output tensor
  tensor::Tensor output_tensor(core::DataType::FP32, input_tokens.size(), dim, true,
                               cpu_memory_manager, nullptr);

  // Set input token values
  for (size_t i = 0; i < input_tokens.size(); i++) {
    input_tokens_tensor.index<int32_t>(i) = input_tokens[i];
  }

  // Set layer inputs and outputs
  cpu_embedding_layer.set_input(0, input_tokens_tensor);
  cpu_embedding_layer.set_input(1, input_token_num);
  cpu_embedding_layer.set_weight(0, weight_tensor);
  cpu_embedding_layer.set_output(0, output_tensor);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), cpu_embedding_layer.check());
  ASSERT_EQ(core::error::Success(), cpu_embedding_layer.forward());

  // Verify the output
  for (int32_t i = 0; i < input_tokens.size(); i++) {
    for (int32_t j = 0; j < dim; j++) {
      EXPECT_FLOAT_EQ(expected_output[i * dim + j], output_tensor.at<float>(i, j))
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }
}

TEST_F(EmbeddingLayerTest, GPU) {
  // Skip GPU test if CUDA is not available
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA is not available, skipping GPU test";
    return;
  }

  // Create embedding layer
  EmbeddingLayer gpu_embedding_layer(core::DeviceType::GPU, dim, seq_len, vocab_size);
  auto cuda_config = std::make_shared<core::CudaConfig>();
  gpu_embedding_layer.set_cuda_config(cuda_config);

  // Create input tensors
  tensor::Tensor input_tokens_tensor(core::DataType::INT32, input_tokens.size(), true,
                                     cpu_memory_manager, nullptr);
  // Create input_token_num tensor, used as a scalar
  tensor::Tensor input_token_num = tensor::make_scalar<int32_t>(seq_len, cpu_memory_manager);

  // Create weight tensor (embedding table)
  tensor::Tensor weight_tensor(core::DataType::FP32, vocab_size, dim, true, cpu_memory_manager,
                               nullptr);

  // Initialize weight tensor with values
  for (int32_t i = 0; i < vocab_size; i++) {
    for (int32_t j = 0; j < dim; j++) {
      weight_tensor.at<float>(i, j) = (i / 10.0f) + ((j + 1) / 10.0f);
    }
  }

  // Create output tensor (on GPU memory)
  tensor::Tensor output_tensor(core::DataType::FP32, input_tokens.size(), dim, true,
                               gpu_memory_manager, nullptr);

  // Set input token values
  for (size_t i = 0; i < input_tokens.size(); i++) {
    input_tokens_tensor.index<int32_t>(i) = input_tokens[i];
  }

  // Transfer tensors to GPU
  input_tokens_tensor.to_cuda();
  weight_tensor.to_cuda();

  // Set layer inputs and outputs
  gpu_embedding_layer.set_input(0, input_tokens_tensor);
  gpu_embedding_layer.set_input(1, input_token_num);
  gpu_embedding_layer.set_weight(0, weight_tensor);
  gpu_embedding_layer.set_output(0, output_tensor);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), gpu_embedding_layer.check());
  ASSERT_EQ(core::error::Success(), gpu_embedding_layer.forward());

  // Copy GPU results back to CPU for verification
  tensor::Tensor output_gpu_cpu = output_tensor.clone();
  output_gpu_cpu.to_cpu();

  // Verify the output
  for (int32_t i = 0; i < input_tokens.size(); i++) {
    for (int32_t j = 0; j < dim; j++) {
      EXPECT_FLOAT_EQ(expected_output[i * dim + j], output_gpu_cpu.at<float>(i, j))
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }
}

}  // namespace
}  // namespace layer
