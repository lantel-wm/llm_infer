#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <vector>
#include "embedding_kernel_cpu.hpp"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class EmbeddingKernelTest : public ::testing::Test {
 protected:
  void SetUp() override { cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance(); }

  void TearDown() override {
#ifndef PYTORCH_NOT_FOUND
    // Clean up generated files
    std::remove("generate_embedding_data.py");
    std::remove("embedding_input.bin");
    std::remove("embedding_weight.bin");
    std::remove("embedding_output.bin");
#endif
  }

  std::shared_ptr<core::CPUMemoryManager> cpu_memory_manager;

  // Helper function to read binary data
  std::vector<float> read_binary_file(const std::string& filename, size_t size) {
    std::vector<float> data(size);
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
      file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
      file.close();
    }
    return data;
  }

  // Helper function to read binary int data
  std::vector<int32_t> read_binary_int_file(const std::string& filename, size_t size) {
    std::vector<int32_t> data(size);
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
      file.read(reinterpret_cast<char*>(data.data()), size * sizeof(int32_t));
      file.close();
    }
    return data;
  }
};

// Test basic embedding functionality with simple values
TEST_F(EmbeddingKernelTest, BasicEmbedding) {
  const int32_t vocab_size = 10;
  const int32_t embedding_dim = 4;
  const int32_t seq_length = 3;

  // Create input tensor (token indices)
  tensor::Tensor input(core::DataType::INT32, seq_length, true, cpu_memory_manager);

  // Create weight tensor (embedding table)
  tensor::Tensor weight(core::DataType::FP32, {vocab_size, embedding_dim}, true,
                        cpu_memory_manager);

  // Create output tensor
  tensor::Tensor output(core::DataType::FP32, {seq_length, embedding_dim}, true,
                        cpu_memory_manager);

  // Initialize input tensor with token indices
  input.at<int32_t>(0) = 1;  // First token is index 1
  input.at<int32_t>(1) = 3;  // Second token is index 3
  input.at<int32_t>(2) = 5;  // Third token is index 5

  // Initialize weight tensor with known values
  for (int32_t i = 0; i < vocab_size; i++) {
    for (int32_t j = 0; j < embedding_dim; j++) {
      weight.at<float>(i, j) = static_cast<float>(i * 0.1f + j);
    }
  }

  // Run embedding
  embedding_kernel_cpu(input, weight, output, vocab_size);

  // Verify results
  for (int32_t i = 0; i < seq_length; i++) {
    int32_t token_idx = input.at<int32_t>(i);
    for (int32_t j = 0; j < embedding_dim; j++) {
      float expected = static_cast<float>(token_idx * 0.1f + j);
      EXPECT_NEAR(output.at<float>(i, j), expected, 1e-6f)
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }
}

// Test embedding with different dimensions
TEST_F(EmbeddingKernelTest, DifferentDimensions) {
  const std::vector<int32_t> embedding_dims = {16, 64, 128};
  const int32_t vocab_size = 100;
  const int32_t seq_length = 5;

  for (const auto& dim : embedding_dims) {
    // Create input tensor (token indices)
    tensor::Tensor input(core::DataType::INT32, seq_length, true, cpu_memory_manager);

    // Create weight tensor (embedding table)
    tensor::Tensor weight(core::DataType::FP32, {vocab_size, dim}, true, cpu_memory_manager);

    // Create output tensor
    tensor::Tensor output(core::DataType::FP32, {seq_length, dim}, true, cpu_memory_manager);

    // Initialize input tensor with token indices
    for (int32_t i = 0; i < seq_length; i++) {
      input.at<int32_t>(i) = i * 10;  // Use different token indices
    }

    // Initialize weight tensor with known values
    for (int32_t i = 0; i < vocab_size; i++) {
      for (int32_t j = 0; j < dim; j++) {
        weight.at<float>(i, j) = static_cast<float>(i) / static_cast<float>(j + 1);
      }
    }

    // Run embedding
    embedding_kernel_cpu(input, weight, output, vocab_size);

    // Verify results
    for (int32_t i = 0; i < seq_length; i++) {
      int32_t token_idx = input.at<int32_t>(i);
      for (int32_t j = 0; j < dim; j++) {
        float expected = static_cast<float>(token_idx) / static_cast<float>(j + 1);
        EXPECT_NEAR(output.at<float>(i, j), expected, 1e-6f)
            << "Mismatch at position [" << i << ", " << j << "] with dim " << dim;
      }
    }
  }
}

// Test embedding with edge case token indices
TEST_F(EmbeddingKernelTest, EdgeCaseTokens) {
  const int32_t vocab_size = 10;
  const int32_t embedding_dim = 4;
  const int32_t seq_length = 3;

  // Create input tensor (token indices)
  tensor::Tensor input(core::DataType::INT32, seq_length, true, cpu_memory_manager);

  // Create weight tensor (embedding table)
  tensor::Tensor weight(core::DataType::FP32, {vocab_size, embedding_dim}, true,
                        cpu_memory_manager);

  // Create output tensor
  tensor::Tensor output(core::DataType::FP32, {seq_length, embedding_dim}, true,
                        cpu_memory_manager);

  // Initialize input tensor with edge case token indices
  input.at<int32_t>(0) = 0;               // First token (index 0)
  input.at<int32_t>(1) = vocab_size - 1;  // Last token in vocabulary
  input.at<int32_t>(2) = 5;               // Middle token

  // Initialize weight tensor with known values
  for (int32_t i = 0; i < vocab_size; i++) {
    for (int32_t j = 0; j < embedding_dim; j++) {
      weight.at<float>(i, j) = static_cast<float>(i + j);
    }
  }

  // Run embedding
  embedding_kernel_cpu(input, weight, output, vocab_size);

  // Verify results
  for (int32_t i = 0; i < seq_length; i++) {
    int32_t token_idx = input.at<int32_t>(i);
    for (int32_t j = 0; j < embedding_dim; j++) {
      float expected = static_cast<float>(token_idx + j);
      EXPECT_NEAR(output.at<float>(i, j), expected, 1e-6f)
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }
}

#ifndef PYTORCH_NOT_FOUND
// Test embedding against PyTorch implementation
TEST_F(EmbeddingKernelTest, CompareWithPyTorch) {
  const int32_t vocab_size = 30000;
  const int32_t embedding_dim = 1024;
  const int32_t seq_length = 10;

  // Create Python script for generating test data
  {
    std::ofstream py_file("generate_embedding_data.py");
    py_file << R"(
import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
vocab_size = 30000
embedding_dim = 1024
seq_length = 10

# Generate random input indices
input_data = torch.randint(0, vocab_size, (seq_length,), dtype=torch.int32)

# Generate random embedding weights
weight_data = torch.randn(vocab_size, embedding_dim, dtype=torch.float32)

# Compute embedding using PyTorch
output_data = torch.nn.functional.embedding(input_data, weight_data)

# Save data to files
input_data.numpy().tofile('embedding_input.bin')
weight_data.numpy().tofile('embedding_weight.bin')
output_data.detach().numpy().tofile('embedding_output.bin')
)";
    py_file.close();
  }

  // Run Python script to generate test data
  ASSERT_EQ(std::system("python3 generate_embedding_data.py"), 0) << "Failed to run Python script";

  // Read test data generated by PyTorch
  std::vector<int32_t> input_data = read_binary_int_file("embedding_input.bin", seq_length);
  std::vector<float> weight_data =
      read_binary_file("embedding_weight.bin", vocab_size * embedding_dim);
  std::vector<float> expected_output =
      read_binary_file("embedding_output.bin", seq_length * embedding_dim);

  // Create and initialize tensors
  tensor::Tensor input(core::DataType::INT32, seq_length, true, cpu_memory_manager);
  tensor::Tensor weight(core::DataType::FP32, {vocab_size, embedding_dim}, true,
                        cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, {seq_length, embedding_dim}, true,
                        cpu_memory_manager);

  // Copy data to tensors
  std::memcpy(input.ptr<int32_t>(), input_data.data(), seq_length * sizeof(int32_t));
  std::memcpy(weight.ptr<float>(), weight_data.data(), vocab_size * embedding_dim * sizeof(float));

  // Run our embedding implementation
  embedding_kernel_cpu(input, weight, output, vocab_size);

  // Compare results with PyTorch
  for (int32_t i = 0; i < seq_length; i++) {
    for (int32_t j = 0; j < embedding_dim; j++) {
      EXPECT_NEAR(output.at<float>(i, j), expected_output[i * embedding_dim + j], 1e-5f)
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }

  // Clean up generated files
  std::remove("generate_embedding_data.py");
  std::remove("embedding_input.bin");
  std::remove("embedding_weight.bin");
  std::remove("embedding_output.bin");
}
#endif

}  // namespace
}  // namespace kernel

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
