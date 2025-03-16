#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <vector>
#include "rmsnorm_kernel_cpu.hpp"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class RMSNormKernelTest : public ::testing::Test {
 protected:
  void SetUp() override { cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance(); }

  void TearDown() override {
#ifndef PYTORCH_NOT_FOUND
    // Clean up generated files
    std::remove("generate_rmsnorm_data.py");
    std::remove("rmsnorm_input.bin");
    std::remove("rmsnorm_weight.bin");
    std::remove("rmsnorm_output.bin");
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
};

// Test basic RMSNorm functionality with simple values
TEST_F(RMSNormKernelTest, BasicNormalization) {
  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f};  // Identity weights
  const int size = 4;

  tensor::Tensor input(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor weight(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, size, true, cpu_memory_manager);

  // Initialize input and weight tensors
  for (int i = 0; i < size; i++) {
    input.at<float>(i) = input_data[i];
    weight.at<float>(i) = weight_data[i];
  }

  rmsnorm_kernel_cpu(input, weight, output, nullptr);

  // Calculate expected values
  float mean_square = 0.0f;
  for (float val : input_data) {
    mean_square += val * val;
  }
  mean_square = mean_square / size + 1e-5f;  // Add epsilon
  float scale = 1.0f / std::sqrt(mean_square);

  // Verify results
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(output.at<float>(i), input_data[i] * scale, 1e-6f);
  }
}

// Test RMSNorm with different tensor sizes
TEST_F(RMSNormKernelTest, DifferentSizes) {
  const std::vector<int> sizes = {1, 16, 256, 4096};

  for (const auto& size : sizes) {
    tensor::Tensor input(core::DataType::FP32, size, true, cpu_memory_manager);
    tensor::Tensor weight(core::DataType::FP32, size, true, cpu_memory_manager);
    tensor::Tensor output(core::DataType::FP32, size, true, cpu_memory_manager);

    // Initialize with alternating values
    for (int i = 0; i < size; i++) {
      input.at<float>(i) = (i % 2 == 0) ? 1.0f : -1.0f;
      weight.at<float>(i) = 1.0f;  // Identity weights
    }

    rmsnorm_kernel_cpu(input, weight, output, nullptr);

    // For alternating +1/-1, the RMS will be 1.0
    // So after normalization, values should remain unchanged
    for (int i = 0; i < size; i++) {
      EXPECT_NEAR(std::abs(output.at<float>(i)), 1.0f / std::sqrt(1.0f + 1e-5f), 1e-6f);
    }
  }
}

// Test RMSNorm with custom weights
TEST_F(RMSNormKernelTest, CustomWeights) {
  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> weight_data = {0.5f, 1.0f, 1.5f, 2.0f};
  const int size = 4;

  tensor::Tensor input(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor weight(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, size, true, cpu_memory_manager);

  // Initialize input and weight tensors
  for (int i = 0; i < size; i++) {
    input.at<float>(i) = input_data[i];
    weight.at<float>(i) = weight_data[i];
  }

  rmsnorm_kernel_cpu(input, weight, output, nullptr);

  // Calculate expected values
  float mean_square = 0.0f;
  for (float val : input_data) {
    mean_square += val * val;
  }
  mean_square = mean_square / size + 1e-5f;
  float scale = 1.0f / std::sqrt(mean_square);

  // Verify results with custom weights
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(output.at<float>(i), input_data[i] * scale * weight_data[i], 1e-6f);
  }
}

// Test RMSNorm with zero input
TEST_F(RMSNormKernelTest, ZeroInput) {
  const int size = 4;
  tensor::Tensor input(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor weight(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, size, true, cpu_memory_manager);

  // Initialize with zeros
  for (int i = 0; i < size; i++) {
    input.at<float>(i) = 0.0f;
    weight.at<float>(i) = 1.0f;
  }

  rmsnorm_kernel_cpu(input, weight, output, nullptr);

  // When input is zero, output should be zero
  // (after normalization by sqrt(eps))
  float expected = 0.0f / std::sqrt(1e-5f);
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(output.at<float>(i), expected, 1e-6f);
  }
}

#ifndef PYTORCH_NOT_FOUND
// Test RMSNorm against PyTorch implementation
TEST_F(RMSNormKernelTest, CompareWithPyTorch) {
  const int size = 1024;  // Must match the size in Python script

  // Create Python script for generating test data
  {
    std::ofstream py_file("generate_rmsnorm_data.py");
    py_file << R"(
import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate random input
input_size = 1024
input_data = torch.randn(input_size, dtype=torch.float32)
weight_data = torch.randn(input_size, dtype=torch.float32)

# Create RMSNorm layer
rmsnorm = torch.nn.RMSNorm(input_size, elementwise_affine=True, eps=1e-5)
rmsnorm.weight = torch.nn.Parameter(weight_data)
rmsnorm.bias = torch.nn.Parameter(torch.zeros(input_size, dtype=torch.float32))

# Compute output
output_data = rmsnorm(input_data)

# Save data to files
input_data.numpy().tofile('rmsnorm_input.bin')
weight_data.numpy().tofile('rmsnorm_weight.bin')
output_data.detach().numpy().tofile('rmsnorm_output.bin')
)";
    py_file.close();
  }

  // Run Python script to generate test data
  ASSERT_EQ(std::system("python3 generate_rmsnorm_data.py"), 0) << "Failed to run Python script";

  // Read test data generated by PyTorch
  std::vector<float> input_data = read_binary_file("rmsnorm_input.bin", size);
  std::vector<float> weight_data = read_binary_file("rmsnorm_weight.bin", size);
  std::vector<float> expected_output = read_binary_file("rmsnorm_output.bin", size);

  // Create and initialize tensors
  tensor::Tensor input(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor weight(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, size, true, cpu_memory_manager);

  // Copy data to tensors
  std::memcpy(input.ptr<float>(), input_data.data(), size * sizeof(float));
  std::memcpy(weight.ptr<float>(), weight_data.data(), size * sizeof(float));

  // Run our RMSNorm implementation
  rmsnorm_kernel_cpu(input, weight, output, nullptr);

  // Compare results with PyTorch
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(output.at<float>(i), expected_output[i], 1e-5f)
        << "Mismatch at index " << i << ". Expected: " << expected_output[i]
        << ", Got: " << output.at<float>(i)
        << ", Diff: " << std::fabs(output.at<float>(i) - expected_output[i]);
  }

  // Clean up generated files
  std::remove("generate_rmsnorm_data.py");
  std::remove("rmsnorm_input.bin");
  std::remove("rmsnorm_weight.bin");
  std::remove("rmsnorm_output.bin");
}
#endif

}  // namespace
}  // namespace kernel

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
