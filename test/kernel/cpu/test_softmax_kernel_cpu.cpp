#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <vector>
#include "softmax_kernel_cpu.hpp"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class SoftmaxKernelTest : public ::testing::Test {
 protected:
  void SetUp() override { cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance(); }

  void TearDown() override {
#ifndef PYTORCH_NOT_FOUND
    // Clean up generated files
    std::remove("generate_softmax_data.py");
    std::remove("softmax_input.bin");
    std::remove("softmax_output.bin");
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

// Test basic softmax functionality with simple values
TEST_F(SoftmaxKernelTest, BasicSoftmax) {
  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  const int size = 4;

  tensor::Tensor input(core::DataType::FP32, size, true, cpu_memory_manager);

  // Initialize input tensor
  for (int i = 0; i < size; i++) {
    input.at<float>(i) = input_data[i];
  }

  // Run softmax
  softmax_kernel_cpu(input, nullptr);

  // Calculate expected values manually
  float max_val = *std::max_element(input_data.begin(), input_data.end());
  std::vector<float> exp_values(size);
  float sum = 0.0f;

  for (int i = 0; i < size; i++) {
    exp_values[i] = std::exp(input_data[i] - max_val);
    sum += exp_values[i];
  }

  // Normalize
  for (int i = 0; i < size; i++) {
    exp_values[i] /= sum;
  }

  // Verify results
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(input.at<float>(i), exp_values[i], 1e-6f);
  }
}

// Test softmax with different tensor sizes
TEST_F(SoftmaxKernelTest, DifferentSizes) {
  const std::vector<int> sizes = {16, 256, 1024, 4096};

  for (const auto& size : sizes) {
    tensor::Tensor input(core::DataType::FP32, size, true, cpu_memory_manager);

    // Initialize with increasing values
    for (int i = 0; i < size; i++) {
      input.at<float>(i) = static_cast<float>(i) / static_cast<float>(size);
    }

    // Run softmax
    softmax_kernel_cpu(input, nullptr);

    // Verify sum is close to 1.0
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
      sum += input.at<float>(i);
      // All values should be positive and less than 1
      EXPECT_GT(input.at<float>(i), 0.0f);
      EXPECT_LT(input.at<float>(i), 1.0f);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
  }
}

// Test softmax with uniform values
TEST_F(SoftmaxKernelTest, UniformValues) {
  const int size = 4;
  tensor::Tensor input(core::DataType::FP32, size, true, cpu_memory_manager);

  // Initialize with same value
  for (int i = 0; i < size; i++) {
    input.at<float>(i) = 1.0f;
  }

  // Run softmax
  softmax_kernel_cpu(input, nullptr);

  // For uniform input, output should be uniform with value 1/size
  float expected = 1.0f / size;
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(input.at<float>(i), expected, 1e-6f);
  }
}

// Test softmax with extreme values
TEST_F(SoftmaxKernelTest, ExtremeValues) {
  const int size = 4;
  tensor::Tensor input(core::DataType::FP32, size, true, cpu_memory_manager);

  // Initialize with extreme values
  input.at<float>(0) = 1000.0f;   // Very large positive
  input.at<float>(1) = -1000.0f;  // Very large negative
  input.at<float>(2) = 0.0f;      // Zero
  input.at<float>(3) = 1.0f;      // Small positive

  // Run softmax
  softmax_kernel_cpu(input, nullptr);

  // The largest value should get probability close to 1
  EXPECT_NEAR(input.at<float>(0), 1.0f, 1e-6f);

  // Other values should be close to 0
  EXPECT_NEAR(input.at<float>(1), 0.0f, 1e-6f);
  EXPECT_NEAR(input.at<float>(2), 0.0f, 1e-6f);
  EXPECT_NEAR(input.at<float>(3), 0.0f, 1e-6f);

  // Sum should still be 1
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += input.at<float>(i);
  }
  EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

#ifndef PYTORCH_NOT_FOUND
// Test softmax against PyTorch implementation
TEST_F(SoftmaxKernelTest, CompareWithPyTorch) {
  const int size = 1024;  // Must match the size in Python script

  // Create Python script for generating test data
  {
    std::ofstream py_file("generate_softmax_data.py");
    py_file << R"(
import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate random input
input_size = 1024
input_data = torch.randn(input_size, dtype=torch.float32)

# Compute softmax using PyTorch
output_data = torch.nn.functional.softmax(input_data, dim=0)

# Save data to files
input_data.numpy().tofile('softmax_input.bin')
output_data.detach().numpy().tofile('softmax_output.bin')
)";
    py_file.close();
  }

  // Run Python script to generate test data
  ASSERT_EQ(std::system("python3 generate_softmax_data.py"), 0) << "Failed to run Python script";

  // Read test data generated by PyTorch
  std::vector<float> input_data = read_binary_file("softmax_input.bin", size);
  std::vector<float> expected_output = read_binary_file("softmax_output.bin", size);

  // Create and initialize tensors
  tensor::Tensor input(core::DataType::FP32, size, true, cpu_memory_manager);

  // Copy data to tensors
  std::memcpy(input.ptr<float>(), input_data.data(), size * sizeof(float));

  // Run our softmax implementation
  softmax_kernel_cpu(input, nullptr);

  // Compare results with PyTorch
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(input.at<float>(i), expected_output[i], 1e-5f)
        << "Mismatch at index " << i << ". Expected: " << expected_output[i]
        << ", Got: " << input.at<float>(i)
        << ", Diff: " << std::fabs(input.at<float>(i) - expected_output[i]);
  }

  // Clean up generated files
  std::remove("generate_softmax_data.py");
  std::remove("softmax_input.bin");
  std::remove("softmax_output.bin");
}
#endif

}  // namespace
}  // namespace kernel

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
