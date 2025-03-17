#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <vector>
#include "swiglu_kernel_cpu.hpp"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class SwiGLUKernelTest : public ::testing::Test {
 protected:
  void SetUp() override { cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance(); }

  void TearDown() override {
#ifndef PYTORCH_NOT_FOUND
    // Clean up generated files
    std::remove("generate_swiglu_data.py");
    std::remove("swiglu_input1.bin");
    std::remove("swiglu_input2.bin");
    std::remove("swiglu_output.bin");
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

// Test basic SwiGLU functionality with simple values
TEST_F(SwiGLUKernelTest, BasicOperation) {
  const std::vector<float> input1_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  const std::vector<float> input2_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const int size = 5;

  tensor::Tensor input1(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor input2(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, size, true, cpu_memory_manager);

  // Initialize input tensors
  for (int i = 0; i < size; i++) {
    input1.at<float>(i) = input1_data[i];
    input2.at<float>(i) = input2_data[i];
  }

  swiglu_kernel_cpu(input1, input2, output, nullptr);

  // Calculate expected values: SwiGLU(x,y) = sigmoid(x) * y
  for (int i = 0; i < size; i++) {
    float x = input1_data[i];
    float y = input2_data[i];
    float swish = x / (1.0f + std::exp(-x));
    float expected = swish * y;
    EXPECT_NEAR(output.at<float>(i), expected, 1e-6f);
  }
}

// Test SwiGLU with different tensor sizes
TEST_F(SwiGLUKernelTest, DifferentSizes) {
  const std::vector<int> sizes = {1, 16, 256, 4096};

  for (const auto& size : sizes) {
    tensor::Tensor input1(core::DataType::FP32, size, true, cpu_memory_manager);
    tensor::Tensor input2(core::DataType::FP32, size, true, cpu_memory_manager);
    tensor::Tensor output(core::DataType::FP32, size, true, cpu_memory_manager);

    // Initialize with alternating values
    for (int i = 0; i < size; i++) {
      input1.at<float>(i) = (i % 2 == 0) ? 1.0f : -1.0f;
      input2.at<float>(i) = (i % 2 == 0) ? 2.0f : 3.0f;
    }

    swiglu_kernel_cpu(input1, input2, output, nullptr);

    // Verify results
    for (int i = 0; i < size; i++) {
      float x = (i % 2 == 0 ? 1.0f : -1.0f);
      float y = (i % 2 == 0 ? 2.0f : 3.0f);
      float swish = x / (1.0f + std::exp(-x));
      float expected = swish * y;
      EXPECT_NEAR(output.at<float>(i), expected, 1e-6f);
    }
  }
}

// Test SwiGLU with zero input
TEST_F(SwiGLUKernelTest, ZeroInput) {
  const int size = 1024;
  tensor::Tensor input1(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor input2(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, size, true, cpu_memory_manager);

  // Initialize with zeros
  for (int i = 0; i < size; i++) {
    input1.at<float>(i) = 0.0f;
    input2.at<float>(i) = 0.0f;
  }

  swiglu_kernel_cpu(input1, input2, output, nullptr);

  // When input is zero, sigmoid(0) = 0.5, and 0.5 * 0 = 0
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(output.at<float>(i), 0.0f, 1e-6f);
  }
}

// Test SwiGLU with extreme values
TEST_F(SwiGLUKernelTest, ExtremeValues) {
  const std::vector<float> input1_data = {-100.0f, 100.0f};
  const std::vector<float> input2_data = {1.0f, 1.0f};
  const int size = 2;

  tensor::Tensor input1(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor input2(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, size, true, cpu_memory_manager);

  // Initialize input tensors
  for (int i = 0; i < size; i++) {
    input1.at<float>(i) = input1_data[i];
    input2.at<float>(i) = input2_data[i];
  }

  swiglu_kernel_cpu(input1, input2, output, nullptr);

  // swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
  // For very negative x, swish(x) ≈ 0
  // For very positive x, sigmoid(x) ≈ x
  EXPECT_NEAR(output.at<float>(0), 0.0f, 1e-6f);
  EXPECT_NEAR(output.at<float>(1), input1_data[1], 1e-6f);
}

#ifndef PYTORCH_NOT_FOUND
// Test SwiGLU against PyTorch implementation
TEST_F(SwiGLUKernelTest, CompareWithPyTorch) {
  const int size = 1024;  // Must match the size in Python script

  // Create Python script for generating test data
  {
    std::ofstream py_file("generate_swiglu_data.py");
    py_file << R"(
import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate random input
input_size = 1024
input1_data = torch.randn(input_size, dtype=torch.float32)
input2_data = torch.randn(input_size, dtype=torch.float32)

# Compute SwiGLU: x * sigmoid(x) * y
def swiglu(x, y):
    return x * torch.sigmoid(x) * y

# Compute output
output_data = swiglu(input1_data, input2_data)

# Save data to files
input1_data.numpy().tofile('swiglu_input1.bin')
input2_data.numpy().tofile('swiglu_input2.bin')
output_data.detach().numpy().tofile('swiglu_output.bin')
)";
    py_file.close();
  }

  // Run Python script to generate test data
  ASSERT_EQ(std::system("python3 generate_swiglu_data.py"), 0) << "Failed to run Python script";

  // Read test data generated by PyTorch
  std::vector<float> input1_data = read_binary_file("swiglu_input1.bin", size);
  std::vector<float> input2_data = read_binary_file("swiglu_input2.bin", size);
  std::vector<float> expected_output = read_binary_file("swiglu_output.bin", size);

  // Create and initialize tensors
  tensor::Tensor input1(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor input2(core::DataType::FP32, size, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, size, true, cpu_memory_manager);

  // Copy data to tensors
  std::memcpy(input1.ptr<float>(), input1_data.data(), size * sizeof(float));
  std::memcpy(input2.ptr<float>(), input2_data.data(), size * sizeof(float));

  // Run our SwiGLU implementation
  swiglu_kernel_cpu(input1, input2, output, nullptr);

  // Compare results with PyTorch
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(output.at<float>(i), expected_output[i], 1e-5f)
        << "Mismatch at index " << i << ". Expected: " << expected_output[i]
        << ", Got: " << output.at<float>(i)
        << ", Diff: " << std::fabs(output.at<float>(i) - expected_output[i]);
  }

  // Clean up generated files
  std::remove("generate_swiglu_data.py");
  std::remove("swiglu_input1.bin");
  std::remove("swiglu_input2.bin");
  std::remove("swiglu_output.bin");
}
#endif

}  // namespace
}  // namespace kernel

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
