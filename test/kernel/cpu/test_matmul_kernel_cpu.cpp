#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <vector>
#include "matmul_kernel_cpu.hpp"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class MatmulKernelTest : public ::testing::Test {
 protected:
  void SetUp() override { cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance(); }

  void TearDown() override {
#ifndef PYTORCH_NOT_FOUND
    // Clean up generated files
    std::remove("generate_matmul_data.py");
    std::remove("matmul_input.bin");
    std::remove("matmul_weight.bin");
    std::remove("matmul_output.bin");
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

// Test basic matrix multiplication functionality with simple values
TEST_F(MatmulKernelTest, BasicOperation) {
  // Setup input tensor (2x3)
  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  // Setup weight tensor (3x2)
  const std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // Expected output (2x2): input * weight
  const std::vector<float> expected_output = {22.0f, 28.0f, 49.0f, 64.0f};

  // Create tensors
  tensor::Tensor input(core::DataType::FP32, 2, 3, true, cpu_memory_manager);
  tensor::Tensor weight(core::DataType::FP32, 3, 2, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, 2, 2, true, cpu_memory_manager);
  tensor::Tensor bias = tensor::zeros(core::DataType::FP32, {2}, cpu_memory_manager);

  // Initialize input tensors
  for (int i = 0; i < input_data.size(); i++) {
    input.index<float>(i) = input_data[i];
  }
  for (int i = 0; i < weight_data.size(); i++) {
    weight.index<float>(i) = weight_data[i];
  }

  // Run the matmul kernel
  matmul_kernel_cpu(input, weight, output, bias, 1.0f, nullptr);

  // Print output for debugging
  std::cout << "Output matrix values:" << std::endl;
  for (int i = 0; i < output.get_dim(0); i++) {
    for (int j = 0; j < output.get_dim(1); j++) {
      std::cout << output.at<float>(i, j) << " ";
    }
    std::cout << std::endl;
  }

  // Verify results
  for (int i = 0; i < expected_output.size(); i++) {
    EXPECT_NEAR(output.index<float>(i), expected_output[i], 1e-6f) << "Mismatch at index " << i;
  }
}

// Test with scale parameter
TEST_F(MatmulKernelTest, ScaleOperation) {
  // Setup input tensor (2x3)
  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  // Setup weight tensor (3x2)， column major
  const std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  const float scale = 0.5f;

  // Create tensors
  tensor::Tensor input(core::DataType::FP32, 2, 3, true, cpu_memory_manager);
  tensor::Tensor weight(core::DataType::FP32, 3, 2, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, 2, 2, true, cpu_memory_manager);
  tensor::Tensor bias = tensor::zeros(core::DataType::FP32, {2}, cpu_memory_manager);

  // Initialize input tensors
  for (int i = 0; i < input_data.size(); i++) {
    input.index<float>(i) = input_data[i];
  }
  for (int i = 0; i < weight_data.size(); i++) {
    weight.index<float>(i) = weight_data[i];
  }

  // Run the matmul kernel with scale = 0.5
  matmul_kernel_cpu(input, weight, output, bias, scale, nullptr);

  // Expected output (2x2): (input * weight) * 0.5
  const std::vector<float> expected_output = {11.0f, 14.0f, 24.5f, 32.0f};

  // Verify results
  for (int i = 0; i < expected_output.size(); i++) {
    EXPECT_NEAR(output.index<float>(i), expected_output[i], 1e-6f) << "Mismatch at index " << i;
  }
}

// Test with 1D input (vector) and 2D weight
TEST_F(MatmulKernelTest, Vector1DInput) {
  // Setup input tensor (1D vector of size 3), column major
  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  const std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // Expected output (1D vector of size 2): input * weight
  const std::vector<float> expected_output = {22.0f, 28.0f};

  // Create tensors
  tensor::Tensor input(core::DataType::FP32, 3, true, cpu_memory_manager);
  tensor::Tensor weight(core::DataType::FP32, 3, 2, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, 2, true, cpu_memory_manager);
  tensor::Tensor bias = tensor::zeros(core::DataType::FP32, {2}, cpu_memory_manager);

  // Initialize input tensors
  for (int i = 0; i < input_data.size(); i++) {
    input.index<float>(i) = input_data[i];
  }
  for (int i = 0; i < weight_data.size(); i++) {
    weight.index<float>(i) = weight_data[i];
  }

  // Run the matmul kernel
  matmul_kernel_cpu(input, weight, output, bias, 1.0f, nullptr);

  // Verify results
  for (int i = 0; i < expected_output.size(); i++) {
    EXPECT_NEAR(output.index<float>(i), expected_output[i], 1e-6f) << "Mismatch at index " << i;
  }
}

// Test with different matrix sizes
TEST_F(MatmulKernelTest, DifferentSizes) {
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
    tensor::Tensor input(core::DataType::FP32, M, K, true, cpu_memory_manager);
    tensor::Tensor weight(core::DataType::FP32, K, N, true, cpu_memory_manager);
    tensor::Tensor output(core::DataType::FP32, M, N, true, cpu_memory_manager);
    tensor::Tensor bias = tensor::zeros(core::DataType::FP32, {N}, cpu_memory_manager);
    tensor::Tensor expected_output(core::DataType::FP32, M, N, true, cpu_memory_manager);

    // Initialize with simple pattern
    for (int i = 0; i < M * K; i++) {
      input.index<float>(i) = static_cast<float>(i % 5);
    }
    for (int i = 0; i < K * N; i++) {
      weight.index<float>(i) = static_cast<float>(i % 7);
    }

    // Compute expected output manually
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
          sum += input.at<float>(i, k) * weight.at<float>(k, j);
        }
        expected_output.at<float>(i, j) = sum;
      }
    }

    // Run the matmul kernel
    matmul_kernel_cpu(input, weight, output, bias, 1.0f, nullptr);

    // Verify results
    for (int i = 0; i < M * N; i++) {
      EXPECT_NEAR(output.index<float>(i), expected_output.index<float>(i), 1e-5f)
          << "Mismatch at index " << i << " for size " << M << "x" << K << " * " << N << "x" << K;
    }
  }
}

// Test with zero inputs
TEST_F(MatmulKernelTest, ZeroInput) {
  const int M = 4;
  const int K = 4;
  const int N = 4;

  // Create tensors
  tensor::Tensor input(core::DataType::FP32, M, K, true, cpu_memory_manager);
  tensor::Tensor weight(core::DataType::FP32, N, K, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, M, N, true, cpu_memory_manager);
  tensor::Tensor bias = tensor::zeros(core::DataType::FP32, {N}, cpu_memory_manager);

  // Initialize with zeros
  for (int i = 0; i < M * K; i++) {
    input.index<float>(i) = 0.0f;
  }
  for (int i = 0; i < N * K; i++) {
    weight.index<float>(i) = 0.0f;
  }

  // Run the matmul kernel
  matmul_kernel_cpu(input, weight, output, bias, 1.0f, nullptr);

  // Verify all results are zeros
  for (int i = 0; i < M * N; i++) {
    EXPECT_NEAR(output.index<float>(i), 0.0f, 1e-6f);
  }
}

// Test identity matrix multiplication
TEST_F(MatmulKernelTest, IdentityMatrix) {
  const int size = 4;

  // Create tensors
  tensor::Tensor input(core::DataType::FP32, size, size, true, cpu_memory_manager);
  tensor::Tensor identity(core::DataType::FP32, size, size, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, size, size, true, cpu_memory_manager);
  tensor::Tensor bias = tensor::zeros(core::DataType::FP32, {size}, cpu_memory_manager);

  // Initialize input with random values
  for (int i = 0; i < size * size; i++) {
    input.index<float>(i) = static_cast<float>(i);
  }

  // Initialize identity matrix
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      identity.at<float>(i, j) = (i == j) ? 1.0f : 0.0f;
    }
  }

  // Run the matmul kernel
  matmul_kernel_cpu(input, identity, output, bias, 1.0f, nullptr);

  // Verify A*I = A
  for (int i = 0; i < size * size; i++) {
    EXPECT_NEAR(output.index<float>(i), input.index<float>(i), 1e-6f) << "Mismatch at index " << i;
  }
}

// Test to confirm that matmul_kernel_cpu works with row major order
TEST_F(MatmulKernelTest, RowMajorOrder) {
  // Create a 2x3 input matrix and a 3x2 weight matrix with specific values
  // to verify row major orientation
  // Input (2x3):
  // [ 1 2 3 ]
  // [ 4 5 6 ]
  // Memory layout in row-major: [1, 2, 3, 4, 5, 6]

  // Weight (3x2):
  // [ 7  8  ]
  // [ 9  10 ]
  // [ 11 12 ]
  // Memory layout in row-major: [7, 8, 9, 10, 11, 12]

  // Expected result (2x2) in row-major:
  // [ 1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12 ]
  // [ 4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12 ]
  // = [ 58, 64 ]
  //   [ 139, 154 ]

  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> weight_data = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  const std::vector<float> expected_output = {58.0f, 64.0f, 139.0f, 154.0f};

  // Create tensors
  tensor::Tensor input(core::DataType::FP32, 2, 3, true, cpu_memory_manager);
  tensor::Tensor weight(core::DataType::FP32, 3, 2, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, 2, 2, true, cpu_memory_manager);
  tensor::Tensor bias = tensor::zeros(core::DataType::FP32, {2}, cpu_memory_manager);

  // Initialize the input and weight tensors with values
  for (int i = 0; i < input_data.size(); i++) {
    input.index<float>(i) = input_data[i];
  }
  for (int i = 0; i < weight_data.size(); i++) {
    weight.index<float>(i) = weight_data[i];
  }

  // Perform matrix multiplication
  matmul_kernel_cpu(input, weight, output, bias, 1.0f, nullptr);

  // Verify that the output matches the expected result (row major ordering)
  for (int i = 0; i < expected_output.size(); i++) {
    EXPECT_NEAR(output.index<float>(i), expected_output[i], 1e-6f)
        << "Mismatch at index " << i << ", which indicates a possible ordering issue";
  }

  // Additionally check individual elements by logical position to confirm row major format
  EXPECT_NEAR(output.index<float>(0), 58.0f, 1e-6f) << "Incorrect value at position (0,0)";
  EXPECT_NEAR(output.index<float>(1), 64.0f, 1e-6f) << "Incorrect value at position (0,1)";
  EXPECT_NEAR(output.index<float>(2), 139.0f, 1e-6f) << "Incorrect value at position (1,0)";
  EXPECT_NEAR(output.index<float>(3), 154.0f, 1e-6f) << "Incorrect value at position (1,1)";
}

// Test for bias functionality
TEST_F(MatmulKernelTest, BiasAddition) {
  // Setup input tensor (2x3)
  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  // Setup weight tensor (3x2)
  const std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  // Setup bias tensor (2)
  const std::vector<float> bias_data = {1.0f, 2.0f};

  // Expected output (2x2): (input * weight) + bias
  // Without bias would be: [22.0f, 28.0f, 49.0f, 64.0f]
  // With bias: [23.0f, 30.0f, 50.0f, 66.0f]
  const std::vector<float> expected_output = {23.0f, 30.0f, 50.0f, 66.0f};

  // Create tensors
  tensor::Tensor input(core::DataType::FP32, 2, 3, true, cpu_memory_manager);
  tensor::Tensor weight(core::DataType::FP32, 3, 2, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, 2, 2, true, cpu_memory_manager);
  tensor::Tensor bias(core::DataType::FP32, 2, true, cpu_memory_manager);

  // Initialize input tensors
  for (int i = 0; i < input_data.size(); i++) {
    input.index<float>(i) = input_data[i];
  }
  for (int i = 0; i < weight_data.size(); i++) {
    weight.index<float>(i) = weight_data[i];
  }
  // Initialize bias with provided values
  for (int i = 0; i < bias_data.size(); i++) {
    bias.index<float>(i) = bias_data[i];
  }

  // Perform matrix multiplication with bias
  matmul_kernel_cpu(input, weight, output, bias, 1.0f, nullptr);

  // Print output for debugging
  std::cout << "Output matrix values with bias:" << std::endl;
  for (int i = 0; i < output.get_dim(0); i++) {
    for (int j = 0; j < output.get_dim(1); j++) {
      std::cout << output.at<float>(i, j) << " ";
    }
    std::cout << std::endl;
  }

  // Verify results
  for (int i = 0; i < expected_output.size(); i++) {
    EXPECT_NEAR(output.index<float>(i), expected_output[i], 1e-6f) << "Mismatch at index " << i;
  }
}

}  // namespace
}  // namespace kernel

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
