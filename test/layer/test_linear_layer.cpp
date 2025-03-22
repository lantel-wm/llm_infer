#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "config.hpp"
#include "core/memory/memory_manager.hpp"
#include "core/status/status.hpp"
#include "layer/linear_layer.hpp"
#include "tensor/tensor.hpp"
#include "type.hpp"

namespace layer {
namespace {

class LinearLayerTest : public ::testing::Test {
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
  const int32_t batch_size = 2;
  const int32_t in_features = 3;
  const int32_t out_features = 4;

  // Input values
  std::vector<float> input_values{
      1.0f, 2.0f, 3.0f,  // First input vector
      4.0f, 5.0f, 6.0f   // Second input vector
  };

  // Weight values - in_features × out_features matrix
  std::vector<float> weight_values{
      0.1f, 0.2f, 0.3f, 0.4f,  // First row
      0.5f, 0.6f, 0.7f, 0.8f,  // Second row
      0.9f, 1.0f, 1.1f, 1.2f   // Third row
  };

  // Bias values - vector of size out_features
  std::vector<float> bias_values{0.1f, 0.2f, 0.3f, 0.4f};

  // Expected output: input × weight + bias
  std::vector<float> expected_output_with_bias{
      // First output vector: (1×0.1 + 2×0.5 + 3×0.9) + 0.1 = 3.9, and so on
      3.9f, 4.6f, 5.3f, 6.0f,
      // Second output vector: (4×0.1 + 5×0.5 + 6×0.9) + 0.1 = 8.4, and so on
      8.4f, 10.0f, 11.6f, 13.2f};

  std::vector<float> expected_output_without_bias{// First output vector without bias
                                                  3.8f, 4.4f, 5.0f, 5.6f,
                                                  // Second output vector without bias
                                                  8.3f, 9.8f, 11.3f, 12.8f};
};

TEST_F(LinearLayerTest, CPUWithBias) {
  // Create linear layer with bias
  LinearLayer linear_layer(core::DeviceType::CPU, in_features, out_features, false);

  // Create input tensor
  tensor::Tensor input_tensor(core::DataType::FP32, batch_size, in_features, true,
                              cpu_memory_manager, nullptr);

  // Create weight tensor
  tensor::Tensor weight_tensor(core::DataType::FP32, in_features, out_features, true,
                               cpu_memory_manager, nullptr);

  // Create output tensor
  tensor::Tensor output_tensor(core::DataType::FP32, batch_size, out_features, true,
                               cpu_memory_manager, nullptr);

  // Create bias tensor
  tensor::Tensor bias_tensor(core::DataType::FP32, out_features, true, cpu_memory_manager, nullptr);

  // Set input values
  for (int32_t i = 0; i < batch_size * in_features; i++) {
    input_tensor.index<float>(i) = input_values[i];
  }

  // Set weight values
  for (int32_t i = 0; i < in_features * out_features; i++) {
    weight_tensor.index<float>(i) = weight_values[i];
  }

  // Set bias values
  for (int32_t i = 0; i < out_features; i++) {
    bias_tensor.index<float>(i) = bias_values[i];
  }

  // Set layer inputs, weights, outputs and bias
  linear_layer.set_input(0, input_tensor);
  linear_layer.set_weight(0, weight_tensor);
  linear_layer.set_output(0, output_tensor);
  linear_layer.set_bias(0, bias_tensor);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), linear_layer.check());
  ASSERT_EQ(core::error::Success(), linear_layer.forward());

  // Verify the output
  for (int32_t i = 0; i < batch_size; i++) {
    for (int32_t j = 0; j < out_features; j++) {
      EXPECT_NEAR(expected_output_with_bias[i * out_features + j], output_tensor.at<float>(i, j),
                  1e-5f)
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }
}

TEST_F(LinearLayerTest, CPUWithoutBias) {
  // Create linear layer without bias
  LinearLayer linear_layer(core::DeviceType::CPU, in_features, out_features, false);

  // Create input tensor
  tensor::Tensor input_tensor(core::DataType::FP32, batch_size, in_features, true,
                              cpu_memory_manager, nullptr);

  // Create weight tensor
  tensor::Tensor weight_tensor(core::DataType::FP32, in_features, out_features, true,
                               cpu_memory_manager, nullptr);

  // Create output tensor
  tensor::Tensor output_tensor(core::DataType::FP32, batch_size, out_features, true,
                               cpu_memory_manager, nullptr);

  // Create bias tensor
  tensor::Tensor bias_tensor =
      tensor::zeros(core::DataType::FP32, {out_features}, cpu_memory_manager);

  // Set input values
  for (int32_t i = 0; i < batch_size * in_features; i++) {
    input_tensor.index<float>(i) = input_values[i];
  }

  // Set weight values
  for (int32_t i = 0; i < in_features * out_features; i++) {
    weight_tensor.index<float>(i) = weight_values[i];
  }

  // Set layer inputs, weights, and outputs
  linear_layer.set_input(0, input_tensor);
  linear_layer.set_weight(0, weight_tensor);
  linear_layer.set_output(0, output_tensor);
  linear_layer.set_bias(0, bias_tensor);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), linear_layer.check());
  ASSERT_EQ(core::error::Success(), linear_layer.forward());

  // Verify the output
  for (int32_t i = 0; i < batch_size; i++) {
    for (int32_t j = 0; j < out_features; j++) {
      EXPECT_NEAR(expected_output_without_bias[i * out_features + j], output_tensor.at<float>(i, j),
                  1e-5f)
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }
}

TEST_F(LinearLayerTest, GPU) {
  // Skip GPU test if CUDA is not available
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA is not available, skipping GPU test";
    return;
  }

  // Create linear layer with bias
  LinearLayer linear_layer(core::DeviceType::GPU, in_features, out_features, false);
  auto cuda_config = std::make_shared<core::CudaConfig>();
  // // Initialize a proper CUDA stream for the test
  // cudaStreamCreate(&cuda_config->stream);
  linear_layer.set_cuda_config(cuda_config);

  // Create input tensor
  tensor::Tensor input_tensor(core::DataType::FP32, batch_size, in_features, true,
                              cpu_memory_manager, nullptr);

  // Create weight tensor
  tensor::Tensor weight_tensor(core::DataType::FP32, in_features, out_features, true,
                               cpu_memory_manager, nullptr);

  // Create output tensor (on GPU memory)
  tensor::Tensor output_tensor(core::DataType::FP32, batch_size, out_features, true,
                               gpu_memory_manager, nullptr);

  // Create bias tensor
  tensor::Tensor bias_tensor(core::DataType::FP32, out_features, true, cpu_memory_manager, nullptr);

  // Set input values
  for (int32_t i = 0; i < batch_size * in_features; i++) {
    input_tensor.index<float>(i) = input_values[i];
  }

  // Set weight values
  for (int32_t i = 0; i < in_features * out_features; i++) {
    weight_tensor.index<float>(i) = weight_values[i];
  }

  // Set bias values
  for (int32_t i = 0; i < out_features; i++) {
    bias_tensor.index<float>(i) = bias_values[i];
  }

  input_tensor.to_cuda();
  weight_tensor.to_cuda();
  bias_tensor.to_cuda();

  // Set layer inputs, weights, and outputs
  linear_layer.set_input(0, input_tensor);
  linear_layer.set_weight(0, weight_tensor);
  linear_layer.set_output(0, output_tensor);
  linear_layer.set_bias(0, bias_tensor);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), linear_layer.check());
  ASSERT_EQ(core::error::Success(), linear_layer.forward());

  // Copy GPU results back to CPU for verification
  tensor::Tensor output_gpu_cpu = output_tensor.clone();
  output_gpu_cpu.to_cpu();

  // Verify the output
  for (int32_t i = 0; i < batch_size; i++) {
    for (int32_t j = 0; j < out_features; j++) {
      EXPECT_NEAR(expected_output_with_bias[i * out_features + j], output_gpu_cpu.at<float>(i, j),
                  1e-5f)
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }
}

}  // namespace
}  // namespace layer
