#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <vector>
#include "config.hpp"
#include "core/memory/memory_manager.hpp"
#include "core/status/status.hpp"
#include "layer/swiglu_layer.hpp"
#include "tensor/tensor.hpp"
#include "type.hpp"

namespace layer {
namespace {

class SwiGLULayerTest : public ::testing::Test {
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
  const int32_t hidden_dim = 8;

  // Input values
  std::vector<float> input1_values{
      1.0f, 2.0f, 3.0f, 4.0f,  // First input vector
      5.0f, 6.0f, 7.0f, 8.0f   // Second input vector
  };

  std::vector<float> input2_values{
      0.5f, 0.6f, 0.7f, 0.8f,  // First input vector
      0.9f, 1.0f, 1.1f, 1.2f   // Second input vector
  };

  // Expected output values
  // Formula: input1 * sigmoid(input1) * input2
  // where sigmoid(x) = 1 / (1 + exp(-x))
  std::vector<float> expected_output{
      0.5f * 1.0f / (1.0f + std::exp(-1.0f)),  // First element
      0.6f * 2.0f / (1.0f + std::exp(-2.0f)), 0.7f * 3.0f / (1.0f + std::exp(-3.0f)),
      0.8f * 4.0f / (1.0f + std::exp(-4.0f)),
      0.9f * 5.0f / (1.0f + std::exp(-5.0f)),  // Second element
      1.0f * 6.0f / (1.0f + std::exp(-6.0f)), 1.1f * 7.0f / (1.0f + std::exp(-7.0f)),
      1.2f * 8.0f / (1.0f + std::exp(-8.0f))};
};

TEST_F(SwiGLULayerTest, CPUTest) {
  // Create SwiGLU layer
  SwiGLULayer swiglu_layer(core::DeviceType::CPU, hidden_dim);

  // Create input tensors
  tensor::Tensor input1_tensor(core::DataType::FP32, hidden_dim, true, cpu_memory_manager, nullptr);
  tensor::Tensor input2_tensor(core::DataType::FP32, hidden_dim, true, cpu_memory_manager, nullptr);

  // Create output tensor
  tensor::Tensor output_tensor(core::DataType::FP32, hidden_dim, true, cpu_memory_manager, nullptr);

  // Set input values
  for (int32_t i = 0; i < input1_tensor.size(); i++) {
    input1_tensor.index<float>(i) = input1_values[i];
    input2_tensor.index<float>(i) = input2_values[i];
  }

  // Set layer inputs and outputs
  swiglu_layer.set_input(0, input1_tensor);
  swiglu_layer.set_input(1, input2_tensor);
  swiglu_layer.set_output(0, output_tensor);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), swiglu_layer.check());
  ASSERT_EQ(core::error::Success(), swiglu_layer.forward());

  // Verify the output
  for (int32_t i = 0; i < output_tensor.size(); i++) {
    EXPECT_NEAR(expected_output[i], output_tensor.index<float>(i), 1e-5f)
        << "Mismatch at position [" << i << "]";
  }
}

TEST_F(SwiGLULayerTest, GPUTest) {
  // Skip GPU test if CUDA is not available
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA is not available, skipping GPU test";
    return;
  }

  // Create SwiGLU layer with GPU device type
  SwiGLULayer swiglu_layer(core::DeviceType::GPU, hidden_dim);

  // Create CUDA configuration
  auto cuda_config = std::make_shared<core::CudaConfig>();
  swiglu_layer.set_cuda_config(cuda_config);

  // Create input tensors
  tensor::Tensor input1_tensor(core::DataType::FP32, hidden_dim, true, cpu_memory_manager, nullptr);
  tensor::Tensor input2_tensor(core::DataType::FP32, hidden_dim, true, cpu_memory_manager, nullptr);

  // Create output tensor (on GPU memory)
  tensor::Tensor output_tensor(core::DataType::FP32, hidden_dim, true, gpu_memory_manager, nullptr);

  // Set input values
  for (int32_t i = 0; i < input1_tensor.size(); i++) {
    input1_tensor.index<float>(i) = input1_values[i];
    input2_tensor.index<float>(i) = input2_values[i];
  }

  // Transfer input tensors to GPU
  input1_tensor.to_cuda();
  input2_tensor.to_cuda();

  // Set layer inputs and outputs
  swiglu_layer.set_input(0, input1_tensor);
  swiglu_layer.set_input(1, input2_tensor);
  swiglu_layer.set_output(0, output_tensor);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), swiglu_layer.check());
  ASSERT_EQ(core::error::Success(), swiglu_layer.forward());

  // Copy GPU results back to CPU for verification
  tensor::Tensor output_cpu = output_tensor.clone();
  output_cpu.to_cpu();

  // Verify the output
  for (int32_t i = 0; i < output_cpu.size(); i++) {
    EXPECT_NEAR(expected_output[i], output_cpu.index<float>(i), 1e-5f)
        << "Mismatch at position [" << i << "]";
  }
}

}  // namespace
}  // namespace layer
