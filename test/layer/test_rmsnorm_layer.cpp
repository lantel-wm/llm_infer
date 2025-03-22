#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <vector>
#include "config.hpp"
#include "core/memory/memory_manager.hpp"
#include "core/status/status.hpp"
#include "layer/rmsnorm_layer.hpp"
#include "tensor/tensor.hpp"
#include "type.hpp"

namespace layer {
namespace {

class RmsNormLayerTest : public ::testing::Test {
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
  const int32_t hidden_size = 4;

  std::vector<float> input_values{
      0.156349f, 0.873002f, -0.185859f, 0.436511f,  // Batch 0
      0.891515f, 1.035043f, -1.262521f, 0.088791f   // Batch 1
  };
  std::vector<float> weight_values{-0.624035f, 0.201925f, 1.800386f, 0.299643f};
  std::vector<float> expected_output{
      -0.194003f, 0.350517f, -0.665354f, 0.260078f,  // Batch 0
      -0.597487f, 0.224461f, -2.441152f, 0.028574f   // Batch 1
  };
};

TEST_F(RmsNormLayerTest, CPU) {
  // Create RmsNorm layer
  RmsNormLayer rms_norm_layer(core::DeviceType::CPU, hidden_size);

  // Create input tensor
  tensor::Tensor input_tensor(core::DataType::FP32, {batch_size, hidden_size}, true,
                              cpu_memory_manager, nullptr);

  // Create weight tensor
  tensor::Tensor weight_tensor(core::DataType::FP32, hidden_size, true, cpu_memory_manager,
                               nullptr);

  // Create output tensor
  tensor::Tensor output_tensor(core::DataType::FP32, {batch_size, hidden_size}, true,
                               cpu_memory_manager, nullptr);

  // Set input values
  for (int32_t i = 0; i < batch_size * hidden_size; i++) {
    input_tensor.index<float>(i) = input_values[i];
  }

  // Set weight values
  for (int32_t i = 0; i < hidden_size; i++) {
    weight_tensor.index<float>(i) = weight_values[i];
  }

  // Set layer inputs, weights, and outputs
  rms_norm_layer.set_input(0, input_tensor);
  rms_norm_layer.set_weight(0, weight_tensor);
  rms_norm_layer.set_output(0, output_tensor);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), rms_norm_layer.check());
  ASSERT_EQ(core::error::Success(), rms_norm_layer.forward());

  // Verify the output
  for (int32_t i = 0; i < batch_size; i++) {
    for (int32_t j = 0; j < hidden_size; j++) {
      EXPECT_NEAR(expected_output[i * hidden_size + j], output_tensor.at<float>(i, j), 1e-5f)
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }
}

TEST_F(RmsNormLayerTest, GPU) {
  // Skip GPU test if CUDA is not available
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA is not available, skipping GPU test";
    return;
  }

  // Create RmsNorm layer
  RmsNormLayer rms_norm_layer(core::DeviceType::GPU, hidden_size);
  auto cuda_config = std::make_shared<core::CudaConfig>();
  rms_norm_layer.set_cuda_config(cuda_config);

  // Create input tensor
  tensor::Tensor input_tensor(core::DataType::FP32, {batch_size, hidden_size}, true,
                              cpu_memory_manager, nullptr);

  // Create weight tensor
  tensor::Tensor weight_tensor(core::DataType::FP32, hidden_size, true, cpu_memory_manager,
                               nullptr);

  // Create output tensor (on GPU memory)
  tensor::Tensor output_tensor(core::DataType::FP32, {batch_size, hidden_size}, true,
                               gpu_memory_manager, nullptr);

  // Set input values
  for (int32_t i = 0; i < batch_size * hidden_size; i++) {
    input_tensor.index<float>(i) = input_values[i];
  }

  // Set weight values
  for (int32_t i = 0; i < hidden_size; i++) {
    weight_tensor.index<float>(i) = weight_values[i];
  }

  // Transfer tensors to GPU
  input_tensor.to_cuda();
  weight_tensor.to_cuda();

  // Set layer inputs, weights, and outputs
  rms_norm_layer.set_input(0, input_tensor);
  rms_norm_layer.set_weight(0, weight_tensor);
  rms_norm_layer.set_output(0, output_tensor);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), rms_norm_layer.check());
  ASSERT_EQ(core::error::Success(), rms_norm_layer.forward());

  // Copy GPU results back to CPU for verification
  tensor::Tensor output_gpu_cpu = output_tensor.clone();
  output_gpu_cpu.to_cpu();

  // Verify the output
  for (int32_t i = 0; i < batch_size; i++) {
    for (int32_t j = 0; j < hidden_size; j++) {
      EXPECT_NEAR(expected_output[i * hidden_size + j], output_gpu_cpu.at<float>(i, j), 1e-5f)
          << "Mismatch at position [" << i << ", " << j << "]";
    }
  }
}

TEST_F(RmsNormLayerTest, ZeroInput) {
  // Create RmsNorm layer
  RmsNormLayer rms_norm_layer(core::DeviceType::CPU, hidden_size);

  // Create input tensor with zeros
  tensor::Tensor input_tensor(core::DataType::FP32, {batch_size, hidden_size}, true,
                              cpu_memory_manager, nullptr);

  // Create weight tensor
  tensor::Tensor weight_tensor(core::DataType::FP32, hidden_size, true, cpu_memory_manager,
                               nullptr);

  // Create output tensor
  tensor::Tensor output_tensor(core::DataType::FP32, {batch_size, hidden_size}, true,
                               cpu_memory_manager, nullptr);

  // Set input values to zero
  for (int32_t i = 0; i < batch_size * hidden_size; i++) {
    input_tensor.index<float>(i) = 0.0f;
  }

  // Set weight values
  for (int32_t i = 0; i < hidden_size; i++) {
    weight_tensor.index<float>(i) = weight_values[i];
  }

  // Set layer inputs, weights, and outputs
  rms_norm_layer.set_input(0, input_tensor);
  rms_norm_layer.set_weight(0, weight_tensor);
  rms_norm_layer.set_output(0, output_tensor);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), rms_norm_layer.check());
  ASSERT_EQ(core::error::Success(), rms_norm_layer.forward());

  // When input is zero, the result is zero / sqrt(epsilon)
  float expected = 0.0f;
  for (int32_t i = 0; i < batch_size * hidden_size; i++) {
    EXPECT_NEAR(expected, output_tensor.index<float>(i), 1e-5f);
  }
}

}  // namespace
}  // namespace layer
