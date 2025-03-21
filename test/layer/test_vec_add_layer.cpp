#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "config.hpp"
#include "core/memory/memory_manager.hpp"
#include "core/status/status.hpp"
#include "layer/vec_add_layer.hpp"
#include "tensor/tensor.hpp"
#include "type.hpp"

namespace layer {
namespace {

class VecAddLayerTest : public ::testing::Test {
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

  std::vector<float> input1{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> input2{1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> expected_output{2.0f, 3.0f, 4.0f, 5.0f};
};

TEST_F(VecAddLayerTest, CPU) {
  VecAddLayer cpu_vec_add_layer(core::DeviceType::CPU);
  tensor::Tensor input1_cpu(core::DataType::FP32, 4, true, cpu_memory_manager, nullptr);
  tensor::Tensor input2_cpu(core::DataType::FP32, 4, true, cpu_memory_manager, nullptr);
  tensor::Tensor output_cpu(core::DataType::FP32, 4, true, cpu_memory_manager, nullptr);

  for (int i = 0; i < input1_cpu.size(); i++) {
    input1_cpu.index<float>(i) = input1[i];
    input2_cpu.index<float>(i) = input2[i];
  }

  cpu_vec_add_layer.set_input(0, input1_cpu);
  cpu_vec_add_layer.set_input(1, input2_cpu);
  cpu_vec_add_layer.set_output(0, output_cpu);

  ASSERT_EQ(core::error::Success(), cpu_vec_add_layer.check());
  ASSERT_EQ(core::error::Success(), cpu_vec_add_layer.forward());

  // Verify CPU results
  for (int i = 0; i < output_cpu.size(); i++) {
    EXPECT_FLOAT_EQ(expected_output[i], output_cpu.index<float>(i));
  }
}

TEST_F(VecAddLayerTest, GPU) {
  // Skip GPU test if CUDA is not available
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA is not available, skipping GPU test";
    return;
  }

  VecAddLayer gpu_vec_add_layer(core::DeviceType::GPU);
  auto cuda_config = std::make_shared<core::CudaConfig>();
  gpu_vec_add_layer.set_cuda_config(cuda_config);
  tensor::Tensor input1_gpu(core::DataType::FP32, 4, true, cpu_memory_manager, nullptr);
  tensor::Tensor input2_gpu(core::DataType::FP32, 4, true, cpu_memory_manager, nullptr);
  tensor::Tensor output_gpu(core::DataType::FP32, 4, true, gpu_memory_manager, nullptr);

  for (int i = 0; i < input1_gpu.size(); i++) {
    input1_gpu.index<float>(i) = input1[i];
    input2_gpu.index<float>(i) = input2[i];
  }
  input1_gpu.to_cuda();
  input2_gpu.to_cuda();

  gpu_vec_add_layer.set_input(0, input1_gpu);
  gpu_vec_add_layer.set_input(1, input2_gpu);
  gpu_vec_add_layer.set_output(0, output_gpu);

  ASSERT_EQ(core::error::Success(), gpu_vec_add_layer.check());
  ASSERT_EQ(core::error::Success(), gpu_vec_add_layer.forward());

  // Copy GPU results back to CPU for verification
  tensor::Tensor output_gpu_cpu = output_gpu.clone();
  output_gpu_cpu.to_cpu();

  // Verify GPU results
  for (int i = 0; i < output_gpu_cpu.size(); i++) {
    EXPECT_FLOAT_EQ(expected_output[i], output_gpu_cpu.index<float>(i));
  }
}

}  // namespace
}  // namespace layer
