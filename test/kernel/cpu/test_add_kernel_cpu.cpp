#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include "add_kernel_cpu.hpp"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class AddKernelTest : public ::testing::Test {
 protected:
  void SetUp() override { cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance(); }

  std::shared_ptr<core::CPUMemoryManager> cpu_memory_manager;
};

// Test basic addition functionality
TEST_F(AddKernelTest, BasicAddition) {
  const std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};

  tensor::Tensor input1(core::DataType::FP32, 4, true, cpu_memory_manager);
  tensor::Tensor input2(core::DataType::FP32, 4, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, 4, true, cpu_memory_manager);

  for (int i = 0; i < 4; i++) {
    input1.at<float>(i) = data1[i];
    input2.at<float>(i) = data2[i];
  }

  add_kernel_cpu(input1, input2, output, nullptr);

  EXPECT_FLOAT_EQ(output.at<float>(0), 6.0f);   // 1 + 5
  EXPECT_FLOAT_EQ(output.at<float>(1), 8.0f);   // 2 + 6
  EXPECT_FLOAT_EQ(output.at<float>(2), 10.0f);  // 3 + 7
  EXPECT_FLOAT_EQ(output.at<float>(3), 12.0f);  // 4 + 8
}

// Test different tensor sizes
TEST_F(AddKernelTest, DifferentSizes) {
  const std::vector<int> sizes = {1, 16, 256, 4096, 65536};

  for (const auto& size : sizes) {
    tensor::Tensor input1(core::DataType::FP32, size, true, cpu_memory_manager);
    tensor::Tensor input2(core::DataType::FP32, size, true, cpu_memory_manager);
    tensor::Tensor output(core::DataType::FP32, size, true, cpu_memory_manager);

    for (int i = 0; i < size; i++) {
      input1.at<float>(i) = 1.0f;
      input2.at<float>(i) = 2.0f;
    }

    add_kernel_cpu(input1, input2, output, nullptr);

    // Check all elements are correctly added (1.0 + 2.0 = 3.0)
    for (int i = 0; i < size; ++i) {
      EXPECT_FLOAT_EQ(output.at<float>(i), 3.0f);
    }
  }
}

}  // namespace
}  // namespace kernel

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
