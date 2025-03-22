#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "config.hpp"
#include "core/memory/memory_manager.hpp"
#include "core/status/status.hpp"
#include "kernel/cpu/rope_kernel_cpu.hpp"
#include "layer/rope_layer.hpp"
#include "tensor/tensor.hpp"
#include "type.hpp"

namespace layer {
namespace {

class RoPELayerTest : public ::testing::Test {
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
  const int32_t seq_len = 4;
  const int32_t hidden_size = 16;
  const int32_t num_q_heads = 4;
  const int32_t num_kv_heads = 2;
  const int32_t head_size = hidden_size / num_q_heads;
  const int32_t kv_size = num_kv_heads * head_size;
};

TEST_F(RoPELayerTest, CPUForward) {
  // Create RoPE layer
  RoPELayer rope_layer(core::DeviceType::CPU, hidden_size, kv_size, head_size);

  // Create input tensors
  tensor::Tensor input_q(core::DataType::FP32, {batch_size, seq_len, num_q_heads, head_size}, true,
                         cpu_memory_manager, nullptr);
  tensor::Tensor input_k(core::DataType::FP32, {batch_size, seq_len, num_kv_heads, head_size}, true,
                         cpu_memory_manager, nullptr);
  tensor::Tensor positions(core::DataType::INT32, seq_len, true, cpu_memory_manager, nullptr);
  tensor::Tensor sin_cache(core::DataType::FP32, {seq_len, head_size}, true, cpu_memory_manager,
                           nullptr);
  tensor::Tensor cos_cache(core::DataType::FP32, {seq_len, head_size}, true, cpu_memory_manager,
                           nullptr);

  // Initialize with simple patterns
  for (int pos = 0; pos < seq_len; pos++) {
    positions.at<int32_t>(pos) = pos;
    for (int batch = 0; batch < batch_size; batch++) {
      for (int n = 0; n < num_q_heads; n++) {
        for (int h = 0; h < head_size; h++) {
          input_q.at<float>(batch, pos, n, h) = (h % 2 == 0) ? 1.0f : -1.0f;
          if (n < num_kv_heads) {
            input_k.at<float>(batch, pos, n, h) = (h % 2 == 0) ? -1.0f : 1.0f;
          }
        }
      }
    }
  }

  // Calculate sin/cos cache
  float rope_theta = 10000.0f;
  kernel::sin_cos_cache_calc_cpu(rope_theta, head_size, seq_len, sin_cache, cos_cache);

  // Create copy of original values for comparison
  tensor::Tensor q_original = input_q.clone();
  tensor::Tensor k_original = input_k.clone();

  // Set inputs
  rope_layer.set_batch_size(batch_size);
  rope_layer.set_input(0, input_q);
  rope_layer.set_input(1, input_k);
  rope_layer.set_input(2, positions);
  rope_layer.set_input(3, sin_cache);
  rope_layer.set_input(4, cos_cache);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), rope_layer.check());
  ASSERT_EQ(core::error::Success(), rope_layer.forward());

  // Verify results:
  // Position 0 should not change (cosine is 1, sine is 0 for pos 0)
  for (int batch = 0; batch < batch_size; batch++) {
    for (int n = 0; n < num_q_heads; n++) {
      EXPECT_NEAR(input_q.at<float>(batch, 0, n, 0), q_original.at<float>(batch, 0, n, 0), 1e-5f)
          << "Q mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=0";
      EXPECT_NEAR(input_q.at<float>(batch, 0, n, 1), q_original.at<float>(batch, 0, n, 1), 1e-5f)
          << "Q mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=1";

      if (n < num_kv_heads) {
        EXPECT_NEAR(input_k.at<float>(batch, 0, n, 0), k_original.at<float>(batch, 0, n, 0), 1e-5f)
            << "K mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=0";
        EXPECT_NEAR(input_k.at<float>(batch, 0, n, 1), k_original.at<float>(batch, 0, n, 1), 1e-5f)
            << "K mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=1";
      }
    }
  }

  // For positions > 0, values should be different (rotated)
  bool found_rotated_values = false;
  for (int pos = 1; pos < seq_len; pos++) {
    for (int batch = 0; batch < batch_size; batch++) {
      for (int n = 0; n < num_q_heads; n++) {
        // Check if at least one value differs from the original
        if (std::abs(input_q.at<float>(batch, pos, n, 0) - q_original.at<float>(batch, pos, n, 0)) >
                1e-5f ||
            std::abs(input_q.at<float>(batch, pos, n, 1) - q_original.at<float>(batch, pos, n, 1)) >
                1e-5f) {
          found_rotated_values = true;
          break;
        }

        if (n < num_kv_heads) {
          if (std::abs(input_k.at<float>(batch, pos, n, 0) -
                       k_original.at<float>(batch, pos, n, 0)) > 1e-5f ||
              std::abs(input_k.at<float>(batch, pos, n, 1) -
                       k_original.at<float>(batch, pos, n, 1)) > 1e-5f) {
            found_rotated_values = true;
            break;
          }
        }
      }
      if (found_rotated_values) break;
    }
    if (found_rotated_values) break;
  }
  EXPECT_TRUE(found_rotated_values) << "No rotated values found for positions > 0";
}

TEST_F(RoPELayerTest, GPU) {
  // Skip GPU test if CUDA is not available
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA is not available, skipping GPU test";
    return;
  }

  // Create RoPE layer
  RoPELayer rope_layer(core::DeviceType::GPU, hidden_size, kv_size, head_size);
  auto cuda_config = std::make_shared<core::CudaConfig>();
  rope_layer.set_cuda_config(cuda_config);

  // Create input tensors on CPU
  tensor::Tensor input_q(core::DataType::FP32, {batch_size, seq_len, num_q_heads, head_size}, true,
                         cpu_memory_manager, nullptr);
  tensor::Tensor input_k(core::DataType::FP32, {batch_size, seq_len, num_kv_heads, head_size}, true,
                         cpu_memory_manager, nullptr);
  tensor::Tensor positions(core::DataType::INT32, seq_len, true, cpu_memory_manager, nullptr);
  tensor::Tensor sin_cache(core::DataType::FP32, {seq_len, head_size}, true, cpu_memory_manager,
                           nullptr);
  tensor::Tensor cos_cache(core::DataType::FP32, {seq_len, head_size}, true, cpu_memory_manager,
                           nullptr);

  // Initialize with simple patterns
  for (int pos = 0; pos < seq_len; pos++) {
    positions.at<int32_t>(pos) = pos;
    for (int batch = 0; batch < batch_size; batch++) {
      for (int n = 0; n < num_q_heads; n++) {
        for (int h = 0; h < head_size; h++) {
          input_q.at<float>(batch, pos, n, h) = (h % 2 == 0) ? 1.0f : -1.0f;
          if (n < num_kv_heads) {
            input_k.at<float>(batch, pos, n, h) = (h % 2 == 0) ? -1.0f : 1.0f;
          }
        }
      }
    }
  }

  // Calculate sin/cos cache
  float rope_theta = 10000.0f;
  kernel::sin_cos_cache_calc_cpu(rope_theta, head_size, seq_len, sin_cache, cos_cache);

  // Create copy of original values for comparison
  tensor::Tensor q_original = input_q.clone();
  tensor::Tensor k_original = input_k.clone();

  // Move tensors to GPU
  input_q.to_cuda();
  input_k.to_cuda();
  sin_cache.to_cuda();
  cos_cache.to_cuda();
  // Note: positions remain on CPU as required by the RoPE layer

  // Set inputs
  rope_layer.set_batch_size(batch_size);
  rope_layer.set_input(0, input_q);
  rope_layer.set_input(1, input_k);
  rope_layer.set_input(2, positions);
  rope_layer.set_input(3, sin_cache);
  rope_layer.set_input(4, cos_cache);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), rope_layer.check());
  ASSERT_EQ(core::error::Success(), rope_layer.forward());

  // Copy results back to CPU for verification
  input_q.to_cpu();
  input_k.to_cpu();

  // Verify results:
  // Position 0 should not change (cosine is 1, sine is 0 for pos 0)
  for (int batch = 0; batch < batch_size; batch++) {
    for (int n = 0; n < num_q_heads; n++) {
      EXPECT_NEAR(input_q.at<float>(batch, 0, n, 0), q_original.at<float>(batch, 0, n, 0), 1e-5f)
          << "Q mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=0";
      EXPECT_NEAR(input_q.at<float>(batch, 0, n, 1), q_original.at<float>(batch, 0, n, 1), 1e-5f)
          << "Q mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=1";

      if (n < num_kv_heads) {
        EXPECT_NEAR(input_k.at<float>(batch, 0, n, 0), k_original.at<float>(batch, 0, n, 0), 1e-5f)
            << "K mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=0";
        EXPECT_NEAR(input_k.at<float>(batch, 0, n, 1), k_original.at<float>(batch, 0, n, 1), 1e-5f)
            << "K mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=1";
      }
    }
  }

  // For positions > 0, values should be different (rotated)
  bool found_rotated_values = false;
  for (int pos = 1; pos < seq_len; pos++) {
    for (int batch = 0; batch < batch_size; batch++) {
      for (int n = 0; n < num_q_heads; n++) {
        // Check if at least one value differs from the original
        if (std::abs(input_q.at<float>(batch, pos, n, 0) - q_original.at<float>(batch, pos, n, 0)) >
                1e-5f ||
            std::abs(input_q.at<float>(batch, pos, n, 1) - q_original.at<float>(batch, pos, n, 1)) >
                1e-5f) {
          found_rotated_values = true;
          break;
        }

        if (n < num_kv_heads) {
          if (std::abs(input_k.at<float>(batch, pos, n, 0) -
                       k_original.at<float>(batch, pos, n, 0)) > 1e-5f ||
              std::abs(input_k.at<float>(batch, pos, n, 1) -
                       k_original.at<float>(batch, pos, n, 1)) > 1e-5f) {
            found_rotated_values = true;
            break;
          }
        }
      }
      if (found_rotated_values) break;
    }
    if (found_rotated_values) break;
  }
  EXPECT_TRUE(found_rotated_values) << "No rotated values found for positions > 0";
}

}  // namespace
}  // namespace layer
