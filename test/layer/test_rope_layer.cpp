#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <vector>
#include "config.hpp"
#include "core/memory/memory_manager.hpp"
#include "core/status/status.hpp"
#include "kernel/cpu/rope_kernel_cpu.hpp"
#include "kernel/gpu/rope_kernel_gpu.cuh"
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
};

TEST_F(RoPELayerTest, CPUPrefill) {
  // Test parameters
  const int32_t batch_size = 2;
  const int32_t q_seq_len = 4;
  const int32_t kv_seq_len = 4;
  const int32_t max_position_embeddings = 16;
  const int32_t hidden_size = 16;
  const int32_t num_q_heads = 4;
  const int32_t num_kv_heads = 2;
  const int32_t head_size = hidden_size / num_q_heads;
  const int32_t kv_size = num_kv_heads * head_size;

  // Create RoPE layer
  RoPELayer rope_layer(core::DeviceType::CPU, hidden_size, kv_size, head_size);

  // Create input tensors
  tensor::Tensor input_q(core::DataType::FP32, {batch_size, q_seq_len, num_q_heads, head_size},
                         true, cpu_memory_manager, nullptr);
  tensor::Tensor input_k(core::DataType::FP32, {batch_size, kv_seq_len, num_kv_heads, head_size},
                         true, cpu_memory_manager, nullptr);
  tensor::Tensor q_positions(core::DataType::INT32, q_seq_len, true, cpu_memory_manager, nullptr);
  tensor::Tensor k_positions(core::DataType::INT32, kv_seq_len, true, cpu_memory_manager, nullptr);
  tensor::Tensor sin_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager, nullptr);
  tensor::Tensor cos_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager, nullptr);

  // Initialize with simple patterns
  for (int pos = 0; pos < q_seq_len; pos++) {
    q_positions.at<int32_t>(pos) = pos;
    k_positions.at<int32_t>(pos) = pos;
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
  kernel::sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, sin_cache,
                                 cos_cache);

  // Create copy of original values for comparison
  tensor::Tensor q_original = input_q.clone();
  tensor::Tensor k_original = input_k.clone();

  // Set inputs
  rope_layer.set_batch_size(batch_size);
  rope_layer.set_q_seq_len(q_seq_len);
  rope_layer.set_kv_seq_len(kv_seq_len);
  rope_layer.set_input(0, input_q);
  rope_layer.set_input(1, input_k);
  rope_layer.set_input(2, q_positions);
  rope_layer.set_input(3, k_positions);
  rope_layer.set_input(4, sin_cache);
  rope_layer.set_input(5, cos_cache);

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
  for (int pos = 1; pos < q_seq_len; pos++) {
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

TEST_F(RoPELayerTest, CPUDecode) {
  // Test parameters
  const int32_t batch_size = 2;
  const int32_t q_seq_len = 1;
  const int32_t kv_seq_len = 4;
  const int32_t max_position_embeddings = 16;
  const int32_t hidden_size = 16;
  const int32_t num_q_heads = 4;
  const int32_t num_kv_heads = 2;
  const int32_t head_size = hidden_size / num_q_heads;
  const int32_t kv_size = num_kv_heads * head_size;

  // Create RoPE layer
  RoPELayer rope_layer(core::DeviceType::CPU, hidden_size, kv_size, head_size);

  // Create input tensors
  tensor::Tensor input_q(core::DataType::FP32, {batch_size, q_seq_len, num_q_heads, head_size},
                         true, cpu_memory_manager, nullptr);
  tensor::Tensor input_k(core::DataType::FP32, {batch_size, kv_seq_len, num_kv_heads, head_size},
                         true, cpu_memory_manager, nullptr);
  tensor::Tensor q_positions(core::DataType::INT32, q_seq_len, true, cpu_memory_manager, nullptr);
  tensor::Tensor k_positions(core::DataType::INT32, kv_seq_len, true, cpu_memory_manager, nullptr);
  tensor::Tensor sin_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager, nullptr);
  tensor::Tensor cos_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager, nullptr);

  // Initialize with simple patterns
  q_positions.at<int32_t>(0) = 0;
  for (int pos = 0; pos < kv_seq_len; pos++) {
    k_positions.at<int32_t>(pos) = pos;
    for (int batch = 0; batch < batch_size; batch++) {
      for (int n = 0; n < num_q_heads; n++) {
        for (int h = 0; h < head_size; h++) {
          if (pos < q_seq_len) {
            input_q.at<float>(batch, pos, n, h) = (h % 2 == 0) ? 1.0f : -1.0f;
          }
          if (n < num_kv_heads) {
            input_k.at<float>(batch, pos, n, h) = (h % 2 == 0) ? -1.0f : 1.0f;
          }
        }
      }
    }
  }

  // Calculate sin/cos cache
  float rope_theta = 10000.0f;
  kernel::sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, sin_cache,
                                 cos_cache);

  // Create copy of original values for comparison
  tensor::Tensor q_original = input_q.clone();
  tensor::Tensor k_original = input_k.clone();

  // Set inputs
  rope_layer.set_batch_size(batch_size);
  rope_layer.set_q_seq_len(q_seq_len);
  rope_layer.set_kv_seq_len(kv_seq_len);
  rope_layer.set_input(0, input_q);
  rope_layer.set_input(1, input_k);
  rope_layer.set_input(2, q_positions);
  rope_layer.set_input(3, k_positions);
  rope_layer.set_input(4, sin_cache);
  rope_layer.set_input(5, cos_cache);

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
  for (int pos = 1; pos < kv_seq_len; pos++) {
    for (int batch = 0; batch < batch_size; batch++) {
      for (int n = 0; n < num_kv_heads; n++) {
        // Check if at least one value differs from the original
        if (std::abs(input_k.at<float>(batch, pos, n, 0) - k_original.at<float>(batch, pos, n, 0)) >
                1e-5f ||
            std::abs(input_k.at<float>(batch, pos, n, 1) - k_original.at<float>(batch, pos, n, 1)) >
                1e-5f) {
          found_rotated_values = true;
          break;
        }
      }
      if (found_rotated_values) break;
    }
    if (found_rotated_values) break;
  }
  EXPECT_TRUE(found_rotated_values) << "No rotated values found for positions > 0";
}

TEST_F(RoPELayerTest, GPUPrefill) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available, skipping GPU test";
  }

  // Test parameters
  const int32_t batch_size = 2;
  const int32_t q_seq_len = 4;
  const int32_t kv_seq_len = 4;
  const int32_t max_position_embeddings = 16;
  const int32_t hidden_size = 16;
  const int32_t num_q_heads = 4;
  const int32_t num_kv_heads = 2;
  const int32_t head_size = hidden_size / num_q_heads;
  const int32_t kv_size = num_kv_heads * head_size;

  // Create RoPE layer for GPU
  RoPELayer rope_layer(core::DeviceType::GPU, hidden_size, kv_size, head_size);

  // Create CPU tensors for initialization and verification
  tensor::Tensor cpu_q(core::DataType::FP32, {batch_size, q_seq_len, num_q_heads, head_size}, true,
                       cpu_memory_manager, nullptr);
  tensor::Tensor cpu_k(core::DataType::FP32, {batch_size, kv_seq_len, num_kv_heads, head_size},
                       true, cpu_memory_manager, nullptr);
  tensor::Tensor cpu_q_pos(core::DataType::INT32, q_seq_len, true, cpu_memory_manager, nullptr);
  tensor::Tensor cpu_k_pos(core::DataType::INT32, kv_seq_len, true, cpu_memory_manager, nullptr);
  tensor::Tensor cpu_sin_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                               cpu_memory_manager, nullptr);
  tensor::Tensor cpu_cos_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                               cpu_memory_manager, nullptr);

  // Create input tensors on GPU
  tensor::Tensor input_q, input_k, q_positions, k_positions, sin_cache, cos_cache;

  // Initialize CPU tensors with simple patterns
  for (int pos = 0; pos < q_seq_len; pos++) {
    cpu_q_pos.at<int32_t>(pos) = pos;
    cpu_k_pos.at<int32_t>(pos) = pos;
    for (int batch = 0; batch < batch_size; batch++) {
      for (int n = 0; n < num_q_heads; n++) {
        for (int h = 0; h < head_size; h++) {
          cpu_q.at<float>(batch, pos, n, h) = (h % 2 == 0) ? 1.0f : -1.0f;
          if (n < num_kv_heads) {
            cpu_k.at<float>(batch, pos, n, h) = (h % 2 == 0) ? -1.0f : 1.0f;
          }
        }
      }
    }
  }

  // Calculate sin/cos cache on CPU
  float rope_theta = 10000.0f;
  kernel::sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, cpu_sin_cache,
                                 cpu_cos_cache);

  input_q = cpu_q.clone();
  input_k = cpu_k.clone();
  q_positions = cpu_q_pos.clone();
  k_positions = cpu_k_pos.clone();
  sin_cache = cpu_sin_cache.clone();
  cos_cache = cpu_cos_cache.clone();

  input_q.to_cuda();
  input_k.to_cuda();
  q_positions.to_cuda();
  k_positions.to_cuda();
  sin_cache.to_cuda();
  cos_cache.to_cuda();

  // Set CUDA config for the layer
  auto cuda_config = std::make_shared<core::CudaConfig>();
  // cudaStreamCreate(&(cuda_config->stream));
  rope_layer.set_cuda_config(cuda_config);

  // Create copy of original values for comparison
  tensor::Tensor q_original = cpu_q.clone();
  tensor::Tensor k_original = cpu_k.clone();

  // Set inputs
  rope_layer.set_batch_size(batch_size);
  rope_layer.set_q_seq_len(q_seq_len);
  rope_layer.set_kv_seq_len(kv_seq_len);
  rope_layer.set_input(0, input_q);
  rope_layer.set_input(1, input_k);
  rope_layer.set_input(2, q_positions);
  rope_layer.set_input(3, k_positions);
  rope_layer.set_input(4, sin_cache);
  rope_layer.set_input(5, cos_cache);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), rope_layer.check());
  ASSERT_EQ(core::error::Success(), rope_layer.forward());

  // // Synchronize stream to ensure all operations are complete
  // cudaStreamSynchronize(cuda_config->stream);

  // Copy results back to CPU for verification
  tensor::Tensor result_q = input_q.clone();
  tensor::Tensor result_k = input_k.clone();
  result_q.to_cpu();
  result_k.to_cpu();

  // Verify results:
  // Position 0 should not change much (cosine is close to 1, sine is close to 0 for pos 0)
  for (int batch = 0; batch < batch_size; batch++) {
    for (int n = 0; n < num_q_heads; n++) {
      EXPECT_NEAR(result_q.at<float>(batch, 0, n, 0), q_original.at<float>(batch, 0, n, 0), 1e-5f)
          << "Q mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=0";
      EXPECT_NEAR(result_q.at<float>(batch, 0, n, 1), q_original.at<float>(batch, 0, n, 1), 1e-5f)
          << "Q mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=1";

      if (n < num_kv_heads) {
        EXPECT_NEAR(result_k.at<float>(batch, 0, n, 0), k_original.at<float>(batch, 0, n, 0), 1e-5f)
            << "K mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=0";
        EXPECT_NEAR(result_k.at<float>(batch, 0, n, 1), k_original.at<float>(batch, 0, n, 1), 1e-5f)
            << "K mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=1";
      }
    }
  }

  // For positions > 0, values should be different (rotated)
  bool found_rotated_values = false;
  for (int pos = 1; pos < q_seq_len; pos++) {
    for (int batch = 0; batch < batch_size; batch++) {
      for (int n = 0; n < num_q_heads; n++) {
        // Check if at least one value differs from the original
        if (std::abs(result_q.at<float>(batch, pos, n, 0) -
                     q_original.at<float>(batch, pos, n, 0)) > 1e-5f ||
            std::abs(result_q.at<float>(batch, pos, n, 1) -
                     q_original.at<float>(batch, pos, n, 1)) > 1e-5f) {
          found_rotated_values = true;
          break;
        }

        if (n < num_kv_heads) {
          if (std::abs(result_k.at<float>(batch, pos, n, 0) -
                       k_original.at<float>(batch, pos, n, 0)) > 1e-5f ||
              std::abs(result_k.at<float>(batch, pos, n, 1) -
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

  // // Clean up CUDA resources
  // cudaStreamDestroy(cuda_config->stream);
}

TEST_F(RoPELayerTest, GPUDecode) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available, skipping GPU test";
  }

  // Test parameters
  const int32_t batch_size = 2;
  const int32_t q_seq_len = 1;  // Single token for decoding
  const int32_t kv_seq_len = 8;
  const int32_t max_position_embeddings = 16;
  const int32_t hidden_size = 16;
  const int32_t num_q_heads = 4;
  const int32_t num_kv_heads = 2;
  const int32_t head_size = hidden_size / num_q_heads;
  const int32_t kv_size = num_kv_heads * head_size;

  // Create RoPE layer for GPU
  RoPELayer rope_layer(core::DeviceType::GPU, hidden_size, kv_size, head_size);

  // Create CPU tensors for initialization and verification
  tensor::Tensor input_q(core::DataType::FP32, {batch_size, q_seq_len, num_q_heads, head_size},
                         true, cpu_memory_manager, nullptr);
  tensor::Tensor input_k(core::DataType::FP32, {batch_size, kv_seq_len, num_kv_heads, head_size},
                         true, cpu_memory_manager, nullptr);
  tensor::Tensor q_positions(core::DataType::INT32, q_seq_len, true, cpu_memory_manager, nullptr);
  tensor::Tensor k_positions(core::DataType::INT32, kv_seq_len, true, cpu_memory_manager, nullptr);
  tensor::Tensor sin_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           gpu_memory_manager, nullptr);
  tensor::Tensor cos_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           gpu_memory_manager, nullptr);

  // Initialize CPU tensors with simple patterns
  q_positions.at<int32_t>(0) = 0;
  for (int pos = 0; pos < kv_seq_len; pos++) {
    k_positions.at<int32_t>(pos) = pos;
    for (int batch = 0; batch < batch_size; batch++) {
      for (int n = 0; n < num_q_heads; n++) {
        for (int h = 0; h < head_size; h++) {
          if (pos < q_seq_len) {
            input_q.at<float>(batch, pos, n, h) = (h % 2 == 0) ? 1.0f : -1.0f;
          }
          if (n < num_kv_heads) {
            input_k.at<float>(batch, pos, n, h) = (h % 2 == 0) ? -1.0f : 1.0f;
          }
        }
      }
    }
  }

  // Calculate sin/cos cache on CPU
  float rope_theta = 10000.0f;
  kernel::sin_cos_cache_calc_gpu(rope_theta, head_size, max_position_embeddings, sin_cache,
                                 cos_cache);

  // Create copy of original values for comparison
  tensor::Tensor q_original = input_q.clone();
  tensor::Tensor k_original = input_k.clone();

  input_q.to_cuda();
  input_k.to_cuda();
  q_positions.to_cuda();
  k_positions.to_cuda();

  // Set CUDA config for the layer
  auto cuda_config = std::make_shared<core::CudaConfig>();
  // cudaStreamCreate(&(cuda_config->stream));
  rope_layer.set_cuda_config(cuda_config);

  // Set inputs
  rope_layer.set_batch_size(batch_size);
  rope_layer.set_q_seq_len(q_seq_len);
  rope_layer.set_kv_seq_len(kv_seq_len);
  rope_layer.set_input(0, input_q);
  rope_layer.set_input(1, input_k);
  rope_layer.set_input(2, q_positions);
  rope_layer.set_input(3, k_positions);
  rope_layer.set_input(4, sin_cache);
  rope_layer.set_input(5, cos_cache);

  // Run checks and forward pass
  ASSERT_EQ(core::error::Success(), rope_layer.check());
  ASSERT_EQ(core::error::Success(), rope_layer.forward());

  // // Synchronize stream to ensure all operations are complete
  // cudaStreamSynchronize(cuda_config->stream);

  // Copy results back to CPU for verification
  tensor::Tensor result_q = input_q.clone();
  tensor::Tensor result_k = input_k.clone();
  result_q.to_cpu();
  result_k.to_cpu();

  // Verify results:
  // Position 0 should not change (cosine is 1, sine is 0 for pos 0)
  for (int batch = 0; batch < batch_size; batch++) {
    for (int n = 0; n < num_q_heads; n++) {
      EXPECT_NEAR(result_q.at<float>(batch, 0, n, 0), q_original.at<float>(batch, 0, n, 0), 1e-5f)
          << "Q mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=0";
      EXPECT_NEAR(result_q.at<float>(batch, 0, n, 1), q_original.at<float>(batch, 0, n, 1), 1e-5f)
          << "Q mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=1";

      if (n < num_kv_heads) {
        EXPECT_NEAR(result_k.at<float>(batch, 0, n, 0), k_original.at<float>(batch, 0, n, 0), 1e-5f)
            << "K mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=0";
        EXPECT_NEAR(result_k.at<float>(batch, 0, n, 1), k_original.at<float>(batch, 0, n, 1), 1e-5f)
            << "K mismatch at batch=" << batch << ", pos=0, head=" << n << ", dim=1";
      }
    }
  }

  // For positions > 0, values should be different (rotated)
  bool found_rotated_values = false;
  for (int pos = 1; pos < kv_seq_len; pos++) {
    for (int batch = 0; batch < batch_size; batch++) {
      for (int n = 0; n < num_kv_heads; n++) {
        // Check if at least one value differs from the original
        if (std::abs(result_k.at<float>(batch, pos, n, 0) -
                     k_original.at<float>(batch, pos, n, 0)) > 1e-5f ||
            std::abs(result_k.at<float>(batch, pos, n, 1) -
                     k_original.at<float>(batch, pos, n, 1)) > 1e-5f) {
          found_rotated_values = true;
          break;
        }
      }
      if (found_rotated_values) break;
    }
    if (found_rotated_values) break;
  }

  // Print k_original and result_k for debugging
  std::cout << "K Original:" << std::endl;
  for (int pos = 0; pos < kv_seq_len; pos++) {
    for (int batch = 0; batch < batch_size; batch++) {
      for (int n = 0; n < num_kv_heads; n++) {
        for (int h = 0; h < head_size; h++) {
          std::cout << k_original.at<float>(batch, pos, n, h) << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  // Print k_original and result_k for debugging
  std::cout << "K Result:" << std::endl;
  for (int pos = 0; pos < kv_seq_len; pos++) {
    for (int batch = 0; batch < batch_size; batch++) {
      for (int n = 0; n < num_kv_heads; n++) {
        for (int h = 0; h < head_size; h++) {
          std::cout << result_k.at<float>(batch, pos, n, h) << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  sin_cache.to_cpu();
  cos_cache.to_cpu();

  // Print sin_cache for debugging
  std::cout << "Sin Cache:" << std::endl;
  for (int pos = 0; pos < max_position_embeddings; pos++) {
    std::cout << "Position " << pos << ": ";
    for (int h = 0; h < head_size; h++) {
      std::cout << sin_cache.at<float>(pos, h) << " ";
    }
    std::cout << std::endl;
  }

  // Print cos_cache for debugging
  std::cout << "Cos Cache:" << std::endl;
  for (int pos = 0; pos < max_position_embeddings; pos++) {
    std::cout << "Position " << pos << ": ";
    for (int h = 0; h < head_size; h++) {
      std::cout << cos_cache.at<float>(pos, h) << " ";
    }
    std::cout << std::endl;
  }

  EXPECT_TRUE(found_rotated_values) << "No rotated values found for positions > 0";

  // // Clean up CUDA resources
  // cudaStreamDestroy(cuda_config->stream);
}

}  // namespace
}  // namespace layer
