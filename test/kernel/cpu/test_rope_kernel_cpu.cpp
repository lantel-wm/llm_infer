#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>
#include "rope_kernel_cpu.hpp"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class RopeKernelTest : public ::testing::Test {
 protected:
  void SetUp() override { cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance(); }

  void TearDown() override {
#ifndef PYTORCH_NOT_FOUND
    // Clean up generated files
    std::remove("generate_rope_data.py");
    std::remove("rope_input_q.bin");
    std::remove("rope_input_k.bin");
    std::remove("rope_output_q.bin");
    std::remove("rope_output_k.bin");
#endif
  }

  std::shared_ptr<core::CPUMemoryManager> cpu_memory_manager;

  // Helper function to read binary data
  template <typename T>
  std::vector<T> read_binary_file(const std::string& filename, size_t size) {
    std::vector<T> data(size);
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
      file.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
      file.close();
    }
    return data;
  }
};

// Basic test for sin_cos_cache_calc_cpu function
TEST_F(RopeKernelTest, SinCosCacheCalc) {
  const int head_size = 32;
  const int max_position_embeddings = 16;
  const float rope_theta = 10000.0f;

  tensor::Tensor sin_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);
  tensor::Tensor cos_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);

  // Calculate sin/cos cache
  sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, sin_cache, cos_cache);

  // Verify a few values manually
  for (int pos = 0; pos < max_position_embeddings; pos++) {
    for (int i = 0; i < head_size / 2; i++) {
      float freq = 1.0f / powf(rope_theta, (2.0f * i) / head_size);
      float angle = pos * freq;

      float expected_sin = sinf(angle);
      float expected_cos = cosf(angle);

      EXPECT_NEAR(sin_cache.at<float>(pos, i * 2), expected_sin, 1e-5f)
          << "Sin mismatch at pos=" << pos << ", i=" << i;
      EXPECT_NEAR(cos_cache.at<float>(pos, i * 2), expected_cos, 1e-5f)
          << "Cos mismatch at pos=" << pos << ", i=" << i;
    }
  }
}

// Test basic RoPE functionality with simple values
TEST_F(RopeKernelTest, MultiHeadAttentionPrefill) {
  const float rope_theta = 1000000.0f;
  const int32_t max_position_embeddings = 512;
  const int32_t hidden_size = 256;
  const int32_t num_attention_heads = 4;
  const int32_t num_key_value_heads = 4;
  const int32_t head_size = hidden_size / num_attention_heads;
  const int32_t key_value_size = head_size * num_key_value_heads;
  const int32_t batch_size = 1;
  const int32_t seq_len = 4;

  // Create tensors
  tensor::Tensor q(core::DataType::FP32, {batch_size, seq_len, num_attention_heads, head_size},
                   true, cpu_memory_manager);
  tensor::Tensor k(core::DataType::FP32, {batch_size, seq_len, num_key_value_heads, head_size},
                   true, cpu_memory_manager);
  tensor::Tensor positions(core::DataType::INT32, seq_len, true, cpu_memory_manager);
  tensor::Tensor sin_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);
  tensor::Tensor cos_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);

  // Initialize with simple patterns:
  // q: [1, 0, 1, 0, ...] for all positions
  // k: [0, 1, 0, 1, ...] for all positions
  for (int pos = 0; pos < seq_len; pos++) {
    positions.at<int32_t>(pos) = pos;
    for (int n = 0; n < num_attention_heads; n++) {
      for (int h = 0; h < head_size; h++) {
        int idx = (pos * num_attention_heads + n) * head_size + h;
        q.at<float>(0, pos, n, h) = (h % 2 == 0) ? 1.0f : -1.0f;
        if (n < num_key_value_heads) {
          k.at<float>(0, pos, n, h) = (h % 2 == 0) ? -1.0f : 1.0f;
        }
      }
    }
  }

  // Calculate sin/cos cache
  sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, sin_cache, cos_cache);

  // Run RoPE kernel
  rope_kernel_cpu(hidden_size, key_value_size, head_size, q, k, positions, positions, sin_cache,
                  cos_cache, nullptr);

  // Manually calculate expected values for a few test cases
  // For position 0, cos=1, sin=0 (approximately), so no change from input
  EXPECT_NEAR(q.at<float>(0, 0, 0, 0), 1.0f, 1e-5f);   // batch=0, pos=0, head=0, dim=0
  EXPECT_NEAR(q.at<float>(0, 0, 0, 1), -1.0f, 1e-5f);  // batch=0, pos=0, head=0, dim=1
  EXPECT_NEAR(k.at<float>(0, 0, 0, 0), -1.0f, 1e-5f);  // batch=0, pos=0, head=0, dim=0
  EXPECT_NEAR(k.at<float>(0, 0, 0, 1), 1.0f, 1e-5f);   // batch=0, pos=0, head=0, dim=1

  // For later positions, verify general pattern (not exact values)
  for (int pos = 1; pos < seq_len; pos++) {
    // Get the first pair of values for this position
    float q0 = q.at<float>(0, pos, 0, 0);
    float q1 = q.at<float>(0, pos, 0, 1);
    float k0 = k.at<float>(0, pos, 0, 0);
    float k1 = k.at<float>(0, pos, 0, 1);

    // Values should be different from the original pattern due to rotation
    // Use EXPECT_FALSE with EXPECT_NEAR to check values are not close to original
    EXPECT_NE(q0, 1.0f);
    EXPECT_NE(q1, -1.0f);
    EXPECT_NE(k0, -1.0f);
    EXPECT_NE(k1, 1.0f);
  }
}

// Test basic RoPE functionality with simple values
TEST_F(RopeKernelTest, MultiHeadAttentionDecode) {
  const float rope_theta = 1000000.0f;
  const int32_t max_position_embeddings = 512;
  const int32_t hidden_size = 256;
  const int32_t num_attention_heads = 4;
  const int32_t num_key_value_heads = 4;
  const int32_t head_size = hidden_size / num_attention_heads;
  const int32_t key_value_size = head_size * num_key_value_heads;
  const int32_t batch_size = 1;
  const int32_t kv_seq_len = 4;
  const int32_t query_seq_len = 1;

  // Create tensors
  tensor::Tensor q(core::DataType::FP32,
                   {batch_size, query_seq_len, num_attention_heads, head_size}, true,
                   cpu_memory_manager);
  tensor::Tensor k(core::DataType::FP32, {batch_size, kv_seq_len, num_key_value_heads, head_size},
                   true, cpu_memory_manager);
  tensor::Tensor q_positions(core::DataType::INT32, query_seq_len, true, cpu_memory_manager);
  tensor::Tensor k_positions(core::DataType::INT32, kv_seq_len, true, cpu_memory_manager);
  tensor::Tensor sin_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);
  tensor::Tensor cos_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);

  // Initialize with simple patterns:
  // q: [1, 0, 1, 0, ...] for all positions
  // k: [0, 1, 0, 1, ...] for all positions
  q_positions.at<int32_t>(0) = 0;
  for (int pos = 0; pos < kv_seq_len; pos++) {
    k_positions.at<int32_t>(pos) = pos;
    for (int n = 0; n < num_attention_heads; n++) {
      for (int h = 0; h < head_size; h++) {
        if (pos < query_seq_len) {
          q.at<float>(0, pos, n, h) = (h % 2 == 0) ? 1.0f : -1.0f;
        }
        if (n < num_key_value_heads) {
          k.at<float>(0, pos, n, h) = (h % 2 == 0) ? -1.0f : 1.0f;
        }
      }
    }
  }

  // Calculate sin/cos cache
  sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, sin_cache, cos_cache);

  // Run RoPE kernel
  rope_kernel_cpu(hidden_size, key_value_size, head_size, q, k, q_positions, k_positions, sin_cache,
                  cos_cache, nullptr);

  // Manually calculate expected values for a few test cases
  // For position 0, cos=1, sin=0 (approximately), so no change from input
  EXPECT_NEAR(q.at<float>(0, 0, 0, 0), 1.0f, 1e-5f);   // batch=0, pos=0, head=0, dim=0
  EXPECT_NEAR(q.at<float>(0, 0, 0, 1), -1.0f, 1e-5f);  // batch=0, pos=0, head=0, dim=1
  EXPECT_NEAR(k.at<float>(0, 0, 0, 0), -1.0f, 1e-5f);  // batch=0, pos=0, head=0, dim=0
  EXPECT_NEAR(k.at<float>(0, 0, 0, 1), 1.0f, 1e-5f);   // batch=0, pos=0, head=0, dim=1

  // For later positions, verify general pattern (not exact values)
  for (int pos = 1; pos < kv_seq_len; pos++) {
    // Get the first pair of values for this position
    float k0 = k.at<float>(0, pos, 0, 0);
    float k1 = k.at<float>(0, pos, 0, 1);

    // Values should be different from the original pattern due to rotation
    // Use EXPECT_FALSE with EXPECT_NEAR to check values are not close to original
    EXPECT_NE(k0, -1.0f);
    EXPECT_NE(k1, 1.0f);
  }
}

// Test basic RoPE functionality with simple values
TEST_F(RopeKernelTest, GroupQueryAttention) {
  const float rope_theta = 1000000.0f;
  const int32_t max_position_embeddings = 16;
  const int32_t hidden_size = 8;
  const int32_t num_attention_heads = 4;
  const int32_t num_key_value_heads = 2;
  const int32_t head_size = hidden_size / num_attention_heads;
  const int32_t key_value_size = head_size * num_key_value_heads;
  const int32_t batch_size = 1;
  const int32_t seq_len = 4;

  // Create tensors
  tensor::Tensor q(core::DataType::FP32, {batch_size, seq_len, num_attention_heads, head_size},
                   true, cpu_memory_manager);
  tensor::Tensor k(core::DataType::FP32, {batch_size, seq_len, num_key_value_heads, head_size},
                   true, cpu_memory_manager);
  tensor::Tensor positions(core::DataType::INT32, seq_len, true, cpu_memory_manager);
  tensor::Tensor sin_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);
  tensor::Tensor cos_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);

  // Initialize with simple patterns:
  // q: [1, 0, 1, 0, ...] for all positions
  // k: [0, 1, 0, 1, ...] for all positions
  for (int pos_idx = 0; pos_idx < seq_len; pos_idx++) {
    positions.at<int32_t>(pos_idx) = pos_idx;
    for (int n = 0; n < num_attention_heads; n++) {
      for (int h = 0; h < head_size; h++) {
        int idx = (pos_idx * num_attention_heads + n) * head_size + h;
        q.at<float>(0, pos_idx, n, h) = (h % 2 == 0) ? 1.0f : -1.0f;
        if (n < num_key_value_heads) {
          k.at<float>(0, pos_idx, n, h) = (h % 2 == 0) ? -1.0f : 1.0f;
        }
      }
    }
  }

  // Calculate sin/cos cache
  sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, sin_cache, cos_cache);

  // Run RoPE kernel
  rope_kernel_cpu(hidden_size, key_value_size, head_size, q, k, positions, positions, sin_cache,
                  cos_cache, nullptr);

  // Manually calculate expected values for a few test cases
  // For position 0, cos=1, sin=0 (approximately), so no change from input
  EXPECT_NEAR(q.at<float>(0, 0, 0, 0), 1.0f, 1e-5f);   // batch=0, pos=0, head=0, dim=0
  EXPECT_NEAR(q.at<float>(0, 0, 0, 1), -1.0f, 1e-5f);  // batch=0, pos=0, head=0, dim=1
  EXPECT_NEAR(k.at<float>(0, 0, 0, 0), -1.0f, 1e-5f);  // batch=0, pos=0, head=0, dim=0
  EXPECT_NEAR(k.at<float>(0, 0, 0, 1), 1.0f, 1e-5f);   // batch=0, pos=0, head=0, dim=1

  // For later positions, verify general pattern (not exact values)
  for (int pos_idx = 1; pos_idx < seq_len; pos_idx++) {
    // Get the first pair of values for this position
    int pos = positions.at<int>(pos_idx);
    float q0 = q.at<float>(0, pos, 0, 0);
    float q1 = q.at<float>(0, pos, 0, 1);
    float k0 = k.at<float>(0, pos, 0, 0);
    float k1 = k.at<float>(0, pos, 0, 1);

    // Values should be different from the original pattern due to rotation
    // Use EXPECT_FALSE with EXPECT_NEAR to check values are not close to original
    EXPECT_NE(q0, 1.0f);
    EXPECT_NE(q1, -1.0f);
    EXPECT_NE(k0, -1.0f);
    EXPECT_NE(k1, 1.0f);
  }
}

// Test basic RoPE functionality with simple values
TEST_F(RopeKernelTest, GroupQueryAttentionDecode) {
  const float rope_theta = 1000000.0f;
  const int32_t max_position_embeddings = 512;
  const int32_t hidden_size = 256;
  const int32_t num_attention_heads = 4;
  const int32_t num_key_value_heads = 2;
  const int32_t head_size = hidden_size / num_attention_heads;
  const int32_t key_value_size = head_size * num_key_value_heads;
  const int32_t batch_size = 1;
  const int32_t kv_seq_len = 4;
  const int32_t query_seq_len = 1;

  // Create tensors
  tensor::Tensor q(core::DataType::FP32,
                   {batch_size, query_seq_len, num_attention_heads, head_size}, true,
                   cpu_memory_manager);
  tensor::Tensor k(core::DataType::FP32, {batch_size, kv_seq_len, num_key_value_heads, head_size},
                   true, cpu_memory_manager);
  tensor::Tensor q_positions(core::DataType::INT32, query_seq_len, true, cpu_memory_manager);
  tensor::Tensor k_positions(core::DataType::INT32, kv_seq_len, true, cpu_memory_manager);
  tensor::Tensor sin_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);
  tensor::Tensor cos_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);

  // Initialize with simple patterns:
  // q: [1, 0, 1, 0, ...] for all positions
  // k: [0, 1, 0, 1, ...] for all positions
  q_positions.at<int32_t>(0) = 0;
  for (int pos = 0; pos < kv_seq_len; pos++) {
    k_positions.at<int32_t>(pos) = pos;
    for (int n = 0; n < num_attention_heads; n++) {
      for (int h = 0; h < head_size; h++) {
        if (pos < query_seq_len) {
          q.at<float>(0, pos, n, h) = (h % 2 == 0) ? 1.0f : -1.0f;
        }
        if (n < num_key_value_heads) {
          k.at<float>(0, pos, n, h) = (h % 2 == 0) ? -1.0f : 1.0f;
        }
      }
    }
  }

  // Calculate sin/cos cache
  sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, sin_cache, cos_cache);

  // Run RoPE kernel
  rope_kernel_cpu(hidden_size, key_value_size, head_size, q, k, q_positions, k_positions, sin_cache,
                  cos_cache, nullptr);

  // Manually calculate expected values for a few test cases
  // For position 0, cos=1, sin=0 (approximately), so no change from input
  EXPECT_NEAR(q.at<float>(0, 0, 0, 0), 1.0f, 1e-5f);   // batch=0, pos=0, head=0, dim=0
  EXPECT_NEAR(q.at<float>(0, 0, 0, 1), -1.0f, 1e-5f);  // batch=0, pos=0, head=0, dim=1
  EXPECT_NEAR(k.at<float>(0, 0, 0, 0), -1.0f, 1e-5f);  // batch=0, pos=0, head=0, dim=0
  EXPECT_NEAR(k.at<float>(0, 0, 0, 1), 1.0f, 1e-5f);   // batch=0, pos=0, head=0, dim=1

  // For later positions, verify general pattern (not exact values)
  for (int pos = 1; pos < kv_seq_len; pos++) {
    // Get the first pair of values for this position
    float k0 = k.at<float>(0, pos, 0, 0);
    float k1 = k.at<float>(0, pos, 0, 1);

    // Values should be different from the original pattern due to rotation
    // Use EXPECT_FALSE with EXPECT_NEAR to check values are not close to original
    EXPECT_NE(k0, -1.0f);
    EXPECT_NE(k1, 1.0f);
  }
}

#ifndef PYTORCH_NOT_FOUND
// Test RoPE against PyTorch implementation
TEST_F(RopeKernelTest, CompareWithPyTorchPrefill) {
  {
    std::ofstream py_file("generate_rope_data.py");
    py_file << R"(
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

def compute_cos_sin_cache(rope_theta, head_size, max_position_embeddings):
    """Calculate sin/cos cache for rotary position embedding"""
    inv_freq = 1.0 / (rope_theta**(torch.arange(
            0, head_size, 2, dtype=torch.float) / head_size))
    t = torch.arange(max_position_embeddings, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return torch.stack((o1, o2), dim=-1).flatten(-2)

# Example usage
rope_theta = 10000.0
max_position_embeddings = 512
hidden_size = 256
batch_size = 1
seq_len = 4
num_heads = 16
num_kv_heads = 8
head_size = hidden_size // num_heads
positions = torch.arange(seq_len)
    
cos_sin_cache = compute_cos_sin_cache(rope_theta, head_size, max_position_embeddings)
cos_sin = cos_sin_cache.index_select(0, positions)
cos, sin = cos_sin.chunk(2, dim=-1)

# Create sample query and key tensors
q = torch.randn(batch_size, seq_len, num_heads, head_size)
k = torch.randn(batch_size, seq_len, num_kv_heads, head_size)

# Save input tensors to files
q.numpy().astype(np.float32).tofile("rope_input_q.bin")
k.numpy().astype(np.float32).tofile("rope_input_k.bin")

q_rot = apply_rotary_emb(q, cos, sin)
k_rot = apply_rotary_emb(k, cos, sin)

# Save output tensors to files
q_rot.numpy().astype(np.float32).tofile("rope_output_q.bin")
k_rot.numpy().astype(np.float32).tofile("rope_output_k.bin")
)";
    py_file.close();
  }

  // Run Python script to generate test data
  ASSERT_EQ(std::system("python3 generate_rope_data.py"), 0) << "Failed to run Python script";

  // Parameters matching the Python script
  const float rope_theta = 10000.0f;
  const int32_t max_position_embeddings = 512;
  const int32_t hidden_size = 256;
  const int32_t batch_size = 1;
  const int32_t seq_len = 4;
  const int32_t num_attention_heads = 16;
  const int32_t num_key_value_heads = 8;
  const int32_t head_size = hidden_size / num_attention_heads;
  const int32_t key_value_size = num_key_value_heads * head_size;

  // Create tensors
  tensor::Tensor q(core::DataType::FP32, {batch_size, seq_len, num_attention_heads, head_size},
                   true, cpu_memory_manager);
  tensor::Tensor k(core::DataType::FP32, {batch_size, seq_len, num_key_value_heads, head_size},
                   true, cpu_memory_manager);
  tensor::Tensor positions(core::DataType::INT32, seq_len, true, cpu_memory_manager);
  tensor::Tensor sin_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);
  tensor::Tensor cos_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);

  // Read input data from file
  std::vector<float> q_data = read_binary_file<float>(
      "rope_input_q.bin", batch_size * seq_len * num_attention_heads * head_size);
  std::vector<float> k_data = read_binary_file<float>(
      "rope_input_k.bin", batch_size * seq_len * num_key_value_heads * head_size);

  // Read expected output data from file
  std::vector<float> q_expected = read_binary_file<float>(
      "rope_output_q.bin", batch_size * seq_len * num_attention_heads * head_size);
  std::vector<float> k_expected = read_binary_file<float>(
      "rope_output_k.bin", batch_size * seq_len * num_key_value_heads * head_size);

  std::memcpy(q.ptr<float>(), q_data.data(), q.size() * sizeof(float));
  std::memcpy(k.ptr<float>(), k_data.data(), k.size() * sizeof(float));

  for (int pos_idx = 0; pos_idx < seq_len; pos_idx++) {
    positions.at<int>(pos_idx) = pos_idx;
  }

  // Calculate sin/cos cache
  sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, sin_cache, cos_cache);

  // Run RoPE kernel
  rope_kernel_cpu(hidden_size, key_value_size, head_size, q, k, positions, positions, sin_cache,
                  cos_cache, nullptr);

  // Compare result with expected output
  for (int b = 0; b < batch_size; b++) {
    for (int s = 0; s < seq_len; s++) {
      for (int n = 0; n < num_attention_heads; n++) {
        for (int h = 0; h < head_size; h++) {
          int idx = ((b * seq_len + s) * num_attention_heads + n) * head_size + h;
          EXPECT_NEAR(q.at<float>(b, s, n, h), q_expected[idx], 1e-5f)
              << "Q mismatch at (batch=" << b << ", seq=" << s << ", head=" << n << ", dim=" << h
              << ")";

          if (n < num_key_value_heads) {
            int k_idx = ((b * seq_len + s) * num_key_value_heads + n) * head_size + h;
            EXPECT_NEAR(k.at<float>(b, s, n, h), k_expected[k_idx], 1e-5f)
                << "K mismatch at (batch=" << b << ", seq=" << s << ", head=" << n << ", dim=" << h
                << ")";
          }
        }
      }
    }
  }
}

// Test RoPE decode against PyTorch implementation
TEST_F(RopeKernelTest, CompareWithPyTorchDecode) {
  {
    std::ofstream py_file("generate_rope_decode_data.py");
    py_file << R"(
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

def compute_cos_sin_cache(rope_theta, head_size, max_position_embeddings):
    """Calculate sin/cos cache for rotary position embedding"""
    inv_freq = 1.0 / (rope_theta**(torch.arange(
            0, head_size, 2, dtype=torch.float) / head_size))
    t = torch.arange(max_position_embeddings, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return torch.stack((o1, o2), dim=-1).flatten(-2)

# Example usage for decode scenario
rope_theta = 10000.0
max_position_embeddings = 512
hidden_size = 256
batch_size = 1
query_seq_len = 1
kv_seq_len = 8
num_heads = 16
num_kv_heads = 8
head_size = hidden_size // num_heads

# Query position is the last position
q_position = torch.tensor([kv_seq_len - 1])
# KV positions are 0 to kv_seq_len-1
k_positions = torch.arange(kv_seq_len)
    
cos_sin_cache = compute_cos_sin_cache(rope_theta, head_size, max_position_embeddings)

# Get cos/sin for query
q_cos_sin = cos_sin_cache.index_select(0, q_position)
q_cos, q_sin = q_cos_sin.chunk(2, dim=-1)

# Get cos/sin for key
k_cos_sin = cos_sin_cache.index_select(0, k_positions)
k_cos, k_sin = k_cos_sin.chunk(2, dim=-1)

# Create sample query and key tensors
q = torch.randn(batch_size, query_seq_len, num_heads, head_size)
k = torch.randn(batch_size, kv_seq_len, num_kv_heads, head_size)

# Save input tensors to files
q.numpy().astype(np.float32).tofile("rope_decode_input_q.bin")
k.numpy().astype(np.float32).tofile("rope_decode_input_k.bin")

# Apply rotary embeddings
q_rot = apply_rotary_emb(q, q_cos, q_sin)
k_rot = apply_rotary_emb(k, k_cos, k_sin)

# Save output tensors to files
q_rot.numpy().astype(np.float32).tofile("rope_decode_output_q.bin")
k_rot.numpy().astype(np.float32).tofile("rope_decode_output_k.bin")

# Save positions to file
q_position.numpy().astype(np.int32).tofile("rope_decode_q_position.bin")
k_positions.numpy().astype(np.int32).tofile("rope_decode_k_positions.bin")
)";
    py_file.close();
  }

  // Run Python script to generate test data
  ASSERT_EQ(std::system("python3 generate_rope_decode_data.py"), 0)
      << "Failed to run Python script";

  // Parameters matching the Python script
  const float rope_theta = 10000.0f;
  const int32_t max_position_embeddings = 512;
  const int32_t hidden_size = 256;
  const int32_t batch_size = 1;
  const int32_t query_seq_len = 1;
  const int32_t kv_seq_len = 8;
  const int32_t num_attention_heads = 16;
  const int32_t num_key_value_heads = 8;
  const int32_t head_size = hidden_size / num_attention_heads;
  const int32_t key_value_size = num_key_value_heads * head_size;

  // Create tensors
  tensor::Tensor q(core::DataType::FP32,
                   {batch_size, query_seq_len, num_attention_heads, head_size}, true,
                   cpu_memory_manager);
  tensor::Tensor k(core::DataType::FP32, {batch_size, kv_seq_len, num_key_value_heads, head_size},
                   true, cpu_memory_manager);
  tensor::Tensor q_positions(core::DataType::INT32, query_seq_len, true, cpu_memory_manager);
  tensor::Tensor k_positions(core::DataType::INT32, kv_seq_len, true, cpu_memory_manager);
  tensor::Tensor sin_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);
  tensor::Tensor cos_cache(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                           cpu_memory_manager);

  // Read input data from file
  std::vector<float> q_data = read_binary_file<float>(
      "rope_decode_input_q.bin", batch_size * query_seq_len * num_attention_heads * head_size);
  std::vector<float> k_data = read_binary_file<float>(
      "rope_decode_input_k.bin", batch_size * kv_seq_len * num_key_value_heads * head_size);
  std::vector<int32_t> q_pos_data =
      read_binary_file<int32_t>("rope_decode_q_position.bin", query_seq_len);
  std::vector<int32_t> k_pos_data =
      read_binary_file<int32_t>("rope_decode_k_positions.bin", kv_seq_len);

  // Read expected output data from file
  std::vector<float> q_expected = read_binary_file<float>(
      "rope_decode_output_q.bin", batch_size * query_seq_len * num_attention_heads * head_size);
  std::vector<float> k_expected = read_binary_file<float>(
      "rope_decode_output_k.bin", batch_size * kv_seq_len * num_key_value_heads * head_size);

  // Copy input data to tensors
  std::memcpy(q.ptr<float>(), q_data.data(), q.size() * sizeof(float));
  std::memcpy(k.ptr<float>(), k_data.data(), k.size() * sizeof(float));
  std::memcpy(q_positions.ptr<int32_t>(), q_pos_data.data(), q_positions.size() * sizeof(int32_t));
  std::memcpy(k_positions.ptr<int32_t>(), k_pos_data.data(), k_positions.size() * sizeof(int32_t));

  // Calculate sin/cos cache
  sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, sin_cache, cos_cache);

  // Run RoPE kernel for decode scenario
  rope_kernel_cpu(hidden_size, key_value_size, head_size, q, k, q_positions, k_positions, sin_cache,
                  cos_cache, nullptr);

  // Compare result with expected output for query
  for (int b = 0; b < batch_size; b++) {
    for (int s = 0; s < query_seq_len; s++) {
      for (int n = 0; n < num_attention_heads; n++) {
        for (int h = 0; h < head_size; h++) {
          int idx = ((b * query_seq_len + s) * num_attention_heads + n) * head_size + h;
          EXPECT_NEAR(q.at<float>(b, s, n, h), q_expected[idx], 1e-5f)
              << "Q mismatch at (batch=" << b << ", seq=" << s << ", head=" << n << ", dim=" << h
              << ")";
        }
      }
    }
  }

  // Compare result with expected output for key
  for (int b = 0; b < batch_size; b++) {
    for (int s = 0; s < kv_seq_len; s++) {
      for (int n = 0; n < num_key_value_heads; n++) {
        for (int h = 0; h < head_size; h++) {
          int k_idx = ((b * kv_seq_len + s) * num_key_value_heads + n) * head_size + h;
          EXPECT_NEAR(k.at<float>(b, s, n, h), k_expected[k_idx], 1e-5f)
              << "K mismatch at (batch=" << b << ", seq=" << s << ", head=" << n << ", dim=" << h
              << ")";
        }
      }
    }
  }
}

#endif
}  // namespace
}  // namespace kernel

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}