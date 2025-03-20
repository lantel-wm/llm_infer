#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>
#include "mha_kernel_cpu.hpp"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class MhaKernelTest : public ::testing::Test {
 protected:
  void SetUp() override { cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance(); }

  void TearDown() override {
#ifndef PYTORCH_NOT_FOUND
    // Clean up generated files
    std::remove("generate_mha_data.py");
    std::remove("mha_query.bin");
    std::remove("mha_key.bin");
    std::remove("mha_value.bin");
    std::remove("mha_output.bin");
#endif
  }

  std::shared_ptr<core::CPUMemoryManager> cpu_memory_manager;
  std::mt19937 rng{42};  // Random number generator with fixed seed

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

  // Helper to generate random tensor
  void generate_random_tensor(tensor::Tensor& tensor) {
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int32_t i = 0; i < tensor.size(); i++) {
      tensor.index<float>(i) = dist(rng);
    }
  }
};

TEST_F(MhaKernelTest, MhaQkT) {
  // Set dimensions as specified
  const int32_t seq_len = 16;
  const int32_t head_size = 128;

  // Create query data (seq_len x head_size)
  std::vector<float> query_data(seq_len * head_size);
  for (int i = 0; i < seq_len * head_size; i++) {
    query_data[i] = static_cast<float>(i % 10) * 0.1f;  // Simple pattern for test data
  }

  // Create key data (seq_len x head_size)
  std::vector<float> key_data(seq_len * head_size);
  for (int i = 0; i < seq_len * head_size; i++) {
    key_data[i] = static_cast<float>((i + 3) % 10) * 0.1f;  // Another pattern for test data
  }

  // Create tensors
  tensor::Tensor query(core::DataType::FP32, seq_len, head_size, true, cpu_memory_manager);
  tensor::Tensor key(core::DataType::FP32, seq_len, head_size, true, cpu_memory_manager);
  tensor::Tensor score(core::DataType::FP32, seq_len, seq_len, true, cpu_memory_manager);

  // Initialize tensors
  for (int i = 0; i < query_data.size(); i++) {
    query.index<float>(i) = query_data[i];
  }
  for (int i = 0; i < key_data.size(); i++) {
    key.index<float>(i) = key_data[i];
  }

  // Compute scaling factor (1/sqrt(head_size))
  float scale = 1.0f / std::sqrt(static_cast<float>(head_size));

  // Run the MHA matmul kernel without causal mask
  mha_qkT_kernel_cpu(query, key, score, scale, false);

  // Verify results
  // Manually compute expected qkT matrix for verification
  std::vector<float> expected_score(seq_len * seq_len, 0.0f);
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < seq_len; j++) {
      float sum = 0.0f;
      for (int k = 0; k < head_size; k++) {
        sum += query_data[i * head_size + k] * key_data[j * head_size + k];
      }
      expected_score[i * seq_len + j] = sum * scale;
    }
  }

  // Verify results
  for (int i = 0; i < seq_len * seq_len; i++) {
    EXPECT_NEAR(score.index<float>(i), expected_score[i], 1e-4f)
        << "Mismatch at index " << i << " (row=" << (i / seq_len) << ", col=" << (i % seq_len)
        << ")";
  }

  // Test with causal masking
  tensor::Tensor causal_score(core::DataType::FP32, seq_len, seq_len, true, cpu_memory_manager);
  mha_qkT_kernel_cpu(query, key, causal_score, scale, true);

  // Verify causal masking is properly applied
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < seq_len; j++) {
      if (j > i) {
        // Upper triangular elements should be -infinity
        EXPECT_EQ(causal_score.at<float>(i, j), -std::numeric_limits<float>::infinity())
            << "Upper triangular element at (" << i << ", " << j << ") is not masked";
      } else {
        // Lower triangular elements should match the non-causal computation
        EXPECT_NEAR(causal_score.at<float>(i, j), expected_score[i * seq_len + j], 1e-4f)
            << "Lower triangular element at (" << i << ", " << j
            << ") doesn't match expected value";
      }
    }
  }
}

TEST_F(MhaKernelTest, MhaSoftmax) {
  const int32_t seq_len = 64;

  // Create tensor for scores
  tensor::Tensor score(core::DataType::FP32, seq_len, seq_len, true, cpu_memory_manager);
  tensor::Tensor score_expected;

  generate_random_tensor(score);
  score_expected = score.clone();

  // Compute expected softmax results manually
  for (int i = 0; i < seq_len; i++) {
    // Find max value in this row for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < seq_len; j++) {
      max_val = std::max(max_val, score_expected.at<float>(i, j));
    }

    // Compute exp(x - max) for each element in the row
    float row_sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
      float exp_val = std::exp(score_expected.at<float>(i, j) - max_val);
      score_expected.at<float>(i, j) = exp_val;
      row_sum += exp_val;
    }

    // Normalize by the sum
    for (int j = 0; j < seq_len; j++) {
      score_expected.at<float>(i, j) /= row_sum;
    }
  }

  // Apply softmax operation
  mha_softmax_kernel_cpu(score);

  // Verify results
  for (int i = 0; i < seq_len; i++) {
    // Check that each row sums to ~1
    float row_sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
      row_sum += score_expected.at<float>(i, j);
    }
    EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "Row " << i << " sum is not 1.0";

    // Check each value
    for (int j = 0; j < seq_len; j++) {
      EXPECT_NEAR(score.at<float>(i, j), score_expected.at<float>(i, j), 1e-5f)
          << "Mismatch at position (" << i << ", " << j << ")";
    }
  }
}

TEST_F(MhaKernelTest, MhaScoreV) {
  // Set dimensions
  const int32_t seq_len = 16;
  const int32_t head_size = 64;

  // Create score data (seq_len x seq_len) - representing softmax output
  std::vector<float> score_data(seq_len * seq_len);
  for (int i = 0; i < seq_len; i++) {
    float row_sum = 0.0f;
    // Create a simple attention pattern - normalized for each row
    for (int j = 0; j < seq_len; j++) {
      // Diagonal-focused attention pattern
      score_data[i * seq_len + j] = std::exp(-0.1f * std::abs(i - j));
      row_sum += score_data[i * seq_len + j];
    }
    // Normalize to sum to 1 (as softmax would)
    for (int j = 0; j < seq_len; j++) {
      score_data[i * seq_len + j] /= row_sum;
    }
  }

  // Create value data (seq_len x head_size)
  std::vector<float> value_data(seq_len * head_size);
  for (int i = 0; i < seq_len * head_size; i++) {
    value_data[i] = static_cast<float>((i + 5) % 10) * 0.1f;  // Simple pattern
  }

  // Create tensors
  tensor::Tensor score(core::DataType::FP32, seq_len, seq_len, true, cpu_memory_manager);
  tensor::Tensor value(core::DataType::FP32, seq_len, head_size, true, cpu_memory_manager);
  tensor::Tensor output(core::DataType::FP32, seq_len, head_size, true, cpu_memory_manager);

  // Initialize tensors
  for (int i = 0; i < score_data.size(); i++) {
    score.index<float>(i) = score_data[i];
  }
  for (int i = 0; i < value_data.size(); i++) {
    value.index<float>(i) = value_data[i];
  }

  // Run the score-value multiplication kernel
  mha_scorev_kernel_cpu(score, value, output);

  // Compute expected output manually
  std::vector<float> expected_output(seq_len * head_size, 0.0f);
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < head_size; j++) {
      float sum = 0.0f;
      for (int k = 0; k < seq_len; k++) {
        sum += score_data[i * seq_len + k] * value_data[k * head_size + j];
      }
      expected_output[i * head_size + j] = sum;
    }
  }

  // Verify results
  for (int i = 0; i < seq_len * head_size; i++) {
    EXPECT_NEAR(output.index<float>(i), expected_output[i], 1e-4f)
        << "Mismatch at index " << i << " (row=" << (i / head_size) << ", col=" << (i % head_size)
        << ")";
  }
}

#ifndef PYTORCH_NOT_FOUND
// Test softmax against PyTorch implementation
struct MhaTestParams {
  int batch_size;
  int query_seq_len;
  int kv_seq_len;
  int hidden_size;
  int num_heads;
  int num_kv_heads;
  std::string test_name;
};

class MhaFullTest : public MhaKernelTest, public ::testing::WithParamInterface<MhaTestParams> {};

TEST_P(MhaFullTest, CompareWithPyTorch) {
  const auto params = GetParam();

  // Create Python script for generating test data with current configuration
  {
    std::ofstream py_file("generate_mha_data.py");
    py_file << R"(
import torch

torch.manual_seed(42)

batch_size = )"
            << params.batch_size << R"(
query_seq_len = )"
            << params.query_seq_len << R"(
kv_seq_len = )"
            << params.kv_seq_len << R"(
hidden_size = )"
            << params.hidden_size << R"(
num_heads = )"
            << params.num_heads << R"(
num_kv_heads = )"
            << params.num_kv_heads << R"(
head_size = hidden_size // num_heads
kv_size = num_kv_heads * head_size

query = torch.randn(batch_size, query_seq_len, num_heads, head_size, dtype=torch.float32)
key = torch.randn(batch_size, kv_seq_len, num_kv_heads, head_size, dtype=torch.float32)
value = torch.randn(batch_size, kv_seq_len, num_kv_heads, head_size, dtype=torch.float32)

heads_per_kv = num_heads // num_kv_heads
key_rep = key.repeat_interleave(heads_per_kv, dim=2)
value_rep = value.repeat_interleave(heads_per_kv, dim=2)

query_input = query.view(batch_size, query_seq_len, hidden_size)
key_input = key_rep.view(batch_size, kv_seq_len, hidden_size)
value_input = value_rep.view(batch_size, kv_seq_len, hidden_size)

mha = torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads,dropout=0.0, batch_first=True, bias=False, dtype=torch.float32)

inproj_weight = torch.nn.Parameter(torch.concat([torch.eye(hidden_size, hidden_size) for _ in range(3)], dim=0))
outproj_weight = torch.nn.Parameter(torch.eye(hidden_size, hidden_size))

mha.in_proj_weight = inproj_weight
mha.out_proj.weight = outproj_weight

if query_seq_len > 1 and query_seq_len == kv_seq_len:
    is_causal = True
    attn_mask = torch.triu(
        torch.ones(query_seq_len, kv_seq_len, dtype=torch.float32), 
        diagonal=1
    ).bool()
else:
    is_causal = False
    attn_mask = None

output = mha(query_input, key_input, value_input, is_causal=is_causal, attn_mask=attn_mask)[0]
output = output.view(batch_size, query_seq_len, num_heads, head_size)

query.numpy().tofile('mha_query.bin')
key.numpy().tofile('mha_key.bin')
value.numpy().tofile('mha_value.bin')
output.detach().numpy().tofile('mha_output.bin')
)";
    py_file.close();
  }

  // Run Python script to generate test data
  ASSERT_EQ(std::system("python3 generate_mha_data.py"), 0) << "Failed to run Python script";

  // Load configuration from params
  const int32_t batch_size = params.batch_size;
  const int32_t query_seq_len = params.query_seq_len;
  const int32_t kv_seq_len = params.kv_seq_len;
  const int32_t num_heads = params.num_heads;
  const int32_t num_kv_heads = params.num_kv_heads;
  const int32_t hidden_size = params.hidden_size;
  const int32_t head_size = hidden_size / num_heads;
  const int32_t layer_idx = 0;
  const int32_t num_layers = 1;
  const int32_t max_position_embeddings = 128;  // Increased for longer sequences

  // Read PyTorch generated data
  std::vector<float> query_data =
      read_binary_file("mha_query.bin", batch_size * query_seq_len * num_heads * head_size);
  std::vector<float> key_data =
      read_binary_file("mha_key.bin", batch_size * kv_seq_len * num_kv_heads * head_size);
  std::vector<float> value_data =
      read_binary_file("mha_value.bin", batch_size * kv_seq_len * num_kv_heads * head_size);
  std::vector<float> torch_output =
      read_binary_file("mha_output.bin", batch_size * query_seq_len * hidden_size);

  // Create tensor shapes
  std::vector<int32_t> query_dims = {batch_size, query_seq_len, num_heads, head_size};
  std::vector<int32_t> score_dims = {query_seq_len, kv_seq_len};
  std::vector<int32_t> kv_cache_dims = {num_layers, batch_size, num_kv_heads,
                                        max_position_embeddings, head_size};

  // Create tensors
  tensor::Tensor query(core::DataType::FP32, query_dims, true, cpu_memory_manager);
  tensor::Tensor mha_output(core::DataType::FP32, query_dims, true, cpu_memory_manager);
  tensor::Tensor score(core::DataType::FP32, score_dims, true, cpu_memory_manager);
  tensor::Tensor key_cache(core::DataType::FP32, kv_cache_dims, true, cpu_memory_manager);
  tensor::Tensor value_cache(core::DataType::FP32, kv_cache_dims, true, cpu_memory_manager);

  // Initialize query tensor
  for (int i = 0; i < query_data.size(); i++) {
    query.index<float>(i) = query_data[i];
  }

  // Initialize key and value cache
  // PyTorch data is [batch_size, seq_len, num_kv_heads, head_size]
  // Need to reshape to [num_layers, batch_size, num_kv_heads, seq_len, head_size]
  for (int l = 0; l < num_layers; l++) {
    for (int b = 0; b < batch_size; b++) {
      for (int s = 0; s < kv_seq_len; s++) {
        for (int n = 0; n < num_kv_heads; n++) {
          for (int h = 0; h < head_size; h++) {
            int pytorch_idx = ((b * kv_seq_len + s) * num_kv_heads + n) * head_size + h;
            key_cache.at<float>(l, b, n, s, h) = key_data[pytorch_idx];
            value_cache.at<float>(l, b, n, s, h) = value_data[pytorch_idx];
          }
        }
      }
    }
  }

  // Run MHA kernel
  mha_kernel_cpu(layer_idx, num_layers, batch_size, query_seq_len, kv_seq_len, mha_output, query,
                 score, key_cache, value_cache);

  // Compare results with PyTorch
  float max_diff = 0.0f;
  float sum_squared_diff = 0.0f;

  for (int i = 0; i < mha_output.size(); i++) {
    float diff = std::abs(mha_output.index<float>(i) - torch_output[i]);
    max_diff = std::max(max_diff, diff);
    sum_squared_diff += diff * diff;
  }

  float rmse = std::sqrt(sum_squared_diff / mha_output.size());

  // We expect some numerical differences due to implementation details,
  // so we use a higher tolerance for the comparison
  EXPECT_LT(rmse, 1e-5f) << "RMSE too high compared to PyTorch implementation";
  EXPECT_LT(max_diff, 1e-4f) << "Max difference too high compared to PyTorch implementation";
}

// Generate different test configurations
INSTANTIATE_TEST_SUITE_P(MhaVariations, MhaFullTest,
                         ::testing::Values(
                             // Multi-head attention prefill
                             MhaTestParams{1, 8, 8, 512, 8, 8, "MHAPrefill"},
                             // Multi-head attention Decode
                             MhaTestParams{1, 1, 8, 512, 8, 8, "MHADecode"},
                             // Larger batch multi-head attention decode
                             MhaTestParams{4, 1, 8, 512, 8, 8, "LargerBatchMHADecode"},
                             // Multi-query attention (MQA) prefill
                             MhaTestParams{2, 8, 8, 512, 8, 1, "MQAPrefill"},
                             // Multi-query attention (MQA) decode
                             MhaTestParams{1, 1, 8, 512, 8, 1, "MQADecode"},
                             // Larger batch multi-query attention decode
                             MhaTestParams{4, 1, 8, 512, 8, 1, "LargerBatchMQADecode"},
                             // Grouped-query attention (GQA) prefill
                             MhaTestParams{2, 8, 8, 512, 8, 2, "GQAPrefill"},
                             // Grouped-query attention (GQA) decode
                             MhaTestParams{1, 1, 8, 512, 8, 2, "GQADecode"},
                             // Larger batch grouped-query attention decode
                             MhaTestParams{4, 1, 8, 512, 8, 2, "LargerBatchGQADecode"}),
                         [](const ::testing::TestParamInfo<MhaTestParams>& info) {
                           return info.param.test_name;
                         });
#endif

}  // namespace
}  // namespace kernel