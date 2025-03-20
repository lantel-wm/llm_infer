#include "mha_kernel_cpu.hpp"
#include <armadillo>
#include <cmath>
#include "memory_manager.hpp"
#include "tensor.hpp"

namespace kernel {
void mha_qkT_kernel_cpu(const tensor::Tensor& query, const tensor::Tensor& key,
                        const tensor::Tensor& score, float scale, bool is_causal) {
  const int32_t query_seq_len = query.get_dim(0);
  const int32_t kv_seq_len = key.get_dim(0);
  const int32_t head_size = query.get_dim(1);

  // Create matrix views with armadillo
  // Note: In armadillo, matrices are stored in column-major order by default
  // We use 'false, true' to indicate that we're using the provided memory without copying,
  // and that the matrices should be interpreted in row-major order
  arma::fmat query_mat(const_cast<float*>(query.ptr<float>()), head_size, query_seq_len, false,
                       true);
  arma::fmat key_mat(const_cast<float*>(key.ptr<float>()), head_size, kv_seq_len, false, true);
  arma::fmat qkT_mat(const_cast<float*>(score.ptr<float>()), kv_seq_len, query_seq_len, false,
                     true);

  // Calculate QK^T
  qkT_mat = (query_mat.t() * key_mat).t() * scale;

  // Apply causal mask if requested
  if (is_causal) {
    // We need to apply the causal mask to ensure each position can only attend
    // to positions up to and including itself
    for (int32_t i = 0; i < query_seq_len; i++) {
      for (int32_t j = i + 1; j < kv_seq_len; j++) {
        // Set the upper triangular elements to -infinity
        // Note: qkT_mat is transposed, so we access (j, i) to modify the (i, j) position
        qkT_mat(j, i) = -std::numeric_limits<float>::infinity();
      }
    }
  }
}

void mha_softmax_kernel_cpu(const tensor::Tensor& score) {
  const int32_t query_seq_len = score.get_dim(0);
  const int32_t kv_seq_len = score.get_dim(1);
  arma::fmat score_mat(const_cast<float*>(score.ptr<float>()), kv_seq_len, query_seq_len, false,
                       false);
  score_mat.each_col([&](arma::fvec& col) {
    double max_val = col.max();
    col -= max_val;
    arma::fvec exp_col = exp(col);
    double sum_exp = sum(exp_col);
    col = exp_col / sum_exp;
  });
}

void mha_scorev_kernel_cpu(const tensor::Tensor& score, const tensor::Tensor& value,
                           const tensor::Tensor& output) {
  const int32_t query_seq_len = score.get_dim(0);
  const int32_t kv_seq_len = score.get_dim(1);
  const int32_t head_size = value.get_dim(1);
  arma::fmat score_mat(const_cast<float*>(score.ptr<float>()), kv_seq_len, query_seq_len, false,
                       true);
  arma::fmat value_mat(const_cast<float*>(value.ptr<float>()), head_size, kv_seq_len, false, true);
  arma::fmat out_mat(const_cast<float*>(output.ptr<float>()), head_size, query_seq_len, false,
                     true);

  // Compute attention-weighted output
  // score_mat.t() is (seq_len, seq_len) and value_mat.t() is (seq_len, head_size)
  // Result is (seq_len, head_size) which we transpose back to (head_size, seq_len)
  out_mat = (score_mat.t() * value_mat.t()).t();
}

void mha_kernel_cpu(int32_t layer_idx, int32_t num_layers, int32_t batch_size,
                    int32_t query_seq_len, int32_t kv_seq_len, tensor::Tensor& mha_output,
                    tensor::Tensor& query_tensor, tensor::Tensor& score_tensor,
                    const tensor::Tensor& key_cache_tensor,
                    const tensor::Tensor& value_cache_tensor, void* stream) {
  // key_cache_tensor: (num_layers, batch_size, num_kv_heads, seq_len, head_size)
  // value_cache_tensor: (num_layers, batch_size, num_kv_heads, seq_len, head_size)
  // query_tensor, mha_out: (batch_size, seq_len, num_heads, head_size)
  // score_tensor: (batch_size, num_heads, seq_len, seq_len)
  CHECK((query_seq_len == 1 && kv_seq_len > 1) ||
        (query_seq_len > 1 && query_seq_len == kv_seq_len));
  CHECK(query_tensor.dims_size() == 4);
  CHECK(key_cache_tensor.dims_size() == 5);
  CHECK(value_cache_tensor.dims_size() == 5);
  CHECK(num_layers == key_cache_tensor.get_dim(0));
  CHECK(num_layers == value_cache_tensor.get_dim(0));
  CHECK(batch_size == query_tensor.get_dim(0));

  const int32_t seq_len = query_tensor.get_dim(1);
  const int32_t num_heads = query_tensor.get_dim(2);
  const int32_t num_kv_heads = key_cache_tensor.get_dim(2);

  CHECK(num_heads % num_kv_heads == 0);

  const int32_t head_size = query_tensor.get_dim(3);
  const int32_t hidden_size = num_heads * head_size;
  const int32_t kv_size = num_kv_heads * head_size;
  const int32_t kv_mul = num_heads / num_kv_heads;
  const bool is_causal = (query_seq_len > 1);

  std::shared_ptr<core::MemoryManager> cpu_memory_manager;
  cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance();
  float scale = 1.f / std::sqrt(static_cast<float>(head_size));

  // [batch, seq_len, num_heads, head_size] -> [batch, num_heads, seq_len, head_size]
  // Transpose for more efficient processing
  query_tensor.transpose<float>(1, 2);
  mha_output.transpose<float>(1, 2);

  for (int32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int32_t head_idx = 0; head_idx < num_heads; head_idx++) {
      // For grouped query attention, map each query head to its corresponding kv head
      int32_t kv_head_idx = head_idx / kv_mul;

      // Extract sub-tensors for the current batch and head
      // key_cache[layer_idx, batch_idx, kv_head_idx]: (seq_len, head_size)
      size_t kv_cache_offset = key_cache_tensor.get_offset(layer_idx, batch_idx, kv_head_idx, 0, 0);
      // q[batch_idx, head_idx]: (seq_len, head_size)
      size_t q_offset = query_tensor.get_offset(batch_idx, head_idx, 0, 0);
      // score[batch_idx, head_idx]
      size_t score_offset = score_tensor.get_offset(batch_idx, head_idx, 0, 0);

      // Create views into the tensor data
      tensor::Tensor query(query_tensor.data_type(), query_seq_len, head_size, false, nullptr,
                           query_tensor.ptr<float>() + q_offset);
      tensor::Tensor key(key_cache_tensor.data_type(), kv_seq_len, head_size, false, nullptr,
                         const_cast<float*>(key_cache_tensor.ptr<float>()) + kv_cache_offset);
      tensor::Tensor score(score_tensor.data_type(), query_seq_len, kv_seq_len, false, nullptr,
                           score_tensor.ptr<float>() + score_offset);
      tensor::Tensor value(value_cache_tensor.data_type(), kv_seq_len, head_size, false, nullptr,
                           const_cast<float*>(value_cache_tensor.ptr<float>()) + kv_cache_offset);
      tensor::Tensor output(mha_output.data_type(), query_seq_len, head_size, false, nullptr,
                            mha_output.ptr<float>() + q_offset);

      // Step 1: Compute QK^T for attention scores
      mha_qkT_kernel_cpu(query, key, score, scale, is_causal);

      // Step 2: Apply softmax to get attention weights
      mha_softmax_kernel_cpu(score);

      // Step 3: Multiply attention weights with values
      mha_scorev_kernel_cpu(score, value, output);
    }
  }

  // Transpose back to the original layout to match PyTorch
  // [batch, num_heads, seq_len, head_size] -> [batch, seq_len, num_heads, head_size]
  mha_output.transpose<float>(1, 2);
}
}  // namespace kernel