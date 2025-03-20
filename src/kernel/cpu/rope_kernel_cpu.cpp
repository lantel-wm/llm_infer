#include "rope_kernel_cpu.hpp"
#include <cmath>

namespace kernel {

/**
 * @brief Computes sine and cosine cache for Rotary Position Embedding (RoPE) on CPU
 *
 * This function precalculates sine and cosine values used for rotary position embeddings.
 * RoPE enables relative position awareness by rotating pairs of vector components by
 * position-dependent angles. The cache is computed using frequencies that
 * decay exponentially with dimension index.
 *
 * @param rope_theta Base value for frequency calculation (typically 10000.0)
 * @param head_size Size of each attention head
 * @param max_seq_len Maximum sequence length to precompute embeddings for
 * @param sin_cache Output tensor to store sine values [max_seq_len, head_size]
 * @param cos_cache Output tensor to store cosine values [max_seq_len, head_size]
 *
 * @note The function handles pairs of dimensions (2i, 2i+1) with the same
 *       frequency but different rotations, as per the RoPE formulation
 */
void sin_cos_cache_calc_cpu(float rope_theta, int head_size, int max_seq_len,
                            const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache) {
  float* sin_cache_ptr = const_cast<float*>(sin_cache.ptr<float>());
  float* cos_cache_ptr = const_cast<float*>(cos_cache.ptr<float>());
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int i = 0; i < head_size / 2; ++i) {
      float freq = 1.0f / powf(rope_theta, (2.0f * i) / head_size);
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      // Store the same values for the paired indices (2*i and 2*i+1)
      sin_cache_ptr[pos * head_size + 2 * i] = fci;
      sin_cache_ptr[pos * head_size + 2 * i + 1] = fci;
      cos_cache_ptr[pos * head_size + 2 * i] = fcr;
      cos_cache_ptr[pos * head_size + 2 * i + 1] = fcr;
    }
  }
}

/**
 * @brief Applies Rotary Position Embedding (RoPE) to query and key tensors on CPU
 *
 * This function applies rotary position embeddings to the query and key tensors used
 * in attention mechanisms. RoPE rotates vector components in 2D subspaces based on
 * token positions, allowing the model to be aware of relative positions without
 * requiring explicit position embeddings to be added.
 *
 * @param hidden_size Total dimension of query tensor (num_q_heads * head_size)
 * @param key_value_size Total dimension of key tensor (num_k_heads * head_size)
 * @param head_size Size of each attention head
 * @param input_q Query tensor [batch_size, seq_len, num_q_heads, head_size]
 * @param input_k Key tensor [batch_size, seq_len, num_k_heads, head_size]
 * @param input_pos Position indices tensor
 * @param sin_cache Precomputed sine values [max_seq_len, head_size]
 * @param cos_cache Precomputed cosine values [max_seq_len, head_size]
 * @param stream Unused in CPU implementation but kept for API consistency with GPU version
 *
 * @note The function applies rotations in place, modifying input_q and input_k directly.
 *       It supports grouped-query attention where num_q_heads may be larger than num_k_heads.
 */
void rope_kernel_cpu(int32_t hidden_size, int32_t key_value_size, int32_t head_size,
                     const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                     const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                     const tensor::Tensor& cos_cache, void* stream) {
  // q.shape: [batch_size, seq_len, hidden_size / head_size, head_size]
  // k.shape: [batch_size, seq_len, key_value_size / head_size, head_size]
  // Get batch_size, num_attention_heads values from input shapes
  const int32_t batch_size = input_q.get_dim(0);
  const int32_t num_q_heads = hidden_size / head_size;
  const int32_t num_k_heads = key_value_size / head_size;
  for (int batch = 0; batch < batch_size; batch++) {
    for (int pos_idx = 0; pos_idx < input_pos.size(); pos_idx++) {
      const int32_t pos = input_pos.index<int32_t>(pos_idx);

      // For each head and position
      // Handle query tensor
      for (int n = 0; n < num_q_heads; n++) {
        for (int h = 0; h < head_size / 2; h++) {
          float sin_val = sin_cache.at<float>(pos, h * 2);
          float cos_val = cos_cache.at<float>(pos, h * 2);

          // Get original x1 and x2 values
          float x1 = input_q.at<float>(batch, pos_idx, n, h * 2);
          float x2 = input_q.at<float>(batch, pos_idx, n, h * 2 + 1);

          // Apply rotation
          float* q_ptr = const_cast<float*>(input_q.ptr<float>());
          int q_idx = ((batch * input_q.get_dim(1) + pos_idx) * num_q_heads + n) * head_size;
          q_ptr[q_idx + h * 2] = x1 * cos_val - x2 * sin_val;
          q_ptr[q_idx + h * 2 + 1] = x2 * cos_val + x1 * sin_val;
        }
      }

      // Handle key tensor
      for (int head = 0; head < num_k_heads; head++) {
        for (int i = 0; i < head_size / 2; i++) {
          float sin_val = sin_cache.at<float>(pos, i * 2);
          float cos_val = cos_cache.at<float>(pos, i * 2);

          // Get original x1 and x2 values
          float x1 = input_k.at<float>(batch, pos_idx, head, i * 2);
          float x2 = input_k.at<float>(batch, pos_idx, head, i * 2 + 1);

          // Apply rotation
          float* k_ptr = const_cast<float*>(input_k.ptr<float>());
          int k_idx = ((batch * input_k.get_dim(1) + pos_idx) * num_k_heads + head) * head_size;
          k_ptr[k_idx + i * 2] = x1 * cos_val - x2 * sin_val;
          k_ptr[k_idx + i * 2 + 1] = x2 * cos_val + x1 * sin_val;
        }
      }
    }
  }
}
// ... existing code ...
}  // namespace kernel