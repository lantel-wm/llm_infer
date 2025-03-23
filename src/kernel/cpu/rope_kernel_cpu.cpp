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
                            const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                            void* stream) {
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
 * @param input_q_pos Position indices tensor for queries
 * @param input_k_pos Position indices tensor for keys
 * @param sin_cache Precomputed sine values [max_seq_len, head_size]
 * @param cos_cache Precomputed cosine values [max_seq_len, head_size]
 * @param stream Unused in CPU implementation but kept for API consistency with GPU version
 *
 * @note The function applies rotations in place, modifying input_q and input_k directly.
 *       It supports grouped-query attention where num_q_heads may be larger than num_k_heads.
 *       For each position, it processes pairs of dimensions (2i, 2i+1) by applying a 2D rotation
 *       where the rotation angle depends on the position and dimension index.
 */
void rope_kernel_cpu(int32_t hidden_size, int32_t key_value_size, int32_t head_size,
                     const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                     const tensor::Tensor& input_q_pos, const tensor::Tensor& input_k_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) {
  // q.shape: [batch_size, seq_len, hidden_size / head_size, head_size]
  // k.shape: [batch_size, seq_len, key_value_size / head_size, head_size]
  // Get batch_size, num_attention_heads values from input shapes
  const int32_t batch_size = input_q.get_dim(0);
  const int32_t num_q_heads = hidden_size / head_size;
  const int32_t num_k_heads = key_value_size / head_size;
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int pos_idx = 0; pos_idx < input_q_pos.size(); pos_idx++) {
      const int32_t q_position = input_q_pos.index<int32_t>(pos_idx);
      // Handle query tensor
      for (int head_idx = 0; head_idx < num_q_heads; head_idx++) {
        for (int pair_idx = 0; pair_idx < head_size / 2; pair_idx++) {
          float q_sin_val = sin_cache.at<float>(q_position, pair_idx * 2);
          float q_cos_val = cos_cache.at<float>(q_position, pair_idx * 2);

          // Get original x1 and x2 values
          float x1 = input_q.at<float>(batch_idx, pos_idx, head_idx, pair_idx * 2);
          float x2 = input_q.at<float>(batch_idx, pos_idx, head_idx, pair_idx * 2 + 1);

          // Apply rotation
          float* q_ptr = const_cast<float*>(input_q.ptr<float>());
          int q_offset = input_q.get_offset(batch_idx, pos_idx, head_idx, 0);
          q_ptr[q_offset + pair_idx * 2] = x1 * q_cos_val - x2 * q_sin_val;
          q_ptr[q_offset + pair_idx * 2 + 1] = x2 * q_cos_val + x1 * q_sin_val;
          // printf(
          //     "[DEBUG_Q_CPU] batch=%d, pos_idx=%d, head_idx=%d, pair_idx=%d, q_sin_val=%lf, "
          //     "q_cos_val=%lf, "
          //     "x1=%lf, "
          //     "x2=%lf, q[%d, %d, %d, %d]=%lf, q[%d, %d, %d, %d]=%lf\n",
          //     batch_idx, pos_idx, head_idx, pair_idx, q_sin_val, q_cos_val, x1, x2, batch_idx,
          //     pos_idx, head_idx, pair_idx * 2, q_ptr[q_offset + pair_idx * 2], batch_idx,
          //     pos_idx, head_idx, pair_idx * 2 + 1, q_ptr[q_offset + pair_idx * 2 + 1]);
        }
      }
    }

    for (int pos_idx = 0; pos_idx < input_k_pos.size(); pos_idx++) {
      const int32_t k_position = input_k_pos.index<int32_t>(pos_idx);
      // Handle key tensor
      for (int head_idx = 0; head_idx < num_k_heads; head_idx++) {
        for (int pair_idx = 0; pair_idx < head_size / 2; pair_idx++) {
          float k_sin_val = sin_cache.at<float>(k_position, pair_idx * 2);
          float k_cos_val = cos_cache.at<float>(k_position, pair_idx * 2);

          // Get original x1 and x2 values
          float x1 = input_k.at<float>(batch_idx, pos_idx, head_idx, pair_idx * 2);
          float x2 = input_k.at<float>(batch_idx, pos_idx, head_idx, pair_idx * 2 + 1);

          // Apply rotation
          float* k_ptr = const_cast<float*>(input_k.ptr<float>());
          int k_offset = input_k.get_offset(batch_idx, pos_idx, head_idx, 0);

          k_ptr[k_offset + pair_idx * 2] = x1 * k_cos_val - x2 * k_sin_val;
          k_ptr[k_offset + pair_idx * 2 + 1] = x2 * k_cos_val + x1 * k_sin_val;
          // printf(
          //     "[DEBUG_K_CPU] batch=%d, pos_idx=%d, head_idx=%d, pair_idx=%d, k_sin_val=%lf, "
          //     "k_cos_val=%lf, "
          //     "x1=%lf, "
          //     "x2=%lf, k[%d, %d, %d, %d]=%lf, k[%d, %d, %d, %d]=%lf\n",
          //     batch_idx, pos_idx, head_idx, pair_idx, k_sin_val, k_cos_val, x1, x2, batch_idx,
          //     pos_idx, head_idx, pair_idx * 2, k_ptr[k_offset + pair_idx * 2], batch_idx,
          //     pos_idx, head_idx, pair_idx * 2 + 1, k_ptr[k_offset + pair_idx * 2 + 1]);
        }
      }
    }
  }
}

}  // namespace kernel