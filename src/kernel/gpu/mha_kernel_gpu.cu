#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel_gpu.cuh"

namespace kernel {

constexpr int block_size_x = 16;
constexpr int block_size_y = 16;
constexpr int block_size = block_size_x * block_size_y;

/**
 * @brief CUDA device function for applying softmax to attention scores
 *
 * This function applies the softmax operation to each row of the attention score matrix.
 * It performs the following steps for numerical stability:
 * 1. Find the maximum value in each row
 * 2. Subtract the maximum from each element
 * 3. Calculate the exponential of each element
 * 4. Normalize by the sum of exponentials
 *
 * @param input Pointer to the attention score matrix [batch_size, num_heads, seq_len, seq_len]
 * @param batch_size Number of sequences in the batch
 * @param num_heads Number of attention heads
 * @param dim0 First dimension size (query sequence length)
 * @param dim1 Second dimension size (key-value sequence length)
 */
__device__ void softmax_kernel_gpu_fp32(float* __restrict__ input, int batch_size, int num_heads,
                                        int dim0, int dim1) {
  // input: (batch_size, num_heads, seq_len, seq_len)
  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int tid = threadIdx.x;
  const int step = blockDim.x;

  for (int row_idx = 0; row_idx < dim0; row_idx++) {
    // input[batch_idx, head_idx, row_idx, 0]
    float* input_row =
        input + row_idx * dim1 + head_idx * dim1 * dim0 + batch_idx * dim1 * dim0 * num_heads;

    // find max value (for numerical stability)
    float row_max = -FLT_MAX;
    for (int col_idx = tid; col_idx < dim1; col_idx += step) {
      if (input_row[col_idx] > row_max) {
        row_max = input_row[col_idx];
      }
    }
    using BlockReduce = cub::BlockReduce<float, block_size>;
    __shared__ BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    row_max = BlockReduce(temp).Reduce(row_max, cub::Max());
    if (threadIdx.x == 0) {
      shared_val = row_max;
    }
    __syncthreads();
    row_max = shared_val;

    float sum = 0.0f;
    for (int col_idx = tid; col_idx < dim1; col_idx += step) {
      input_row[col_idx] = expf(input_row[col_idx] - row_max);
      sum += input_row[col_idx];
    }
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) {
      shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    for (int col_idx = tid; col_idx < dim1; col_idx += step) {
      input_row[col_idx] /= sum;
    }
  }
}

__global__ void test_softmax(float* __restrict__ input, int batch_size, int num_heads, int dim0,
                             int dim1) {
  softmax_kernel_gpu_fp32(input, batch_size, num_heads, dim0, dim1);
}

// exposed for test
void mha_softmax_kernel_gpu(int32_t num_heads, int32_t batch_size, tensor::Tensor& score) {
  dim3 grid_size(num_heads, batch_size);
  test_softmax<<<grid_size, block_size>>>(score.ptr<float>(), batch_size, num_heads,
                                          score.get_dim(0), score.get_dim(1));
}

/**
 * @brief CUDA kernel for multi-head attention in prefill phase
 *
 * This kernel performs the full self-attention calculation for the prefill phase
 * where query_seq_len equals kv_seq_len. It handles all steps:
 * 1. Computing scaled dot-product attention scores
 * 2. Applying causal masking
 * 3. Applying softmax normalization
 * 4. Computing weighted average of value vectors
 *
 * @param batch_size Number of sequences in the batch
 * @param seq_len Sequence length (same for query and key-value)
 * @param max_position_embedding Maximum supported sequence length in KV cache
 * @param num_heads Number of attention heads
 * @param num_kv_heads Number of key-value heads (for grouped-query attention)
 * @param kv_mul Multiplier for key-value sharing (num_heads / num_kv_heads)
 * @param head_size Size of each attention head
 * @param query Query tensor
 * @param score Score tensor to store attention weights
 * @param output Output tensor for attention results
 * @param key_cache KV cache tensor for keys
 * @param value_cache KV cache tensor for values
 */
__global__ void multi_head_attention_kernel_fp32(int32_t batch_size, int32_t seq_len,
                                                 int32_t max_position_embedding, int32_t num_heads,
                                                 int32_t num_kv_heads, int32_t kv_mul,
                                                 int32_t head_size, float* query, float* score,
                                                 float* output, float* key_cache,
                                                 float* value_cache) {
  // prefill, kv_seq_len == query_seq_len
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / kv_mul;
  const int batch_idx = blockIdx.y;

  if (head_idx >= num_heads || kv_head_idx >= num_kv_heads || batch_idx >= batch_size) {
    return;
  }

  float scale = 1.f / sqrtf(static_cast<float>(head_size));

  // Initialize score tensor for this head
  // score: (batch_size, num_heads, seq_len, seq_len)
  // score[batch_idx, head_idx, 0, 0]
  float* score_head =
      score + head_idx * seq_len * seq_len + batch_idx * num_heads * seq_len * seq_len;

  // Calculate attention scores (Q*K^T)
  for (int t_q = threadIdx.x / block_size_y; t_q < seq_len; t_q += block_size_x) {
    // query: (batch_size, seq_len, num_heads, head_size)
    // query[batch_idx, t_q, head_idx, 0]
    float* query_head = query + head_idx * head_size + t_q * num_heads * head_size +
                        batch_idx * seq_len * num_heads * head_size;

    for (int t_kv = threadIdx.x % block_size_y; t_kv < seq_len; t_kv += block_size_y) {
      // Apply causal mask - only attend to positions up to the current one
      if (t_kv > t_q) {
        // Set to a very low value but not -inf to avoid potential numerical issues
        score_head[t_q * seq_len + t_kv] -= 10000.0f;
        continue;
      }

      // key_cache: (batch_size, num_kv_heads, max_position_embedding, head_size)
      // key_cache[batch_idx, kv_head_idx, t_kv, 0]
      float* key_head = key_cache + batch_idx * num_kv_heads * max_position_embedding * head_size +
                        kv_head_idx * max_position_embedding * head_size + t_kv * head_size;

      // Calculate dot product between query and key vectors
      float score_temp = 0.0f;
#pragma unroll
      for (int i = 0; i < head_size; i += 4) {
        float4 key_head_float4 = *reinterpret_cast<float4*>(key_head + i);
        float4 query_head_float4 = *reinterpret_cast<float4*>(query_head + i);
        score_temp += key_head_float4.x * query_head_float4.x;
        score_temp += key_head_float4.y * query_head_float4.y;
        score_temp += key_head_float4.z * query_head_float4.z;
        score_temp += key_head_float4.w * query_head_float4.w;
      }
      //   for (int i = 0; i < head_size; i++) {
      //     score_temp += key_head[i] * query_head[i];
      //   }

      // Apply scaling and store the score
      score_temp *= scale;
      score_head[t_q * seq_len + t_kv] = score_temp;
    }
  }

  // Ensure all threads have finished computing scores
  __syncthreads();

  // Apply softmax to each row of the score matrix
  softmax_kernel_gpu_fp32(score, batch_size, num_heads, seq_len, seq_len);

  // Ensure softmax is complete before computing attention output
  __syncthreads();

  // Calculate attention output (Softmax(Q*K^T)*V)
  for (int t_q = threadIdx.x / block_size_y; t_q < seq_len; t_q += block_size_x) {
    // output: (batch_size, seq_len, num_heads, head_size)
    // output[batch_idx, t_q, head_idx, 0]
    float* output_head = output + batch_idx * seq_len * num_heads * head_size +
                         t_q * num_heads * head_size + head_idx * head_size;

    for (int i = threadIdx.x % block_size_y; i < head_size; i += block_size_y) {
      float value_temp = 0.0f;

      // Calculate weighted sum of values for each attention score
      for (int t_kv = 0; t_kv <= t_q; t_kv++) {  // Only iterate up to t_q (causal attention)
        float attn_score = score_head[t_q * seq_len + t_kv];

        // value_cache: (batch_size, num_kv_heads, max_position_embedding, head_size)
        // value_cache[batch_idx, kv_head_idx, t_kv, i]
        float value_scalar =
            value_cache[batch_idx * num_kv_heads * max_position_embedding * head_size +
                        kv_head_idx * max_position_embedding * head_size + t_kv * head_size + i];

        value_temp += attn_score * value_scalar;
      }

      // Store the final output value
      output_head[i] = value_temp;
    }
  }
}

/**
 * @brief CUDA kernel for multi-head attention in decoding phase
 *
 * This kernel performs the full self-attention calculation for the decoding phase
 * where query_seq_len is 1 and kv_seq_len is the context length. It calculates:
 * 1. Attention scores between current query and all previous keys
 * 2. Softmax normalization of scores
 * 3. Weighted average of value vectors
 *
 * @param batch_size Number of sequences in the batch
 * @param kv_seq_len Number of key-value pairs in the context
 * @param max_position_embedding Maximum supported sequence length in KV cache
 * @param num_heads Number of attention heads
 * @param num_kv_heads Number of key-value heads (for grouped-query attention)
 * @param kv_mul Multiplier for key-value sharing (num_heads / num_kv_heads)
 * @param head_size Size of each attention head
 * @param query Query tensor (single token)
 * @param score Score tensor to store attention weights
 * @param output Output tensor for attention results
 * @param key_cache KV cache tensor for keys
 * @param value_cache KV cache tensor for values
 */
__global__ void decoding_attention_kernel_fp32(int32_t batch_size, int32_t kv_seq_len,
                                               int32_t max_position_embedding, int32_t num_heads,
                                               int32_t num_kv_heads, int32_t kv_mul,
                                               int32_t head_size, float* query, float* score,
                                               float* output, float* key_cache,
                                               float* value_cache) {
  // decode, kv_seq_len > 1, query_seq_len == 1
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / kv_mul;
  const int batch_idx = blockIdx.y;

  if (head_idx >= num_heads || kv_head_idx >= num_kv_heads || batch_idx >= batch_size) {
    return;
  }

  float scale = 1.f / sqrtf(static_cast<float>(head_size));

  // Initialize score tensor for this head
  // score: (batch_size, num_heads, 1, kv_seq_len)
  // score[batch_idx, head_idx, 0, 0]
  float* score_head = score + head_idx * kv_seq_len * 1 + batch_idx * num_heads * kv_seq_len * 1;

  // Calculate attention scores (Q*K^T)
  // query: (batch_size, 1, num_heads, head_size)
  // query[batch_idx, 0, head_idx, 0]
  float* query_head = query + head_idx * head_size + batch_idx * 1 * num_heads * head_size;

  for (int t_kv = threadIdx.x; t_kv < kv_seq_len; t_kv += blockDim.x) {
    // key_cache: (batch_size, num_kv_heads, max_position_embedding, head_size)
    // key_cache[batch_idx, kv_head_idx, t_kv, 0]
    float* key_head = key_cache + batch_idx * num_kv_heads * max_position_embedding * head_size +
                      kv_head_idx * max_position_embedding * head_size + t_kv * head_size;

    // Calculate dot product between query and key vectors
    float score_temp = 0.0f;
#pragma unroll
    for (int i = 0; i < head_size; i += 4) {
      float4 key_head_float4 = *reinterpret_cast<float4*>(key_head + i);
      float4 query_head_float4 = *reinterpret_cast<float4*>(query_head + i);
      score_temp += key_head_float4.x * query_head_float4.x;
      score_temp += key_head_float4.y * query_head_float4.y;
      score_temp += key_head_float4.z * query_head_float4.z;
      score_temp += key_head_float4.w * query_head_float4.w;
    }
    //   for (int i = 0; i < head_size; i++) {
    //     score_temp += key_head[i] * query_head[i];
    //   }

    // Apply scaling and store the score
    score_temp *= scale;
    score_head[t_kv] = score_temp;
  }

  // Ensure all threads have finished computing scores
  __syncthreads();

  // Apply softmax to each row of the score matrix
  softmax_kernel_gpu_fp32(score, batch_size, num_heads, 1, kv_seq_len);

  // Ensure softmax is complete before computing attention output
  __syncthreads();

  // Calculate attention output (Softmax(Q*K^T)*V)
  // output: (batch_size, 1, num_heads, head_size)
  // output[batch_idx, 0, head_idx, 0]
  float* output_head = output + batch_idx * num_heads * head_size + head_idx * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value_temp = 0.0f;

    // Calculate weighted sum of values for each attention score
    for (int t_kv = 0; t_kv < kv_seq_len; t_kv++) {
      float attn_score = score_head[t_kv];

      // value_cache: (batch_size, num_kv_heads, max_position_embedding, head_size)
      // value_cache[batch_idx, kv_head_idx, t_kv, i]
      float value_scalar =
          value_cache[batch_idx * num_kv_heads * max_position_embedding * head_size +
                      kv_head_idx * max_position_embedding * head_size + t_kv * head_size + i];

      value_temp += attn_score * value_scalar;
    }

    // Store the final output value
    output_head[i] = value_temp;
  }
}

/**
 * @brief Performs complete multi-head attention operation on GPU
 *
 * This function executes the full multi-head attention computation by selecting
 * the appropriate kernel based on whether it's in prefill or decoding phase.
 * It handles both the computation of attention scores and the application of
 * those scores to value vectors.
 *
 * @param layer_idx Current layer index
 * @param num_layers Total number of layers
 * @param batch_size Batch size (typically 1 for inference)
 * @param query_seq_len Length of query sequence
 * @param kv_seq_len Length of key-value sequence
 * @param mha_output Output tensor for multi-head attention results
 * @param query_tensor Query tensor
 * @param score_tensor Tensor to store intermediate attention scores
 * @param key_cache_tensor KV cache tensor for keys
 * @param value_cache_tensor KV cache tensor for values
 * @param stream CUDA stream for asynchronous execution
 */
void mha_kernel_gpu(int32_t layer_idx, int32_t num_layers, int32_t batch_size,
                    int32_t query_seq_len, int32_t kv_seq_len, tensor::Tensor& mha_output,
                    tensor::Tensor& query_tensor, tensor::Tensor& score_tensor,
                    const tensor::Tensor& key_cache_tensor,
                    const tensor::Tensor& value_cache_tensor, void* stream) {
  // key_cache_tensor: (num_layers, batch_size, num_kv_heads, max_position_embedding, head_size)
  // value_cache_tensor: (num_layers, batch_size, num_kv_heads, max_position_embedding, head_size)
  // query_tensor, mha_out: (batch_size, query_seq_len, num_heads, head_size)
  // score_tensor: (seq_len, seq_len)
  CHECK((query_seq_len == 1 && kv_seq_len > 1) ||
        (query_seq_len > 1 && query_seq_len == kv_seq_len));
  CHECK(query_tensor.dims_size() == 4);
  CHECK(key_cache_tensor.dims_size() == 5);
  CHECK(value_cache_tensor.dims_size() == 5);
  CHECK(query_tensor.get_dim(1) == query_seq_len);
  CHECK(num_layers == key_cache_tensor.get_dim(0));
  CHECK(num_layers == value_cache_tensor.get_dim(0));
  CHECK(batch_size == query_tensor.get_dim(0));

  const int32_t max_position_embedding = key_cache_tensor.get_dim(3);
  CHECK(kv_seq_len <= max_position_embedding);
  CHECK(query_seq_len <= max_position_embedding);

  const int32_t num_heads = query_tensor.get_dim(2);
  const int32_t num_kv_heads = key_cache_tensor.get_dim(2);

  CHECK(num_heads % num_kv_heads == 0);

  const int32_t head_size = query_tensor.get_dim(3);
  //   const int32_t hidden_size = num_heads * head_size;
  //   const int32_t kv_size = num_kv_heads * head_size;
  const int32_t kv_mul = num_heads / num_kv_heads;
  const bool is_prefill = (query_seq_len > 1);

  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_output.ptr<float>());

  int32_t kv_cache_offset = key_cache_tensor.get_offset(layer_idx, 0, 0, 0, 0);
  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>()) + kv_cache_offset;
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>()) + kv_cache_offset;

  dim3 grid_size(num_heads, batch_size);
  //   dim3 block_size(16, 16);

  if (stream) {
    auto stream_ = static_cast<cudaStream_t>(stream);
    if (is_prefill) {
      multi_head_attention_kernel_fp32<<<grid_size, block_size, 0, stream_>>>(
          batch_size, query_seq_len, max_position_embedding, num_heads, num_kv_heads, kv_mul,
          head_size, query, score, output, key_cache, value_cache);
    } else {
      decoding_attention_kernel_fp32<<<grid_size, block_size, 0, stream_>>>(
          batch_size, kv_seq_len, max_position_embedding, num_heads, num_kv_heads, kv_mul,
          head_size, query, score, output, key_cache, value_cache);
    }
  } else {
    if (is_prefill) {
      multi_head_attention_kernel_fp32<<<grid_size, block_size>>>(
          batch_size, query_seq_len, max_position_embedding, num_heads, num_kv_heads, kv_mul,
          head_size, query, score, output, key_cache, value_cache);
    } else {
      decoding_attention_kernel_fp32<<<grid_size, block_size>>>(
          batch_size, kv_seq_len, max_position_embedding, num_heads, num_kv_heads, kv_mul,
          head_size, query, score, output, key_cache, value_cache);
    }
  }
}

}  // namespace kernel