#include <cstdint>
#include "memory_manager.hpp"
#include "rope_kernel_gpu.cuh"

namespace kernel {

/**
 * @brief CUDA kernel for calculating sine and cosine cache for Rotary Position Embedding
 *
 * This kernel precomputes sine and cosine values used in rotary position embeddings (RoPE).
 * It calculates values for each position up to max_seq_len using frequencies that
 * decay exponentially with the dimension index. These cached values are later used
 * for efficient rotation of query and key vectors during attention computation.
 *
 * @param rope_theta Base value for frequency calculation (typically 10000.0)
 * @param head_size Size of each attention head
 * @param max_seq_len Maximum sequence length to precompute embeddings for
 * @param sin_cache Output array for sine values [max_seq_len, head_size]
 * @param cos_cache Output array for cosine values [max_seq_len, head_size]
 *
 * @note The kernel processes the head_size in pairs, computing values for
 *       dimensions (2i, 2i+1) with the same frequency as per the RoPE formulation
 */
__global__ void sin_cos_cache_calc_gpu_fp32(float rope_theta, int head_size, int max_seq_len,
                                            float* sin_cache, float* cos_cache) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i >= head_size / 2) {
    return;
  }

  for (int pos = 0; pos < max_seq_len; ++pos) {
    // Calculate frequency for paired indices
    float freq = 1.0f / powf(rope_theta, (2.0f * i) / head_size);
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    // Store the same values for the paired indices (2*i and 2*i+1)
    sin_cache[pos * head_size + 2 * i] = fci;
    sin_cache[pos * head_size + 2 * i + 1] = fci;
    cos_cache[pos * head_size + 2 * i] = fcr;
    cos_cache[pos * head_size + 2 * i + 1] = fcr;
  }
}

/**
 * @brief CUDA kernel for applying Rotary Position Embedding (RoPE) to query and key tensors
 *
 * This kernel applies rotary position embeddings to query and key tensors by rotating
 * vector components in 2D subspaces based on token positions. This allows the model to
 * be aware of relative positions without requiring explicit position embeddings to be added.
 * The implementation is optimized for processing multiple batches, positions, and heads in
 * parallel.
 *
 * @param positions Array of position indices
 * @param batch_size Number of sequences in the batch
 * @param seq_len Length of each sequence
 * @param hidden_size Total dimension of query tensor (num_q_heads * head_size)
 * @param key_value_size Total dimension of key tensor (num_k_heads * head_size)
 * @param head_size Size of each attention head
 * @param input_q Query tensor [batch_size, seq_len, num_q_heads, head_size]
 * @param input_k Key tensor [batch_size, seq_len, num_k_heads, head_size]
 * @param sin_cache Precomputed sine values [max_seq_len, head_size]
 * @param cos_cache Precomputed cosine values [max_seq_len, head_size]
 *
 * @note The kernel handles grouped-query attention where num_q_heads may be larger than
 * num_k_heads. It applies rotations in-place, modifying input_q and input_k directly. Each thread
 * processes one (head, pair) combination.
 */
__global__ void rope_kernel_gpu_fp32(const int* q_pos, const int* k_pos, const int batch_size,
                                     const int query_seq_len, const int kv_seq_len,
                                     const int hidden_size, const int key_value_size,
                                     const int head_size, float* input_q, float* input_k,
                                     const float* sin_cache, const float* cos_cache) {
  // Calculate indices
  //   const int batch_idx = blockIdx.z;
  //   const int pos_idx = blockIdx.y;
  //   const int batch_idx = blockIdx.y;
  //   if (batch_idx >= batch_size) return;

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Calculate head and pair indices
  const int num_q_heads = hidden_size / head_size;
  const int num_k_heads = key_value_size / head_size;
  const int total_head_pairs = num_q_heads * (head_size / 2);

  if (tid >= total_head_pairs) {
    return;
  }

  // Extract indices for q tensor
  const int pair_idx = tid % (head_size / 2);
  const int head_idx = tid / (head_size / 2);

  //   // Debug print for thread indices
  //   if (tid == 0) {
  //     printf("pos_idx: %d, batch_idx: %d, query_seq_len: %d, kv_seq_len: %d\n", pos_idx,
  //     batch_idx,
  //            query_seq_len, kv_seq_len);
  //   }
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int pos_idx = 0; pos_idx < query_seq_len; pos_idx++) {
      // if (pos_idx < query_seq_len) {
      const int q_position = q_pos[pos_idx];
      // Get the sin/cos values
      float q_sin_val = sin_cache[q_position * head_size + pair_idx * 2];
      float q_cos_val = cos_cache[q_position * head_size + pair_idx * 2];

      // Calculate the base index for this batch, position, head
      int q_offset = ((batch_idx * query_seq_len + pos_idx) * num_q_heads + head_idx) * head_size;
      // Get the paired x values
      float x1 = input_q[q_offset + pair_idx * 2];
      float x2 = input_q[q_offset + pair_idx * 2 + 1];

      // Apply rotation
      input_q[q_offset + pair_idx * 2] = x1 * q_cos_val - x2 * q_sin_val;
      input_q[q_offset + pair_idx * 2 + 1] = x2 * q_cos_val + x1 * q_sin_val;

      // printf(
      //     "[DEBUG_Q_GPU] batch=%d, pos_idx=%d, head_idx=%d, pair_idx=%d, q_sin_val=%lf, "
      //     "q_cos_val=%lf, x1=%lf, "
      //     "x2=%lf, q[%d, %d, %d, %d]=%lf, q[%d, %d, %d, %d]=%lf\n",
      //     batch_idx, pos_idx, head_idx, pair_idx, q_sin_val, q_cos_val, x1, x2, batch_idx,
      //     pos_idx, head_idx, pair_idx * 2, input_q[q_base_idx + pair_idx * 2], batch_idx,
      //     pos_idx, head_idx, pair_idx * 2 + 1, input_q[q_base_idx + pair_idx * 2 + 1]);
    }

    // If this thread also handles a key head (for grouped queries, not all q heads have k
    // counterparts)
    if (head_idx >= num_k_heads) {
      // return;
      break;
    }
    for (int pos_idx = 0; pos_idx < kv_seq_len; pos_idx++) {
      //   if (pos_idx < kv_seq_len && head_idx < num_k_heads) {
      const int k_position = k_pos[pos_idx];
      float k_sin_val = sin_cache[k_position * head_size + pair_idx * 2];
      float k_cos_val = cos_cache[k_position * head_size + pair_idx * 2];
      // Calculate the base index for this batch, position, head in k tensor
      int k_offset = ((batch_idx * kv_seq_len + pos_idx) * num_k_heads + head_idx) * head_size;
      // Get the paired x values
      float x1 = input_k[k_offset + pair_idx * 2];
      float x2 = input_k[k_offset + pair_idx * 2 + 1];

      // Apply rotation
      input_k[k_offset + pair_idx * 2] = x1 * k_cos_val - x2 * k_sin_val;
      input_k[k_offset + pair_idx * 2 + 1] = x2 * k_cos_val + x1 * k_sin_val;
      // printf(
      //     "[DEBUG_K_GPU] batch=%d, pos_idx=%d, head_idx=%d, pair_idx=%d, k_sin_val=%lf, "
      //     "k_cos_val=%lf, x1=%lf, "
      //     "x2=%lf, k[%d, %d, %d, %d]=%lf, k[%d, %d, %d, %d]=%lf\n",
      //     batch_idx, pos_idx, head_idx, pair_idx, k_sin_val, k_cos_val, x1, x2, batch_idx,
      //     pos_idx, head_idx, pair_idx * 2, input_k[k_base_idx + pair_idx * 2], batch_idx,
      //     pos_idx, head_idx, pair_idx * 2 + 1, input_k[k_base_idx + pair_idx * 2 + 1]);
    }
  }
}

/**
 * @brief Computes sine and cosine cache for Rotary Position Embedding on GPU
 *
 * This function launches a CUDA kernel to precalculate sine and cosine values
 * for rotary position embeddings. These precomputed values are used during
 * inference to efficiently rotate query and key vectors.
 *
 * @param rope_theta Base value for frequency calculation (typically 10000.0)
 * @param head_size Size of each attention head
 * @param max_seq_len Maximum sequence length to precompute embeddings for
 * @param sin_cache Output tensor to store sine values [max_seq_len, head_size]
 * @param cos_cache Output tensor to store cosine values [max_seq_len, head_size]
 * @param stream Optional CUDA stream for asynchronous execution
 *
 * @note The function launches a kernel with grid dimensions optimized for the
 *       head_size, with each thread handling a pair of dimensions
 */
void sin_cos_cache_calc_gpu(float rope_theta, int head_size, int max_seq_len,
                            const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                            void* stream) {
  CHECK_EQ(sin_cache.is_empty(), false);
  CHECK_EQ(cos_cache.is_empty(), false);
  int block_size = 256;
  int grid_size = (head_size / 2 + block_size - 1) / block_size;

  if (stream) {
    auto stream_ = static_cast<cudaStream_t>(stream);
    sin_cos_cache_calc_gpu_fp32<<<grid_size, block_size, 0, stream_>>>(
        rope_theta, head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
        const_cast<float*>(cos_cache.ptr<float>()));
  } else {
    sin_cos_cache_calc_gpu_fp32<<<grid_size, block_size>>>(
        rope_theta, head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
        const_cast<float*>(cos_cache.ptr<float>()));
  }
}

/**
 * @brief Applies Rotary Position Embedding (RoPE) to query and key tensors on GPU
 *
 * This function launches a CUDA kernel to apply rotary position embeddings to
 * the query and key tensors used in attention mechanisms. It uses a 3D grid to
 * efficiently process multiple batches, positions, and heads in parallel.
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
 * @param stream Optional CUDA stream for asynchronous execution
 *
 * @note The function supports grouped-query attention where the number of query heads
 *       may be larger than the number of key-value heads. The 3D grid is configured with
 *       dimensions for head pairs (x), sequence positions (y), and batch size (z) for
 *       optimal parallelization.
 */
void rope_kernel_gpu(int32_t hidden_size, int32_t key_value_size, int32_t head_size,
                     const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                     const tensor::Tensor& input_q_pos, const tensor::Tensor& input_k_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) {
  const int batch_size = input_q.get_dim(0);
  const int query_seq_len = input_q_pos.size();
  const int kv_seq_len = input_k_pos.size();
  const int num_q_heads = hidden_size / head_size;
  const int pairs_per_head = head_size / 2;

  const int total_head_pairs = num_q_heads * pairs_per_head;
  const int block_size = 256;
  const int grid_size_x = (total_head_pairs + block_size - 1) / block_size;
  //   const int grid_size_y = kv_seq_len;
  //   const int grid_size_y = query_seq_len;
  //   const int grid_size_z = batch_size;
  const int grid_size_y = 1;
  const int grid_size_z = 1;

  // Print parameters for debugging
  printf("RoPE GPU Kernel Parameters:\n");
  printf("  batch_size: %d\n", batch_size);
  printf("  query_seq_len: %d\n", query_seq_len);
  printf("  kv_seq_len: %d\n", kv_seq_len);
  printf("  hidden_size: %d\n", hidden_size);
  printf("  key_value_size: %d\n", key_value_size);
  printf("  head_size: %d\n", head_size);
  printf("  num_q_heads: %d\n", num_q_heads);
  printf("  pairs_per_head: %d\n", pairs_per_head);
  printf("  total_head_pairs: %d\n", total_head_pairs);
  printf("  grid_size: (%d, %d, %d)\n", grid_size_x, grid_size_y, grid_size_z);
  printf("  block_size: %d\n", block_size);

  dim3 grid_size(grid_size_x, grid_size_y, grid_size_z);
  int* q_pos_ptr = const_cast<int*>(input_q_pos.ptr<int>());
  int* kv_pos_ptr = const_cast<int*>(input_k_pos.ptr<int>());
  float* input_q_ptr = const_cast<float*>(input_q.ptr<float>());
  float* input_k_ptr = const_cast<float*>(input_k.ptr<float>());

  if (stream) {
    auto stream_ = static_cast<cudaStream_t>(stream);
    rope_kernel_gpu_fp32<<<grid_size, block_size, 0, stream_>>>(
        q_pos_ptr, kv_pos_ptr, batch_size, query_seq_len, kv_seq_len, hidden_size, key_value_size,
        head_size, input_q_ptr, input_k_ptr, sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    rope_kernel_gpu_fp32<<<grid_size, block_size>>>(
        q_pos_ptr, kv_pos_ptr, batch_size, query_seq_len, kv_seq_len, hidden_size, key_value_size,
        head_size, input_q_ptr, input_k_ptr, sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}
}  // namespace kernel