#include "embedding_kernel_gpu.cuh"

namespace kernel {
/**
 * @brief CUDA kernel for embedding lookup operation
 *
 * This kernel retrieves embedding vectors from the weight tensor based on token indices
 * in the input tensor. Each thread block processes one token and copies its corresponding
 * embedding vector to the output tensor, with threads within a block cooperating to
 * copy different parts of the embedding vector in parallel.
 *
 * @param vocab_size Size of the vocabulary (maximum allowed token index)
 * @param token_num Number of tokens to process
 * @param embedding_dim Dimension of each embedding vector
 * @param input_ptr Pointer to input tensor containing token indices
 * @param weight_ptr Pointer to weight tensor containing embedding vectors [vocab_size,
 * embedding_dim]
 * @param output_ptr Pointer to output tensor to store the retrieved embeddings [token_num,
 * embedding_dim]
 *
 * @note The kernel performs bounds checking to ensure that token indices are less than vocab_size
 */
__global__ void embedding_kernel_gpu_fp32(int32_t vocab_size, int32_t token_num,
                                          int32_t embedding_dim, const int32_t* input_ptr,
                                          const float* weight_ptr, float* output_ptr) {
  for (int32_t token_idx = blockIdx.x; token_idx < token_num; token_idx += gridDim.x) {
    if (token_idx >= token_num) {
      return;
    }
    int32_t token = input_ptr[token_idx];
    if (token >= vocab_size) {
      return;
    }

    // output: [seq_len, embedding_dim]
    // weight: [vocab_size, embedding_dim]
    float* output_ptr_start = output_ptr + token_idx * embedding_dim;
    const float* weight_ptr_start = weight_ptr + token * embedding_dim;
    // threadIdx.x :  0, 1, 2, .., 127
    // blockDim.x  :  128
    for (int32_t i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
      output_ptr_start[i] = weight_ptr_start[i];
    }
  }
}

/**
 * @brief Performs embedding lookup operation on GPU
 *
 * This function takes token indices from the input tensor and retrieves the corresponding
 * embedding vectors from the weight tensor. The retrieved embeddings are copied to the
 * output tensor. This is typically the first operation in many NLP models, converting
 * token IDs to dense vector representations.
 *
 * @param input Input tensor containing token indices
 * @param weight Weight tensor containing embedding vectors for the vocabulary [vocab_size,
 * embedding_dim]
 * @param output Output tensor to store the retrieved embedding vectors [token_num, embedding_dim]
 * @param vocab_size Size of the vocabulary (maximum allowed token index)
 * @param stream Optional CUDA stream for asynchronous execution
 *
 * @note The function expects:
 *       - All tensors to be on GPU
 *       - input tensor to contain int32_t token indices
 *       - weight tensor to have shape [vocab_size, embedding_dim]
 *       - output tensor to have enough space to store embeddings for all input tokens
 */
void embedding_kernel_gpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == core::DeviceType::GPU &&
        weight.device_type() == core::DeviceType::GPU &&
        output.device_type() == core::DeviceType::GPU);

  const int32_t token_num = static_cast<int32_t>(input.size());
  const int32_t embedding_dim = weight.get_dim(1);

  constexpr int32_t grid_size = 512;
  const int32_t block_size = 256;
  int32_t* in_ptr = const_cast<int32_t*>(input.ptr<int32_t>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    embedding_kernel_gpu_fp32<<<grid_size, block_size, 0, stream_>>>(
        vocab_size, token_num, embedding_dim, in_ptr, wei_ptr, out_ptr);
  } else {
    embedding_kernel_gpu_fp32<<<grid_size, block_size>>>(vocab_size, token_num, embedding_dim,
                                                         in_ptr, wei_ptr, out_ptr);
  }
}
}  // namespace kernel