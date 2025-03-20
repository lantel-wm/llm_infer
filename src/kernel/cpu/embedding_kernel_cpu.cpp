#include "embedding_kernel_cpu.hpp"
#include "memory_manager.hpp"

namespace kernel {

/**
 * @brief Performs embedding lookup operation on CPU
 *
 * This function takes token indices from the input tensor and retrieves the corresponding
 * embedding vectors from the weight tensor. The retrieved embeddings are copied to the
 * output tensor. This is the first operation in many NLP models, converting token IDs
 * to dense vector representations.
 *
 * @param input Input tensor containing token indices
 * @param weight Weight tensor containing embedding vectors for the vocabulary
 * @param output Output tensor to store the retrieved embedding vectors
 * @param vocab_size Size of the vocabulary (maximum allowed token index)
 * @param stream Unused in CPU implementation but kept for API consistency with GPU version
 *
 * @note The function expects:
 *       - input tensor to contain int32_t token indices
 *       - weight tensor to have shape [vocab_size, embedding_dim]
 *       - output tensor to have enough space to store embeddings for all input tokens
 *       - all token indices to be less than vocab_size
 */
void embedding_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(input.device_type() == core::DeviceType::CPU);

  const auto memory_manager = core::CPUMemoryManagerFactory::get_instance();
  for (int32_t i = 0; i < input_num; ++i) {
    int32_t token = input.at<int32_t>(i);
    if (token >= vocab_size) {
      LOG(FATAL) << "Token index is greater than vocab size.";
    } else {
      float* dst_ptr = const_cast<float*>(output.ptr<float>(i * weight_dim));
      float* src_ptr = const_cast<float*>(weight.ptr<float>(token * weight_dim));
      if (weight.device_type() == core::DeviceType::CPU) {
        memory_manager->memcpy(src_ptr, dst_ptr, weight_dim * sizeof(float),
                               core::MemcpyKind::MemcpyCPU2CPU);
      } else {
        LOG(FATAL) << "Unknown device type of weight tensor in the embedding layer.";
      }
    }
  }
}

}  // namespace kernel