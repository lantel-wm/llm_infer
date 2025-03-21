
#ifndef EMBEDDING_LAYER_HPP
#define EMBEDDING_LAYER_HPP
#include "layer.hpp"

namespace op {

struct EmbeddingOutput {
  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;
  explicit EmbeddingOutput(tensor::Tensor input_tokens, tensor::Tensor input_embeddings,
                           tensor::Tensor input_token_num)
      : input_tokens(std::move(input_tokens)),
        input_embeddings(std::move(input_embeddings)),
        input_token_num(std::move(input_token_num)) {}
};

class EmbeddingLayer : public LayerParam {
 public:
  explicit EmbeddingLayer(core::DeviceType device_type, int32_t dim, int32_t seq_len,
                          int32_t vocab_size);

  core::Status check() const override;

  core::Status forward() override;

 private:
  int32_t m_dim = 0;
  int32_t m_seq_len = 0;
  int32_t m_vocab_size = 0;
};
}  // namespace op

#endif  // EMBEDDING_LAYER_HPP
