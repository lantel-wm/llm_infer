#include "softmax_kernel_cpu.hpp"
#include <armadillo>

namespace kernel {
/**
 * @brief Applies the softmax function to a tensor in-place on CPU
 *
 * This function computes the softmax of the input tensor:
 * softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
 * The computation is performed in-place, modifying the input tensor.
 *
 * @param input Input tensor to be transformed by softmax (will be modified in-place)
 * @param stream Unused in CPU implementation but kept for API consistency with GPU version
 *
 * @note The function first finds the maximum value to subtract for numerical stability,
 *       then computes exponentials and normalizes by their sum
 */
void softmax_kernel_cpu(const tensor::Tensor& input, void* stream) {
  int32_t size = static_cast<int32_t>(input.size());
  const float* input_ptr = input.ptr<float>();

  float max_value = *std::max_element(input_ptr, input_ptr + size);

  arma::fvec input_mat(const_cast<float*>(input_ptr), size, false, true);
  input_mat = arma::exp(input_mat - max_value);

  float sum_value = arma::sum(input_mat);
  input_mat = input_mat / sum_value;
}

/**
 * @brief Applies the softmax function to a raw float array on CPU
 *
 * This is a convenience wrapper that creates a tensor from the input pointer
 * and then calls the tensor-based softmax implementation.
 *
 * @param input_ptr Pointer to the input float array
 * @param size Number of elements in the input array
 *
 * @note The function modifies the input array in-place
 */
void softmax_kernel_cpu(const float* input_ptr, int32_t size) {
  tensor::Tensor input(core::DataType::FP32, size);
  std::shared_ptr<core::Buffer> buffer =
      std::make_shared<core::Buffer>(size * sizeof(float), nullptr, (void*)input_ptr, true);
  input.assign(buffer);
  return softmax_kernel_cpu(input);
}
}  // namespace kernel