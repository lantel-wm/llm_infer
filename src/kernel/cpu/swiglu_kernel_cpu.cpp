#include "swiglu_kernel_cpu.hpp"
#include <armadillo>

namespace kernel {
/**
 * @brief Applies SwiGLU activation function to input tensors on CPU
 *
 * This function implements the SwiGLU (Swish-Gated Linear Unit) activation,
 * which is a variant of GLU used in many recent transformer models like Llama.
 * The computation performed is: output = input1 * sigmoid(input1) * input2
 * where the Swish function (x * sigmoid(x)) is applied to input1 before
 * multiplying element-wise with input2.
 *
 * @param input1 First input tensor for Swish activation (gate value)
 * @param input2 Second input tensor to be gated
 * @param output Output tensor to store the result
 * @param stream Unused in CPU implementation but kept for API consistency with GPU version
 *
 * @note The function expects:
 *       - All tensors must have the same shape and size
 *       - All tensors must be on CPU
 *       - sigmoid(x) is implemented as 1/(1+exp(-x))
 *       - The operation is performed element-wise
 */
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  CHECK(input1.device_type() == core::DeviceType::CPU);
  CHECK(input2.device_type() == core::DeviceType::CPU);
  CHECK(output.device_type() == core::DeviceType::CPU);

  arma::fvec input1_vec(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
  arma::fvec input2_vec(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
  arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);

  input1_vec %= (1.0f / (1.0f + arma::exp(-input1_vec)));
  output_vec = input1_vec % input2_vec;
}
}  // namespace kernel