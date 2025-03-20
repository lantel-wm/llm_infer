#include "add_kernel_cpu.hpp"
#include <armadillo>

namespace kernel {
/**
 * @brief Performs element-wise addition of two tensors on CPU
 *
 * This function adds two input tensors element-wise and stores the result in the output tensor.
 * It uses the Armadillo library to efficiently perform the vector addition operation.
 *
 * @param input1 First input tensor
 * @param input2 Second input tensor
 * @param output Output tensor to store the result
 * @param stream Unused in CPU implementation but kept for API consistency with GPU version
 *
 * @note The function expects all tensors to be non-empty and of the same size
 */
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  CHECK_EQ(input1.size(), input2.size());
  CHECK_EQ(input1.size(), output.size());

  arma::fvec input_vec1(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
  arma::fvec input_vec2(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
  arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);
  output_vec = input_vec1 + input_vec2;
}

}  // namespace kernel
