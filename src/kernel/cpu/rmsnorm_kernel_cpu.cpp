#include "rmsnorm_kernel_cpu.hpp"
#include <armadillo>
#include <cmath>

namespace kernel {
/**
 * @brief Applies Root Mean Square Normalization to input tensor on CPU
 *
 * This function normalizes the input tensor using RMSNorm:
 * output = weight * (input / sqrt(mean(input^2) + epsilon))
 * RMSNorm differs from LayerNorm in that it doesn't subtract the mean and only
 * normalizes by the root-mean-square of the inputs.
 *
 * @param input Input tensor to be normalized
 * @param weight Weight tensor for element-wise scaling after normalization
 * @param output Output tensor to store the normalized result
 * @param stream Unused in CPU implementation but kept for API consistency with GPU version
 *
 * @note Both input and weight tensors must have the same shape
 */
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == core::DeviceType::CPU &&
        weight.device_type() == core::DeviceType::CPU &&
        output.device_type() == core::DeviceType::CPU);

  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  const float* out_ptr = output.ptr<float>();
  const int32_t dim = static_cast<int32_t>(input.size());

  arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
  arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);
  arma::fvec weight_tensor(const_cast<float*>(wei_ptr), dim, false, true);

  const float eps = 1e-5f;
  const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
  const float rsqrt = 1.f / std::sqrt(mean);
  out_tensor = weight_tensor % (rsqrt * in_tensor);
}
}  // namespace kernel