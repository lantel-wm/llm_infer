#include "rmsnorm_kernel_cpu.hpp"
#include <armadillo>
#include <cmath>
#include <cstdint>

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

  CHECK_EQ(input.dims_size(), 2);
  CHECK_EQ(output.dims_size(), 2);

  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  const float* out_ptr = output.ptr<float>();
  const int32_t batch_size = static_cast<int32_t>(input.get_dim(0));
  const int32_t hidden_size = static_cast<int32_t>(input.get_dim(1));

  arma::fmat in_tensor(const_cast<float*>(in_ptr), hidden_size, batch_size, false, true);
  arma::fmat out_tensor(const_cast<float*>(out_ptr), hidden_size, batch_size, false, true);
  arma::fmat weight_tensor(const_cast<float*>(wei_ptr), hidden_size, 1, false, true);

  const float eps = 1e-5f;

  // Process each batch item separately
  for (int32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    // Extract column for current batch item
    arma::subview_col<float> in_vec = in_tensor.col(batch_idx);
    arma::subview_col<float> out_vec = out_tensor.col(batch_idx);

    // Compute RMSNorm for this batch item
    const float mean_square = arma::mean(arma::square(in_vec)) + eps;
    const float rsqrt = 1.f / std::sqrt(mean_square);

    // Apply normalization and scaling
    out_vec = weight_tensor % (rsqrt * in_vec);
  }
}
}  // namespace kernel