#ifndef RMSNORM_KERNEL_GPU_CUH
#define RMSNORM_KERNEL_GPU_CUH

#include "tensor.hpp"

namespace kernel {
void rmsnorm_kernel_gpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream = nullptr);
}
#endif  // RMSNORM_KERNEL_GPU_CUH
