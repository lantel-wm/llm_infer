#ifndef SWIGLU_KERNEL_GPU_CUH
#define SWIGLU_KERNEL_GPU_CUH

#include <tensor.hpp>

namespace kernel {
void swiglu_kernel_gpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream);
}

#endif  // SWIGLU_KERNEL_GPU_CUH
