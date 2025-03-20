#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <functional>
#include "../core/type/type.hpp"
#include "tensor.hpp"

namespace kernel {

using AddKernel =
    std::function<void(const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, void*)>;

using EmbeddingKernel = std::function<void(const tensor::Tensor&, const tensor::Tensor&,
                                           const tensor::Tensor&, int32_t, void*)>;

using MatMulKernel = std::function<void(const tensor::Tensor&, const tensor::Tensor&,
                                        const tensor::Tensor&, float, void*)>;

using MHAKernel = std::function<void(int32_t, int32_t, int32_t, int32_t, int32_t, tensor::Tensor&,
                                     tensor::Tensor&, tensor::Tensor&, const tensor::Tensor&,
                                     const tensor::Tensor&, void*)>;

using RMSNormKernel =
    std::function<void(const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, void*)>;

using RoPEKernel =
    std::function<void(int32_t, int32_t, int32_t, const tensor::Tensor&, const tensor::Tensor&,
                       const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, void*)>;

using SinCosCacheCalcKernel =
    std::function<void(float, int, int, const tensor::Tensor&, const tensor::Tensor&, void*)>;

using SwiGLUKernel =
    std::function<void(const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, void*)>;

AddKernel get_add_kernel(core::DeviceType device_type);

EmbeddingKernel get_embedding_kernel(core::DeviceType device_type);

MatMulKernel get_matmul_kernel(core::DeviceType device_type);

MHAKernel get_mha_kernel(core::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(core::DeviceType device_type);

RoPEKernel get_rope_kernel(core::DeviceType device_type);

SinCosCacheCalcKernel get_sin_cos_cache_calc_kernel(core::DeviceType device_type);

SwiGLUKernel get_swiglu_kernel(core::DeviceType device_type);

}  // namespace kernel

#endif  // KERNEL_HPP
