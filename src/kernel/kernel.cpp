#include "kernel.hpp"
#include <glog/logging.h>
#include "cpu/add_kernel_cpu.hpp"
#include "cpu/embedding_kernel_cpu.hpp"
#include "cpu/matmul_kernel_cpu.hpp"
#include "cpu/mha_kernel_cpu.hpp"
#include "cpu/rmsnorm_kernel_cpu.hpp"
#include "cpu/rope_kernel_cpu.hpp"
#include "cpu/swiglu_kernel_cpu.hpp"
#include "gpu/add_kernel_gpu.cuh"
#include "gpu/embedding_kernel_gpu.cuh"
#include "gpu/matmul_kernel_gpu.cuh"
#include "gpu/mha_kernel_gpu.cuh"
#include "gpu/rmsnorm_kernel_gpu.cuh"
#include "gpu/rope_kernel_gpu.cuh"
#include "gpu/swiglu_kernel_gpu.cuh"

namespace kernel {
AddKernel get_add_kernel(core::DeviceType device_type) {
  if (device_type == core::DeviceType::CPU) {
    return add_kernel_cpu;
  } else if (device_type == core::DeviceType::GPU) {
    return add_kernel_gpu;
  } else {
    LOG(FATAL) << "Unknown device type for get_add_kernel";
    return nullptr;
  }
}

EmbeddingKernel get_embedding_kernel(core::DeviceType device_type) {
  if (device_type == core::DeviceType::CPU) {
    return embedding_kernel_cpu;
  } else if (device_type == core::DeviceType::GPU) {
    return embedding_kernel_gpu;
  } else {
    LOG(FATAL) << "Unknown device type for get_embedding_kernel";
    return nullptr;
  }
}

MatMulKernel get_matmul_kernel(core::DeviceType device_type) {
  if (device_type == core::DeviceType::CPU) {
    return matmul_kernel_cpu;
  } else if (device_type == core::DeviceType::GPU) {
    return matmul_kernel_gpu;
  } else {
    LOG(FATAL) << "Unknown device type for get_matmul_kernel";
    return nullptr;
  }
}

MHAKernel get_mha_kernel(core::DeviceType device_type) {
  if (device_type == core::DeviceType::CPU) {
    return mha_kernel_cpu;
  } else if (device_type == core::DeviceType::GPU) {
    return mha_kernel_gpu;
  } else {
    LOG(FATAL) << "Unknown device type for get_mha_kernel";
    return nullptr;
  }
}

RMSNormKernel get_rmsnorm_kernel(core::DeviceType device_type) {
  if (device_type == core::DeviceType::CPU) {
    return rmsnorm_kernel_cpu;
  } else if (device_type == core::DeviceType::GPU) {
    return rmsnorm_kernel_gpu;
  } else {
    LOG(FATAL) << "Unknown device type for get_rmsnorm_kernel";
    return nullptr;
  }
}

RoPEKernel get_rope_kernel(core::DeviceType device_type) {
  if (device_type == core::DeviceType::CPU) {
    return rope_kernel_cpu;
  } else if (device_type == core::DeviceType::GPU) {
    return rope_kernel_gpu;
  } else {
    LOG(FATAL) << "Unknown device type for get_rope_kernel";
    return nullptr;
  }
}

SinCosCacheCalcKernel get_sin_cos_cache_calc_kernel(core::DeviceType device_type) {
  if (device_type == core::DeviceType::CPU) {
    return sin_cos_cache_calc_cpu;
  } else if (device_type == core::DeviceType::GPU) {
    return sin_cos_cache_calc_gpu;
  } else {
    LOG(FATAL) << "Unknown device type for get_sin_cos_cache_calc_kernel";
    return nullptr;
  }
}

SwiGLUKernel get_swiglu_kernel(core::DeviceType device_type) {
  if (device_type == core::DeviceType::CPU) {
    return swiglu_kernel_cpu;
  } else if (device_type == core::DeviceType::GPU) {
    return swiglu_kernel_gpu;
  } else {
    LOG(FATAL) << "Unknown device type for get_swiglu_kernel";
    return nullptr;
  }
}

}  // namespace kernel
