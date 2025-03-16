#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cuda_runtime.h>

namespace core {
struct CudaConfig {
  cudaStream_t stream = nullptr;
  ~CudaConfig() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
};
}  // namespace core

#endif  // CONFIG_HPP