#include "memory_manager.hpp"
#include <cuda_runtime.h>

namespace core {

CPUMemoryManager::CPUMemoryManager() : MemoryManager(DeviceType::CPU) {}

void* CPUMemoryManager::allocate(size_t size) const {
  if (!size) {
    return nullptr;
  }
  void* data = malloc(size);
  return data;
}

void CPUMemoryManager::deallocate(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}

std::shared_ptr<CPUMemoryManager> CPUMemoryManagerFactory::instance = nullptr;

GPUMemoryManager::GPUMemoryManager() : MemoryManager(DeviceType::GPU) {}

void* GPUMemoryManager::allocate(size_t size) const {
  int device_id = -1;
  cudaError_t state = cudaGetDevice(&device_id);
  CHECK(state == cudaSuccess);

  if (size > 1024 * 1024) {  // size > 1MB: big buffer
    auto& big_buffers = m_big_buffers_map[device_id];
    int sel_id = -1;
    // Search for the smallest available buffer that:
    // 1. Is large enough to fit the requested size
    // 2. Is not currently in use (not busy)
    // 3. Won't waste more than 1MB of space (size difference < 1MB)
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].size >= size && !big_buffers[i].busy &&
          big_buffers[i].size - size < 1 * 1024 * 1024) {
        if (sel_id == -1 || big_buffers[sel_id].size > big_buffers[i].size) {
          sel_id = i;
        }
      }
    }
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return big_buffers[sel_id].data;
    }

    // Create new large buffer
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, size);
    if (cudaSuccess != state) {
      char buf[256];
      snprintf(buf, 256,
               "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
               "left on  device.",
               size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    big_buffers.emplace_back(ptr, size, true);
    return ptr;
  }

  // size <= 1MB: small buffer
  auto& cuda_buffers = m_cuda_buffers_map[device_id];
  // Search for an existing buffer that:
  // 1. Is large enough to fit the requested size
  // 2. Is not currently in use (not busy)
  // If found, mark it as busy and update the free memory counter
  for (int i = 0; i < cuda_buffers.size(); i++) {
    if (cuda_buffers[i].size >= size && !cuda_buffers[i].busy) {
      cuda_buffers[i].busy = true;
      m_fragments_size[device_id] -= cuda_buffers[i].size;
      return cuda_buffers[i].data;
    }
  }

  // Create new small buffer
  void* ptr = nullptr;
  state = cudaMalloc(&ptr, size);
  if (cudaSuccess != state) {
    char buf[256];
    snprintf(buf, 256,
             "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
             "left on  device.",
             size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  cuda_buffers.emplace_back(ptr, size, true);
  return ptr;
}

void GPUMemoryManager::deallocate(void* ptr) const {
  if (ptr == nullptr) {
    return;
  }

  if (m_cuda_buffers_map.empty()) {
    return;
  }

  cudaError_t state = cudaSuccess;
  // Garbage collection:
  // Iterate through all GPU devices and their buffer maps
  // For each device, if size of fragments exceed 1GB,
  // free all available buffers to reclaim memory and only keep busy ones
  for (auto& it : m_cuda_buffers_map) {
    if (m_fragments_size[it.first] > 1024 * 1024 * 1024) {
      auto& cuda_buffers = it.second;
      std::vector<GPUMemoryBuffer> temp;
      for (int i = 0; i < cuda_buffers.size(); i++) {
        if (!cuda_buffers[i].busy) {
          state = cudaSetDevice(it.first);
          state = cudaFree(cuda_buffers[i].data);
          CHECK(state == cudaSuccess)
              << "Error: CUDA error when release memory on device " << it.first;
        } else {
          temp.push_back(cuda_buffers[i]);
        }
      }
      cuda_buffers.clear();
      it.second = temp;
      m_fragments_size[it.first] = 0;
    }
  }

  for (auto& it : m_cuda_buffers_map) {
    auto& cuda_buffers = it.second;
    for (int i = 0; i < cuda_buffers.size(); i++) {
      if (cuda_buffers[i].data == ptr) {
        m_fragments_size[it.first] += cuda_buffers[i].size;
        cuda_buffers[i].busy = false;
        return;
      }
    }
    auto& big_buffers = m_big_buffers_map[it.first];
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
        return;
      }
    }
  }
  state = cudaFree(ptr);
  CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}
std::shared_ptr<GPUMemoryManager> GPUMemoryManagerFactory::instance = nullptr;

void MemoryManager::memcpy(const void* src, void* dst, size_t count, MemcpyKind kind, void* stream,
                           bool sync) const {
  CHECK_NE(src, nullptr);
  CHECK_NE(dst, nullptr);
  if (!count) {
    return;
  }

  cudaStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<CUstream_st*>(stream);
  }
  if (kind == MemcpyKind::MemcpyCPU2CPU) {
    std::memcpy(dst, src, count);
  } else if (kind == MemcpyKind::MemcpyCPU2GPU) {
    if (!stream_) {
      cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream_);
    }
  } else if (kind == MemcpyKind::MemcpyGPU2CPU) {
    if (!stream_) {
      cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream_);
    }
  } else if (kind == MemcpyKind::MemcpyGPU2GPU) {
    if (!stream_) {
      cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream_);
    }
  } else {
    LOG(FATAL) << "Unknown memcpy kind: " << int(kind);
  }
  if (sync) {
    cudaDeviceSynchronize();
  }
}

void MemoryManager::memset0(void* ptr, size_t size, void* stream, bool sync) {
  CHECK(m_device_type != DeviceType::Unknown);
  if (m_device_type == DeviceType::CPU) {
    std::memset(ptr, 0, size);
  } else {
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      cudaMemsetAsync(ptr, 0, size, stream_);
    } else {
      cudaMemset(ptr, 0, size);
    }
    if (sync) {
      cudaDeviceSynchronize();
    }
  }
}

}  // namespace core