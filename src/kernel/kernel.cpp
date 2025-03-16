#include "kernel.hpp"
#include "cpu/add_kernel_cpu.hpp"

namespace kernel {
AddKernel get_add_kernel(core::DeviceType device_type) {
  if (device_type == core::DeviceType::CPU) {
    return add_kernel_cpu;
    // } else if (device_type == core::DeviceType::GPU) {
    //   return add_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}

}  // namespace kernel
