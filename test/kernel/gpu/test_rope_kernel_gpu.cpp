#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include "kernel/cpu/rope_kernel_cpu.hpp"
#include "memory_manager.hpp"
#include "rope_kernel_gpu.cuh"
#include "tensor.hpp"
#include "type.hpp"

namespace kernel {
namespace {

class RopeKernelGPUTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cpu_memory_manager = core::CPUMemoryManagerFactory::get_instance();
    gpu_memory_manager = core::GPUMemoryManagerFactory::get_instance();

    // Initialize random number generator
    rng.seed(42);  // Fixed seed for reproducibility

    // Check if CUDA is available and initialize CUDA context
    int device_count = 0;
    cuda_available = (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0);
  }

  void TearDown() override {
    if (cuda_available) {
      // Ensure all CUDA operations are complete
      cudaDeviceSynchronize();

      // Check for any errors that occurred during the test
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::cerr << "CUDA error during test: " << cudaGetErrorString(err) << std::endl;
      }
    }
  }

  std::shared_ptr<core::CPUMemoryManager> cpu_memory_manager;
  std::shared_ptr<core::GPUMemoryManager> gpu_memory_manager;
  std::mt19937 rng;
  bool cuda_available = false;

  // Helper to generate random tensor
  void generate_random_tensor(tensor::Tensor& tensor) {
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int32_t i = 0; i < tensor.size(); i++) {
      tensor.index<float>(i) = dist(rng);
    }
  }
};

// test sin_cos_cache_calc kernel
TEST_F(RopeKernelGPUTest, TestSinCosCacheCalc) {
  if (!cuda_available) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  const float rope_theta = 1000000.0f;
  const int32_t max_position_embeddings = 512;
  const int32_t head_size = 256;
  tensor::Tensor sin_cpu(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                         cpu_memory_manager, nullptr);
  tensor::Tensor cos_cpu(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                         cpu_memory_manager, nullptr);
  tensor::Tensor sin_gpu(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                         gpu_memory_manager, nullptr);
  tensor::Tensor cos_gpu(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                         gpu_memory_manager, nullptr);

  sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, sin_cpu, cos_cpu);
  sin_cos_cache_calc_gpu(rope_theta, head_size, max_position_embeddings, sin_gpu, cos_gpu, nullptr);

  sin_gpu.to_cpu();
  cos_gpu.to_cpu();

  for (int i = 0; i < max_position_embeddings; i++) {
    for (int j = 0; j < head_size; j++) {
      EXPECT_NEAR(sin_cpu.at<float>(i, j), sin_gpu.at<float>(i, j), 1e-5f);
      EXPECT_NEAR(cos_cpu.at<float>(i, j), cos_gpu.at<float>(i, j), 1e-5f);
    }
  }
}

TEST_F(RopeKernelGPUTest, TestRopePrefill) {
  if (!cuda_available) {
    GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
  }

  const float rope_theta = 1000000.0f;
  const int32_t max_position_embeddings = 512;
  const int32_t hidden_size = 512;
  const int32_t num_attention_heads = 16;
  const int32_t num_key_value_heads = 4;
  const int32_t head_size = hidden_size / num_attention_heads;
  const int32_t key_value_size = head_size * num_key_value_heads;
  const int32_t batch_size = 2;
  const int32_t seq_len = 64;

  tensor::Tensor sin_cpu(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                         cpu_memory_manager, nullptr);
  tensor::Tensor cos_cpu(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                         cpu_memory_manager, nullptr);
  tensor::Tensor q_cpu(core::DataType::FP32, {batch_size, seq_len, num_attention_heads, head_size},
                       true, cpu_memory_manager, nullptr);
  tensor::Tensor k_cpu(core::DataType::FP32, {batch_size, seq_len, num_key_value_heads, head_size},
                       true, cpu_memory_manager, nullptr);
  tensor::Tensor pos_cpu(core::DataType::INT32, seq_len, true, cpu_memory_manager, nullptr);

  tensor::Tensor sin_gpu(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                         gpu_memory_manager, nullptr);
  tensor::Tensor cos_gpu(core::DataType::FP32, {max_position_embeddings, head_size}, true,
                         gpu_memory_manager, nullptr);
  tensor::Tensor pos_gpu;
  tensor::Tensor q_gpu;
  tensor::Tensor k_gpu;

  sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, sin_cpu, cos_cpu);
  sin_cos_cache_calc_gpu(rope_theta, head_size, max_position_embeddings, sin_gpu, cos_gpu);

  generate_random_tensor(q_cpu);
  generate_random_tensor(k_cpu);

  for (int pos_idx = 0; pos_idx < seq_len; pos_idx++) {
    pos_cpu.at<int32_t>(pos_idx) = pos_idx;
  }

  rope_kernel_cpu(hidden_size, key_value_size, head_size, q_cpu, k_cpu, pos_cpu, pos_cpu, sin_cpu,
                  cos_cpu, nullptr);

  pos_gpu = pos_cpu.clone();
  q_gpu = q_cpu.clone();
  k_gpu = k_cpu.clone();
  pos_gpu.to_cuda();
  q_gpu.to_cuda();
  k_gpu.to_cuda();

  rope_kernel_gpu(hidden_size, key_value_size, head_size, q_gpu, k_gpu, pos_gpu, pos_gpu, sin_gpu,
                  cos_gpu, nullptr);

  q_gpu.to_cpu();
  k_gpu.to_cpu();

  // Synchronize to ensure the copy is complete
  cudaDeviceSynchronize();

  // Calculate RMSE and max difference for q tensor
  float q_sum_squared_diff = 0.0f;
  float q_max_diff = 0.0f;
  for (int i = 0; i < q_cpu.size(); i++) {
    float diff = std::abs(q_cpu.index<float>(i) - q_gpu.index<float>(i));
    q_sum_squared_diff += diff * diff;
    q_max_diff = std::max(q_max_diff, diff);
  }
  float q_rmse = std::sqrt(q_sum_squared_diff / q_cpu.size());

  // Calculate RMSE and max difference for k tensor
  float k_sum_squared_diff = 0.0f;
  float k_max_diff = 0.0f;
  for (int i = 0; i < k_cpu.size(); i++) {
    float diff = std::abs(k_cpu.index<float>(i) - k_gpu.index<float>(i));
    k_sum_squared_diff += diff * diff;
    k_max_diff = std::max(k_max_diff, diff);
  }
  float k_rmse = std::sqrt(k_sum_squared_diff / k_cpu.size());

  std::cout << "Query tensor - RMSE: " << q_rmse << ", Max diff: " << q_max_diff << std::endl;
  std::cout << "Key tensor - RMSE: " << k_rmse << ", Max diff: " << k_max_diff << std::endl;

  // Verify that differences are within acceptable tolerance
  EXPECT_LT(q_rmse, 1e-5f) << "Query tensor RMSE exceeds tolerance";
  EXPECT_LT(q_max_diff, 1e-4f) << "Query tensor max difference exceeds tolerance";
  EXPECT_LT(k_rmse, 1e-5f) << "Key tensor RMSE exceeds tolerance";
  EXPECT_LT(k_max_diff, 1e-4f) << "Key tensor max difference exceeds tolerance";
}

// TEST_F(RopeKernelGPUTest, TestRopeDecode) {
//   if (!cuda_available) {
//     GTEST_SKIP() << "Skipping GPU kernel tests because no CUDA device is available";
//   }

//   const float rope_theta = 1000000.0f;
//   const int32_t max_position_embeddings = 512;
//   const int32_t hidden_size = 128;
//   const int32_t num_attention_heads = 4;
//   const int32_t num_key_value_heads = 2;
//   const int32_t head_size = hidden_size / num_attention_heads;
//   const int32_t key_value_size = head_size * num_key_value_heads;
//   const int32_t batch_size = 2;
//   const int32_t query_seq_len = 16;
//   const int32_t kv_seq_len = 16;

//   tensor::Tensor sin_cpu(core::DataType::FP32, {max_position_embeddings, head_size}, true,
//                          cpu_memory_manager, nullptr);
//   tensor::Tensor cos_cpu(core::DataType::FP32, {max_position_embeddings, head_size}, true,
//                          cpu_memory_manager, nullptr);
//   tensor::Tensor q_cpu(core::DataType::FP32,
//                        {batch_size, query_seq_len, num_attention_heads, head_size}, true,
//                        cpu_memory_manager, nullptr);
//   tensor::Tensor k_cpu(core::DataType::FP32,
//                        {batch_size, kv_seq_len, num_key_value_heads, head_size}, true,
//                        cpu_memory_manager, nullptr);
//   tensor::Tensor q_pos_cpu(core::DataType::INT32, query_seq_len, true, cpu_memory_manager,
//   nullptr); tensor::Tensor k_pos_cpu(core::DataType::INT32, kv_seq_len, true, cpu_memory_manager,
//   nullptr);

//   tensor::Tensor sin_gpu(core::DataType::FP32, {max_position_embeddings, head_size}, true,
//                          gpu_memory_manager, nullptr);
//   tensor::Tensor cos_gpu(core::DataType::FP32, {max_position_embeddings, head_size}, true,
//                          gpu_memory_manager, nullptr);
//   tensor::Tensor q_pos_gpu, k_pos_gpu, q_gpu, k_gpu;

//   sin_cos_cache_calc_cpu(rope_theta, head_size, max_position_embeddings, sin_cpu, cos_cpu);
//   // sin_cos_cache_calc_gpu(rope_theta, head_size, max_position_embeddings, sin_gpu, cos_gpu,
//   // nullptr);

//   generate_random_tensor(q_cpu);
//   generate_random_tensor(k_cpu);
//   for (int pos_idx = 0; pos_idx < query_seq_len; pos_idx++) {
//     q_pos_cpu.at<int32_t>(pos_idx) = pos_idx;
//   }
//   for (int pos_idx = 0; pos_idx < kv_seq_len; pos_idx++) {
//     k_pos_cpu.at<int32_t>(pos_idx) = pos_idx;
//   }

//   rope_kernel_cpu(hidden_size, key_value_size, head_size, q_cpu, k_cpu, q_pos_cpu, k_pos_cpu,
//                   sin_cpu, cos_cpu, nullptr);

//   q_pos_gpu = q_pos_cpu.clone();
//   k_pos_gpu = k_pos_cpu.clone();
//   q_gpu = q_cpu.clone();
//   k_gpu = k_cpu.clone();
//   q_pos_gpu.to_cuda();
//   k_pos_gpu.to_cuda();
//   q_gpu.to_cuda();
//   k_gpu.to_cuda();

//   rope_kernel_gpu(hidden_size, key_value_size, head_size, q_gpu, k_gpu, q_pos_gpu, k_pos_gpu,
//                   sin_gpu, cos_gpu, nullptr);
//   q_gpu.to_cpu();
//   k_gpu.to_cpu();

//   // Calculate RMSE and max difference for q tensor
//   float q_sum_squared_diff = 0.0f;
//   float q_max_diff = 0.0f;
//   for (int i = 0; i < q_cpu.size(); i++) {
//     float diff = std::abs(q_cpu.index<float>(i) - q_gpu.index<float>(i));
//     q_sum_squared_diff += diff * diff;
//     q_max_diff = std::max(q_max_diff, diff);
//   }
//   float q_rmse = std::sqrt(q_sum_squared_diff / q_cpu.size());

//   // Calculate RMSE and max difference for k tensor
//   float k_sum_squared_diff = 0.0f;
//   float k_max_diff = 0.0f;
//   for (int i = 0; i < k_cpu.size(); i++) {
//     float diff = std::abs(k_cpu.index<float>(i) - k_gpu.index<float>(i));
//     k_sum_squared_diff += diff * diff;
//     k_max_diff = std::max(k_max_diff, diff);
//   }
//   float k_rmse = std::sqrt(k_sum_squared_diff / k_cpu.size());

//   std::cout << "Query tensor - RMSE: " << q_rmse << ", Max diff: " << q_max_diff << std::endl;
//   std::cout << "Key tensor - RMSE: " << k_rmse << ", Max diff: " << k_max_diff << std::endl;

//   // Verify that differences are within acceptable tolerance
//   EXPECT_LT(q_rmse, 1e-5f) << "Query tensor RMSE exceeds tolerance";
//   EXPECT_LT(q_max_diff, 1e-4f) << "Query tensor max difference exceeds tolerance";
//   EXPECT_LT(k_rmse, 1e-5f) << "Key tensor RMSE exceeds tolerance";
//   EXPECT_LT(k_max_diff, 1e-4f) << "Key tensor max difference exceeds tolerance";
// }

}  // namespace
}  // namespace kernel

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  // google::SetStderrLogging(google::GLOG_INFO);
  // google::InstallFailureSignalHandler();
  return RUN_ALL_TESTS();
}