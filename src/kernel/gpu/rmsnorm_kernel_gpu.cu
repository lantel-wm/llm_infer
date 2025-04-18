#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel_gpu.cuh"

namespace kernel {
/**
 * @brief CUDA kernel for applying Root Mean Square Normalization to a row of data
 *
 * This kernel normalizes a row of input data using RMSNorm:
 * output = weight * (input / sqrt(mean(input^2) + epsilon))
 * It operates efficiently by:
 * 1. Computing the sum of squares across the row
 * 2. Using block reduction to aggregate results
 * 3. Applying normalization with the scaling weights
 *
 * @tparam BLOCK_DIM Block size for the CUDA kernel
 * @param in Pointer to input data
 * @param wei Pointer to weight data for scaling
 * @param out Pointer to output buffer
 * @param size Number of elements in the row
 * @param eps Small epsilon value for numerical stability
 *
 * @note Uses vectorized loads/stores when possible for improved performance
 */
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* input, float* weight, float* output, int size,
                                       float eps) {
  const int tid = threadIdx.x;
  const int batch_idx = blockIdx.x;

  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float* input_batch = input + batch_idx * size;
  float* output_batch = output + batch_idx * size;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(input_batch);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += input_batch[i] * input_batch[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(weight);
  float4* out_pack = reinterpret_cast<float4*>(output_batch);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    output_batch[i] = weight[i] * input[i] * scale;
  }
}

/**
 * @brief Applies Root Mean Square Normalization to input tensor on GPU
 *
 * This function normalizes the input tensor using RMSNorm by launching the appropriate
 * CUDA kernel with the right block size. It selects the optimal kernel based on
 * the input size and optionally uses a CUDA stream for asynchronous execution.
 *
 * @param input Input tensor to be normalized
 * @param weight Weight tensor for element-wise scaling after normalization
 * @param output Output tensor to store the normalized result
 * @param stream Optional CUDA stream for asynchronous execution
 *
 * @note Both input and weight tensors must have the same shape and be on GPU
 */
void rmsnorm_kernel_gpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == core::DeviceType::GPU &&
        weight.device_type() == core::DeviceType::GPU &&
        output.device_type() == core::DeviceType::GPU);

  CHECK_EQ(input.dims_size(), 2);
  CHECK_EQ(output.dims_size(), 2);

  const float eps = 1e-5f;
  const int32_t batch_size = static_cast<int32_t>(input.get_dim(0));
  const int32_t hidden_size = static_cast<int32_t>(input.get_dim(1));
  float* input_ptr = const_cast<float*>(input.ptr<float>());
  float* weight_ptr = const_cast<float*>(weight.ptr<float>());
  float* output_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int block_size = 128;
  const int grid_size = batch_size;
  if (stream) {
    auto stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32<128><<<grid_size, block_size, 0, stream_>>>(input_ptr, weight_ptr, output_ptr,
                                                                hidden_size, eps);
  } else {
    row_rmsnorm_f32<128>
        <<<grid_size, block_size>>>(input_ptr, weight_ptr, output_ptr, hidden_size, eps);
  }
}
}  // namespace kernel