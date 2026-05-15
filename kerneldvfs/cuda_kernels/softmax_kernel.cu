#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <stdexcept>

namespace {

__global__ void row_softmax_kernel(const float* input, float* output, int rows, int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float shared[];
  float* max_buffer = shared;
  float* sum_buffer = shared;
  int row_offset = row * cols;

  float local_max = -FLT_MAX;
  for (int col = tid; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, input[row_offset + col]);
  }
  max_buffer[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      max_buffer[tid] = fmaxf(max_buffer[tid], max_buffer[tid + stride]);
    }
    __syncthreads();
  }

  float max_value = max_buffer[0];
  float local_sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    local_sum += expf(input[row_offset + col] - max_value);
  }
  sum_buffer[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_buffer[tid] += sum_buffer[tid + stride];
    }
    __syncthreads();
  }

  float inv_sum = 1.0f / sum_buffer[0];
  for (int col = tid; col < cols; col += blockDim.x) {
    output[row_offset + col] = expf(input[row_offset + col] - max_value) * inv_sum;
  }
}

void check_softmax_input(const torch::Tensor& input) {
  if (!input.is_cuda()) {
    throw std::runtime_error("row_softmax_forward expects a CUDA tensor");
  }
  if (input.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("row_softmax_forward currently supports float32 tensors only");
  }
  if (!input.is_contiguous() || input.dim() != 2) {
    throw std::runtime_error("row_softmax_forward expects a contiguous 2D tensor");
  }
}

}  // namespace

torch::Tensor row_softmax_forward(torch::Tensor input) {
  check_softmax_input(input);
  auto output = torch::empty_like(input);
  int rows = static_cast<int>(input.size(0));
  int cols = static_cast<int>(input.size(1));
  int threads = 256;
  int shared_mem = threads * static_cast<int>(sizeof(float));
  auto stream = at::cuda::getDefaultCUDAStream();
  row_softmax_kernel<<<rows, threads, shared_mem, stream>>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      rows,
      cols);
  return output;
}
