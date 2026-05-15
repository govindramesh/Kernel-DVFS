#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace {

__global__ void rmsnorm_kernel(
    const float* input,
    const float* weight,
    float* output,
    int rows,
    int cols,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float shared[];
  float sum = 0.0f;
  const int row_offset = row * cols;
  for (int col = tid; col < cols; col += blockDim.x) {
    float value = input[row_offset + col];
    sum += value * value;
  }
  shared[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  float inv_rms = rsqrtf(shared[0] / static_cast<float>(cols) + eps);
  for (int col = tid; col < cols; col += blockDim.x) {
    output[row_offset + col] = input[row_offset + col] * inv_rms * weight[col];
  }
}

void check_rmsnorm_inputs(const torch::Tensor& input, const torch::Tensor& weight) {
  if (!input.is_cuda() || !weight.is_cuda()) {
    throw std::runtime_error("rmsnorm_forward expects CUDA tensors");
  }
  if (input.scalar_type() != torch::kFloat32 || weight.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("rmsnorm_forward currently supports float32 tensors only");
  }
  if (!input.is_contiguous() || !weight.is_contiguous()) {
    throw std::runtime_error("rmsnorm_forward expects contiguous tensors");
  }
  if (input.dim() != 2 || weight.dim() != 1 || input.size(1) != weight.size(0)) {
    throw std::runtime_error("rmsnorm_forward expects input [rows, cols] and weight [cols]");
  }
}

}  // namespace

torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, double eps) {
  check_rmsnorm_inputs(input, weight);
  auto output = torch::empty_like(input);
  int rows = static_cast<int>(input.size(0));
  int cols = static_cast<int>(input.size(1));
  int threads = 256;
  int shared_mem = threads * static_cast<int>(sizeof(float));
  auto stream = at::cuda::getDefaultCUDAStream();
  rmsnorm_kernel<<<rows, threads, shared_mem, stream>>>(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      output.data_ptr<float>(),
      rows,
      cols,
      static_cast<float>(eps));
  return output;
}
