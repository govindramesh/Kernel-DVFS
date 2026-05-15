#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace {

__global__ void silu_kernel(const float* input, float* output, int elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  float x = input[index];
  output[index] = x / (1.0f + expf(-x));
}

void check_silu_input(const torch::Tensor& input) {
  if (!input.is_cuda()) {
    throw std::runtime_error("silu_forward expects a CUDA tensor");
  }
  if (input.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("silu_forward currently supports float32 tensors only");
  }
  if (!input.is_contiguous()) {
    throw std::runtime_error("silu_forward expects a contiguous tensor");
  }
}

}  // namespace

torch::Tensor silu_forward(torch::Tensor input) {
  check_silu_input(input);
  auto output = torch::empty_like(input);
  int elements = static_cast<int>(input.numel());
  int threads = 256;
  int blocks = (elements + threads - 1) / threads;
  auto stream = at::cuda::getDefaultCUDAStream();
  silu_kernel<<<blocks, threads, 0, stream>>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      elements);
  return output;
}
