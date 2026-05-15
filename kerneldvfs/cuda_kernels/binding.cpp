#include <torch/extension.h>

torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, double eps);
torch::Tensor silu_forward(torch::Tensor input);
torch::Tensor row_softmax_forward(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
  m.def("silu_forward", &silu_forward, "SiLU forward (CUDA)");
  m.def("row_softmax_forward", &row_softmax_forward, "Row softmax forward (CUDA)");
}
