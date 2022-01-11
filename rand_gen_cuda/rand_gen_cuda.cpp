#include <torch/extension.h>

// CUDA forward declarations
std::vector<torch::Tensor> rand_gen_cuda(torch::Tensor offset, torch::Tensor output, unsigned long seed);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> rand_gen(torch::Tensor offset, unsigned long seed) {
  CHECK_INPUT(offset);
  auto output = torch::zeros_like(offset);
  return rand_gen_cuda(offset, output, seed);
  //return {output};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rand_gen", &rand_gen, "rand_gen (CUDA)");
}