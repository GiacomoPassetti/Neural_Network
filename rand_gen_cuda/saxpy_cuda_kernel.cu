#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

namespace {
template <typename scalar_t>
__global__ void saxpy_cuda_kernel(scalar_t* __restrict__ input, float a, float b, size_t input_len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input_len; i += stride)
        input[i] = a * input[i] + b;
}
} //namespace
std::vector<torch::Tensor> saxpy_cuda( torch::Tensor input, float a, float b) {
    int blockSize = 1024;
    int numBlocks = 256;
    size_t input_len = input.size(0) * input.size(1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "saxpy_forward_cuda", ([&] {
        saxpy_cuda_kernel<scalar_t><<<numBlocks, blockSize>>>( input.data<scalar_t>(), a, b, input_len );
    }));

  return {input};
}