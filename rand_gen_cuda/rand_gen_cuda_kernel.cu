#include <torch/extension.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state, unsigned long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(seed, 0u, 0u, &state[id]);
}

template <typename scalar_t>
__global__ void rand_gen_cuda_kernel(const curandState *state,
                                const scalar_t* offset,
                                scalar_t* result, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride){
        float x = 0;
        
        /* Copy state to local memory for efficiency */
        curandState localState = *state;
        skipahead(int(offset[i]), &localState);
        
        /* Generate pseudo-random normals */
        x = curand_normal(&localState);

        /* Store results */
        result[i] = x;
    }
}

std::vector<torch::Tensor> rand_gen_cuda(torch::Tensor offset, torch::Tensor output, unsigned long seed){
    int blockSize = 1024;
    int numBlocks = 256;
    size_t input_len = offset.size(0) * offset.size(1);
    curandState *State;
    cudaMallocManaged(&State, sizeof(curandState));
    setup_kernel<<<1, 1>>>(State,seed);


    AT_DISPATCH_FLOATING_TYPES(offset.type(), "rand_gen_cuda", ([&] {
        rand_gen_cuda_kernel<scalar_t><<<numBlocks, blockSize>>>( State, offset.data<scalar_t>(), output.data<scalar_t>(), input_len );
    }));
    cudaFree(State);
    return {output};
}
