from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rand_gen_cuda',
    ext_modules=[
        CUDAExtension('rand_gen_cuda', [
            'rand_gen_cuda.cpp',
            'rand_gen_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })