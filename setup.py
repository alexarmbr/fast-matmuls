from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_args = [
    '-std=c++17',
    '-arch=sm_90a',
    '--expt-relaxed-constexpr',
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    '--use_fast_math',
    "-Xptxas=-v",
    "-O3",
    "-lineinfo"
    # "-G"
]

if '-G' in nvcc_args:
    print("COMPILING WITH DEBUG INFO")

setup(
    name='gemm_ext',
    ext_modules=[
        CUDAExtension(
            name='gemm_ext',
            sources=['src/gemm_wrapper.cpp', 'src/kernel1.cu', 'src/kernel2.cu', 'src/kernel3.cu'],
            extra_compile_args={
                'nvcc': nvcc_args
            },
            extra_link_args=['-lcuda'],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)

