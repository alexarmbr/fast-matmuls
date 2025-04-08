from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



nvcc_args = [
    '-std=c++17',
    '-arch=sm_90a',  # Ensure your GPU supports this architecture
    '--expt-relaxed-constexpr',
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    '--use_fast_math',
    "-Xptxas=-v",
    "-O3"
    # "-G"
]

if '-G' in nvcc_args:
    print("COMPILING WITH DEBUG INFO")

setup(
    name='gemm_ext',
    ext_modules=[
        CUDAExtension(
            name='gemm_ext',
            sources=['src/gemm_wrapper.cpp', 'src/kernel1.cu'],
            extra_compile_args={
                # 'cxx': ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
                'nvcc': nvcc_args
            },
            # library_dirs=['/opt/conda/lib/python3.11/site-packages/torch/lib'],
            extra_link_args=['-lcuda'],
            # include_dirs=['path/to/your/include/directory']  # Adjust as needed
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)

