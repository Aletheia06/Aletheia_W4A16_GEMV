from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_w4a16_gemv",
    ext_modules=[
        CUDAExtension(
            name="custom_w4a16_gemv",
            sources=[
                "csrc/binding.cpp",
                "csrc/gemv_w4a16.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)