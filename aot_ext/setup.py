from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('torch_x_aot_cpu.lltm_cpp',
                                              ['torch_x_aot/cpu/lltm.cpp']),
                   cpp_extension.CUDAExtension('torch_x_aot_cuda.lltm_cpp',
                                               ['torch_x_aot/cuda/lltm_cuda.cpp',
                                                'torch_x_aot/cuda/lltm_cuda_kernel.cu',
                                                ])
                   ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
