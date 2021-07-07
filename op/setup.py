from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from pathlib import Path

# Usage:
# python setup.py install (or python setup.py bdist_wheel)
# NB: Windows: run from VS2017 x64 Native Tool Command Prompt

rootdir = (Path(__file__).parent / '..' / 'op').resolve()

setup(
    name='upfirdn2d',
    ext_modules=[
        CUDAExtension('upfirdn2d_op',
            [str(rootdir / 'upfirdn2d.cpp'), str(rootdir / 'upfirdn2d_kernel.cu')],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

setup(
    name='fused',
    ext_modules=[
        CUDAExtension('fused',
            [str(rootdir / 'fused_bias_act.cpp'), str(rootdir / 'fused_bias_act_kernel.cu')],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)