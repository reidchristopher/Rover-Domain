# 
# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# from Cython.Compiler import Options
# 
# Options.annotate = True
# 
# setup(
#     name = 'Test app',
#     ext_modules=[
#     Extension('rover_domain',
#         sources=['rover_domain.pyx'],
#                 extra_compile_args=['-std=c++11'],
#                 language='c++')
#     ],
#     cmdclass = {'build_ext': build_ext}
# )

import sys
old_sys_argv = sys.argv[:]
sys.argv = ['', 'install']

from setuptools import setup 
from setuptools.extension import Extension
from Cython.Build import cythonize


ext_modules = [ 
    Extension( 
        "rover_domain",
        ["rover_domain.pyx"],
         extra_compile_args=['-std=c++11'] ) ]
        
setup(ext_modules = cythonize(ext_modules), zip_safe=False)

sys.argv = old_sys_argv