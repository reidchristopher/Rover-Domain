# Run setup
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
from distutils.extension import Extension
from Cython.Distutils import build_ext

#Options.annotate = True

extensions = [Extension('rover_domain', sources=['rover_domain.pyx'], extra_compile_args=['-std=c++11'])]

setup(
    name="rover_domain",
    ext_modules=cythonize(extensions, gdb_debug=True),
    cmdclass={'build_ext': build_ext}
)

