from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

NAME = "HyperSpec-utils"
VERSION = "1.0"
DESCR = "HyperSpec's utils functions written in Cython"
URL = "http://varys.ucsd.edu/"
REQUIRES = ['numpy', 'cython']

AUTHOR = "Weihong Xu, Jaeyoung Kang, Wout Bittremieux, Niema Moshiri, and Tajana Rosing"
EMAIL = "wexu@ucsd.edu"

LICENSE = "BSD"

SRC_DIR = "utils"
PACKAGES = [SRC_DIR]

ext = Extension(
    SRC_DIR + ".wrapped",
    [SRC_DIR + "/wrapped.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3"], language="c++"
    )


EXTENSIONS = [ext]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=cythonize(EXTENSIONS)
          )
