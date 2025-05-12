"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
Modified for Python 3.11 compatibility
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Configuration pour Python 3.11+
extra_compile_args = ["-O3"]  # Optimisation niveau 3
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]  # Éviter les avertissements d'API dépréciée

package = Extension(
    'bbox', 
    ['box_overlaps.pyx'], 
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
    define_macros=define_macros
)

setup(
    name="bbox",
    ext_modules=cythonize([package], language_level=3),  # Spécification explicite du niveau de langage Python 3
)
