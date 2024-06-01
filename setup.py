from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
ext_options = {"compiler_directives": {"profile": True,
                                       "embedsignature":True},
               "annotate": True,
               "gdb_debug":True,}

extensions = [
Extension(
            name="_overlap_intervals",
            sources=["MORE/_overlap_intervals.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O0","-std=c99"],
            extra_link_args=["-std=c99"],
            language="c",
        )
]

setup(
    # "mutlOutRegr/*.pyx","*.pyx"
    ext_modules=cythonize(
        extensions,
        **ext_options,
    ),
    include_dirs = [np.get_include()]
)