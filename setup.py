from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "ewma",
        ["ewma.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="ewma",
    ext_modules=ext_modules,
)
