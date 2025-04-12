from setuptools import setup, Extension
import pybind11

# Stier fra vcpkg
#vcpkg_path = "C:/Users/andre/PycharmProjects/Master/vcpkg/installed/x64-windows"
#vcpkg_path = "./vcpkg/installed/x64-windows"
#armadillo_include = f"{vcpkg_path}/include"
#armadillo_lib = f"{vcpkg_path}/lib"

ext_modules = [
    Extension(
        "ewma",
        ["ewma.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
    Extension(
        "sqrtm_cpp",
        ["sqrtm_cpp.cpp"],
        include_dirs=[pybind11.get_include(), armadillo_include],
        library_dirs=[armadillo_lib],
        libraries=["armadillo", "openblas", "lapack"],  # Fjernet "lapacke"
        language="c++"
    ),
]

setup(
    name="cpp_extensions",
    ext_modules=ext_modules,
)
