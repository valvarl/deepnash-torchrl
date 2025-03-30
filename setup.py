import os
import re
import sys
import subprocess
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        build_args = []

        if platform.system() == "Windows":
            cmake_args += ["-A", "x64"]
            build_args += ["--config", cfg]
        else:
            cmake_args += ["-DCMAKE_POSITION_INDEPENDENT_CODE=ON"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--target", ext.name] + build_args, cwd=build_temp)

setup(
    name="stratego_cpp",
    version="0.1",
    author="Your Name",
    description="Stratego C++ bindings using pybind11",
    ext_modules=[CMakeExtension("stratego_cpp", sourcedir="csrc")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)