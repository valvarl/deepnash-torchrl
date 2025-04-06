from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
import platform
import subprocess
from pathlib import Path


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="csrc"):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # 📍 Имя модуля (например, stratego.cpp.stratego_cpp)
        ext_name = ext.name  # stratego.cpp.stratego_cpp
        ext_path = Path(self.get_ext_fullpath(ext_name))  # полный путь, где setuptools ожидает .so

        cfg = "Release"
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        # 📍 Папка, куда должен попасть .so (где его будет искать Python)
        extdir = ext_path.parent.resolve()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",  # путь, куда CMake положит .so
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        if platform.system() != "Windows":
            cmake_args += ["-DCMAKE_POSITION_INDEPENDENT_CODE=ON"]

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--target", "stratego_cpp"], cwd=build_temp)


setup(
    name="deepnash-stratego",
    version="0.1.0",
    author="Varlachev Valery",
    description="Stratego environment with C++ backend and Python training pipeline",
    ext_modules=[CMakeExtension("stratego.cpp.stratego_cpp")],  # 💡 это Python-импорт-путь
    cmdclass={"build_ext": CMakeBuild},
    packages=find_packages("python"),
    package_dir={"": "python"},
    zip_safe=False,
)