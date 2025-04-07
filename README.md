Debug with ASan:
```Shell
export CMAKE_BUILD_TYPE=Debug
export LD_PRELOAD=$(gcc-13 -print-file-name=libasan.so) 
python setup.py build_ext --inplace
```

Release mode:
```Shell
export CMAKE_BUILD_TYPE=Release
python setup.py build_ext --inplace
```

Run:
```Shell
python -c "import stratego.cpp.stratego_cpp;
```