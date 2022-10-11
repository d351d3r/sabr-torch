# SABR Torch Cpp

## Install torch
`wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip`

`unzip libtorch-shared-with-deps-latest.zip`

## Install

 `mkdir build`

 `cd build`

`cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..`

`cmake --build . --config Release`