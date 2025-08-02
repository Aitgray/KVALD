#!/bin/bash
set -e
export TORCH_CUDA_ARCH_LIST="5.0;8.0;8.6;8.9;9.0"
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S . -B build -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DTorch_DIR="$TORCH_PREFIX/Torch"
cmake --build build -j"$(nproc)"