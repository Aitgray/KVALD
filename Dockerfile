# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel AS build

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# 1) OS deps (cached unless this list changes)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git pkg-config \
        libeigen3-dev libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# 2) Copy only what's needed to configure (preserves cache on source changes)
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/
COPY tests/ tests/

# 3) Configure & build (resolve Torch cmake path via python reliably)
RUN bash -c 'set -euo pipefail;     TORCH_PREFIX="$(python -c "import torch; print(torch.utils.cmake_prefix_path)")";     echo "Torch CMake prefix: $TORCH_PREFIX";     export TORCH_CUDA_ARCH_LIST="8.6";     cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DTorch_DIR="$TORCH_PREFIX/Torch" &&     cmake --build build -j"$(nproc)"'

# -------- Slim runtime stage (optional) --------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
      libopencv-core4.5 libopencv-imgproc4.5 libopencv-highgui4.5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /app/build/src/kvald ./kvald
# Bring PyTorch (and deps) along; simplest is copying /opt/conda
COPY --from=build /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

CMD ["./kvald"]
