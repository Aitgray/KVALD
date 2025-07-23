# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# 1) OS deps (cached unless this list changes)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git pkg-config \
        libeigen3-dev libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copy files in an order that optimizes Docker's layer caching.
# Files needed for C++ compilation are copied first, based on the original Dockerfile's
# caching strategy. Changes to these files will only invalidate subsequent layers.
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/
COPY tests/ tests/

# Copy the rest of the project files. This creates a complete development
# environment inside the container. Changes to these other files won't
# require reinstalling dependencies or recopying the C++ source files above.
COPY . .

# 3) Start a shell for interactive debugging instead of compiling the code.
CMD ["bash"]