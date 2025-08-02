FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

WORKDIR /app

# Install OS dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git pkg-config \
    libeigen3-dev libopencv-dev nlohmann-json3-dev libcxxopts-dev \
    libcurl4-openssl-dev libtiff-dev libgdal-dev libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


# Copy project files
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/
COPY tests/ tests/
COPY . .

# Start a shell for interactive debugging instead of compiling the code.
CMD ["bash"]
