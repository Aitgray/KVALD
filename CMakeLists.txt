cmake_minimum_required(VERSION 3.10)

project(KVALD)

# Find LibTorch
find_package(Torch REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG         v1.17.0
)

# cxxopts
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG         v3.1.1
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
FetchContent_MakeAvailable(cxxopts)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include directories
include_directories(include /usr/include/eigen3 ${OpenCV_INCLUDE_DIRS} ${cxxopts_SOURCE_DIR}/include)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Enable testing
enable_testing()
