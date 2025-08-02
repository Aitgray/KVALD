cmake_minimum_required(VERSION 3.15)
project(kvald CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find LibTorch

find_package(Torch REQUIRED)

# Find OpenCV with necessary components

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui)

# Optional support for GDAL, NetCDF, CURL, TIFF

find_package(GDAL QUIET)
find_package(CURL QUIET)
find_package(TIFF QUIET)
find_package(NetCDF QUIET)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json REQUIRED)
find_package(cxxopts REQUIRED)

# Executable sources

add_executable(kvald_app
main.cpp
kvald/data_loader.cpp
kvald/kalman_filter.cpp
kvald/model.cpp
kvald/proof_of_concept.cpp
kvald/synthetic_data.cpp
kvald/trainer.cpp
)

target_include_directories(kvald_app PRIVATE ../include)

# Link libraries with conditional support

target_link_libraries(kvald_app
PRIVATE
${TORCH_LIBRARIES}
${OpenCV_LIBS}
$<$BOOL:${GDAL_FOUND}:GDAL::GDAL>
$<$BOOL:${CURL_FOUND}:CURL::libcurl>
$<$BOOL:${TIFF_FOUND}:TIFF::TIFF>
$<$BOOL:${NetCDF_FOUND}:NetCDF::NetCDF_C>
Eigen3::Eigen
nlohmann_json::nlohmann_json
cxxopts::cxxopts
)

# Enable rpath so runtime can find libtorch

set_target_properties(kvald_app PROPERTIES
BUILD_RPATH "$ORIGIN/../lib"
)

