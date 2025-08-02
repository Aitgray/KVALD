cmake_minimum_required(VERSION 3.15)
project(kvald CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Manually specify paths for all external libraries
set(OpenCV_INCLUDE_DIRS /usr/include/opencv4)
set(OpenCV_LIBS
    /usr/lib/x86_64-linux-gnu/libopencv_videoio.so
    /usr/lib/x86_64-linux-gnu/libopencv_highgui.so
    /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so
    /usr/lib/x86_64-linux-gnu/libopencv_core.so
)

set(NLOHMANN_JSON_INCLUDE_DIRS /usr/include/nlohmann)

set(CXXOPTS_INCLUDE_DIRS /usr/include)

find_library(CURL_LIBRARY NAMES curl PATHS /usr/lib/x86_64-linux-gnu)
if(NOT CURL_LIBRARY)
    message(FATAL_ERROR "cURL library not found.")
endif()
set(CURL_INCLUDE_DIRS /usr/include/x86_64-linux-gnu/curl)

find_library(GDAL_LIBRARY NAMES gdal PATHS /usr/lib)
if(NOT GDAL_LIBRARY)
    message(FATAL_ERROR "GDAL library not found.")
endif()
set(GDAL_INCLUDE_DIRS /usr/include/gdal)

set(NETCDF_INCLUDE_DIRS /usr/include)
set(NETCDF_LIBRARIES /usr/lib/x86_64-linux-gnu/libnetcdf.so)

find_library(TIFF_LIBRARY NAMES tiff PATHS /usr/lib/x86_64-linux-gnu)
if(NOT TIFF_LIBRARY)
    message(FATAL_ERROR "TIFF library not found.")
endif()
set(TIFF_INCLUDE_DIRS /usr/include)

add_executable(kvald_app
    src/main.cpp
    src/kvald/data_loader.cpp
    src/kvald/kalman_filter.cpp
    src/kvald/model.cpp
    src/kvald/proof_of_concept.cpp
    src/kvald/synthetic_data.cpp
    src/kvald/trainer.cpp
)

target_include_directories(kvald_app PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${NLOHMANN_JSON_INCLUDE_DIRS}
    ${CXXOPTS_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${GDAL_INCLUDE_DIRS}
    ${NETCDF_INCLUDE_DIRS}
    ${TIFF_INCLUDE_DIRS}
)

target_link_libraries(kvald_app
    PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    ${CURL_LIBRARY}
    ${GDAL_LIBRARY}
    ${NETCDF_LIBRARIES}
    ${TIFF_LIBRARY}
)

set_target_properties(kvald_app PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib"
)