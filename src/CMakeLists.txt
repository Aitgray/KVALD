add_library(kvald
    kvald/data_loader.cpp
    kvald/kalman_filter.cpp
    kvald/model.cpp
    kvald/synthetic_data.cpp
    kvald/trainer.cpp
    kvald/proof_of_concept.cpp
)

target_include_directories(kvald PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${Eigen3_INCLUDE_DIRS}
    ${Torch_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(kvald PUBLIC
    Eigen3::Eigen
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

add_executable(main main.cpp)
target_link_libraries(main kvald)
