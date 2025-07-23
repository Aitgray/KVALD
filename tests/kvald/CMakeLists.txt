add_executable(test_data_loader test_data_loader.cpp)
add_executable(test_kalman_filter test_kalman_filter.cpp)

target_link_libraries(test_data_loader
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  kvald
)

target_link_libraries(test_kalman_filter
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  kvald
)

add_test(NAME DataLoaderTest COMMAND test_data_loader)
add_test(NAME KalmanFilterTest COMMAND test_kalman_filter)