cmake_minimum_required(VERSION 3.10)

cmake_minimum_required(VERSION 3.10)

add_executable(run_tests test_main.cpp)

target_link_libraries(run_tests
  PRIVATE
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(run_tests)

add_subdirectory(kvald)
