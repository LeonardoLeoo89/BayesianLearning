cmake_minimum_required(VERSION 3.14)
project(BayesianLearning)

add_subdirectory(external/aGrUM)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(bayesian_learning ${SOURCES})
target_include_directories(bayesian_learning PRIVATE src)

target_link_libraries(bayesian_learning PRIVATE agrumBASE)
