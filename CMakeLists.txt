cmake_minimum_required(VERSION 3.14)
project(BayesianLearning)

# 1. Tell CMake to build the aGrUM folder locally
add_subdirectory(external/aGrUM)

# 2. Gather your sources
file(GLOB_RECURSE FOO_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# 3. Create your executable
add_executable(foo ${FOO_SOURCE})
target_include_directories(foo PRIVATE src)

# 4. Link aGrUM
target_link_libraries(foo PRIVATE agrumBASE)
